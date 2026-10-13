"""Add a single task to a Nextcloud task list.

Real integration: the Nextcloud bridge over HTTP, which fronts the user's
Nextcloud (CalDAV VTODO). The bridge URL/token and the LLM bridge URL/token are
operational values supplied through `ctx.config` (declared in CONTRACT); the task
summary and description are derived from `request.text`.

`GET /tasklists` returns only VTODO-capable task lists (the bridge excludes
event-only calendars and Nextcloud Deck boards), so a task can never be written
to a collection that rejects it. When the account has several task lists the
target is resolved by a single SemIf choice over them; otherwise the sole list is
used. The task is created for real with `POST /tasks` and the real outcome is
reported — a failure is never dressed up as success.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from semif_agent.decisions import DecisionRequest, Option
from semif_agent.skills import ActionResult

INTEGRATION = {
    "service": "nextcloud",
    "transport": "http",
    "config_vars": [
        "nextcloud_bridge_url",
        "nextcloud_bridge_token",
        "llm_bridge_url",
        "llm_bridge_token",
    ],
}

CONTRACT = {
    "nextcloud_bridge_url": "Base URL of the local Nextcloud bridge, e.g. http://127.0.0.1:5230 (no trailing slash needed); used to list task lists and create the task.",
    "nextcloud_bridge_token": "Shared secret for the Nextcloud bridge, if one is configured; sent as the X-Semif-Token header. Leave blank when the bridge requires no auth.",
    "llm_bridge_url": "Base URL of the local LLM bridge, e.g. http://127.0.0.1:5229 (no trailing slash needed); used to phrase the task title and description.",
    "llm_bridge_token": "Shared secret for the LLM bridge, if one is configured; sent as the X-Semif-Token header. Leave blank when the bridge requires no auth.",
}


def _headers(token):
    """Build request headers, sending the bridge token only when one is set."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Semif-Token"] = token
    return headers


def _call(method, url, headers, payload=None, timeout=25):
    """Perform one HTTP call, returning (status, decoded_json_or_error_dict)."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return response.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8"))
        except Exception:
            detail = {"error": str(exc)}
        return exc.code, detail
    except (urllib.error.URLError, ValueError) as exc:
        return None, {"error": str(exc)}


def _parse_json_object(text):
    """Pull the first JSON object out of a model reply, or return None."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _extract_task(ctx, request):
    """Ask the LLM bridge for (summary, description) of the task; else an error."""
    llm_url = str(ctx.config.get("llm_bridge_url", "") or "").rstrip("/")
    headers = _headers(ctx.config.get("llm_bridge_token", ""))
    messages = [
        {
            "role": "system",
            "content": (
                "You turn a user's request into a single to-do task. Reply with "
                'only a JSON object: {"summary": "<short task title>", '
                '"description": "<extra detail, or an empty string>"}.'
            ),
        },
        {"role": "user", "content": request.text or ""},
    ]
    status, payload = _call(
        "POST",
        f"{llm_url}/chat",
        headers,
        {"messages": messages, "max_tokens": 200},
    )
    if status != 200:
        return (
            None,
            None,
            f"could not phrase the task via the LLM bridge at {llm_url}: "
            f"status {status}, {payload.get('error', payload)}",
        )
    data = _parse_json_object(payload.get("text") or "")
    if not data or not str(data.get("summary") or "").strip():
        return (
            None,
            None,
            f"LLM bridge at {llm_url} returned no usable task text: "
            f"{payload.get('text')!r}",
        )
    return (
        str(data["summary"]).strip(),
        str(data.get("description") or "").strip(),
        None,
    )


def act(ctx, request) -> ActionResult:
    """Add one task to the chosen Nextcloud task list and report the result."""
    decisions = []
    base_url = str(ctx.config.get("nextcloud_bridge_url", "") or "").rstrip("/")
    nc_headers = _headers(ctx.config.get("nextcloud_bridge_token", ""))

    # 1. Phrase the task from the request via the LLM bridge.
    summary, description, llm_error = _extract_task(ctx, request)
    if llm_error:
        return ActionResult(
            action_log=llm_error,
            new_state=request.text,
            decisions=decisions,
        )

    # 2. Find the task list to write to (the bridge returns only VTODO-capable
    #    task lists, so an event-only calendar can never be chosen).
    status, payload = _call("GET", f"{base_url}/tasklists", nc_headers)
    if status != 200:
        return ActionResult(
            action_log=(
                f"could not list Nextcloud task lists from {base_url}: "
                f"status {status}, {payload.get('error', payload)}"
            ),
            new_state=request.text,
            decisions=decisions,
        )
    task_lists = [c for c in (payload.get("tasklists") or []) if c.get("name")]
    if not task_lists:
        return ActionResult(
            action_log="no Nextcloud task lists are available to add a task to",
            new_state=request.text,
            decisions=decisions,
        )

    if len(task_lists) == 1:
        target = task_lists[0]["name"]
    else:
        decision = DecisionRequest(
            state=request.text,
            question="Which Nextcloud task list should the new task be added to?",
            options=[
                Option(c["name"], c.get("label") or c["name"]) for c in task_lists
            ],
        )
        result = ctx.engine.call(decision)
        decisions.append((decision, result))
        target = result.selected
    target_label = next(
        (c.get("label") or c["name"] for c in task_lists if c["name"] == target),
        target,
    )

    # 3. Create the task for real through the bridge.
    status, payload = _call(
        "POST",
        f"{base_url}/tasks",
        nc_headers,
        {"calendar": target, "summary": summary, "description": description},
    )
    if status != 200 or not payload.get("ok"):
        return ActionResult(
            action_log=(
                f"failed to add task \"{summary}\" to Nextcloud task list "
                f"\"{target_label}\": status {status}, {payload.get('error', payload)}"
            ),
            new_state=request.text,
            decisions=decisions,
        )

    uid = payload.get("uid", "")
    calendar = payload.get("calendar", target)
    report = (
        f'added task "{summary}" to Nextcloud task list "{target_label}" '
        f"(calendar={calendar}, uid={uid})"
    )
    return ActionResult(action_log=report, new_state=report, decisions=decisions)
