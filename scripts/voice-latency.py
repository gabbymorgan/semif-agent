#!/usr/bin/env python3
"""Break down voice-gateway latency from the run trace.

Reads `data/runs.jsonl` (written by the running gateway) and, for each spoken
interaction, reports the time between the **last word spoken** and the **first
reply audio** split by section:

    last word ── endpointing ── STT ── adapter ── [scheduler: guards + act + assess]
              ── reply routing (humanize) ── TTS synth ── playback start

The voice transport emits `voice_speech_end` at the last speech frame and
`voice_playback_start` right before the first reply samples reach the device;
the scheduler emits `submit`/`actionability`/`category_scope`/`intent_guard`/
`assessed`/`ran`; the service emits `humanize`/`gateway_reply_queued` and the
adapter `gateway_reply_dequeued`/`voice_speak_start`/`voice_tts_synth`.

Usage:
    python scripts/voice-latency.py [data/runs.jsonl] [--json] [--decisions PATH]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def load_rows(path: str) -> list[dict]:
    rows = []
    if not Path(path).is_file():
        return rows
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _first(events: list[dict], kind: str) -> dict | None:
    return next((e for e in events if e.get("kind") == kind), None)


def _ts(event: dict | None) -> float | None:
    return None if event is None else event.get("ts")


def build_interactions(rows: list[dict]) -> list[dict]:
    """Split the trace into spoken interactions, each from speech-end to reply-end."""
    interactions: list[dict] = []
    current: dict | None = None
    for event in rows:
        kind = event.get("kind")
        if kind == "voice_speech_end" or (
            kind == "voice_utterance" and current is None
        ):
            # A new utterance begins; close any prior interaction at its reply
            # end. `voice_utterance` also starts one on legacy traces that
            # predate the `voice_speech_end` probe (last-word boundary unknown).
            if current is not None:
                interactions.append(current)
            current = {"events": [event]}
        elif current is not None:
            current["events"].append(event)
            if kind in ("voice_spoke", "voice_speak_failed", "voice_stt_empty",
                        "voice_stt_failed", "voice_short_utterance"):
                interactions.append(current)
                current = None
    if current is not None:
        interactions.append(current)
    return interactions


def _decisions_timing(decisions_path: str) -> dict[str, list[dict]]:
    """Map run_id -> list of {phase, timing} from the decision log."""
    by_run: dict[str, list[dict]] = {}
    if not Path(decisions_path).is_file():
        return by_run
    with open(decisions_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            extra = row.get("extra") or {}
            run_id = extra.get("run_id")
            timing = extra.get("timing")
            if run_id and timing:
                by_run.setdefault(run_id, []).append(
                    {"phase": extra.get("phase"), **timing}
                )
    return by_run


def analyze(interaction: dict, decisions: dict[str, list[dict]]) -> dict | None:
    events = interaction["events"]
    speech_end = _first(events, "voice_speech_end")
    utterance = _first(events, "voice_utterance")
    transcribed = _first(events, "voice_transcribed")
    submit = _first(events, "submit")
    if utterance is None or submit is None:
        return None  # empty/short utterance: nothing was dispatched
    # On legacy traces without `voice_speech_end`, measure from the utterance.
    origin = speech_end or utterance

    actionability = _first(events, "actionability")
    category_scope = _first(events, "category_scope")
    intent_guard = _first(events, "intent_guard")
    assessed = _first(events, "assessed")
    ran = _first(events, "ran")
    reply_queued = _first(events, "gateway_reply_queued")
    reply_dequeued = _first(events, "gateway_reply_dequeued")
    speak_start = _first(events, "voice_speak_start")
    tts = _first(events, "voice_tts_synth")
    playback = _first(events, "voice_playback_start")
    spoke = _first(events, "voice_spoke")
    humanize = _first(events, "humanize")

    t0 = _ts(origin)
    run_id = submit.get("run_id")

    def delta(a: dict | None, b: dict | None) -> float | None:
        ta, tb = _ts(a), _ts(b)
        return None if ta is None or tb is None else round(tb - ta, 3)

    out = {
        "run_id": run_id,
        "text": submit.get("text"),
        "skill": (ran or {}).get("skill"),
        "summary": (ran or {}).get("summary"),
        "last_word_ts": t0,
        "endpointing_s": delta(speech_end, utterance),
        "stt_s": delta(utterance, transcribed),
        "adapter_s": delta(transcribed, submit),
        "scheduler_s": delta(submit, ran),
        "guard_actionability_s": delta(submit, actionability),
        "guard_category_s": delta(actionability, category_scope),
        "guard_intent_s": delta(category_scope, intent_guard),
        "act_assess_s": delta(intent_guard, assessed),
        "assess_ran_s": delta(assessed, ran),
        "reply_s": delta(ran, playback),
        "humanize_s": (humanize or {}).get("elapsed_s"),
        "tts_synth_s": (tts or {}).get("synth_s"),
        "tts_audio_s": (tts or {}).get("audio_s"),
        "queue_s": delta(reply_queued, reply_dequeued),
        "speak_to_playback_s": delta(speak_start, playback),
        "playback_s": delta(playback, spoke),
        "total_to_first_audio_s": delta(origin, playback),
        "total_to_reply_end_s": delta(origin, spoke),
        "decisions": decisions.get(run_id, []),
    }
    # Anything after `ran` but before playback that is not humanize/TTS is queue.
    if out["reply_s"] is not None:
        accounted = (out["humanize_s"] or 0.0) + (out["tts_synth_s"] or 0.0)
        out["reply_unaccounted_s"] = round(out["reply_s"] - accounted, 3)
    return out


def fmt(value) -> str:
    return "—" if value is None else f"{value:.2f}"


def render(results: list[dict]) -> None:
    for r in results:
        print(f"\nrun {r['run_id']}  {r['text']!r}  skill={r['skill']}")
        print(f"  last word → endpointing done : {fmt(r['endpointing_s'])} s")
        print(f"  STT                          : {fmt(r['stt_s'])} s")
        print(f"  adapter → scheduler submit   : {fmt(r['adapter_s'])} s")
        print(f"  SCHEDULER total              : {fmt(r['scheduler_s'])} s")
        print(f"      actionability guard      : {fmt(r['guard_actionability_s'])} s")
        print(f"      category softmax + scope : {fmt(r['guard_category_s'])} s")
        print(f"      leaf softmax + intent    : {fmt(r['guard_intent_s'])} s")
        print(f"      act + assess:outcome     : {fmt(r['act_assess_s'])} s")
        print(f"      assess → ran             : {fmt(r['assess_ran_s'])} s")
        print(f"  reply (ran → first audio)    : {fmt(r['reply_s'])} s")
        print(f"      humanize                 : {fmt(r['humanize_s'])} s")
        print(f"      TTS synth                : {fmt(r['tts_synth_s'])} s")
        print(f"      other (queue/open)       : {fmt(r.get('reply_unaccounted_s'))} s")
        print(f"  playback (audio duration)    : {fmt(r['playback_s'])} s")
        print(f"  ── LAST WORD → FIRST AUDIO   : {fmt(r['total_to_first_audio_s'])} s")
        if r["decisions"]:
            for d in r["decisions"]:
                print(
                    f"      [{d.get('phase')}] tokens={d.get('input_tokens')} "
                    f"forward={d.get('forward_seconds')} total={d.get('total_seconds')}"
                )

    def agg(key: str) -> str:
        vals = [r[key] for r in results if r.get(key) is not None]
        if not vals:
            return "—"
        return f"median {statistics.median(vals):.2f}s  mean {statistics.mean(vals):.2f}s  (n={len(vals)})"

    print("\n===== aggregate =====")
    for key, label in (
        ("endpointing_s", "endpointing"),
        ("stt_s", "STT"),
        ("scheduler_s", "SCHEDULER"),
        ("guard_actionability_s", "  actionability"),
        ("guard_category_s", "  category softmax+scope"),
        ("guard_intent_s", "  leaf softmax+intent"),
        ("act_assess_s", "  act+assess"),
        ("reply_s", "reply (humanize+TTS)"),
        ("humanize_s", "  humanize"),
        ("tts_synth_s", "  TTS synth"),
        ("total_to_first_audio_s", "TOTAL last word → first audio"),
    ):
        print(f"{label:28s}: {agg(key)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="?", default="data/runs.jsonl")
    parser.add_argument("--decisions", default="data/decisions.jsonl")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    rows = load_rows(args.trace)
    decisions = _decisions_timing(args.decisions)
    results = []
    for interaction in build_interactions(rows):
        analyzed = analyze(interaction, decisions)
        if analyzed is not None:
            results.append(analyzed)
    if args.json:
        print(json.dumps(results, indent=2))
        return 0
    if not results:
        print("no spoken interactions found in", args.trace)
        return 1
    render(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
