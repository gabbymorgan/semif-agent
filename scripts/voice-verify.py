#!/usr/bin/env python3
"""Check a recorded Piper dataset against its prompts with Whisper.

The recording front end (`scripts/voice-record.py`) resumes by prompt id and
trusts you to accept each take, so a silent false start or a take read from the
wrong line slips in silently. This verifier transcribes every `wavs/<id>.wav`
with faster-whisper and compares it to the prompt recorded for that id in
`metadata.csv`, flagging the ones to re-record.

Statuses:

- ``missing``  — a metadata row whose wav is gone.
- ``empty``    — Whisper heard no speech (a silent / false-start take).
- ``mismatch`` — the transcript differs from the prompt by more than `--max-wer`.
- ``review``   — between `--review-wer` and `--max-wer`: listen, because
  Whisper's own spelling/number variants land here (e.g. "two plus seven" ->
  "2 plus 7"); not treated as a failure.
- ``ok``       — below `--review-wer`.

Run it from the checkout with the voice stack installed:

    .runtime/venv/bin/python scripts/voice-verify.py --model small.en
    .runtime/venv/bin/python scripts/voice-verify.py --json .runtime/voice-train/verify.json
    # mark the flagged takes for re-recording (bad + review) from that report:
    .runtime/venv/bin/python scripts/voice-verify.py \
        --from-json .runtime/voice-train/verify.json --prune --prune-review

`--prune` moves each flagged wav into `<dataset>/rejected/` and drops its
metadata row, so re-running the recorder re-records exactly those prompts; it is
off by default (run once and listen, or keep the `--json` report, first).
`--prune-review` also marks the `review` band. `--from-json` reuses a previous
report instead of transcribing again. Recordings are biometric — never commit
them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
# The voice models live in the checkout's .runtime/hf cache (bootstrap sets this
# for the gateway); default it here so the verifier finds the cached Whisper
# model instead of downloading one into ~/.cache.
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".runtime" / "hf"))

from semif_agent.voice_train import (  # noqa: E402
    BAD_STATUSES,
    SAMPLE_RATE,
    DatasetStore,
    compare_transcript,
    duration_s,
    read_wav,
    resample_pcm16,
)
from semif_agent.voice_transport import WhisperTranscriber  # noqa: E402

#: faster-whisper's model input rate (the transcriber contract).
WHISPER_RATE = 16000


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset", default=str(REPO_ROOT / ".runtime" / "voice-train" / "dataset"),
        help="LJSpeech dataset dir (metadata.csv + wavs/)",
    )
    parser.add_argument("--model", default="base.en",
                        help="faster-whisper model size or path (default: base.en)")
    parser.add_argument("--device", default="cpu", help="inference device (default: cpu)")
    parser.add_argument("--compute-type", default="int8",
                        help="CTranslate2 compute type (default: int8)")
    parser.add_argument("--language", default="en", help="spoken language (default: en)")
    parser.add_argument("--beam-size", type=int, default=5,
                        help="Whisper beam size; higher is more accurate/slower (default: 5)")
    parser.add_argument("--max-wer", type=float, default=0.5,
                        help="word error rate above which a take fails (default: 0.5)")
    parser.add_argument("--review-wer", type=float, default=0.35,
                        help="word error rate at/above which a take is listed for "
                             "review but does not fail (default: 0.35)")
    parser.add_argument("--limit", type=int, default=0,
                        help="verify at most N recordings (for a quick smoke test)")
    parser.add_argument("--all", action="store_true",
                        help="print every recording, not just the flagged ones")
    parser.add_argument("--json", dest="json_path", default="",
                        help="write the full report to this JSON file")
    parser.add_argument("--prune", action="store_true",
                        help="move flagged wavs to <dataset>/rejected/ and drop their rows")
    parser.add_argument("--prune-review", action="store_true",
                        help="also prune the 'review' band (Whisper-variant candidates)")
    parser.add_argument("--reject-dir", default="",
                        help="where --prune moves bad takes (default: <dataset>/rejected)")
    parser.add_argument("--from-json", default="",
                        help="reuse a previous --json report instead of transcribing again")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress the per-file progress and print only the summary")
    return parser.parse_args(argv)


def _resample_to_whisper(pcm16: bytes, rate: int) -> bytes:
    if rate == WHISPER_RATE:
        return pcm16
    return resample_pcm16(pcm16, rate, WHISPER_RATE)


def _transcribe_dataset(store, ids, args) -> tuple[list[dict], float]:
    """Transcribe each wav and compare it to its prompt."""
    transcriber = WhisperTranscriber(
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        beam_size=args.beam_size,
    )
    report: list[dict] = []
    total_seconds = 0.0
    for index, uid in enumerate(ids, start=1):
        reference = store.entries[uid]
        wav = store.wav_path(uid)
        result = {"id": uid, "reference": reference}
        if not wav.is_file():
            result.update(compare_transcript(reference, "", args.max_wer, args.review_wer))
            result["status"] = "missing"
            result["hypothesis"] = ""
            result["seconds"] = 0.0
        else:
            pcm16, rate = read_wav(wav)
            seconds = duration_s(pcm16, rate)
            total_seconds += seconds
            hypothesis = transcriber.transcribe(_resample_to_whisper(pcm16, rate))
            result.update(
                compare_transcript(reference, hypothesis, args.max_wer, args.review_wer)
            )
            result["hypothesis"] = hypothesis
            result["seconds"] = round(seconds, 2)
        report.append(result)
        if not args.quiet and sys.stderr.isatty() and index % 25 == 0:
            print(f"\r  {index}/{len(ids)} …", end="", file=sys.stderr)
    if not args.quiet and sys.stderr.isatty():
        print("\r" + " " * 24 + "\r", end="", file=sys.stderr)
    return report, total_seconds


def _prune(store, ids, reject_dir: Path) -> int:
    """Move each id's wav aside and drop its metadata row; return how many moved."""
    reject_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for uid in ids:
        src = store.wav_path(uid)
        if src.is_file():
            src.replace(reject_dir / src.name)
            moved += 1
        store.remove(uid)
    return moved


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    store = DatasetStore(args.dataset, sample_rate=SAMPLE_RATE)
    if store.count() == 0:
        print(f"no dataset at {store.out} (metadata.csv is empty or missing)", file=sys.stderr)
        print("record one first: python scripts/voice-record.py", file=sys.stderr)
        return 2

    if args.from_json:
        payload = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        report = payload.get("results", [])
        counts = dict(payload.get("counts", {}))
        bad = list(payload.get("bad_ids", []))
        review = list(payload.get("review_ids", []))
        total_seconds = sum(float(r.get("seconds", 0.0)) for r in report)
        args.model = payload.get("model", args.model)
        args.beam_size = payload.get("beam_size", args.beam_size)
    else:
        ids = sorted(store.entries)
        if args.limit:
            ids = ids[: args.limit]
        report, total_seconds = _transcribe_dataset(store, ids, args)
        counts = {}
        bad, review = [], []
        for result in report:
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            if result["status"] in BAD_STATUSES:
                bad.append(result["id"])
            elif result["status"] == "review":
                review.append(result["id"])

    if not args.quiet:
        for result in report:
            status = result["status"]
            if args.all or status in BAD_STATUSES or status == "review":
                print(f"{result['id']}  {status:<8} wer={result['wer']:.2f}  "
                      f"ref: {result['reference']}")
                print(f"         hyp: {result['hypothesis']!r}")

    prune_ids = list(bad) + (list(review) if args.prune_review else [])
    if args.prune and prune_ids:
        reject_dir = Path(args.reject_dir) if args.reject_dir else store.out / "rejected"
        moved = _prune(store, prune_ids, reject_dir)
        print(f"pruned {moved} take(s) to {reject_dir}; re-run the recorder to redo them")

    print(
        f"\nchecked {len(report)} recording(s), {total_seconds / 60:.1f} min audio  "
        f"model={args.model} beam={args.beam_size} "
        f"max-wer={args.max_wer} review-wer={args.review_wer}"
    )
    order = ["ok", "review", "mismatch", "empty", "missing"]
    print("  " + "  ".join(f"{name}: {counts.get(name, 0)}" for name in order))
    if bad:
        print(f"  re-record {len(bad)}: {' '.join(bad)}")
    else:
        print("  no take failed the fidelity check.")
    if review:
        print(f"  review (listen; likely Whisper variants) {len(review)}: {' '.join(review)}")

    if args.json_path:
        payload = {
            "dataset": str(store.out),
            "model": args.model,
            "beam_size": args.beam_size,
            "max_wer": args.max_wer,
            "review_wer": args.review_wer,
            "counts": counts,
            "bad_ids": bad,
            "review_ids": review,
            "results": report,
        }
        Path(args.json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"report: {args.json_path}")

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
