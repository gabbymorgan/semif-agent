# Training a custom Piper voice

The voice gateway speaks through **Piper** (`gateway.voice.tts.voice`). You can
replace the stock voice with one trained on your own recordings. This guide is
the end-to-end procedure; the tooling is:

- `scripts/voice-record.py` — record an LJSpeech dataset from a microphone.
- `scripts/voice-verify.py` — transcribe the takes with Whisper and flag the ones
  to re-record.
- `scripts/voice-train.sh` — fine-tune + export a Piper voice on a GPU host.
- `pins.json` `voice` block — the pinned piper1-gpl commit and base checkpoint.

None of this runs as part of `scripts/bootstrap.sh`; training is a one-off,
hardware-specific job. The gateway needs **no code change** to use the result.

## The three hosts

| Role | What it needs |
|---|---|
| **Record** | a microphone + the optional voice stack (`scripts/bootstrap.sh --voice`) |
| **Train** | a CUDA/ROCm GPU + PyTorch |
| **Use** | the voice gateway host |

Only the training host needs PyTorch; the recording host only needs the voice
stack for `sounddevice`. Per-deployment host choices live in the gitignored
`local/` notes, not here.

## Important expectations

- **Piper cannot clone a voice from a clip.** It fine-tunes a new voice from a
  recorded dataset, starting from an existing checkpoint. Budget real recording
  time.
- **Data quality beats quantity.** A quiet room, one consistent mic and
  distance, and a steady tone matter more than raw minutes. ~1 h gives a good
  voice; ~15–30 min gives a usable but rougher one; below ~50 utterances is
  barely worth it.
- **Training is an overnight job**, even for a small dataset (hours on the GPU).

## 1. Record

On the recording host, with the voice stack installed. Three modes:

```sh
# interactive (default): shows each prompt; Enter to record, then accept/redo/skip
.runtime/venv/bin/python scripts/voice-record.py --list-devices
.runtime/venv/bin/python scripts/voice-record.py

# hands-free, spoken: the recorder reads each sentence aloud, you repeat it
.runtime/venv/bin/python scripts/voice-record.py --speak-prompts

# hands-free, read-along: prints each line + beeps (no speech), you read it
.runtime/venv/bin/python scripts/voice-record.py --read-prompts
```

- Prompts default to `scripts/voice-prompts-en.txt` — the 720 public-domain
  Harvard (IEEE) sentences, phonetically balanced and ideal for TTS. Override
  with `--prompts FILE` (one sentence per line; `#` comments skipped).
- `--device`/`--output-device` default to the voice gateway's configured
  `gateway.voice.audio.input_device`/`output_device` from `config.json`; set
  them by PortAudio name substring or index.
- The dataset is written as `metadata.csv` + `wavs/NNNN.wav`, and **resumes**:
  re-running skips prompts already recorded and prints `N of 720 prompt(s) to
  record` up front, so a resume over a nearly-complete dataset is never mistaken
  for a fresh run (`--overwrite` re-records everything).
- **Interactive**: a mic check runs first (reports RMS/peak/dBFS, warns on
  clipping / too-quiet); `Enter` records, then `Enter` accept / `r` redo /
  `p` play / `s` skip / `q` quit.
- **Hands-free** (`--speak-prompts` / `--read-prompts`): no keypresses. Each
  prompt waits for you to speak (up to `--max-seconds`), records until you stop,
  saves, and moves on. It stops after `--max-misses` consecutive silent prompts
  (default 3). `--read-prompts` prints the line and beeps and needs no TTS voice.
- `Ctrl-C` stops and keeps what is saved.

Output is under the gitignored `.runtime/` tree by default. **Recordings are
biometric — never commit them.**

Tuning: `--max-seconds` (hard cap per take), `--silence-ms` (trailing silence
that ends a take), `--vad-threshold` (int16 RMS speech gate), `--read-timeout`
(seconds with no audio frame before a take is abandoned — guards a dropped
device), `--limit N`.

## 1b. Verify (Whisper fidelity check)

The recorder trusts you to accept each take, so a silent false start or a line
read with a word or two wrong slips in silently. Transcribe every take and
compare it to its prompt before training:

```sh
.runtime/venv/bin/python scripts/voice-verify.py --model small.en \
    --json .runtime/voice-train/verify.json
```

- `--model` picks the faster-whisper size; `base.en` is the gateway default and
  is cached, but `small.en` is noticeably more accurate (a one-time ~700 MB
  download into `.runtime/hf`) and is the better choice here — it resolves most
  of base's false positives.
- Each take is classified: `empty` (Whisper heard nothing — a silent/false-start
  take), `mismatch` (word error rate above `--max-wer`, default 0.5), `review`
  (between `--review-wer` 0.35 and `--max-wer` — worth a listen, because
  Whisper's own spelling/number variants land here: "two plus seven" -> "2 plus
  7", "junk yard" -> "junkyard"), and `ok`. Number words are normalized to
  digits before comparison.
- The flagged lines print the prompt and the transcript side by side, so a
  spelling/number variant (`review`) is obvious versus a real word swap.
- `--json FILE` writes the full report (every transcript + WER), `--limit N`
  does a quick smoke test.

To re-record the flagged takes, mark them first — this moves their wavs into
`<dataset>/rejected/` and drops their metadata rows, so the recorder resumes and
re-records exactly those prompts:

```sh
# --prune-review also marks the 'review' band; drop it to only redo hard fails
.runtime/venv/bin/python scripts/voice-verify.py \
    --from-json .runtime/voice-train/verify.json --prune --prune-review

.runtime/venv/bin/python scripts/voice-record.py --read-prompts   # only the gaps
```

Nothing is deleted (the rejected wavs stay under `dataset/rejected/`), and
re-running the verifier after re-recording confirms the fixes. Recordings are
biometric — never commit them.

## 2. Train

Copy the dataset to the training host, then run the trainer there. It installs
everything under `.runtime/voice-train/` (gitignored) and fine-tunes from the
checkpoint pinned in `pins.json`.

```sh
rsync -av .runtime/voice-train/dataset <train-host>:'<checkout>/.runtime/voice-train/dataset'

# on the training host, from the checkout:
scripts/voice-train.sh --name en_US-myvoice-medium \
    --dataset .runtime/voice-train/dataset \
    --max-epochs 2000 --batch-size 16 --free-gpu
```

- `--name` becomes the `.onnx` basename and the `gateway.voice.tts.voice` value.
  Use Piper's convention `<locale>-<name>-<quality>`, e.g. `en_US-myvoice-medium`.
- **`--free-gpu`** stops `winnow.service` and unloads ollama models first. On a
  host that also serves the decision engine / codegen, the GPU must be free or
  training OOMs (without the flag the script only warns).
- **Quality must match the checkpoint**: the pinned base is `medium` (22050 Hz).
  A `high`/`low` checkpoint needs `--quality` and a matching `--checkpoint`.
- **torch must be a ROCm/CUDA build.** The script installs torch from
  `--torch-index-url` (default `https://download.pytorch.org/whl/nightly/rocm7.1`,
  matching this deployment's ROCm). If the device check reports
  `cuda_available=False`, change the index for your GPU/ROCm version — this is
  the one host-specific piece.
- Stages are skippable (`--skip-setup`, `--skip-train`, `--skip-export`) and
  `--resume` continues from the newest run checkpoint.

The run writes checkpoints under `.runtime/voice-train/runs/` and exports:

- `.runtime/voice-train/out/<name>.onnx`
- `.runtime/voice-train/out/<name>.onnx.json`

Fine-tuning from the base checkpoint, ~1000–3000 epochs is the useful range;
watch `loss_disc_all` level off (tensorboard logs under `runs/`).

## 3. Deploy and verify

The gateway resolves `tts.voice` inside `.runtime/voice/tts/` automatically
(`cli._build_gateway_adapter` anchors `voice_dir`), so deploy is just two files:

```sh
rsync -av .runtime/voice-train/out/<name>.onnx .runtime/voice-train/out/<name>.onnx.json \
    <gateway-host>:'<checkout>/.runtime/voice/tts/'
```

Then on the gateway host set:

```json
"gateway": { "voice": { "tts": { "voice": "<name>" } } }
```

Restart the gateway and speak a reply. **Only a real spoken reply proves the
integration** — a generated test passing proves mechanics, not that the voice
loads and sounds right.

## Troubleshooting

- **`could not open input device` / silence**: `--list-devices`; on a host where
  PortAudio hides a held device, pin by name substring. Another process may hold
  the mic (`arecord -l` shows `Subdevices: 0/1`) — e.g. the running gateway
  (`gateway --platform all`); stop it for the recording session.
- **No prompt audio** (`--speak-prompts`): the output is muted at the ALSA
  level. Unmute with `alsamixer`, or `amixer -c <card> sset Master 100% unmute`
  and set `Auto-Mute Mode Disabled` (it silences the speakers when the headphone
  jack is sensed). PortAudio has no volume control, so this is the only knob.
- **Recorder hangs mid-session**: usually the audio device dropped (a USB
  re-enumeration) and a blocking read stalled. `--read-timeout` (default 3 s)
  now ends the take instead of hanging. Prefer a **direct USB port** over a hub,
  and check `journalctl -k | grep 'USB disconnect'` for re-enumeration storms.
- **Training `cuda_available=False`**: wrong torch wheel for the GPU — set
  `--torch-index-url` to a ROCm/CUDA build matching the host (e.g.
  `https://download.pytorch.org/whl/cu124` on a CUDA cloud instance).
- **Training OOMs**: pass `--free-gpu`; lower `--batch-size` (16 → 8).
- **`Weights only load failed` / `Unsupported global: pathlib.PosixPath`**:
  PyTorch ≥ 2.6 defaults `torch.load` to `weights_only=True` and Lightning passes
  it explicitly, so the pinned base checkpoint's hyperparameters are rejected.
  `scripts/voice-train.sh` handles this with a `sitecustomize.py` that
  allowlists `PosixPath` on `PYTHONPATH` for the train/export commands; if you
  run `piper.train` by hand, do the same.
- **`--ckpt_path` aborts with `does not accept option 'model.sample_bytes'`**:
  the pinned older checkpoint's saved model hyperparameters predate the current
  `VitsModel` signature, so `LightningCLI._parse_ckpt_path` fails. The script
  fine-tunes with `--model.warmstart_ckpt` (non-strict weight copy) instead and
  reserves `--ckpt_path` for `--resume`.
- **`ImportError: cannot import name 'espeakbridge'`**: the editable install
  builds the CMake extension in its isolated env but does not expose it. Run the
  documented dev build in the piper checkout:
  `python setup.py build_ext --inplace` (install `scikit-build cmake ninja
  cython` first). `scripts/voice-train.sh` does this automatically when the
  import fails.
- **Training aborts at epoch 0 with `ModelCheckpoint(monitor='val_mos') could
  not find the monitored key`**, or finishes having written **no checkpoints**:
  piper's default callbacks monitor `val_mel` (from the validation split) and
  `val_mos` (scored over the test split), and Lightning ≥ 2.5 raises rather than
  skips when a monitor is missing. Do not zero those splits
  (`--data.validation_split` / `--data.num_test_examples`) — leave piper's
  defaults. `val_mos` also needs the UTMOS predictor, which downloads on first
  use (`torch.hub`, ~392 MB); a host that cannot reach it will not log
  `val_mos`.
- **Voice loads but sounds wrong**: too little/clipped data, or too many epochs
  (artifacts). Re-record quieter/cleaner; stop training earlier.
- **`piper1-gpl` checkout fails to build**: it needs `espeak-ng`,
  `build-essential`, `cmake`, `python3-dev` (the script apt-installs them) and a
  compatible Python. Very new interpreters may need a dedicated venv or the
  project's Docker image.

## Maintenance

The piper1-gpl commit and base checkpoint are pinned in `pins.json` (`voice`).
Bump them via git and rerun `scripts/voice-train.sh`; the checkpoint sha256 is
verified on download.
