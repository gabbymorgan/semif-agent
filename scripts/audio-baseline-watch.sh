#!/bin/sh
# Re-apply the audio baseline whenever a sound device is added, changed, or
# removed, so the internal speaker stays at full volume and unmuted (the
# baseline) after a USB mic, HDMI, or headset is plugged or unplugged.
#
# `udevadm monitor` is readable without root, so this runs as a long-lived
# systemd user service (see `audio-baseline.service`); each sound event
# triggers one idempotent apply of `audio-baseline.sh`.
set -u

BASELINE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/audio-baseline.sh"

stdbuf -oL udevadm monitor --udev --subsystem-match=sound 2>/dev/null \
  | while IFS= read -r line; do
      # Event lines look like `UDEV  [1696.123] add /devices/... (sound)`;
      # the `[` distinguishes them from the 3-line startup banner.
      case "$line" in
        UDEV*\[*) "$BASELINE" ;;
      esac
    done
