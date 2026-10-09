#!/bin/sh
# Restore the internal analog speaker to full volume, unmuted.
#
# PortAudio/ALSA has no volume control, so the voice gateway's cue beep and
# spoken replies are silent when the card's Master is muted or at 0%. The mixer
# also resets when a sound device is added or removed (e.g. a USB mic or HDMI
# unplug), so this runs both at startup and on every hotplug event (see
# `audio-baseline-watch.sh`).
#
# The card is identified by its 'Speaker' control (HDMI cards only expose
# IEC958), so a shifting card index after a device change does not matter.
# Idempotent and safe to run repeatedly; exits 0 when no card is present yet
# (the watcher retries on the next hotplug event).
set -u

for dir in /proc/asound/card*/; do
  [ -e "$dir" ] || continue
  idx=${dir#/proc/asound/card}
  idx=${idx%/}
  amixer -c "$idx" scontrols 2>/dev/null | grep -q "'Speaker'" || continue
  amixer -c "$idx" sset Master    100% unmute >/dev/null 2>&1 || true
  amixer -c "$idx" sset Speaker   100% unmute >/dev/null 2>&1 || true
  amixer -c "$idx" sset Headphone 100% unmute >/dev/null 2>&1 || true
done

exit 0
