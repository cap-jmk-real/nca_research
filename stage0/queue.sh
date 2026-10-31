#!/usr/bin/env bash
# Detached job runner. Usage: queue.sh <queue-name>   (run inside tmux)
# Jobs: stage0/queues/<name>.txt, one shell command per line ('#' = comment).
# Stop rules: touch ~/nca_data/STOP (or STOP_<name>) to stop after current job;
# halts on <30 GB free disk or 3 consecutive failures; per-job timeout via JOB_TIMEOUT (default 12h).
set -u
Q=$1; D=$HOME/nca_data/queue_state/$Q; mkdir -p "$D"; touch "$D/done" "$D/failed"
cd "$HOME/nca_research/stage0"; fails=0
while true; do
  [ -e "$HOME/nca_data/STOP" ] || [ -e "$HOME/nca_data/STOP_$Q" ] && { echo "$(date -Is) STOP file" >> "$D/log"; exit 0; }
  free=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc 0-9); [ "$free" -lt 30 ] && { echo "$(date -Is) disk low ${free}G" >> "$D/log"; exit 1; }
  job=""; n=0
  while IFS= read -r line; do n=$((n+1)); [[ -z "$line" || "$line" == \#* ]] && continue
    id="$n:$(echo "$line" | md5sum | cut -c1-8)"; grep -qxF "$id" "$D/done" "$D/failed" 2>/dev/null && continue; job=$line; break
  done < "queues/$Q.txt"
  [ -z "$job" ] && { echo "$(date -Is) queue empty, idle" >> "$D/log"; sleep 600; continue; }
  echo "$(date -Is) START $id $job" >> "$D/log"
  if timeout "${JOB_TIMEOUT:-12h}" bash -c "$job" >> "$D/job_$(echo "$id" | tr : _).log" 2>&1; then
    echo "$id" >> "$D/done"; fails=0; echo "$(date -Is) OK $id" >> "$D/log"
  else echo "$id" >> "$D/failed"; fails=$((fails+1)); echo "$(date -Is) FAIL $id" >> "$D/log"
    [ $fails -ge 3 ] && { echo "$(date -Is) 3 consecutive failures, halting" >> "$D/log"; exit 1; }; fi
done
