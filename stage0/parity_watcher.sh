#!/usr/bin/env bash
# Waits for queue lines 62-91 (parity) + 93-95 (sota_v2 clean) in queues/train.txt to be done/failed,
# then analyze_parity (incl. Hermes session bootstrap), restic backup (append-only, tag stage0_results), DONE.json.
set -u; S=$HOME/nca_research/stage0; D=$HOME/nca_data/queue_state/train; OUT=$HOME/nca_data/stage0_results/parity_20261026; mkdir -p $OUT; L=$OUT/watcher.log
ids=(); n=0; while IFS= read -r line; do n=$((n+1)); { [ $n -ge 62 ] && [ $n -le 91 ]; } || { [ $n -ge 93 ] && [ $n -le 95 ]; } || continue; ids+=("$n:$(echo "$line" | md5sum | cut -c1-8)"); done < $S/queues/train.txt
echo "$(date -Is) watching ${#ids[@]} ids" >> $L
while true; do ok=0; bad=0; for id in "${ids[@]}"; do grep -qxF "$id" $D/done && ok=$((ok+1)) || { grep -qxF "$id" $D/failed && bad=$((bad+1)); }; done
  [ $((ok+bad)) -ge ${#ids[@]} ] && break; sleep 300; done
echo "$(date -Is) all finished ok=$ok failed=$bad" >> $L
cd $S && CUDA_VISIBLE_DEVICES= ~/nca_venv/bin/python analyze_parity.py --out $OUT >> $L 2>&1; ar=$?
set -a; . $HOME/.config/nca_restic.env; set +a
restic backup --tag stage0_results --tag nca_research --host engine $HOME/nca_data/stage0_results $HOME/nca_data/stage0_runs --exclude="*.pt" > $OUT/restic.log 2>&1; rr=$?
snap=$(grep -oE "snapshot [0-9a-f]+ saved" $OUT/restic.log | awk "{print \$2}")
printf "{\"finished\":\"%s\",\"jobs_ok\":%d,\"jobs_failed\":%d,\"analyze_exit\":%d,\"restic_exit\":%d,\"snapshot\":\"%s\",\"analysis\":\"%s/parity_analysis.json\"}\n" "$(date -Is)" $ok $bad $ar $rr "$snap" "$OUT" > $OUT/DONE.json
echo "$(date -Is) DONE" >> $L
