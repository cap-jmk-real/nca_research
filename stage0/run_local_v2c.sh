#!/bin/bash
cd ~/nca_research/stage0; O=~/nca_data/stage0_results/local_v2c_20261026; mkdir -p $O
while pgrep -f "[p]ython.* score_teachers" >/dev/null; do sleep 60; done
for m in d1:3b granite4.2:8b; do
  ollama pull $m >> $O/pull.log 2>&1 || { echo "PULLFAIL $m" >> $O/run.log; continue; }
  python3 score_local_v2c.py $m val 200 $O >> $O/run.log 2>&1
  t=$(python3 -c "import json;print(json.load(open(\"$O/teacher_${m//:/_}_val.json\"))[\"top1\"])" 2>/dev/null)
  echo "$m val top1=$t" >> $O/run.log
  if python3 -c "import sys;sys.exit(0 if float(\"${t:-0}\")>52.9 else 1)"; then python3 score_local_v2c.py $m test 0 $O >> $O/run.log 2>&1; fi
done
echo DONE >> $O/run.log
