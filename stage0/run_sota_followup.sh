#!/usr/bin/env bash
cd ~/nca_research/stage0; O=~/nca_data/stage0_results/sota_baselines_20261026
while kill -0 3889334 2>/dev/null; do sleep 60; done
for s in 0 1 2; do CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 nice -n 10 ~/nca_venv/bin/python sota_v2.py $s > $O/v2_s$s.log 2>&1 & done; wait
CUDA_VISIBLE_DEVICES= nice -n 10 ~/nca_venv/bin/python sota_analysis.py > $O/analysis.log 2>&1
echo $? > $O/FOLLOWUP_DONE
