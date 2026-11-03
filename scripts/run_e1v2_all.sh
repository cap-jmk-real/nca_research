#!/bin/bash
# E1v2/E2v2 real-only then real+synthetic; backups mandatory; then re-arm Seam.
set -euo pipefail
cd ~/nca_research_git
export NCA_DATA_ROOT=$HOME/nca_data_e1 PATH=$HOME/nca_venv/bin:$HOME/bin:$PATH CUBLAS_WORKSPACE_CONFIG=:4096:8
P=$HOME/nca_venv/bin/python; SYN=$NCA_DATA_ROOT/synthetic/bfcl_e1v2_synth_train.filtered.jsonl
echo "== e1v2 start $(date -Is) sha=$(git rev-parse --short HEAD)"
$P stage0/run_e1v2_benchmark.py --seeds 0,1,2 --topologies tree,meta,star,1d --steps 150000; research-backup-now e1v2_real
$P stage0/run_e2v2_router_ablation.py --seeds 0,1,2 --variant run; research-backup-now e2v2_real
$P stage0/run_e1v2_benchmark.py --seeds 0,1,2 --topologies tree,meta,star,1d --steps 150000 --synthetic-jsonl "$SYN"; research-backup-now e1v2_synth
$P stage0/run_e2v2_router_ablation.py --seeds 0,1,2 --variant run_synth; research-backup-now e2v2_synth
echo "== e1v2 done $(date -Is)"; touch ~/nca_e_logs/e1v2.done
if [[ -f ~/seam_q_logs/smoke_ok.paused_1103 ]]; then mv ~/seam_q_logs/smoke_ok.paused_1103 ~/seam_q_logs/smoke_ok; systemctl --user start --no-block seam-q.service; fi
