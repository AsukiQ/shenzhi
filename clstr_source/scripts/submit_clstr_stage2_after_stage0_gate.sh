#!/bin/bash
set -euo pipefail

cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr

echo "DEPRECATED: Stage0 now hands off to Stage1 heads initialization before Stage2." >&2
echo "Delegating to scripts/submit_clstr_stage1_after_stage0_gate.sh." >&2

exec bash scripts/submit_clstr_stage1_after_stage0_gate.sh
