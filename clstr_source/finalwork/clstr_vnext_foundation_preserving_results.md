# CLSTR vNext foundation-preserving results

Date: 2026-07-19  
Source commit: `7a40699`  
Method: `clstr_vnext_foundation_preserving_route`  
Backbone: frozen Qwen3-Embedding-0.6B

## Same-protocol primary results

| Benchmark | Deployment expert | R@1 | R@5 | MRR | Best SR/ToolREx MRR | MRR gap | Result |
|---|---|---:|---:|---:|---:|---:|---|
| ToolBench-G3 global pool | adapted static + recurrent dynamic | 0.187959 | 0.460352 | 0.311853 | 0.140362 | +0.171491 | win |
| ToolSandbox benchmark-local legal pool | preserved closed-set foundation | 0.339130 | 1.000000 | 0.605942 | 0.561988 | +0.043954 | win |
| Tau2-base benchmark-local legal pool, 1208 routing rows | preserved closed-set foundation | 0.172185 | 0.457781 | 0.330649 | 0.266530 | +0.064119 | win |

The Tau2 ToolREx MRR target is converted from `0.264559` on all 1217 source
rows to the equivalent `0.266530` on the 1208 routing-row denominator used by
the CLSTR evaluator.

## Secondary-metric audit

| Benchmark | Metric | CLSTR | Best aligned SR/ToolREx | Gap |
|---|---|---:|---:|---:|
| ToolBench-G3 | R@1 | 0.187959 | 0.073421 | +0.114538 |
| ToolBench-G3 | R@5 | 0.460352 | 0.192364 | +0.267988 |
| ToolSandbox | R@1 | 0.339130 | 0.330435 | +0.008695 |
| ToolSandbox | R@5 | 1.000000 | 0.956522 | +0.043478 |
| Tau2-base | R@1 | 0.172185 | 0.090232 | +0.081953 |
| Tau2-base | R@5 | 0.457781 | 0.462748 | -0.004967 |

Thus CLSTR clears all three primary MRR targets. The only remaining strict
"every reported metric" miss is Tau2 R@5 by approximately `0.005`.

## Deployment and credibility checks

- ToolBench selects the open/global-pool Stage2 expert. Recurrent memory is
  used on all 1362 rows, and factual routing improves over its own static route
  by `+0.003976` MRR.
- ToolSandbox and Tau2 select the preserved closed-set foundation based only on
  whether the complete decision-time legal pool fits the natural Top500
  budget; benchmark identity is not used.
- On both closed-set benchmarks, `factual == static == deployment` exactly.
  Recurrent memory, shuffled-history, mismatched-history, action-cache, and
  result-cache paths are not executed.
- No GT-positive candidate injection is used. The Stage2 and Stage0 artifacts
  are bound by the same frozen-backbone, prompt, skill-prefix, and checkpoint
  lineage contract.

## Authoritative reports

- ToolBench-G3:
  `/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-recallfix/outputs/unified_restore_scratch_v1/foundation_dispatch_full_7a40699/toolbench/toolbench_g3_vnext_eval_report.json`
- ToolSandbox:
  `/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-recallfix/outputs/unified_restore_scratch_v1/foundation_dispatch_full_7a40699/toolsandbox/toolsandbox_vnext_eval_report.json`
- Tau2-base:
  `/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-recallfix/outputs/unified_restore_scratch_v1/foundation_dispatch_full_7a40699/tau2/tau2_vnext_eval_report.json`

## Checkpoints

- Release-selected Stage2:
  `/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-recallfix/outputs/unified_restore_scratch_v1/stage2_querymix_depth_abstain055_a800_pilot1000_7ab206a/checkpoints/clstr_vnext_stage2-step1000.pt`
- Preserved closed-set foundation:
  `/data/run01/scyb713/xzf/AAAI/autodl-tmp/clstrs/clstr-qwen06-recallfix/outputs/unified_restore_scratch_v1/stage0_weakpos_pilot500_v3/checkpoints/clstr_vnext_stage0-step0.pt`
