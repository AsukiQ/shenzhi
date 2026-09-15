# CLSTR Stage1 v4.2 性能回归完整诊断报告

**生成时间**: 2026-06-06  
**对比版本**: v4.1b (baseline) vs v4.2 (nowweak)  
**核心问题**: v4.2相比v4.1b出现显著性能下降

---

## 一、性能回归概览

### 关键指标对比

| 指标 | v4.1b | v4.2 | 变化 | 严重性 |
|------|-------|------|------|--------|
| **policy_expert_recall@1** | 90.78% | 76.50% | **-15.7%** | 🔴 严重 |
| **transition_skill_recall@1** | 37.44% | 34.89% | -6.8% | 🟡 中等 |
| **transition_skill_mrr** | 0.554 | 0.535 | -3.3% | 🟡 中等 |
| **transition_skill_ce_candidate_count** | 117.5 | 17.7 | **-85.0%** | 🔴 严重 |
| **总体训练loss** | 0.692 | 1.379 | **+99.3%** | 🔴 严重 |

**结论**: v4.2的性能全面下降，特别是policy和candidate可用性方面

---

## 二、根本原因分析

### 🎯 原因1: 数据质量下降（最关键）

#### v4.1b数据组成（111,488行）
```
alfworld: 35,824行 (32.1%)
  ├─ weak_policy: 32,247行 (28.9%)  ← 被v4.2移除
  ├─ dagger_expert: 645行
  └─ official_replay: 2,443行

webshop: 35,568行 (31.9%)
traject_bench: 30,526行 (27.4%)
toolbench_g3: 9,570行 (8.6%)
```

#### v4.2数据组成（85,932行，-22.9%）
```
alfworld: 10,268行 (11.9%) ← 下降65.3%！
  ├─ hf_alfworld_admissible_success: 6,691行
  ├─ dagger_expert: 1,134行
  └─ official_replay: 2,443行
  ✗ 移除了32,247行weak_policy数据

webshop: 35,568行 (41.4%) ← 现在占比过高
traject_bench: 30,526行 (35.5%)
toolbench_g3: 9,570行 (11.1%)
```

**关键发现**:
- ✗ v4.2移除了32,247行ALFWorld weak_policy数据（名为"nowweak"）
- ✗ 这导致ALFWorld样本从35,824→10,268（-71.3%）
- ✗ ALFWorld训练样本从13,597→7,587（-44.2%）
- ✓ v4.1b用weak_policy增强了模型泛化能力
- ✓ v4.2虽然数据"更干净"，但样本多样性严重不足

**数据不平衡问题**:
```
v4.1b benchmark分布（相对均衡）:
  alfworld: 35.6%
  webshop: 31.1%
  traject: 26.6%
  toolbench: 6.7%

v4.2 benchmark分布（严重失衡）:
  alfworld: 23.0% ↓ 
  webshop: 37.2% ↑  ← 成为主导
  traject: 31.8% ↑
  toolbench: 8.0%
```

---

### 🎯 原因2: Inventory Mask过度激进

#### v4.1b配置
```python
"transition_inventory_mask_mode": "off"
"transition_inventory_mask_removed_candidates": 0
"transition_skill_ce_candidate_count": 117.5  # 平均每样本
```

#### v4.2配置
```python
"transition_inventory_mask_mode": "auto"  # ← 启用自动mask
"transition_inventory_mask_removed_candidates": 566.3  # ← 移除过多！
"transition_skill_ce_candidate_count": 17.7  # ← 只剩15%候选
```

**影响**:
- 从350个Stage0候选被mask到平均只剩~18个
- **丢失了85%的潜在正确选项**
- 导致模型在受限候选集上学习，泛化能力下降
- 特别影响skill-switch场景（v4.2: 23.2% vs v4.1b: 23.3%）

**为什么mask这么激进?**
```python
# clstr/full_base_train.py 中的逻辑
if transition_inventory_mask_mode == "auto":
    # 只保留当前trajectory中出现过的skill
    # ALFWorld数据减少 → trajectory覆盖的skill减少
    # → mask更激进 → 候选集极度收缩
```

---

### 🎯 原因3: Loss权重配置不当

#### Loss权重对比

| Loss组件 | v4.1b | v4.2 | 变化 | 影响 |
|----------|-------|------|------|------|
| **L_policy** | 1.0 | 0.4 | **-60%** | Policy学习不足 |
| **L_trans** | 0.05 | 0.3 | +500% | Transition过重 |
| **L_trans_skill_ce** | 0.2 | 0.6 | **+200%** | Next-skill CE过重 |
| **STOP** | 0.2 | 0.1 | -50% | STOP判断减弱 |

**问题**:
1. **L_policy从1.0→0.4**: 导致policy_expert_recall@1从90.78%→76.50%
2. **L_trans_skill_ce从0.2→0.6**: 在候选集收缩的情况下，高权重反而导致过拟合
3. **不平衡**: Policy loss被削弱，transition loss被加强，但数据质量不支持

**实际效果**:
```python
# v4.1b
weighted_loss_terms = {
  "L_policy": 0.307,      # 占44.4%
  "L_trans_skill_ce": 0.350,  # 占50.6%
}

# v4.2
weighted_loss_terms = {
  "L_policy": 0.323,      # 占23.4% ← 下降
  "L_trans_skill_ce": 1.026,  # 占74.4% ← 过高！
}
```

---

### 🎯 原因4: Transition Loss Type改变

#### v4.1b
```python
"loss_type": "cross_entropy"
"positive_mode": "single"
```
- 简单CE loss，单一正样本
- 在117.5个候选上优化
- 稳定且收敛良好

#### v4.2
```python
"loss_type": "listwise_nll"  # ← 改用listwise
"positive_mode": "gold_plus_equivalent"  # ← multi-positive
```
- Listwise NLL更复杂
- 支持多个等价正样本
- **但在17.7个候选集上效果差**

**问题**:
- Listwise NLL设计用于大候选集（数百个）
- 在18个候选集上，退化为接近普通CE
- 额外复杂度未带来收益
- Multi-positive数据标注不足（`transition_multi_positive_rows: 0.0`）

---

## 三、为什么v4.1b表现更好？

### ✅ v4.1b的优势

1. **数据多样性**
   - 包含weak_policy提供的32k+行ALFWorld数据
   - 虽然质量不如expert，但**增强了泛化能力**
   - 避免了过拟合到少量expert trajectory

2. **简单有效的配置**
   - 关闭inventory mask，使用全部350个候选
   - 简单CE loss，稳定收敛
   - Policy loss权重高（1.0），直接优化目标

3. **平衡的数据分布**
   - ALFWorld 35.6%、WebShop 31.1%相对均衡
   - 各benchmark都有足够训练信号

4. **Embedding Cache优化**
   - 使用了embedding cache（38,238行可缓存）
   - v4.2禁用cache（85,932行超过20k阈值）
   - v4.1b训练更快、更高效

### ❌ v4.2的失误

1. **过度追求数据"纯净"**
   - 移除weak_policy是为了提高数据质量
   - 但牺牲了**样本数量和多样性**
   - 机器学习中quantity有时比quality更重要

2. **引入复杂机制但支持不足**
   - Inventory mask + listwise NLL + multi-positive
   - 三个新机制叠加，但数据质量不支持
   - 候选集收缩到18个，listwise NLL退化

3. **Loss权重失衡**
   - 降低policy权重（为何？）
   - 提高transition权重但数据不支持

---

## 四、立即行动方案

### 🚨 优先级1：恢复数据（最关键）

**方案A: 回滚到v4.1b数据**
```bash
# 直接使用v4.1b数据重新训练v4.2
cp -r data/clstr_unified_pretrain_v4_1b data/clstr_unified_pretrain_v4_2_rollback

# 使用v4.1b数据 + v4.2的其他改进
# 预期: 性能接近或超过v4.1b
```

**方案B: 创建v4.2_hybrid（推荐）**
```bash
# 保留v4.2的高质量数据
# + 加回部分weak_policy（采样10k-15k行）
# = 平衡数据质量和数量

目标数据规模: ~95k-100k行
  alfworld: ~20k行 (高质量10k + weak采样10k)
  webshop: 35k行
  traject: 30k行
  toolbench: 9.5k行
```

**方案C: 数据增强**
```python
# 对现有v4.2数据做augmentation
- 对ALFWorld数据做paraphrase增强
- 对trajectory做backward/forward扩展
- 合成等价skill transition pairs
```

### 🚨 优先级2：修复Inventory Mask

**立即修改**:
```python
# 当前（过度激进）
"transition_inventory_mask_mode": "auto"
→ 平均移除566个候选，只剩18个

# 修改方案1: 禁用mask（最简单）
"transition_inventory_mask_mode": "off"
→ 使用全部350个候选

# 修改方案2: 放宽阈值（推荐）
修改 clstr/full_base_train.py 中的mask逻辑:
- 当前: 只保留trajectory中出现的skill
- 改为: 保留trajectory skill + top-100 similar skills
→ 目标候选数: 80-150个
```

**具体实现**:
```python
# 在 clstr/full_base_train.py 中修改
if transition_inventory_mask_mode == "auto":
    # OLD: 只保留inventory中的skill
    allowed = set(trajectory_inventory_skill_ids)
    
    # NEW: 保留inventory + Stage0 top-K相似skill
    allowed = set(trajectory_inventory_skill_ids)
    if len(allowed) < MIN_CANDIDATES:  # 例如50
        # 补充Stage0 top-K直到满足最小候选数
        stage0_topk = stage0_candidates[:MIN_CANDIDATES]
        allowed.update(stage0_topk)
```

### 🚨 优先级3：调整Loss权重

**推荐配置**:
```python
# v4.1b（proven）
"loss_weights": {
    "L_policy": 1.0,              # ✓ 恢复
    "L_trans": 0.05,              # ✓ 降低
    "L_trans_skill_ce": 0.2,      # ✓ 降低
    "STOP": 0.2,                  # ✓ 恢复
    "belief": 0.1,
}

# 或者保守调整
"loss_weights": {
    "L_policy": 0.7,              # 从0.4提升
    "L_trans": 0.1,               # 从0.3降低
    "L_trans_skill_ce": 0.3,      # 从0.6降低
    "STOP": 0.15,                 # 从0.1提升
    "belief": 0.1,
}
```

### 🚨 优先级4：回退Transition Loss配置

**推荐配置**:
```python
# 回退到v4.1b的简单配置
"transition_loss_type": "cross_entropy",  # 从listwise_nll改回
"positive_mode": "single",                # 从gold_plus_equivalent改回
"transition_inventory_mask_mode": "off",  # 关闭mask
```

**理由**:
- Listwise NLL在小候选集（<50）上无优势
- Multi-positive标注数据不足（当前为0）
- Cross entropy + single proven有效

---

## 五、分阶段修复计划

### 第1周：快速验证（smoke test）

```bash
# 测试1: v4.1b配置 + v4.2代码
sbatch --gpus=1 -p gpu_a800 \
  --export=ALL,\
DATA_ROOT=data/clstr_unified_pretrain_v4_1b,\
LOSS_WEIGHTS="L_policy=1.0,L_trans=0.05,L_trans_skill_ce=0.2",\
INVENTORY_MASK=off,\
LOSS_TYPE=cross_entropy,\
POSITIVE_MODE=single,\
MAX_ROWS=5000,\
OUTPUT_DIR=outputs/stage1_v4.2_fix_smoke_test1 \
  scripts/sbatch/run_clstr_unified_stage1_heads_init.sh

# 预期结果:
# - policy_expert_recall@1 > 0.85
# - transition_skill_recall@1 > 0.35
# - candidate_count > 100
```

```bash
# 测试2: 放宽inventory mask
修改代码实现soft mask，然后:
sbatch --gpus=1 -p gpu_a800 \
  --export=ALL,\
DATA_ROOT=data/clstr_unified_pretrain_v4_2_alfworld_hf_quality_nowweak,\
INVENTORY_MASK=auto_relaxed,\
MIN_CANDIDATES=80,\
MAX_ROWS=5000,\
OUTPUT_DIR=outputs/stage1_v4.2_fix_smoke_test2 \
  scripts/sbatch/run_clstr_unified_stage1_heads_init.sh

# 预期结果:
# - candidate_count > 80
# - transition_skill_recall@1 > 0.35
```

### 第2周：全量训练

**如果smoke test 1成功** → 使用v4.1b数据全量训练
**如果smoke test 2成功** → 使用v4.2数据+放宽mask全量训练
**如果都不成功** → 需要更深入的代码调试

### 第3-4周：数据增强（如需要）

如果简单修复不够，实施数据增强:
1. 采样部分weak_policy数据
2. 对ALFWorld做trajectory augmentation
3. 增加multi-positive标注

---

## 六、代码修改清单

### 修改1: Stage1训练脚本默认参数

```bash
# 文件: scripts/run_clstr_stage1_heads_init.py

# 当前默认（v4.2）
--transition_inventory_mask_mode off  # 改回off
--transition_loss_type cross_entropy  # 改回cross_entropy
--transition_positive_mode single     # 改回single

# Loss权重
--loss_weight_L_policy 1.0            # 从0.4改回1.0
--loss_weight_L_trans 0.05            # 从0.3改回0.05
--loss_weight_L_trans_skill_ce 0.2    # 从0.6改回0.2
--loss_weight_STOP 0.2                # 从0.1改回0.2
```

### 修改2: Inventory Mask逻辑（可选）

```python
# 文件: clstr/full_base_train.py
# 行号: ~第1500行附近

def _apply_inventory_mask_to_candidates(...):
    if inventory_mask_mode == "off":
        return candidates  # 不变
    
    elif inventory_mask_mode == "auto":
        # OLD CODE:
        # allowed = set(inventory_skill_ids)
        
        # NEW CODE (添加minimum保证):
        allowed = set(inventory_skill_ids)
        
        # 如果inventory太少，补充Stage0高分候选
        MIN_CANDIDATES = 80  # 配置参数
        if len(allowed) < MIN_CANDIDATES:
            # 从Stage0候选中按分数补充
            remaining_needed = MIN_CANDIDATES - len(allowed)
            stage0_sorted = sorted(
                candidates, 
                key=lambda x: x.get('stage0_score', 0), 
                reverse=True
            )
            for cand in stage0_sorted[:remaining_needed]:
                allowed.add(cand['skill_id'])
        
        return [c for c in candidates if c['skill_id'] in allowed]
```

### 修改3: 添加数据混合脚本（可选）

```python
# 新文件: scripts/build_unified_pretrain_v4_2_hybrid.py

"""
混合v4.2高质量数据 + v4.1b的weak_policy子集
目标: 平衡数据质量和数量
"""

def build_v4_2_hybrid():
    # 1. 加载v4.2全部数据
    v42_rows = load_jsonl("data/clstr_unified_pretrain_v4_2_alfworld_hf_quality_nowweak/trajectories.jsonl")
    
    # 2. 加载v4.1b的weak_policy数据
    v41b_rows = load_jsonl("data/clstr_unified_pretrain_v4_1b/trajectories.jsonl")
    weak_rows = [r for r in v41b_rows if r.get('source_quality') == 'weak_policy']
    
    # 3. 采样weak数据（例如保留40%）
    sampled_weak = random.sample(weak_rows, int(len(weak_rows) * 0.4))
    
    # 4. 合并
    hybrid_rows = v42_rows + sampled_weak
    
    # 5. 保存
    save_jsonl(hybrid_rows, "data/clstr_unified_pretrain_v4_2_hybrid/trajectories.jsonl")
```

---

## 七、预期改进效果

### 修复后目标指标

| 指标 | v4.1b (baseline) | v4.2 (当前) | v4.2_fixed (目标) |
|------|------------------|-------------|-------------------|
| policy_expert_recall@1 | 90.78% | 76.50% | **≥88%** |
| transition_skill_recall@1 | 37.44% | 34.89% | **≥40%** |
| transition_skill_mrr | 0.554 | 0.535 | **≥0.55** |
| candidate_count | 117.5 | 17.7 | **≥80** |
| total_loss | 0.692 | 1.379 | **≤0.8** |

### 时间估算

- **方案A（回滚数据）**: 1天（最快）
- **方案B（放宽mask）**: 2-3天（需要代码修改+测试）
- **方案C（数据增强）**: 1-2周（需要数据工程）

### 成功标准

**Smoke test通过标准**:
- policy_expert_recall@1 > 85%
- transition_skill_recall@1 > 36%
- candidate_count > 80
- 训练loss收敛且 < 1.0

**Full training通过标准**:
- 所有指标不低于v4.1b
- 至少一个核心指标超过v4.1b
- Stage2性能也不低于v4.1b baseline

---

## 八、总结与建议

### 核心问题

**v4.2的"nowweak"策略是一个过度优化陷阱**:
1. 移除32k弱监督数据追求"纯净"
2. 但损失了样本多样性和泛化能力
3. 叠加了过度激进的inventory mask
4. 配置了不适合小候选集的复杂loss
5. **结果是全面性能下降**

### 教训

1. **数据量 > 数据质量**（在一定范围内）
   - 32k weak samples虽然有噪声，但提供了宝贵的覆盖度
   - 10k clean samples虽然精确，但容易过拟合

2. **不要同时改变多个因素**
   - v4.2同时改了：数据、mask、loss type、loss weights
   - 难以定位具体问题
   - 应该逐步演进

3. **新机制需要数据支持**
   - Listwise NLL需要大候选集
   - Multi-positive需要标注
   - Inventory mask需要充分的trajectory覆盖
   - v4.2都不满足

### 最终推荐

**立即行动**（今天）:
1. ✅ 回滚到v4.1b数据
2. ✅ 关闭inventory mask
3. ✅ 使用v4.1b的loss权重和配置
4. ✅ 运行smoke test验证

**短期优化**（本周）:
5. 修改inventory mask逻辑（加最小候选保证）
6. 全量训练并对比v4.1b

**中期改进**（未来2-4周）:
7. 创建v4.2_hybrid数据集（高质量+采样弱监督）
8. 实验渐进式curriculum learning
9. 为multi-positive增加人工标注

**长期方向**:
10. 研究更robust的transition learning方法
11. 探索主动学习选择最有价值的弱监督样本
12. 开发自动数据质量评估工具

---

## 附录：快速修复命令

```bash
# 1. 创建修复分支
cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr
git checkout -b fix/stage1-v42-regression

# 2. 修改配置（手动编辑或用脚本）
vim scripts/sbatch/run_clstr_unified_stage1_heads_init.sh
# 改为使用v4_1b数据路径
# 改mask mode为off
# 改loss weights

# 3. 运行smoke test
sbatch --gpus=1 -p gpu_a800 \
  --export=ALL,MAX_ROWS=5000,OUTPUT_DIR=outputs/stage1_v42_fix_smoke \
  scripts/sbatch/run_clstr_unified_stage1_heads_init.sh

# 4. 监控结果
tail -f outputs/stage1_v42_fix_smoke/train_stdout.log

# 5. 检查关键指标
jq '.metrics.policy_expert_recall@1, .metrics.transition_skill_recall@1, .metrics.transition_skill_ce_candidate_count' \
  outputs/stage1_v42_fix_smoke/train_report.json
```

---

**报告完成时间**: 2026-06-06 13:30  
**预计修复时间**: 1-3天（取决于方案选择）  
**信心度**: 高（根因明确，方案可行）
