完成 CLSTR 的 AppWorld + SkillX 最小运行时结合

工作目录硬约束：
  只能修改、生成、移动 /data/home/scyb713/run/xzf/AAAI/autodl-tmp 下的文件。
  不得在 /data/home/scyb713/run 以外写入任何文件。

当前已完成状态：
  1. SkillX 已准备在：
     /data/home/scyb713/run/xzf/AAAI/autodl-tmp/SkillX
  2. SkillX AppWorld skills 已规范化为：
     /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/data/appworld_skill_pool/skill_pool.jsonl
     共 188 条。
  3. AppWorld 已安装在本地 venv：
     /data/home/scyb713/run/xzf/AAAI/autodl-tmp/.venvs/appworld
  4. AppWorld 数据已准备在：
     /data/home/scyb713/run/xzf/AAAI/autodl-tmp/appworld_root/data
  5. AppWorld smoke 已通过 import/data/split-metadata 检查。
  6. 目前尚未完成 AppWorld runtime adapter，也尚未把 SkillX skill pool 接入 CLSTR routing loop。

本阶段目标：
  将 AppWorld task runtime 和 SkillX AppWorld skill pool 做最小可运行结合。
  本阶段不训练 CLSTR，不跑完整 benchmark，不做大规模数据生成。
  只验证：
    1. AppWorld 单个 task 能被 CLSTR adapter 加载；
    2. adapter 能暴露 task instruction、可用 apps/API 信息、执行入口、verifier 或 reward/done 信息；
    3. SkillX skill_pool.jsonl 能被加载并构建最小 skill embedding/index 输入；
    4. 一个 AppWorld task 可以拿到 SkillX skill candidates，形成后续 routing 的输入格式。

必须遵守：
  1. 不要把 AppWorld 逻辑塞进 clstr/alfworld_eval.py 或 clstr/full_base_train.py。
  2. 不要启动 CLSTR 训练。
  3. 不要使用 SkillRouter/SKILLRET warm-start，除非先做 leakage audit。
  4. 登录节点只做轻量 smoke 和检查。
  5. 如果需要 reset/step 多个任务、完整 verify、GPU embedding、批量生成，必须写 sbatch 脚本。
  6. sbatch 脚本放在：
     /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/scripts/sbatch/

主要任务：
  1. 阅读现有文件：
     - README.md
     - docs/clstr_pivot_routing_appworld.md
     - clstr/appworld_smoke.py
     - clstr/bridges/skillx/appworld_adapter.py
     - scripts/run_appworld_smoke.py
     - outputs/appworld_smoke/report.json
     - outputs/skillx_audit/report.json
     - data/appworld_skill_pool/manifest.json

  2. 新增最小 AppWorld adapter：
     - clstr/envs/appworld_env.py

     adapter 至少支持：
       - 配置 APPWORLD_ROOT 和 APPWORLD_CACHE；
       - 加载指定 task_id；
       - 返回 task instruction；
       - 返回 allowed apps 或 API docs summary；
       - 判断 ground_truth/verifier 信息是否存在；
       - 可选：只在安全前提下尝试 reset，不做长循环；
       - 输出 machine-readable smoke report。

  3. 新增 smoke 脚本：
     - scripts/run_appworld_adapter_smoke.py

     功能：
       - 默认选择 train split 的一个 task；
       - 通过 clstr/envs/appworld_env.py 加载；
       - 输出：
         outputs/appworld_adapter_smoke/report.json
       - 报告包含：
         task_id、instruction 是否非空、allowed_apps 数量、api docs 是否可用、
         db/ground_truth 是否存在、reset 是否尝试、reset 是否成功、blocker。

  4. 新增 SkillX skill pool loader 或复用现有 adapter：
     - 确认 data/appworld_skill_pool/skill_pool.jsonl 能加载为 CLSTR skill candidate list；
     - 写一个轻量 smoke：
       scripts/run_appworld_skill_pool_smoke.py
     - 输出：
       outputs/appworld_skill_pool_smoke/report.json
     - 报告包含：
       skill_count、字段覆盖率、body 长度统计、executor_desc 覆盖率、样例 skill。

  5. 如果可以轻量完成，新增最小 embedding 输入构建脚本：
     - scripts/build_appworld_skill_embedding_inputs.py
     - 不调用大模型，不跑 GPU；
     - 只把每条 skill 序列化为 embedding text；
     - 输出：
       data/appworld_skill_pool/embedding_inputs.jsonl
       data/appworld_skill_pool/embedding_manifest.json

  6. 更新 README.md：
     写清楚当前状态：
       - AppWorld 数据已准备；
       - SkillX skill pool 已准备；
       - AppWorld adapter smoke 是否通过；
       - skill embedding input 是否生成；
       - 尚未训练；
       - 下一步是否可以进入真正 embedding/index 构建。

预期交付物：
  1. clstr/envs/appworld_env.py
  2. scripts/run_appworld_adapter_smoke.py
  3. outputs/appworld_adapter_smoke/report.json
  4. scripts/run_appworld_skill_pool_smoke.py
  5. outputs/appworld_skill_pool_smoke/report.json
  6. 可选但推荐：
     data/appworld_skill_pool/embedding_inputs.jsonl
     data/appworld_skill_pool/embedding_manifest.json
  7. 如需要集群运行：
     scripts/sbatch/run_appworld_adapter_smoke.sh
  8. 更新 README.md

验收标准：
  1. AppWorld adapter smoke 状态明确为 ok / partial / blocked。
  2. 能明确加载至少一个 AppWorld task，并读出 instruction。
  3. 能明确确认 ground_truth/verifier 或 reward/done 相关信息是否可访问。
  4. SkillX skill pool 能作为 CLSTR skill candidates 加载。
  5. 如生成 embedding_inputs.jsonl，每行必须对应一个 skill，并包含 skill_id 和 embedding_text。
  6. 所有新增报告都是 JSON。
  7. 所有新增/修改文件都在 /data/home/scyb713/run/xzf/AAAI/autodl-tmp 内。
  8. 没有启动训练，没有跑完整 AppWorld benchmark，没有扩大 ALFWorld/Qwen 历史路线。

验证命令：
  cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr

  python3 -m py_compile \
    clstr/envs/appworld_env.py \
    scripts/run_appworld_adapter_smoke.py \
    scripts/run_appworld_skill_pool_smoke.py

  /data/home/scyb713/run/xzf/AAAI/autodl-tmp/.venvs/appworld/bin/python \
    scripts/run_appworld_adapter_smoke.py

  python3 scripts/run_appworld_skill_pool_smoke.py

  git diff --check