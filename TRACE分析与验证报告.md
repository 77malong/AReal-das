# Qwen3.5-4B AWEX 参数布局诊断与修复候选（2026-10-15）

## 输入来源与边界

分析来源：用户提供 `qwen35-awex-trace.tar.gz`，总计 8 份 JSON：FSDP DP4 × 4，SGLang TP4 × 4。每个训练端 Rank 在原有诊断 Patch 下产生 723 个参数元数据记录；每个推理 Rank 产生 771 个记录。未包含真实权重值、梯度、NCCL 收发记录。

## 原始问题

- AWEX 初始化报 `Train/infer key mismatch for transfer plan: train=723 infer=771`。
- 仅规范化 `model.language_model.`/`model.visual.` 和 Full Attention 的 `self_attn` 后：579 个名称交集，455 个 Global Shape 一致，124 个 Shape 不一致；训练端剩余 144 个 key，推理端剩余 192 个 key。
- Vision `patch_embed` 推理张量是复制的完整权重，但 Sharding Strategy 将其错误归为 TP0，致 Global Shape 乘以 4。`attn.proj`/`mlp.linear_fc2`/`merger.linear_fc2` 的矩阵被错误归类为 TP0，实际应为 TP1；对应输出 Bias 应为 NO_SHARDING。
- Full Attention 的门控 Q 行数是普通 Q 行数的 2 倍；默认 `_unfuse_params` 仍使用普通 GQA 头数比例导致错误的 1706 / 426 / 426 本地行数（存在两个行位置未被分出）。正确对应本次配置的 TP4 本地行数应为 Q=2048、K=256、V=256。
- GDN `in_proj_qkvz` 在每个 TP Rank 内的顺序为 [Q, K, V, Z]；FSDP 源侧是独立 `in_proj_qkv` + `in_proj_z`，且 `in_proj_qkv` 物理顺序为 [Q_all,K_all,V_all]。需要两端拆成统一逻辑分量，以免错误的范围复制。
- GDN `conv1d` 同样需要拆成 Q/K/V 逻辑片段；BA 需要拆成 B/A。
- 24 个 GDN 层的 `A_log` 训练端被原 AWEX `_wire_dtype()` 报为 `torch.bfloat16`，SGLang 接收端为 `torch.float32`。候选修复中发送端为 A_log 单独声明并转换为 float32。

## 候选 Runtime Patch

采用 `hcu_areal` 项目内 Feature + Manager 的 Megatron-LM-das 风格。扩展文件 `hcu_areal/features_manager/weight_update/qwen35_awex_layout.py`，将以下五个 AReaL Adapter 方法以运行时包装方式替换：

1. FSDP `get_weight_metadata()`
2. FSDP `get_local_shard_parameters()`
3. SGLang `_unfuse_params()`
4. SGLang `_build_sharding_strategy()`
5. SGLang `get_weight_metadata()`（仅附加可选的 stderr/JSON 诊断）

补丁仅对检测到 Qwen3.5 的模型实例生效；非 Qwen3.5 实例原样调用源方法。训练端转换将每个原始 FSDP 分片与规范参数逻辑区间求交，得到共享底层张量存储的 View；推理端转换也是 fused 参数的 View；AWEX TransferPlanBuilder 可以对各 View 的全局偏移计算交集并传输，避免误拼接/多余全量 AllGather。

通过同一套几何规则离线展开 8 份元数据后：两端均有 **915** 个规范参数 key，**915/915** 的全局形状与 dtype 一致；FSDP 源切片无区间孔洞/重叠；推理端的目标分片落在全局张量边界之内。

> 这里的 915/915 只是**元数据合同**，不是对真实权重数值、NCCL 传输或两步训练成功的证明。之前的 819/819 是在不完整转换条件下得到的理论数量，**应以本次 915/915 取代**。

## 测试

- ZIP 中 Python 运行时注册/移除/子进程模拟正常。
- CPU 数值模拟包括：GDN QKVZ、BA、Conv1D，Gated Full Attention QKV，Vision QKV Weight/Bias；全部模拟了四个源分片 → 四个推理分片的矩形区间复制，断言输出与原始权重的正确 TP 排列一致。
- `python -m unittest discover -s hcu_areal/tests -v` 共 **15 个测试通过**。
- 尚未在 bw67 运行真实 AWEX NCCL 传输。因此这是**候选修复 V3**，不是 PASS 状态。

## 重试及回退

`unzip -o hcu_areal_Qwen35_AWEX_RuntimePatch_v3.zip` 安装后，**无需额外 `export HCU_AREAL_PATCH_QWEN35_AWEX`**：补丁默认注册且通过模型识别条件生效。调试时可使用 `HCU_AREAL_DISABLE_QWEN35_AWEX=1` 暂时禁用，或还原备份目录。需要确保 `.pth` 仍指向 `/code/AReaL-das-fix-awex` 并重启 Ray，且不能影响其他作业。

再次运行 Qwen3.5-4B FSDP DP4 + SGLang TP4 + V2 AWEX 的 2-Step；主要检查 AWEX 双端元数据、TransferPlan 构建、Tensor dtype、NCCL 初始化、两个 Step 非零梯度。即使两步 PASS，权重数值一致性仍需单独验证。
