# Gemma3-4B / V2 AWEX：2026-10-15 20:51 失败复盘

## 观测结论

- 入口：`examples/math/gsm8k_rl.py`（测试文本链路），`Gemma3-4B`；FSDP DP4、SGLang TP4、V2、AWEX。
- Hydra 已经能解析，4 个训练 Rank 均加载 **883/883** 权重，模型加载时间约 73–77 秒。
- AWEX `TransferPlanBuilder.build_local_transfer_plan` 拒绝传输：`train=883 infer=883`，两端 key 原始交集 **0/883**。
- 解析报错中的 `missing_on_infer` 与 `extra_on_infer`：对训练端进行 3 个顶层前缀变换后匹配 **829/883**；再将 Vision `self_attn.out_proj` 变为 `self_attn.proj`，**883/883 一一对应**（54 项需要额外 Vision 重命名）。
- 错误发生在 **NCCL AWEX Process Group 初始化/权重传输之前**；未进入训练 Step，不能标 PASS。

## 为什么不能只做重命名

检查打包的 SGLang `gemma3_mm.py`、`gemma3_causal.py`、`siglip.py`、`layers/attention/vision.py`，Gemma3 下列张量不服从通用的按第 0 维 TP4 分片假设：

- `language_model.model.embed_tokens.weight` 来自 `Gemma3TextScaledWordEmbedding(nn.Embedding)`：**复制**。
- `vision_tower.vision_model.embeddings.patch_embedding.*` 来自 `Conv2dLayer`：**复制**。
- `multi_modal_projector.mm_input_projection_weight` 为普通 `nn.Parameter`：**复制**。
- Vision `self_attn.proj.weight`、`mlp.fc2.weight` 来自 `RowParallelLinear`：沿 **第 1 维** 分片；Bias 复制。
- Vision `self_attn.qkv_proj` 来自 `QKVParallelLinear`，三段 **Q/K/V 等长**。不能直接照文本模型 Q/KV head 比例拆。

这些推论依赖当前 SGLang 源码，不等价于实际 HCU 权重一致性检验。V3.2 添加双方元数据预检，确保若 Global Shape 或 dtype 不匹配，先报简短诊断而非冒险 NCCL 传输。

## 对现有项目的修改范围

所有代码更改仅在 `/code/AReaL-das-fix-awex/hcu_areal/` 内；保留 `hcu_megatron` 风格 Feature 注册、Patch Manager 和 `.pth` 启动方式；不修改 `areal/`、`awex/`、`sglang/`。

Qwen3.5 V3.1 的两步验证已成功，本包不调整 Qwen3.5 已实现的映射/分片计算规则，新增适用于 Gemma3 的模型守卫路径。但**Qwen3.5 在 V3.2 包下仍需适当回归验证**，不能将之前成功直接外推为新代码端到端验证。
