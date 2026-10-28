# hcu_areal：Qwen3.5 + Gemma3 + GLM-5 + DeepSeek AWEX Runtime Patch V3.5

本目录模仿 `Megatron-LM-das/hcu_megatron` 的 Feature/Manager 结构。`hcu_areal` 通过 `.pth` 与延迟 import hook 注入 AReaL V2 Ray/SGLang 子进程，**不修改** `areal/`、`sglang/` 和 `awex/` 原始源码。使用容器原生 `/usr/local/bin/python`。

## DeepSeek BF16/FP8 自动分流（V3.5，2026-10-28）

按模型 `quantization_config` 和 SGLang 实际 `model.quant_config` 分流；仅有
`torch_dtype=bfloat16` 不代表权重没有 FP8。入口读取推理目标的配置，将目标量化信息
显式传到两端 Ray worker；直接使用 adapter 时默认读取训练模型的量化配置。
无量化配置继续走 V3.4.1 的 BF16/FP32 权重路径；普通动态 E4M3 分块配置走 FP8 路径。

新增 FP8 行为：

* 原生 DeepSeek Bridge 读取 FP8 检查点时，结合对应 `weight_scale_inv` 反量化后再交给
  原有 HF→Megatron 转换；FP32 correction bias 保持 FP32。训练是否启用 TE FP8 算子
  仍由原有训练配置和硬件决定，FP8 检查点不等于强制 FP8 GEMM。
* BF16/FP32 或可反量化的训练参数，经原生 TP/EP 聚合和 Bridge 名称转换后，按推理目标
  配置量化为 E4M3FN 字节和 FP32 scale；非量化层、router、norm 等保留原有精度。
  普通 Torch FP8 源参数必须带 scale，缺失 scale 会明确失败。
* 传输用 uint8 表示 FP8 原始存储，避免依赖通信库原生 Float8 支持。scale 同步参与
  名称、形状和类型预检，其 TP 分片轴跟随对应权重。DCU 转置后的线性层权重和 scale
  都暴露正确的可写视图，融合 Q/KV-A、gate/up、MoE 专家及融合共享专家分别拆分。
* 运行时若使用 E4M3FNUZ，接收端先收 checkpoint FN 字节及 scale，再同时转换为 FNUZ
  字节和双倍 scale。每个新版本重新计算，避免重复乘 scale。MLA 刷新时临时恢复原生
  loader 所需的权重方向与格式，完成后恢复运行时参数对象，在缓存刷新之后确认更新。

核心补丁没有四层限制，独立入口也已去掉四层断言，改为检查正层数及训练/推理几何一致。
单元测试覆盖 2、8、61 层布局与配置分流。默认 smoke 配置仍使用用户四层样本和两台
8 卡节点；实际内存需求取决于完整模型，结构测试不代表完整 61 层训练已通过。
HF 配置声明可选 MTP 不会拒绝普通 base inference；实际启用的 MTP 训练、NextN 推理
仍明确拒绝。其他边界：LoRA、MXFP8/FP4、AITER/Marlin/UE8M0 等预打包布局、推理 EP>1
未覆盖；普通 block FP8 的训练 EP 分片及推理 TP 分片已验证。

V3.5 验证：66 项测试、16 个实际 Hook 检查通过。16 进程 CPU/Gloo 的真实 AWEX P2P
连续四次更新通过，覆盖 1621 项普通 FP8 权重/scale，以及 1619 项混合精度数据、融合
共享专家和一个真实 Torch FP8 训练源参数。权重字节、FP32 scale、MLA 缓存和计算探针
与原生 v1 Bridge/原生 SGLang loader 的缩小张量对照一致，最大误差 0；检测缺失 scale
更新、过期缓存的负对照通过。测试不启动 v1 控制器，不保存权重。

当前 BW 硬件不支持 FP8 GEMM；本轮计算验证均为代码/CPU 数据路径，未宣称 nmz5 上的
FP8 GEMM、GPU/RCCL FP8 传输或完整模型训练通过。V3.4.1 的双机 BF16 实测记录保留在下文。

```bash
python -m unittest discover -s hcu_areal/tests -v
python -m hcu_areal.check_install --full --megatron --flashmla
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 torchrun --standalone --nproc_per_node=16 \
  -m hcu_areal.validation.test_deepseek_fp8_awex_update_parity --backend gloo \
  --report /tmp/deepseek-fp8-awex.json
```

FP8 格式、缺失 scale 或未知布局会明确失败，不静默改成另一种精度。补丁按模型格式
分流，不代替硬件的算子支持检查。nmz5 的 FP8 计算需要后续独立实测。

## DeepSeek-R1 四层 BF16（V3.4，2026-10-27）

新增 `DeepseekAwexFeature`，只处理 `model_type=deepseek_v3`。使用现有原生
`DeepSeekV3Bridge` 生成权重名称，TP gather 显式处理 GLU gate/up 排列，保留参数覆盖
语义及 FP32 expert correction bias。推理侧把融合 Q/KV-A 参数拆成两个可写视图，并将
这两个投影及 correction bias 标为复制参数。通信组建立前检查两端名称、全局形状和类型；
接收权重后刷新 SGLang `post_load_weights()` 及 MLA 缓存，colocate 完成事件也在刷新后发布。
DeepSeek 不使用 GLM 的 DSA indexer 精度转换。

沿用已有 Hook 的模型分发，不放宽 Patch Manager 的重复目标检查。AWEX 部分仍为 13 个 Hook，
其中既有 `glm5.*` Hook 的名称保留，内部增加 DeepSeek 分发。禁用其他模型 Feature 时，
DeepSeek 会独立注册缺少的 9 个 Hook。单独禁用：`HCU_AREAL_DISABLE_DEEPSEEK_AWEX=1`。
当前支持未量化的 BF16/FP32 完整模型；LoRA、量化模型、MTP 明确拒绝。

本地模型 `deepseek-r1-4layers-bf16` 已检查：821 个张量，820 BF16 + 1 FP32，
总计 15,111,101,696 个参数、28.15 GiB。测试夹具只保存真实权重的名称、形状、类型。
两个数值测试分别对照原生 v1 Bridge 转换、原生 v1 SGLang 加载后的参数和 MLA 缓存；
后者使用真实 AWEX TP8/PP1/EP8/ETP1 -> TP8 收发，连续更新两次，覆盖共享专家独立/融合存储。
数值张量缩小，运行后需以 JSON `status=passed` 和退出码 0 判定。
这不代表完整模型 logits 或双机 RCCL/IB 训练已经验证。

V3.4 验证：49 项单元/回归测试、13 个实际 Hook 检查通过；DeepSeek 的两轮
原生 v1 转换对照、四组 Gloo 收发对照、关闭其他三个 Feature 后的四组独立收发对照
全部通过，最大误差 0；原 GLM 四组收发数值回归也通过。DeepSeek 实际双机训练尚未执行。

### BF16 FlashMLA 接口兼容（V3.4.1）

首次双机实测在推理 warmup 阶段报错：
`tile_scheduler_metadata must be of type FlashMLASchedMeta`。当前 DCU backend
将通用元数据生成器绑定到 FP8 接口，返回 Tensor；已安装的新版 BF16 decode 接口
要求 `FlashMLASchedMeta` 对象。该对象描述调度信息，并非权重精度。

新增 `DeepseekMlaApiFeature` 的两个 Hook，仅当 `model_type=deepseek_v3`、
KV cache 为 BF16、且原生 FlashMLA 提供新版接口时启用。每次 decode 使用原生
BF16 API 创建新调度对象，沿用现有 block table、长度、缩放和 Q/KV 布局。
不修改第三方文件，不转换权重或 KV cache；FP8、其他模型、旧版 API 保留原路径。
单独关闭兼容处理：`HCU_AREAL_DISABLE_DEEPSEEK_MLA_API=1`。

V3.4.1 验证：52 项单元/回归测试、15 个实际 Hook 检查通过。在 bw68 上以
真实 BF16 FlashMLA 内核连续测试两组 cache 长度，对照 FP32 Torch attention，
最大绝对误差分别为 0.0002575740 和 0.0001453310。该测试不代表完整模型 logits
一致，也不验证原始 FP8 权重的反量化；原 FP8 目录缺少数据分片，转换数值对照尚未完成。

**双机 HCU 实测通过（V3.4.1，2026-10-27）：**bw48 训练、bw61 推理，v2/AWEX，
训练 TP8/PP1/EP8/ETP1，推理 TP8，并发 4。完成两步训练和两次权重更新，耗时分别
15.956 秒、4.749 秒；`update_successful` 均为 1，梯度范数为 8.873e-5、5.3765e-5。
16 个真实模型进程均建立 world-size=16 的 RCCL IB 通信；821 个逻辑参数的名称、
全局形状、类型及推理 TP8 分片覆盖检查通过。主程序退出码 0，未生成训练权重文件，
结束后两个自有测试容器均已重启。两步奖励为 0；本次验证运行和更新链路，
不代表学习效果或完整模型 logits 与 v1 的数值一致性。

实际运行采用 `AWEX_WU_USE_GROUP=0`。该模式另在 bw48/bw68 上通过四组缩小张量的
GPU 收发数值对照，权重、MLA 缓存及计算探针与原生 v1 加载结果一致，最大误差 0。
此前 bw48/bw68 的完整模型 Socket 测试选中 1 Gbps 管理网，传输 28.15 GiB 的理论
下限约 242 秒，超过 gateway 默认的 120 秒更新期限；分组及逐项收发均在该期限超时。
随后的 `Broken pipe` 不能作为分组收发缺陷的证据。实际训练通过的传输方式为 IB。

实测目录：`areal_runs/deepseek-dual-20261027/deepseek-r1-v341-4layers-v2-ib_rdma-group0-bw48-bw61-20261027-184036`。
其中 `result.json`、`metrics.json`、`real-metadata-check.json`、`real-awex-network-proof.json`
分别记录结果、逐步指标、真实元数据检查及 IB 证据；`export-check.json` 记录无权重导出。

```bash
python -m unittest discover -s hcu_areal/tests -v
python -m hcu_areal.check_install --full --megatron --flashmla
python -m hcu_areal.validation.test_deepseek_flashmla_gpu --report /tmp/deepseek-flashmla.json
python -m hcu_areal.validation.test_deepseek_v1_conversion_parity --report /tmp/deepseek-v1.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 torchrun --standalone --nproc_per_node=16 \
  -m hcu_areal.validation.test_deepseek_awex_update_parity --backend gloo \
  --report /tmp/deepseek-awex.json
```

Gloo 模式下测试权重和传输均在 CPU；当前 Hygon Transformer Engine 仍要求导入时能查询
GPU 属性，因此应在空闲 DCU 节点使用现有环境。数值验证不启动 v1 控制器、不写出权重。

提供独立的双机运行配置，不修改外层 `hcu_example` 启动脚本：

* `validation/deepseek_dual_profile.sh`：训练 TP8/EP8/PP1，推理 TP8，BF16，
  SGLang 最大并发 **4**，rollout 并发 4，context 2048，显存比例 0.45；初始 2 step。
* `validation/deepseek_dual_config.yaml`：所有 saver/recover 的 epoch/step/second
  频率均为 null，避免 base config 原有的 3600 秒恢复快照。
* `validation/deepseek_dual_train.py`、`deepseek_v2_actor.py`：只允许两台 8 卡节点及
  v2/AWEX，拒绝权重导出，并保留已验证的 Torch FP32 router 与 gateway 退出清理。
* `validation/deepseek_ib_env.sh`：在 DTK 初始化之后对齐 LD_LIBRARY_PATH，选择各节点
  活跃的 12.12.12.* RoCE v2/GID3 端口和 10.16.1.* 管理网卡；无匹配端口就失败。
* `validation/worker_env/sitecustomize.py`：沿用原双机验证的 HSA 物理卡映射。

完整启动、只检查配置和回滚方法见 `validation/DEEPSEEK_USAGE.txt`。启动前须选择两台
8 卡空闲且通信预检通过的节点；优先 IB。无活跃 IB 端口时，须明确选择并验证
Socket 配置。失败后重启本次自有容器清理，不添加第三台机器。

## GLM-5 权重更新修复（2026-10-27）

此前四层 GLM 的实际错误为 `Train/infer key mismatch: train=841 infer=837`：SGLang 每层将 `q_a_proj`、`kv_a_proj_with_mqa` 合成一个 `fused_qkv_a_proj_with_mqa`。Megatron 训练侧另有通用 HF 转换表缺少 `glm_moe_dsa` 的问题。

V3.3 新增独立的 `Glm5AwexFeature`，仅对 `model_type=glm_moe_dsa` 启用：

1. 训练侧沿用原有 TP gather、PP 层号和 EP 专家编号，使用本地 `GLM5Bridge` 的映射生成 AWEX 的 HF 参数名；覆盖 MLA、DSA indexer、稠密 MLP、共享专家、路由专家及 correction bias。TP gather 显式处理 gate/up 的排列，兼容 stride=1/2，并保留 delta 更新的参数覆盖语义。
2. 推理侧按 `q_lora_rank` 与 `kv_lora_rank + qk_rope_head_dim` 拆分融合 A 投影，返回原张量的可写视图，接收权重时直接更新融合参数。
3. A 投影、DSA indexer 和 expert correction bias 按复制参数处理；其他参数保留既有的 TP/EP 分片规则。
4. 两端建立分离式通信组之前检查参数名、Global Shape 和 Dtype。遇到不一致直接失败，避免传输范围错误。
5. 权重接收后调用 SGLang 原有 `post_load_weights()` 刷新吸收式 MLA 的 `w_kc/w_vc`，并同步完成后再确认更新；colocate 路径在发布 completion 之前刷新。
6. V3.3 的真实双机预检已确认 841/841 参数名与形状一致，同时发现 8 个 DSA `indexer.k_norm` 参数训练侧为 BF16、推理侧为 FP32。V3.3.1 在训练侧发送时将这 8 个参数转为推理端对应的精度，训练参数和优化器精度保持原样。选择规则与 SGLang NSA indexer 一致：HIP AITER 路径为 BF16，其余为 FP32；两端的 `SGLANG_USE_AITER` 设置应一致。

原有 Qwen/Gemma 共享推理 Hook 增加 GLM 分发，Patch Manager 的冲突检测保持原样。GLM 自身注册训练侧及刷新 Hook；即使单独禁用 Qwen/Gemma Feature，GLM 所需的共享 Hook 仍会注册。

**验证范围：**38 项单元/回归测试通过；实际适配器 13 个 Hook 检查通过。以用户 `GLM-5-4Layers` 的 safetensors 头部构造 CPU/meta 测试：841 个训练侧参数映射，以及推理 TP8 全部参数名称、形状、精度、分片覆盖检查通过；另检查可写视图、gate/up 顺序、刷新与 ACK 的先后关系，并覆盖真实 GPU 日志中发现的 indexer LayerNorm 存储精度。测试夹具只含名称、形状与精度。

**双机 HCU 实测通过（2026-10-27）：**bw48 训练、bw61 推理，GLM-5 四层 BF16，v2 控制器，训练 attention TP4/PP2、FFN TP1/EP4/PP2，推理 TP8。完成 10 个训练步骤和 10 次 AWEX 权重更新，所有 `update_successful=1`，梯度范数均有限且非零，主程序退出码 0。真实 16 个模型进程均建立了 world-size=16 的 RCCL IB 通信组；841 个逻辑参数的名称、形状、精度与推理分片覆盖检查全部通过，传输精度为 832 个 BF16、9 个 FP32。测试未导出训练权重，结束后两个测试容器已重启。

本次沿用独立双机测试入口里的 Torch FP32 router 兼容设置（`moe_utils.te_general_gemm=None`），该设置不属于权重更新补丁。GPU 验证为 `SGLANG_USE_AITER=0`；AITER 路径仅覆盖 CPU 精度规则测试。四层裁剪模型此次奖励均为 0，因此上述结果验证运行和传输链路，不表示模型任务效果或 GRPO 学习效果已验证。

实测日志目录：`areal_runs/glm5-dual-20261027/glm5-v331-4layers-v2-ib-bw48-bw61-20261027-142856`，其中 `result.json`、`metrics.json`、`real-gpu-metadata-check.json`、`real-awex-ib-proof.json` 分别记录结果、逐步指标、真实元数据检查与 IB 证据。Ray 子进程需要通过 `SchedulingSpec.env_vars` 传递可选的 `HCU_AREAL_PATCH_TRACE_DIR`，两端同时传递 `SGLANG_USE_AITER`。

依赖现有 DCU `Megatron-Bridge` 中的 `glm_moe_dsa/glm5_bridge.py`；模型本身的注册仍需正确选择 `/workspace/dcu_megatron/Megatron-Bridge`。本补丁支持未量化的完整模型权重；量化训练/推理及 LoRA 会明确拒绝。

在 AReaL 仓库根目录解压 ZIP 后，使用已有 GLM 环境执行：

```bash
unzip -o /path/to/hcu_areal_GLM5_AWEX_RuntimePatch_v3.3.1.zip
/usr/local/bin/python -m unittest discover -s hcu_areal/tests -v
/usr/local/bin/python -m hcu_areal.check_install --full --megatron
```

已有 `.pth` 指向同一仓库时，新 Python 进程自动加载；尚未安装时执行 `/usr/local/bin/python hcu_areal/install_hook.py install`。旧进程不热更新，在自己的作业结束后重新启动训练/Ray 进程。`--full` 检查 10 个 FSDP/SGLang Hook，增加 `--megatron` 检查 13 个 AWEX Hook 和一个 DeepSeek FP8 加载 Hook；再加 `--flashmla` 检查两个 BF16 接口 Hook，共 16 个。

可选诊断：`export HCU_AREAL_PATCH_TRACE_DIR=/path/to/glm5-metadata` 保存各 rank 的参数元数据，不保存权重或凭据。禁用 GLM 补丁：`export HCU_AREAL_DISABLE_GLM5_AWEX=1`。回滚时恢复解压前的 `hcu_areal` 目录，并重新启动自己的 Python/Ray 进程。

## 验证状态（2026-10-15）

- Qwen3.5-4B V2 / FSDP DP4 / SGLang TP4 / AWEX **2-Step PASS**（此前 V3.1，在 bw67，`SGLANG_MEM_FRACTION_STATIC=0.55`）。
- Gemma3-4B V2 / FSDP DP4 / SGLang TP4 / AWEX：原始 Gemma3 测试日志 **FAIL**，报 `Train/infer key mismatch: train=883 infer=883`。
- 此 V3.2 **离线单元测试通过**，包括真实报错中 883 对权重名的映射和 CPU 分片测试；**尚未在 bw67 真实执行 Gemma3 AWEX**。不得标记 Gemma3 PASS。

## 实现的 Gemma3 适配

1. FSDP 名称转换：`model.language_model.*` → `language_model.model.*`；`model.vision_tower.*` → `vision_tower.*`；`model.multi_modal_projector.*` → `multi_modal_projector.*`。
2. Vision `self_attn.out_proj.*` → SGLang 的 `self_attn.proj.*`。基于最新日志 **883/883 名称一一对应**。
3. Vision QKV fused tensor 依照 Siglip `VisionAttention` 在第 0 维等长拆成 Q/K/V 的 **原张量零拷贝视图**，不借用 Gemma3 文本 Attention 的头数比例。
4. 修正 SGLang TP4 分片：Gemma3 Text Embedding（`nn.Embedding`）、Vision Conv2D Patch Embedding、Multimodal projector 为复制参数；Vision `self_attn.proj.weight` 和 `mlp.fc2.weight` 沿第 1 维分片，二者 Bias 复制。其余遵循原有 AWEX 策略。
5. FSDP 和 SGLang **两端** `init_weight_update_group` 先从原有 KV Store 获取元数据，检查双方参数名称、Global Shape 和 Dtype **全部一致**。有不一致则在首次 NCCL 通信前失败，给出前 8 个差异，避免“静默缺失权重”的假成功。
6. 只对 `model_type=gemma3` 执行 Gemma3 逻辑。`gemma3_text`、`gemma3n` 及 Qwen3.5/其他模型不走这部分规则；Qwen3.5 原有转换公式未改。

**实现取舍：**当前 `AReaLPatchesManager` 不允许对同一方法重复注册。Gemma3 专属 Feature 自己注册 FSDP `_to_hf_name` 与双方 `init_weight_update_group`，并通过既有 `_unfuse_params`、`_build_sharding_strategy` Hook 的模型分发调用 Gemma3 的纯逻辑模块。**不放宽 manager 冲突检测、不改 Qwen3.5 分支**。

## 安装

将 ZIP 上传到 bw67 的 `/code/hcu_areal_Gemma3_AWEX_RuntimePatch_v3.2.zip` 后：

```bash
cd /code/AReaL-das-fix-awex
# 现有 V3.1 已通过 Qwen3.5 冒烟，务必保留备份
test ! -e hcu_areal.before-gemma32 && cp -a hcu_areal hcu_areal.before-gemma32
unzip -o /code/hcu_areal_Gemma3_AWEX_RuntimePatch_v3.2.zip
/usr/local/bin/python -m unittest discover -s hcu_areal/tests -v
/usr/local/bin/python -m hcu_areal.check_install --full
```

V3.2 当时为 27 项测试、8 个 Hook；V3.3 的测试与 Hook 数量见上方 GLM 章节。`.pth` 已经指向同一目录，无需重新注册；如未安装，请用 `install_hook.py install`。启动/诊断输出只写 stderr。

运行 Gemma3 文本两步测试（**不是图文 VLM 验证**）：

```bash
cd /code/AReaL-das-fix-awex/hcu_example/grpo
SGLANG_MEM_FRACTION_STATIC=0.55 \
CONTROLLER_VERSION=v2 WEIGHT_UPDATE_MODE=awex \
GRPO_ENTRYPOINT=examples/math/gsm8k_rl.py \
GRPO_CONFIG=examples/math/gsm8k_grpo.yaml \
DATASET_PATH=openai/gsm8k \
TOTAL_TRAIN_STEPS=2 TRAIN_BATCH_SIZE=4 VALID_BATCH_SIZE=4 N_SAMPLES=2 \
MAX_NEW_TOKENS=256 ACTOR_MAX_TOKENS_PER_MB=2048 \
TRIAL_NAME=gemma3-4b-v2-awex-text-smoke-v32 \
bash run.sh --model=gemma3 --variant=vl --backend=fsdp --rollout=sglang \
  --model-path=/code/gemma3_4b --nodes=1 --gpus-per-node=8 --restart-ray
```

`--restart-ray` **会停止本机其他 Ray 作业**，请确认空闲后执行。Gemma3 专用启动脚本还应使用已确认的 `++sglang.enable_multimodal=True`（如果已修复，无需重复修改）。

## 回滚

在退出当前训练并停止旧的 Ray Worker 后：

```bash
cd /code/AReaL-das-fix-awex
mv hcu_areal hcu_areal.failed-gemma32
mv hcu_areal.before-gemma32 hcu_areal
```

原 `.pth` 路径保持不变；恢复后**重启 Ray**才能使子进程使用旧代码。

## 限制

此包没有在真实 HCU 上验证 Gemma3 推理端 Global Shape / dtype / 传输结果。即使预检通过，还要看 AWEX 的真实权重更新、`update_successful=1`、非零 `grad_norm` 以及两次 GRPO 更新；逐张量权重一致性仍需单独数值核对。真正的 Geometry3K VLM 还需要解决 V2 视觉张量轨迹传递，不能以 GSM8K 通过替代。
