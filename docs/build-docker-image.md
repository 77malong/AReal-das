# 构建 DTK 26.10 和 Torch 2.13 镜像

Dockerfile 默认使用光源社区 Harbor 的 Ubuntu 22.04、Python 3.12 和 DTK 26.10 开发镜像，固定了
`26.10-ubuntu22.04-mpi5.0-gcc11.4-cmake3.29-py3.12` 标签对应的 digest。基础镜像应提供 `hipcc`、C/C++
编译器、Git、curl 和 Python venv，可通过 `BASE_IMAGE` 指定其他满足条件的镜像。默认镜像同时为 BW1000 和 BW1100 编译
SGLang 原生算子。

在 AReal-das 仓库根目录执行以下命令。

```bash
docker build -t areal-das:v2.1.0-ubuntu22.04-dtk26.10-py3.12-torch2.13-20261125 .
```

如果宿主机没有安装 Buildx 插件，在 `docker build` 前加上 `DOCKER_BUILDKIT=0`，使用 Docker 的传统构建器。
如果构建容器无法直连包索引，宿主机已配置代理时可给 `docker build` 加上
`--build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg NO_PROXY`。

`HCU_TARGET=dual` 将 BW1000 的 `gfx936` 和 BW1100 的 `gfx938` 编入同一个 SGLang 原生库。
需要构建单卡型镜像时，可以通过 `--build-arg HCU_TARGET=bw1000` 或 `--build-arg HCU_TARGET=bw1100` 指定。
`io.hygon.hcu.target` 记录构建目标。预编译 wheel 使用各自发布时的架构配置，需在目标卡上验证对应算子。 各构建目标可以复用 AICC、SGLang
Python 包和运行依赖的 Docker 缓存。

Dockerfile 在独立的 Python 环境里安装 Torch 2.13 和 HCU wheel，并从同一 SGLang-DAS 提交 构建 SGLang 和 HCU
kernel。Triton 在 HCU 上进行即时编译时需要额外的 AICC LLVM，Dockerfile 通过 SHA-256 校验并安装 DTK 26.10 的
clang-22 构建。安装包来源见
[FlagOS 构建配置](https://github.com/flagos-ai/build-infra/blob/main/base/hygon-dtk26.10)。

构建时为 SGLang 的 RoPE kernel 声明 512 线程的上限，与固定源码中的启动参数一致。 在 BW1000 上使用 `fa3` 时，示例 YAML 通过
`page_size: 64` 满足 FlashAttention 的 KV 缓存页大小要求。

Megatron 在导入时读取 NVRx 的版本，Dockerfile 从固定提交构建 NVRx，并关闭依赖 CUDA CUPTI 的性能分析扩展。 镜像中的
`mbridge` 兼容固定版本 Megatron 的模型类型枚举，也能从 Transformers 的 `rope_parameters` 读取 Qwen3 的
`rope_theta`。 Megatron 的 GatedDeltaNet 使用 `flash-linear-attention[rocm]==0.5.1` 提供的
Triton kernel。 安装 FLA 时沿用已固定的 HCU Torch 和 Triton 依赖。

AReaL v2 数据代理和数学训练工作流分别需要 LiteLLM 和 Agents SDK。镜像沿用已发布 AReal-das 镜像的 LiteLLM
`1.103.0`、OpenAI SDK `2.33.0` 和 Agents SDK `0.14.8`。 Geometry3K 示例用 `pylatexenc==2.11`
解析 LaTeX 答案。

SGLang 的 HIP DSA 索引器依赖
[HCU TileLang wheel](../docker/requirements-dtk2610-torch213-dsa.txt)。 安装时保留运行环境的依赖版本，包括
NumPy `1.26.0`。 固定的 fast Hadamard wheel 提供 `gfx938` 原生算子。在 BW1000 上，镜像中的
[DSA 兼容补丁](../hcu_areal/third_party/sglang/dsa-hadamard.patch) 让索引器使用 SGLang 自带的
Hadamard 实现。

在 HCU 上，[Gemma RMSNorm 补丁](../hcu_areal/third_party/sglang/gemma-rmsnorm.patch) 让 SGLang
使用自身的 Triton 实现，修复 vLLM 算子在无残差路径上未写入输出的问题。

原生 wheel 的下载地址及哈希见
[HCU 依赖清单](../docker/requirements-dtk2610-torch213.txt)。可在光源社区查询其他适配版本的
[Release wheel](https://pypi.sourcefind.cn/das_release/dtk2610/+simple/) 和
[Nightly wheel](https://pypi.sourcefind.cn/das_nightly/dtk2610/+simple/)，并参考
[SGLang-DAS 的 HCU 构建说明](https://github.com/HYGON-AI/sglang-das/blob/453edca64008bdaf162bb42dcebda14bd1b4927a/README_HCU.md)。

构建完成后，可在宿主机检查版本。

```bash
image=areal-das:v2.1.0-ubuntu22.04-dtk26.10-py3.12-torch2.13-20261125
docker image inspect --format '{{index .Config.Labels "io.hygon.hcu.target"}}' "$image"
docker run --rm --entrypoint python3 "$image" \
  -c 'from importlib.metadata import version; import torch; print(torch.version.hip); print({p: version(p) for p in ("torch", "sglang", "sglang-kernel", "awex", "flash-linear-attention", "fla-core")})'
```

## HCU 验证

按[快速上手](quickstart.md#3-%E5%88%9B%E5%BB%BA%E5%AE%B9%E5%99%A8)在宿主机挂载 HCU
设备、`/opt/hyhal`、仓库和模型。在两种卡型上使用同一个双目标镜像 ID 分别验证，将 Qwen3-4B BF16 检查点只读挂载到
`/models/Qwen3-4B`。 基础镜像中的 DTK 设备库用于编译和链接，HCU 运行时依赖宿主机驱动。

在容器内先检查 8 张可见设备及基础算子，再执行两步 GRPO。

```bash
python3 -c 'import torch; assert torch.cuda.device_count() == 8; print([torch.cuda.get_device_properties(i).gcnArchName for i in range(8)]); x = torch.ones((64, 64), device="cuda", dtype=torch.bfloat16); print((x @ x)[0, 0].item())'

export PYTHONPATH="/opt/hyhal/share/smi${PYTHONPATH:+:${PYTHONPATH}}"
NCCL_P2P_DISABLE=1 python3 -m hcu_areal.launch --config examples/configs/gsm8k_qwen3_megatron.yaml \
  --set 'training.actor.path=/models/Qwen3-4B' \
  --set 'runtime.ray.num_gpus=8' \
  --set 'runtime.env.PYTORCH_ALLOC_CONF=expandable_segments:False' \
  --set 'runtime.env.PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False' \
  --set 'training.total_train_steps=2'
```

验证时关闭了可扩展显存分段。`NCCL_P2P_DISABLE=1` 让同机 RCCL 选用其他通信传输， 规避训练端和推理端使用不同设备可见范围时遇到的 HIP
`invalid argument`。 运行 Qwen3-VL 示例也应使用命令中的两项 `expandable_segments:False` 覆盖。保持
示例中默认的可扩展分段时，实测 SGLang 在构造视觉编码器期间发生段错误。

在训练日志里核对采样、两个训练步、AWEX 权重同步和梯度范数，并检查进程退出状态。当前 Transformer Engine wheel 在导入时会查询 HCU
设备属性，涉及 Megatron 的 AWEX 导入须在 实际的训练环境中验证。已安装的 vLLM wheel 与 SGLang 使用的部分 Python 依赖版本不同， 涉及
vLLM 的路径仍需单独验证。

BW1000 和 BW1100 上，GLM-5 四层 BF16 的默认 DSA 路径会触发 `FP8 index K cache is not enabled`。
运行[GLM-5 示例](../examples/README.md#glm5-megatron)时，在训练命令末尾追加
`--set 'training.sglang.attention_backend=hcu_mla'`。该后端已通过 BW1000 单机和双机两步 GRPO，也在 BW1100
上通过单卡生成验证。

DeepSeek-R1 四层 FP8 在 BW1000 和 BW1100 上均用 SGLang 的 Triton GEMM 完成两步 GRPO。复现时沿用上述 RCCL
和显存分配设置，并把 `HF_MODULES_CACHE` 指向本次容器可写的缓存目录。

DeepSeek‑V4‑Flash 四层示例分别配置训练端和推理端的显存分配器。Megatron 使用原生缓存分配，对超过 4 GiB 的块限制拆分，并在分配量达到 75%
时回收可释放的缓存。SGLang 关闭可扩展分段。这个配置在 BW1000 上使用 4 张训练卡和两个 TP2 推理副本完成两步训练和两次 AWEX 同步，两步梯度均为
0。启动方式见[模型示例](../examples/README.md#deepseek-v4-flash-megatron)。

## 单卡型镜像历史验证

BW1000 和 BW1100 镜像的 SGLang AOT 二进制分别包含 `gfx936` 和 `gfx938`。两种镜像各通过 49 项 CPU 单测，并成功展开 16
份示例 YAML。BW1000 已完成 14 份单机和 2 份双机示例的两步短训，也通过了 Qwen3-4B CI 基线。各模型使用的检查点及运行覆盖项见
[HCU 示例](../examples/README.md)。BW1100 已在 `gfx938` 实卡验证原生 KV 分配、FP8 输入 GEMM、fast
Hadamard，以及 FLA 和 Transformer Engine 的 BF16 前向与反向。

在 BW1100 上，Qwen2.5-1.5B 的 FSDP 和 Megatron 示例，以及 Qwen3.5-4B 的 FSDP 示例，均使用四张卡完成两步短训及两次
AWEX 同步。 三个四卡用例的梯度均为 0，覆盖采样、训练和同步流程。单机八卡的 Qwen3-4B CI 基线与 DeepSeek-R1 四层 FP8 也完成了两步及两次
AWEX 同步，后者两步梯度均为非零。

## BW1000 基线镜像验证记录

双目标构建改造前的 BW1000 基线镜像使用 Python `3.12.12`、Torch `2.13.0+dtk2610.2610261432.gefb4e1`、
SGLang `0.5.18+g453edca`、SGLang kernel `0.4.6.post1`、AWEX `0.8.1`。 镜像保留项目的 NumPy
`1.26.0`，并使用兼容的 PyArrow `25.0.1` 来创建训练数据集。 在无卡容器中验证了 Torch CPU 运算、SGLang 参数模块导入和 HCU
kernel 动态加载。 `tests/unit` 通过 46 项、跳过 1 项。16 个示例 YAML 均通过配置展开检查，其中包括 Qwen3-4B 单机 8 卡两步配置。

在 BW1000 上验证了 8 卡枚举、Torch BF16 矩阵乘法、Triton 即时编译、Transformer Engine BF16 前向和反向，以及双卡 RCCL
`all_reduce`。 在单张 BW1000 上，SGLang 使用 FA3 和 64 token 缓存页加载 Qwen3-4B BF16 检查点，成功完成生成请求。

Qwen3-4B BF16 在 `bw51` 单机 8 张 BW1000 上完成两步 GRPO。训练和推理各使用 4 张卡， 运行仓库的 CI 短训脚本，使用 8
条训练样本的算术数据集、每条提示采样 2 次，最多生成 256 token。 两步梯度范数分别为 `13.234` 和 `11.008`，v2 AWEX 分别在约
`5.55 s` 和 `5.47 s` 内完成权重同步。 这项验证使用上述显存分配和 RCCL 设置，CI 短训脚本退出码为 0。退出时 SGLang 服务触发了强制清理。
