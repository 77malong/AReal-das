# hcu_areal — AReaL HCU 运行时适配层

本目录借鉴 [HYGON-AI/Megatron-LM-das](https://github.com/HYGON-AI/Megatron-LM-das/tree/core_v0.18.2)
中的 `hcu_megatron/patch_utils.py`、`megatron_adaptor.py`、`features_manager/` 分层。
**与 Megatron-LM-das 的差异**：AReaL V2 的训练和推理服务分别在 Ray 子进程中运行，
因此需增加 `bootstrap.py` 的按模块延迟导入，并用 `.pth` 确保每个 Python 进程加载。

> 当前版本 `v0.1.0` **只提供 Patch 基础框架和 Qwen3.5 AWEX 元数据诊断**，
> **尚未实现 Qwen3.5 的参数融合/拆分与 TP4 张量重排**。
> 当前 AWEX `Train/infer key mismatch` 仍会出现，不要据此声称 AWEX 已经修复。

## 目录职责

| 文件/目录 | 职责（对照 hcu_megatron） |
| --- | --- |
| `patch_utils.py` | `Patch`、`AReaLPatchesManager`；注册、应用、撤销补丁 |
| `areal_adaptor.py` | 汇集启用的 Feature、`patch_features()`、`repatch()` |
| `features_manager/feature.py` | `AbstractFeature` 抽象接口 |
| `features_manager/weight_update/qwen35_awex_feature.py` | Qwen3.5 专用 AWEX 元数据诊断 Feature |
| `bootstrap.py` | Lazy import hook；避免 Python 初始化阶段加载 torch/SGLang |
| `install_hook.py` | 在当前 Python 的 site-packages 中注册/移除 `.pth` 入口 |
| `python_startup/sitecustomize.py` | 备选入口（可能被其他 sitecustomize 覆盖） |
| `check_install.py` | 检查启动注入是否真实生效 |
| `tests/` | 无 GPU/无 AWEX 的 CPU 单元测试 |

## 1. 安装到已有 AReaL 源码目录

将包的 `hcu_areal` 文件夹直接解压到仓库根目录：

```bash
cd /code/AReaL-das-fix-awex
unzip /tmp/hcu_areal_MegatronStyle_RuntimePatch_v2.zip
/usr/local/bin/python -m unittest discover -s hcu_areal/tests -v
/usr/local/bin/python -m compileall -q hcu_areal
```

不会修改 `areal/`、`awex/` 或 `/workspace/sglang/` 下的任何 Python 文件。

## 2. 从旧 areal_patches 迁移

如果之前使用过旧包，**先卸载旧的 `.pth` 注册文件**：

```bash
cd /code/AReaL-das-fix-awex
if test -f areal_patches/install_hook.py; then
  /usr/local/bin/python areal_patches/install_hook.py uninstall
fi
unset AREAL_PATCH_QWEN35_AWEX AREAL_PATCH_TRACE_DIR AREAL_EXTRA_PYTHONPATH
```

旧 `areal_patches/` 可以暂时留作备份。确认新方案工作正常后再人工清理。
新包的命名和变量如下：

- 目录：`hcu_areal/`
- Patch 开关：`HCU_AREAL_PATCH_QWEN35_AWEX=1`
- 诊断输出目录：`HCU_AREAL_PATCH_TRACE_DIR=/code/areal_runs/qwen35-awex-trace`
- 入口文件：`zz_hcu_areal_runtime.pth`

## 3. 注册运行时启动 Hook

当前 Worker 使用 `/usr/local/bin/python`；一定要以和 Ray Worker 相同的解释器执行：

```bash
cd /code/AReaL-das-fix-awex
/usr/local/bin/python hcu_areal/install_hook.py install
/usr/local/bin/python hcu_areal/install_hook.py status
```

该命令只添加一个 `.pth` 文件；**补丁默认关闭**。
不修改已安装的 AWEX、AReaL 或 SGLang 包。

## 4. 验证注入

```bash
export HCU_AREAL_PATCH_QWEN35_AWEX=1
export HCU_AREAL_PATCH_TRACE_DIR=/code/areal_runs/qwen35-awex-trace

/usr/local/bin/python -m hcu_areal.check_install
/usr/local/bin/python -m hcu_areal.check_install --full
```

快速检查只判断本进程的启动 Hook 是否存在；`--full` 还需要成功导入
FSDP 和 SGLang 的 AWEX Adapter，并确认其 `get_weight_metadata()` 已打补丁。
`--full` 可能加载依赖，但不分配训练任务。

为了让 Ray、FSDP、SGLang 全部使用新补丁，必须在**启动 Ray 之前**设置环境变量，
随后使用你原有且已验证的 `--restart-ray` 命令。仅修改已经运行的进程环境没有作用。
对实际多节点任务，需在每个节点的同一 Python 环境安装 Hook。

## 5. Qwen3.5 当前诊断范围

只有 `HCU_AREAL_PATCH_QWEN35_AWEX=1` 时启用 Qwen3.5 AWEX Trace Feature。
它只包装如下方法，仍调用并返回原始实现的结果：

```python
areal.v2.weight_update.awex.fsdp_adapter.AwexFSDPAdapter.get_weight_metadata
areal.v2.weight_update.awex.sglang_adapter.AwexSGLangAdapter.get_weight_metadata
```

指定 `HCU_AREAL_PATCH_TRACE_DIR` 后，尝试保存参数名、shape、shard offset、dtype 等元数据，
不导出权重张量。当前映射失败来自 Qwen3.5 Transformers/FSDP 与 SGLang 的
`language_model` 前缀、Gated DeltaNet 融合参数和视觉 QKV 布局差异。
接下来需要独立实现两端 `tensor + metadata + sharding` 一致的转换，做 TP4 数值验证。
不能通过关闭严格校验、丢弃参数或仅调整名字假装修复。

## 6. 关闭或移除

```bash
unset HCU_AREAL_PATCH_QWEN35_AWEX HCU_AREAL_PATCH_TRACE_DIR
/usr/local/bin/python hcu_areal/install_hook.py uninstall
```

然后重启 Ray/Python Worker。进程退出即可恢复原始方法，无须 `git reset --hard`。
注：`.pth` 执行时需要仓库目录在所有训练节点上可见。
