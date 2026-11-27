# HCU 适配测试

`unit/` 覆盖补丁应用与回滚、进程启动和 Ray 环境、版本兼容，以及模型权重名称、布局与 FP8 数值。默认测试使用小型 CPU 张量，不需要模型权重或 HCU
设备；运行环境需要 Python 3.11/3.12、pytest、psutil、PyYAML 和支持 float8 的 PyTorch。

在仓库根目录运行：

```bash
export PYTHONPATH="$(python3 -m hcu_areal.launch --pythonpath)"
python3 -m pytest -q -rs tests/unit
```

完整运行 Guard 测试还需要可导入的 AReaL 子模块及其依赖。使用项目镜像并初始化子模块，检查 pytest 的跳过原因。

PR 的 CPU 测试入口是 `.github/scripts/run_hcu_unit_tests.sh`，CI 在配置的 HCU
镜像中运行且不暴露设备。多卡、真实模型和通信链路不由这些单元测试证明。模型短训按 PR 中可选的 `CI_MODELS:` 选择；未填写时运行 Qwen3-4B 两步基线。可用
case 及资源要求见 `.github/workflows/README.md`。

## GDN 前向精度

`unit/test_gdn_precision.py` 的 CPU 用例核对 L2Norm、衰减门和逐 token 递推参考。指定
`GDN_PRECISION_BACKEND=nvidia` 或 `hcu` 后，同一入口用原生 FLA 和 Megatron 执行硬件用例。两端共用含
`fixture.json`、`fixture.pt` 和 `reference-tp{1,2}.pt` 的 定值目录，由实验的 `prepare-fixture.py`
生成。用例加载时检查数据 SHA-256。

在已配置 Megatron、TE、FLA 和设备可见性的环境运行下列命令。请设置实际的 `GDN_PRECISION_FIXTURE` 与
`GDN_PRECISION_OUTPUT` 路径。

```bash
export GDN_PRECISION_BACKEND=hcu
export GDN_PRECISION_FIXTURE=/path/to/fixture
GDN_PRECISION_OUTPUT=/path/to/results/single \
  python3 -m torch.distributed.run --standalone --nproc-per-node=1 --module pytest \
  -q tests/unit/test_gdn_precision.py -k 'test_device_operator or (test_device_module and short)'
GDN_PRECISION_OUTPUT=/path/to/results/tp2-1 \
  python3 -m torch.distributed.run --standalone --nproc-per-node=2 --module pytest \
  -q tests/unit/test_gdn_precision.py -k 'test_device_module and not short'
```

每个用例连续执行三遍，保存每 rank 的精度误差、首个超限边界和逐位重复性数据。 独立进程组使用独立输出目录，便于比较同一调用序号。
