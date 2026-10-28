# hcu_areal — Qwen3.5 HCU AWEX Runtime Patch v3（候选版）

## 适用范围
- **模型**：Qwen3.5-4B `Qwen3_5ForConditionalGeneration`（Dense，全量模型含视觉模块）
- **训练端**：AReaL V2 + FSDP DP4；**推理端**：SGLang TP4；**同步**：AWEX
- **源码隔离**：仅放在仓库根目录 `hcu_areal/`，不修改 `areal/`、SGLang、AWEX 源码。
- **生效方式**：基于 Megatron-LM-das 的 Feature / Manager，并通过 `.pth` + bootstrap 注入各 Ray 子进程。
- **默认自动注册**：不需要 `HCU_AREAL_PATCH_QWEN35_AWEX=1`。只在 Qwen3.5 Adapter 实例执行转换；其他模型回退原方法。

## 重要状态
使用你提供的八份元数据，对 `4x FSDP + 4x SGLang` 建立 canonical 权重契约：
**两端名称数量都为 915，915/915 名称与 Global Shape、dtype 完全一致；FSDP 分片无孔洞/重叠**。
本候选实现尚未在 bw67 运行 AWEX NCCL 数据传输及训练，**不等于端到端训练通过**。

## 安装
在 bw67 的 `/code/AReaL-das-fix-awex` 目录，先复制并备份当前 `hcu_areal`，
然后将 ZIP 内的 `hcu_areal/` 覆盖旧目录的对应代码。原 `.pth` 已指向仓库根目录，通常无需重新安装。

```bash
cd /code/AReaL-das-fix-awex
cp -a hcu_areal hcu_areal.before-qwen35-awex-v3
unzip -o /tmp/hcu_areal_Qwen35_AWEX_RuntimePatch_v3.zip
/usr/local/bin/python -m unittest discover -s hcu_areal/tests -v
/usr/local/bin/python -m hcu_areal.check_install --full
```

> 再运行之前 AWEX 的 2-step 命令（`--restart-ray` 仅当本机没有其他 Ray 任务）。
> 不需要额外 export `HCU_AREAL_PATCH_QWEN35_AWEX`；如果要收集新日志，可选 `HCU_AREAL_PATCH_TRACE_DIR=...`。

## 内容
1. 训练端规范化 `model.language_model.*`、视觉参数前缀和 full-attention `self_attn`；
2. 训练端将 GDN `in_proj_qkv` / `conv1d`、Vision `qkv` 根据 FSDP 原始 shard 交集拆为零拷贝视图；
3. 推理端将 `in_proj_qkvz` / `in_proj_ba`、GDN conv、门控 Full Attention QKV、Vision QKV 以 TP4 本地布局拆为视图；
4. 修正视觉 `patch_embed` 复制参数、`attn.proj/linear_fc2` dim1 分片与 Bias 类型；
5. GDN `A_log` 发送端转 float32，匹配推理端；
6. AWEX 仍然执行严格 key-match；不设置 `strict=False` 以强行掩盖错误；
7. 双端元数据和 tensor 参数使用相同转换规则，数据通过真实原始张量视图传输，不生成无关副本。

## 验证标准
- 不再出现 `Train/infer key mismatch` 或 global Shape 错位。
- AWEX `init_weights_update_group` 完整返回，且至少两个训练 Step 成功。
- `grad_norm > 0` 且 `update_successful=1`。
- 除上述冒烟指标外，若要判定“权重数值正确”，仍需比对更新前后训练端和 SGLang 实际张量的数值/生成分布；只看通信成功不够。

## 诊断
默认诊断信息写 stderr；需要 JSON 时可加：
```
HCU_AREAL_PATCH_TRACE_DIR=/code/areal_runs/qwen35-awex-v3-trace
```

## 回滚
停止任务并恢复旧文件夹：
```
cd /code/AReaL-das-fix-awex
mv hcu_areal hcu_areal.v3-failed
mv hcu_areal.before-qwen35-awex-v3 hcu_areal
```
`.pth` 仍指向同一项目路径，无需重装；需要重启 Ray 才会载入旧代码。

### 启动流程说明

当前版本启用 .pth 后会默认安装导入监听，但不会在 Python 启动时导入 `torch`、`areal`、`awex`、`sglang`。
仅在目标 Adapter 被实际导入后包装方法，并且仅在 Qwen3.5 模型上执行特殊逻辑；其他模型调用原方法。
如果启动脚本捕获 Python stdout 获取虚拟环境路径，hcu_areal 的自身诊断只会写入 stderr，避免再发生 `/usr/local/bin/python/bin/activate` 错误。

### 相关文档

此次 ZIP 同级还提供 `TRACE分析与验证报告.md`，列出本次 8 份 trace 的分析结论与验证边界。
