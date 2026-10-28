"""GLM regression coverage: real checkpoint shapes, writable views and refresh."""
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from hcu_areal.features_manager.weight_update.glm5_awex_layout import (
    is_glm5, glm5_inference_parts, Glm5ShardingProxy,
    glm5_mapping_registry, glm5_convert_gathered, glm5_wire_tensor, assert_glm5_transfer_metadata,
)
from hcu_areal.features_manager.weight_update.glm5_awex_feature import _refresh_factory, _train_iter_factory, _colocate_refresh_factory
from hcu_areal.features_manager.weight_update.qwen35_awex_feature import _infer_parts_factory, _infer_strategy_factory


def fixture():
    return json.loads((Path(__file__).parent / "data/glm5_4layers_shapes.json").read_text())


class TestGlmViews(unittest.TestCase):
    def test_indexer_norm_wire_dtype_matches_runtime_backend(self):
        import torch
        name = "model.layers.2.self_attn.indexer.k_norm.weight"
        source = torch.tensor([0.25, 1.5, -2.0], dtype=torch.bfloat16)
        for hip in (None, "6.3"):
            for flag in ("0", "1"):
                with patch.object(torch.version, "hip", hip), patch.dict(os.environ, {"SGLANG_USE_AITER":flag}):
                    wire = glm5_wire_tensor(name, source)
                    dtype = torch.bfloat16 if hip and flag == "1" else torch.float32
                    self.assertEqual(wire.dtype, dtype)
                    self.assertTrue(torch.equal(wire.float(), source.float()))
                    self.assertEqual(source.dtype, torch.bfloat16)
                    self.assertIs(glm5_wire_tensor("model.layers.2.self_attn.q_b_proj.weight", source), source)

    def test_fused_q_kv_split_writes_through_without_overlap(self):
        import torch
        cfg = NS(q_lora_rank=4, kv_lora_rank=2, qk_rope_head_dim=2, hidden_size=3)
        x = torch.arange(24).reshape(8, 3).clone()
        name = "model.layers.2.self_attn.fused_qkv_a_proj_with_mqa.weight"
        parts = glm5_inference_parts(name, x, cfg)
        self.assertEqual([n for n, _ in parts], ["model.layers.2.self_attn.q_a_proj.weight", "model.layers.2.self_attn.kv_a_proj_with_mqa.weight"])
        self.assertTrue(torch.equal(torch.cat([p for _, p in parts]), x))
        before = x.clone()
        parts[1][1].fill_(-2)
        self.assertTrue(torch.equal(x[:4], before[:4]))
        self.assertTrue(torch.all(x[4:] == -2))
        with self.assertRaisesRegex(ValueError, "Invalid GLM fused"):
            glm5_inference_parts(name, x[:7], cfg)
        self.assertIsNone(glm5_inference_parts("unrelated.weight", x, cfg))

    def test_guard_and_shared_hooks(self):
        import torch
        from awex.sharding.param_sharding import ShardingType
        cfg = NS(model_type="glm_moe_dsa", q_lora_rank=4, kv_lora_rank=2, qk_rope_head_dim=2, hidden_size=3)
        adapter = NS(_get_model=lambda: NS(config=cfg))
        parts = _infer_parts_factory(lambda self, n, t: [(n, t)])
        x = torch.zeros(8, 3)
        name = "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight"
        self.assertEqual(len(parts(adapter, name, x)), 2)
        base = NS(get_sharding_strategy=lambda n, **k: ("base", 0, 8))
        strategy = _infer_strategy_factory(lambda self, rank: base)(adapter, NS(tp_size=8))
        for n in ["model.layers.0.self_attn.q_a_proj.weight", "model.layers.0.self_attn.kv_a_proj_with_mqa.weight", "model.layers.0.self_attn.indexer.wq_b.weight", "model.layers.3.mlp.gate.e_score_correction_bias"]:
            self.assertEqual(strategy.get_sharding_strategy(n), (ShardingType.NO_SHARDING, 0, 1))
        self.assertEqual(strategy.get_sharding_strategy("model.layers.0.self_attn.q_b_proj.weight"), ("base", 0, 8))
        for model_type in ("glm4_moe", "deepseek_v2", "qwen2"):
            cfg.model_type = model_type
            self.assertEqual(len(parts(adapter, name, x)), 1)
            self.assertIs(_infer_strategy_factory(lambda self, rank: base)(adapter, NS(tp_size=8)), base)
        cfg.model_type = "glm_moe_dsa"
        with patch.dict(os.environ, {"HCU_AREAL_DISABLE_GLM5_AWEX":"1"}):
            self.assertFalse(is_glm5(adapter, "infer"))
            self.assertEqual(len(parts(adapter, name, x)), 1)

    def test_metadata_rejects_missing_names_shapes_dtypes_and_duplicates(self):
        meta = lambda name, shape=(3, 4), dtype="bf16": NS(name=name, global_shape=shape, dtype=dtype)
        a, b = [meta("a"), meta("b")], [meta("a"), meta("b")]
        self.assertEqual(assert_glm5_transfer_metadata(a, b), 2)
        with self.assertRaisesRegex(ValueError, "missing=1"):
            assert_glm5_transfer_metadata(a[:1], b)
        with self.assertRaisesRegex(ValueError, "shape_mismatch=1"):
            assert_glm5_transfer_metadata([meta("a", (4, 3)), meta("b")], b)
        with self.assertRaisesRegex(ValueError, "dtype_mismatch=1"):
            assert_glm5_transfer_metadata([meta("a", dtype="fp32"), meta("b")], b)
        with self.assertRaisesRegex(ValueError, "duplicate infer"):
            assert_glm5_transfer_metadata([a[0], a[0]], b)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            assert_glm5_transfer_metadata([], [])


class TestGlmRefresh(unittest.TestCase):
    def test_refresh_after_success_and_never_after_failed_transfer(self):
        import torch
        w = torch.eye(3)
        model = NS(config=NS(model_type="glm_moe_dsa"), quant_config=None, cache=w.clone())
        events = []
        def refresh():
            events.append("refresh")
            model.cache.copy_(w)
        model.post_load_weights = refresh
        adapter = NS(_get_model=lambda: model)
        def receive(self, version):
            events.append("receive")
            w.fill_(version)
            return "ack"
        with patch("areal.infra.platforms.current_platform", NS(synchronize=lambda: events.append("sync"))):
            wrapped = _refresh_factory(receive)
            self.assertEqual(wrapped(adapter, 2), "ack")
            self.assertEqual(events, ["receive", "refresh", "sync"])
            self.assertTrue(torch.equal(torch.ones(1, 3) @ model.cache, torch.ones(1, 3) @ w))
            events.clear()
            def failed(self, version):
                raise RuntimeError("transfer failed")
            with self.assertRaisesRegex(RuntimeError, "transfer failed"):
                _refresh_factory(failed)(adapter, 3)
            self.assertEqual(events, [])
            model.config.model_type = "qwen2"
            wrapped(adapter, 4)
            self.assertEqual(events, ["receive"])

    def test_missing_refresh_fails_before_receiving(self):
        model = NS(config=NS(model_type="glm_moe_dsa"))
        adapter = NS(_get_model=lambda: model)
        def unexpected(*args):
            self.fail("Weights must not change before refresh compatibility is checked")
        with self.assertRaisesRegex(RuntimeError, "post_load_weights"):
            _refresh_factory(unexpected)(adapter, 1)

    def test_colocate_refresh_precedes_completion_and_restores_client(self):
        events = []
        model = NS(config=NS(model_type="glm_moe_dsa"), quant_config=None,
                   post_load_weights=lambda: events.append("refresh"))
        client = NS(put=lambda *a, **k: events.append("ack"))
        adapter = NS(_get_model=lambda: model, _colocate_http_client=client)
        def update(self, version):
            events.append("receive")
            self._colocate_http_client.put("http://kv/weight_meta/pair/colocate_done_rank0_1", json={"value":True})
        with patch("areal.infra.platforms.current_platform", NS(synchronize=lambda: events.append("sync"))):
            _colocate_refresh_factory(update)(adapter, 1)
            self.assertEqual(events, ["receive", "refresh", "sync", "ack"])
            self.assertIs(adapter._colocate_http_client, client)
            def failed(self, version):
                raise RuntimeError("failed before ACK")
            with self.assertRaisesRegex(RuntimeError, "failed before ACK"):
                _colocate_refresh_factory(failed)(adapter, 2)
            self.assertIs(adapter._colocate_http_client, client)


class TestGlmCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = fixture()

    def test_all_841_bridge_names_shapes_and_dtypes(self):
        import torch
        try:
            registry = glm5_mapping_registry()
        except ImportError as exc:
            self.skipTest(f"Existing DCU GLM5Bridge required: {exc}")
        headers, groups, converted = self.data["tensors"], {}, {}
        for name in headers:
            mapping = registry.hf_to_megatron_lookup(name)
            self.assertIsNotNone(mapping, name)
            groups[mapping.megatron_param] = mapping
        for name, mapping in groups.items():
            target = mapping.hf_param
            if isinstance(target, dict):
                info = headers[target["gate"]]
                shape = [info["shape"][0] * 2] + info["shape"][1:]
            else:
                info = headers[target]
                shape = info["shape"]
            dtype = {"BF16":torch.bfloat16, "F32":torch.float32}[info["dtype"]]
            for hf_name, value in glm5_convert_gathered("module.module." + name, torch.empty(shape, dtype=dtype, device="meta")):
                self.assertNotIn(hf_name, converted)
                self.assertEqual(list(value.shape), headers[hf_name]["shape"], hf_name)
                self.assertEqual(value.dtype, {"BF16":torch.bfloat16,"F32":torch.float32}[headers[hf_name]["dtype"]])
                converted[hf_name] = value
        self.assertEqual(len(converted), 841)
        self.assertEqual(set(converted), set(headers))
        with self.assertRaisesRegex(ValueError, "Unsupported GLM AWEX mapping"):
            glm5_convert_gathered("unknown.weight", torch.empty(1))

    @patch.dict(os.environ, {"SGLANG_USE_AITER":"0"})
    def test_actual_sglang_adapter_all_checkpoint_shapes_at_tp8(self):
        """Exercise the real adapter with meta tensors, no GPU/checkpoint load."""
        import torch
        from areal.v2.weight_update.awex.sglang_adapter import AwexSGLangAdapter
        from awex.sharding.param_sharding import ShardingStrategy

        cfg = NS(**self.data["config"])
        cfg.tie_word_embeddings = False
        headers = self.data["tensors"]
        meta = lambda s, d="BF16": torch.nn.Parameter(torch.empty(s, dtype={"BF16":torch.bfloat16,"F32":torch.float32}[d], device="meta"), requires_grad=False)
        runtime = []
        for name, info in headers.items():
            shape = list(info["shape"])
            if name.endswith(".self_attn.q_a_proj.weight"):
                shape[0] += cfg.kv_lora_rank + cfg.qk_rope_head_dim
                runtime.append((name.replace("q_a_proj", "fused_qkv_a_proj_with_mqa"), meta(shape)))
                continue
            if name.endswith(".self_attn.kv_a_proj_with_mqa.weight"):
                continue
            replicated = any(x in name for x in ("layernorm", "model.norm", ".indexer.", ".mlp.gate."))
            if not replicated:
                axis = 1 if name.endswith((".o_proj.weight", ".down_proj.weight")) else 0
                self.assertEqual(shape[axis] % 8, 0, name)
                shape[axis] //= 8
            # Actual NSA indexer LayerNorm storage observed in the GPU run.
            dtype = "F32" if ".indexer.k_norm." in name else info["dtype"]
            runtime.append((name, meta(shape, dtype)))
        self.assertEqual(len(runtime), 837)
        model = NS(config=cfg, named_parameters=lambda: iter(runtime))
        all_meta = []
        for rank in range(8):
            server = NS(tp_size=8, pp_size=1, dp_size=1, ep_size=1, enable_dp_attention=False, enable_dp_lm_head=False, moe_dense_tp_size=None)
            scheduler = NS(server_args=server, tp_rank=rank, pp_rank=0, tp_worker=NS(model_runner=NS(model=model)))
            adapter = AwexSGLangAdapter(scheduler)
            metadata = adapter.get_weight_metadata()
            self.assertEqual(len(metadata), 841)
            expected = [NS(name=n, global_shape=i["shape"], dtype=glm5_wire_tensor(n, torch.empty(0, dtype={"BF16":torch.bfloat16,"F32":torch.float32}[i["dtype"]])).dtype) for n, i in headers.items()]
            self.assertEqual(assert_glm5_transfer_metadata(metadata, expected), 841)
            views = adapter.get_local_shard_parameters()
            self.assertEqual(set(views), set(headers))
            all_meta.append({m.name:m for m in metadata})
        # Every sharded tensor covers the full target dimension exactly once;
        # replicated tensors retain offset 0 on all ranks.
        for name, info in headers.items():
            shards = [row[name].shards[0] for row in all_meta]
            s = shards[0]
            if int(s.num_shards) == 1:
                self.assertTrue(all(not any(v.global_offset) for v in shards))
            else:
                axis = s.sharding_dim
                ranges = sorted((v.global_offset[axis], v.global_offset[axis]+v.shape[axis]) for v in shards)
                self.assertEqual(ranges[0][0], 0)
                self.assertEqual(ranges[-1][1], info["shape"][axis])
                self.assertTrue(all(a[1] == b[0] for a,b in zip(ranges, ranges[1:])))

    def test_train_override_and_tp_gate_up_gather(self):
        import torch
        try:
            glm5_mapping_registry()
        except ImportError as exc:
            self.skipTest(str(exc))
        from areal.engine.megatron_utils.megatron import _all_gather_and_concat
        p = torch.nn.Parameter(torch.arange(24, dtype=torch.float32).reshape(8, 3))
        p.tensor_model_parallel, p.partition_dim, p.partition_stride = True, 0, 1
        theta = p.detach().clone() + 100
        engine = NS(hf_config=NS(model_type="glm_moe_dsa"), config=NS(use_lora=False), model=None,
                    tf_config=NS(num_moe_experts=256, gated_linear_unit=True), _duplicated_param_names=set())
        adapter = NS(_engine=engine)
        original = lambda *a, **k: iter(())
        norm = torch.nn.Parameter(torch.ones(128, dtype=torch.bfloat16))
        overrides = {id(p):theta}
        named = [("module.module.decoder.layers.0.mlp.linear_fc1.weight", p),
                 ("module.module.decoder.layers.0.self_attention.core_attention.indexer.k_norm.weight", norm)]
        with patch.dict(os.environ, {"SGLANG_USE_AITER":"0"}), patch("areal.engine.megatron_utils.megatron.get_named_parameters", return_value=named), patch("areal.engine.megatron_utils.megatron.all_gather_param", side_effect=lambda name, src, **k: src) as gather:
            parts = list(_train_iter_factory(original)(adapter, overrides, True))
        self.assertTrue(gather.call_args.kwargs["gated_linear_unit"])
        self.assertEqual(overrides, {})
        self.assertTrue(torch.equal(parts[0][1], theta[:4]))
        self.assertTrue(torch.equal(parts[1][1], theta[4:]))
        self.assertEqual(parts[2][1].dtype, torch.float32)
        self.assertEqual(norm.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(parts[2][1], norm.float()))
        # Simulate real rank-local [gate;up] storage with both MCore stride APIs.
        rank_parts = [torch.tensor([[1.],[2.],[11.],[12.]]), torch.tensor([[3.],[4.],[13.],[14.]])]
        def all_gather(outputs, local, group):
            for out, value in zip(outputs, rank_parts): out.copy_(value)
        with patch("areal.engine.megatron_utils.megatron.dist.all_gather", side_effect=all_gather):
            for stride in (1, 2):
                joined = _all_gather_and_concat(rank_parts[0], 2, None, 0, stride,
                                                "mlp.linear_fc1.weight", gated_linear_unit=True)
                self.assertEqual(joined.flatten().tolist(), [1.,2.,3.,4.,11.,12.,13.,14.])


if __name__ == "__main__":
    unittest.main()
