import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from hcu_areal.features_manager.weight_update import deepseek_mla_api_feature as feature


class TestDeepseekMlaApi(unittest.TestCase):
    def api(self, calls):
        import torch
        class Metadata:pass
        def decode(**kwargs):
            calls.append(kwargs)
            return torch.zeros((*kwargs['q'].shape[:-1],512)),None
        return NS(metadata_type=Metadata,get_metadata=lambda:(Metadata(),None),decode=decode)

    def test_modern_bf16_metadata_is_fresh_and_fp8_metadata_untouched(self):
        import torch
        calls=[]
        api=self.api(calls)
        backend=NS(data_type=torch.bfloat16,kv_lora_rank=512,kv_cache_dim=576,qk_rope_head_dim=64,
                   forward_metadata=NS(block_kv_indices=torch.zeros(2,1,dtype=torch.int32),flashmla_metadata='old FP8 metadata'))
        runner=NS(model_config=NS(hf_config=NS(model_type='deepseek_v3')))
        layer=NS(tp_q_head_num=16,head_dim=576,scaling=.1)
        with patch.object(feature,'native_bf16_api',return_value=api):
            feature._init_factory(lambda *a,**k:None)(backend,runner)
            fn=feature._decode_factory(lambda *a,**k:self.fail('BF16 must use the modern metadata API'))
            for length in (17,31):
                out=fn(backend,torch.zeros(2,16,576),torch.zeros(128,576),torch.full((2,),length),2,layer)
                self.assertEqual(tuple(out.shape),(2,1,16,512))
        self.assertIsNot(calls[0]['tile_scheduler_metadata'],calls[1]['tile_scheduler_metadata'])
        self.assertTrue(all(isinstance(c['tile_scheduler_metadata'],api.metadata_type) for c in calls))
        self.assertTrue(all(c['num_splits'] is None and c['is_fp8_kvcache'] is False for c in calls))
        self.assertEqual(backend.forward_metadata.flashmla_metadata,'old FP8 metadata')

    def test_other_models_dtypes_and_legacy_api_fall_through(self):
        import torch
        for model,dtype,api in (('glm_moe_dsa',torch.bfloat16,self.api([])),
                               ('deepseek_v3',torch.float8_e4m3fn,self.api([])),
                               ('deepseek_v3',torch.bfloat16,None)):
            backend=NS(data_type=dtype)
            runner=NS(model_config=NS(hf_config=NS(model_type=model)))
            with patch.object(feature,'native_bf16_api',return_value=api):
                feature._init_factory(lambda *a,**k:None)(backend,runner)
                self.assertFalse(backend._hcu_deepseek_bf16_flashmla)
                self.assertEqual(feature._decode_factory(lambda *a,**k:'original')(backend,None,None,None,1,None),'original')

    def test_separate_rope_query_is_concatenated_in_native_order(self):
        import torch
        calls=[]
        backend=NS(_hcu_deepseek_bf16_flashmla=True,kv_lora_rank=512,kv_cache_dim=576,qk_rope_head_dim=64,
                   forward_metadata=NS(block_kv_indices=torch.zeros(1,1,dtype=torch.int32)))
        layer=NS(tp_q_head_num=16,head_dim=576,scaling=.1)
        with patch.object(feature,'native_bf16_api',return_value=self.api(calls)):
            feature._decode_factory(lambda *a,**k:None)(backend,torch.ones(1,16,512),torch.zeros(64,576),
                torch.tensor([1]),1,layer,q_rope=torch.full((1,16,64),2.))
        query=calls[0]['q']
        self.assertTrue(torch.all(query[...,:512]==1))
        self.assertTrue(torch.all(query[...,512:]==2))
