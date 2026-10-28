"""Real AWEX byte/scale transfer versus native v1 loading, with FP8 storage.

Uses 8 training ranks (TP8/EP8) and 8 inference ranks (TP8). Gloo is a CPU
transport test, not a claim of FP8 GEMM support. --backend nccl additionally
uses the native GPU quantizer. Neither mode runs a controller or exports weights.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import torch.distributed as dist

from .deepseek_fp8_numeric_common import (Fp8InferenceFixture,Fp8TrainingFixture,
    quant_config,reference_hf,compare_fp8_models)
from .deepseek_numeric_common import config,provenance
from .test_deepseek_awex_update_parity import set_mpu,inference_adapter,merged_metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('gloo','nccl'),default='gloo')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    os.environ['SGLANG_USE_AITER']='0'
    os.environ['DTE_DELTA_TRANSFER']='0'
    os.environ['DTE_SEPARATION_WEIGHT_UPDATE']='0'
    os.environ.pop('HCU_AREAL_PATCH_TRACE_DIR',None)
    torch.set_num_threads(1)
    rank=int(os.environ['RANK']);assert int(os.environ['WORLD_SIZE'])==16
    device='cpu'
    if args.backend=='nccl':
        device='cuda:'+os.environ['LOCAL_RANK'];torch.cuda.set_device(device)
    dist.init_process_group('gloo',timeout=datetime.timedelta(seconds=240))
    control=dist.new_group(list(range(16)),backend='gloo')
    transfer=dist.new_group(list(range(16)),backend=args.backend)
    singles=[dist.new_group([i],backend='gloo') for i in range(16)]
    tp=dist.new_group(list(range(8,16)),backend=args.backend)
    set_mpu(singles[rank],singles[rank],singles[rank],singles[rank],singles[rank])
    rounds=[]
    try:
        for mixed,fused,source_fp8 in ((False,False,False),(True,True,True)):
            cfg=quant_config(mixed)
            os.environ['HCU_AREAL_DEEPSEEK_FP8_CONFIG']=json.dumps(cfg)
            if rank<8:
                model=Fp8InferenceFixture(rank,device,fused,mixed,v1=False)
                reference=Fp8InferenceFixture(rank,device,fused,mixed,v1=True)
                adapter=inference_adapter(model,rank)
                refs={version:reference_hf(version,cfg,device,source_fp8) for version in (1,2)}
            else:
                set_mpu(tp,singles[rank],tp,singles[rank],singles[rank])
                model=Fp8TrainingFixture(rank-8,device,source_fp8)
                engine=NS(model=[model],tf_config=model.config,hf_config=config(),config=NS(use_lora=False),
                          quantization_config=cfg,_duplicated_param_names=set(),rank=rank-8,world_size=8,
                          data_parallel_rank=0,data_parallel_world_size=1)
                from areal.v2.weight_update.awex.megatron_adapter import AwexMegatronAdapter
                adapter=AwexMegatronAdapter(engine)
            local=adapter.get_weight_metadata();gathered=[None]*16
            dist.all_gather_object(gathered,local,group=control)
            infer,train=merged_metadata(gathered[:8]),merged_metadata(gathered[8:])
            from hcu_areal.features_manager.weight_update.deepseek_awex_layout import assert_deepseek_transfer_metadata
            expected=1619 if mixed else 1621
            assert assert_deepseek_transfer_metadata(infer,train)==expected
            from awex.transfer.transfer_plan import TransferPlanBuilder
            adapter._transfer_plan=TransferPlanBuilder(infer_world_size=8,train_world_size=8,num_infer_engines=1).build_local_transfer_plan(infer,train,global_transfer_rank=rank)
            adapter._weights_update_group=transfer;adapter._weights_update_group_gloo=control
            adapter._transfer_rank=rank;adapter._world_size=16
            if rank==0:print('FP8_METADATA_READY',expected,'mixed',mixed,flush=True)
            if args.backend=='nccl':
                from areal.v2.weight_update.nccl_group import setup_batch_isend_irecv
                setup_batch_isend_irecv(transfer,rank,16,barrier_group=control)
            for version in (1,2):
                if rank>=8:model.update(version)
                dist.barrier(group=control)
                adapter.execute_weight_update(version)
                dist.barrier(group=control)
                if rank<8:
                    reference.load_v1(refs[version]);compare_fp8_models(model,reference)
                    assert model.refreshes==version
                    if version==2:
                        parameter=next(p for n,p in model.named_parameters() if n.endswith('weight_scale_inv'))
                        saved=parameter.data.clone();parameter.data.zero_()
                        try:compare_fp8_models(model,reference)
                        except AssertionError:pass
                        else:raise AssertionError('Missing scale update was not detected')
                        parameter.data.copy_(saved)
                        cache=model.model.layers[0].self_attn.w_kc
                        saved=cache.clone();cache.zero_()
                        try:compare_fp8_models(model,reference)
                        except AssertionError:pass
                        else:raise AssertionError('Stale MLA cache was not detected')
                        cache.copy_(saved)
                        compare_fp8_models(model,reference)
                dist.barrier(group=control)
                if rank==0:
                    record=dict(mixed_precision=mixed,fused_shared_expert=fused,
                        plain_fp8_source=source_fp8,version=version,logical_parameters=expected,
                        weight_storage_bitwise_equal=True,fp32_scales_bitwise_equal=True,
                        mla_cache_equal=True,mla_probe_equal=True,max_abs_error=0.0)
                    rounds.append(record);print('FP8_UPDATE_PASS '+json.dumps(record),flush=True)
        if rank==0:
            result=dict(status='passed',backend=args.backend,device=device,world_size=16,
                wire_dtype='uint8 for E4M3FN storage; float32 for block scales',
                train_layout='TP8/PP1/EP8/ETP1',infer_layout='TP8',rounds=rounds,
                oracle='Native v1 Bridge conversion and native SGLang do_load_weights/post_load_weights; scaled FP8 tensors',
                fp8_gemm_verified=False,full_model_training_verified=False,weights_written=False,
                source_hashes=provenance())
            args.report.parent.mkdir(parents=True,exist_ok=True)
            args.report.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    finally:
        dist.destroy_process_group()


if __name__=='__main__':main()
