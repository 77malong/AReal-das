"""Real AWEX P2P updates versus v1 SGLang loading, using numeric GLM fixtures.

CPU: torchrun --standalone --nproc_per_node=16 -m hcu_areal.validation.test_glm5_awex_update_parity --backend gloo --report result.json
GPU: the same module with --backend nccl, 2 nodes x 8 ranks. Node 0 is inference,
node 1 is training. Source layout is TP4/PP2/EP4; destination layout is TP8.
No controllers or checkpoint export are used. This is not a full-model logits test.
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace as NS

import torch
import torch.distributed as dist

from .glm5_numeric_common import (
    config, values, mcore_full, v1_reference, hf_axis,
    InferenceFixture, compare_models, provenance,
)


def set_mpu(tp,pp,ep,etp,single):
    """Bind the fixture's actual process groups without starting an engine."""
    from megatron.core import parallel_state as mpu
    for key,value in dict(_TENSOR_MODEL_PARALLEL_GROUP=tp,
                          _PIPELINE_MODEL_PARALLEL_GROUP=pp,
                          _EXPERT_MODEL_PARALLEL_GROUP=ep,
                          _EXPERT_TENSOR_PARALLEL_GROUP=etp,
                          _DATA_PARALLEL_GROUP=single,
                          _DATA_PARALLEL_GROUP_WITH_CP=single,
                          _CONTEXT_PARALLEL_GROUP=single).items():
        setattr(mpu,key,value)
    for key in ('_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE','_MPU_TENSOR_MODEL_PARALLEL_RANK',
                '_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE','_MPU_PIPELINE_MODEL_PARALLEL_RANK',
                '_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE','_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK'):
        setattr(mpu,key,None)


class TrainingFixture:
    """Rank-local native Megatron names, parameter flags and PP/EP placement."""
    def __init__(self,local_rank,device):
        self.rank=local_rank
        self.pp,self.tp=divmod(local_rank,4)
        self.device=device
        self.config=NS(num_layers=4,pipeline_model_parallel_size=2,
                       pipeline_model_parallel_layout=None,
                       num_layers_in_first_pipeline_stage=None,num_layers_in_last_pipeline_stage=None,
                       virtual_pipeline_model_parallel_size=None,
                       account_for_embedding_in_pipeline_split=False,
                       account_for_loss_in_pipeline_split=False,
                       num_moe_experts=256,gated_linear_unit=True)
        self._params=[]
        self._buffers=[]
        self.update(1,initialize=True)

    def update(self,version,initialize=False):
        full,mappings=mcore_full(values(version))
        slots={name:param for name,param in self._params+self._buffers}
        for global_name,tensor in full.items():
            layer=re.search(r'decoder.layers\.(\d+)\.',global_name)
            owner=int(layer.group(1))//2 if layer else (0 if global_name.startswith('embedding.') else 1)
            if owner!=self.pp:continue
            local=global_name
            if layer:local=local.replace('layers.'+layer.group(1)+'.','layers.'+str(int(layer.group(1))%2)+'.',1)
            expert=re.search(r'\.experts\.linear_fc[12]\.weight(\d+)$',global_name)
            if expert:
                index=int(expert.group(1))
                if index//64!=self.tp:continue
                local=local[:local.rfind('weight')]+f'weight{index%64}'
            dst=mappings[global_name].hf_param
            axis=hf_axis(dst['gate'] if isinstance(dst,dict) else dst)
            if not expert and axis is not None:
                if isinstance(dst,dict):
                    gate,up=tensor.chunk(2,dim=0)
                    tensor=torch.cat([gate.chunk(4,dim=0)[self.tp],up.chunk(4,dim=0)[self.tp]],dim=0)
                else:tensor=tensor.chunk(4,dim=axis)[self.tp]
            tensor=tensor.to(self.device).contiguous()
            name='module.'+local
            if initialize:
                param=torch.nn.Parameter(tensor,requires_grad=False)
                param.tensor_model_parallel=axis is not None
                param.partition_dim=axis if axis is not None else 0
                param.partition_stride=2 if isinstance(dst,dict) else 1
                (self._buffers if 'expert_bias' in local else self._params).append((name,param))
            else:slots[name].data.copy_(tensor)

    def named_parameters(self):return iter(self._params)
    def named_buffers(self):return iter(self._buffers)


def inference_adapter(model,rank):
    from areal.v2.weight_update.awex.sglang_adapter import AwexSGLangAdapter
    server=NS(tp_size=8,pp_size=1,dp_size=1,ep_size=1,enable_dp_attention=False,
              enable_dp_lm_head=False,moe_dense_tp_size=None)
    scheduler=NS(server_args=server,tp_rank=rank,pp_rank=0,
                 tp_worker=NS(model_runner=NS(model=model)))
    adapter=AwexSGLangAdapter(scheduler)
    adapter._get_model_context=lambda:dict(scheduler=scheduler,tp_rank=rank,tp_size=8,
        pp_rank=0,pp_size=1,dp_size=1,world_size=8,global_rank=rank,local_rank=rank,
        attn_tp_rank=rank,attn_tp_size=8,attn_dp_rank=0)
    return adapter


def merged_metadata(entries):
    from areal.infra.rpc.serialization import serialize_value,deserialize_value
    from areal.v2.weight_update.gateway.app import _merge_training_meta_by_name
    flat=[m for rank in entries for m in rank]
    return deserialize_value(_merge_training_meta_by_name(serialize_value(flat)))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend',choices=('gloo','nccl'),default='gloo')
    ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args()
    os.environ['SGLANG_USE_AITER']='0'
    os.environ['AWEX_WU_USE_GROUP']='1'
    os.environ['DTE_DELTA_TRANSFER']='0'
    os.environ['DTE_SEPARATION_WEIGHT_UPDATE']='0'
    os.environ.pop('HCU_AREAL_PATCH_TRACE_DIR',None)
    torch.set_num_threads(1)
    rank=int(os.environ['RANK'])
    assert int(os.environ['WORLD_SIZE'])==16,'Run 8 infer + 8 train ranks'
    device='cpu'
    if args.backend=='nccl':
        device='cuda:'+os.environ['LOCAL_RANK']
        torch.cuda.set_device(device)
    dist.init_process_group('gloo',timeout=datetime.timedelta(seconds=180))
    control=dist.new_group(list(range(16)),backend='gloo')
    transfer=dist.new_group(list(range(16)),backend=args.backend)
    singles=[dist.new_group([i],backend='gloo') for i in range(16)]
    set_mpu(singles[rank],singles[rank],singles[rank],singles[rank],singles[rank])
    # Real v1 conversion on singleton CPU groups, independently of v2 gathers.
    references={v:v1_reference(values(v)) for v in (1,2)}
    if rank==0:print('V1_REFERENCE_READY',flush=True)
    train_tps=[dist.new_group(list(range(start,start+4)),backend=args.backend) for start in (8,12)]
    train_pps=[dist.new_group([8+i,12+i],backend='gloo') for i in range(4)]
    dist.barrier(group=control)
    reports=[]
    try:
        for fused_shared in (False,True):
            if rank>=8:
                local=rank-8
                pp,tp=divmod(local,4)
                set_mpu(train_tps[pp],train_pps[tp],train_tps[pp],singles[rank],singles[rank])
                from areal.v2.weight_update.awex.megatron_adapter import AwexMegatronAdapter
                model=TrainingFixture(local,device)
                engine=NS(model=[model],tf_config=model.config,hf_config=config(),config=NS(use_lora=False),
                          quantization_config=None,_duplicated_param_names=set(),rank=local,world_size=8,
                          data_parallel_rank=0,data_parallel_world_size=1)
                adapter=AwexMegatronAdapter(engine)
            else:
                model=InferenceFixture(rank,device=device,fused_shared=fused_shared,v1=False)
                reference=InferenceFixture(rank,device=device,fused_shared=fused_shared,v1=True)
                adapter=inference_adapter(model,rank)
            from awex.transfer.transfer_plan import TransferPlanBuilder
            local_meta=adapter.get_weight_metadata()
            gathered=[None]*16
            dist.all_gather_object(gathered,local_meta,group=control)
            inf,train=merged_metadata(gathered[:8]),merged_metadata(gathered[8:])
            from hcu_areal.features_manager.weight_update.glm5_awex_layout import assert_glm5_transfer_metadata
            assert assert_glm5_transfer_metadata(inf,train)==841
            if rank==0:print('AWEX_METADATA_READY',fused_shared,flush=True)
            adapter._transfer_plan=TransferPlanBuilder(infer_world_size=8,train_world_size=8,num_infer_engines=1).build_local_transfer_plan(inf,train,global_transfer_rank=rank)
            adapter._weights_update_group=transfer
            adapter._weights_update_group_gloo=control
            adapter._transfer_rank=rank
            adapter._world_size=16
            # Initialize every peer before issuing the real batched P2P plan.
            if args.backend=='nccl':
                from areal.v2.weight_update.nccl_group import setup_batch_isend_irecv
                setup_batch_isend_irecv(transfer,rank,16,barrier_group=control)
            else:
                # The runtime warmup constructs accelerator:index devices and
                # does not accept CpuPlatform.current_device() == 'cpu'. Use
                # real CPU P2P to initialize the Gloo-only test group instead.
                warmup=torch.full((100,),rank,dtype=torch.float32)
                received=torch.zeros_like(warmup)
                op=dist.P2POp(dist.irecv,received,rank+8,group=transfer) if rank<8 else dist.P2POp(dist.isend,warmup,rank-8,group=transfer)
                for work in dist.batch_isend_irecv([op]):work.wait()
                dist.barrier(group=control)
                assert rank>=8 or torch.all(received==rank+8)
            for version in (1,2):
                if rank>=8:model.update(version)
                dist.barrier(group=control)
                adapter.execute_weight_update(version)
                dist.barrier(group=control)
                if rank<8:
                    assert model.refreshes==version,('Expected one MLA refresh per update',model.refreshes,version)
                    reference.load_v1(references[version])
                    compare_models(model,reference)
                    # Independently check logical destination slices too.
                    actual=adapter.get_local_shard_parameters()
                    assert len(actual)==841
                    for name,full in references[version].items():
                        axis=hf_axis(name)
                        expected=full if axis is None else full.chunk(8,dim=axis)[rank]
                        expected=expected.to(device=device,dtype=actual[name].dtype)
                        assert torch.equal(actual[name],expected),(name,'readback mismatch')
                    if version==2:
                        # Poison a cache and a weight: both must be detectable.
                        cache=model.model.layers[0].self_attn.w_kc
                        old=cache.clone();cache.zero_()
                        try:compare_models(model,reference)
                        except AssertionError:pass
                        else:raise AssertionError('Stale cache negative control was missed')
                        cache.copy_(old)
                        first=next(model.parameters());old=first.detach().clone()
                        first.data.zero_()
                        try:compare_models(model,reference)
                        except AssertionError:pass
                        else:raise AssertionError('Missing update negative control was missed')
                        first.data.copy_(old)
                    compare_models(model,reference)
                dist.barrier(group=control)
                if rank==0:
                    reports.append(dict(fused_shared_expert=fused_shared,version=version,
                                        weights_per_infer_rank=841,infer_ranks=8,
                                        raw_parameter_blocks_per_rank=65 if fused_shared else 67,
                                        max_abs_error=0.0,raw_parameters_bitwise_equal=True,
                                        mla_cache_bitwise_equal=True,mla_matmul_probe_equal=True,
                                        refreshes_per_rank=version,
                                        negative_controls_detected=['stale MLA cache','missing update'] if version==2 else []))
                    print('NUMERIC_UPDATE_PASS '+json.dumps(reports[-1]),flush=True)
            adapter._weights_update_group=None
            adapter._weights_update_group_gloo=None
            del adapter,model
        if rank==0:
            result=dict(status='passed',test='awex_update_vs_v1_native_sglang_loading',backend=args.backend,tensor_device=device,
                        train_layout='TP4/PP2/EP4/ETP1',infer_layout='TP8',world_size=16,
                        rounds=reports,sources=provenance(),weights_written=False,
                        oracle='Native v1 Bridge conversion + native SGLang do_load_weights/post_load_weights source; scaled parameter modules and loader callbacks',
                        limitations='All 841 names and distributed layouts, scaled tensor dimensions. MLA matmul probe, not full-model logits. Gloo mode does not validate RCCL/IB.')
            args.report.parent.mkdir(parents=True,exist_ok=True)
            args.report.write_text(json.dumps(result,indent=2))
    finally:
        dist.destroy_process_group()


if __name__=='__main__':main()
