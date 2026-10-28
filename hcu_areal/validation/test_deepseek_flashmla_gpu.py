"""Small real GPU decode versus FP32 Torch attention, without model weights."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace as NS

import torch


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args()
    from sglang.srt.layers.attention.dcu_mla_backend import DCUMLABackend
    markers=getattr(DCUMLABackend._call_decode,'__hcu_areal_patch_ids__',())
    assert 'deepseek.mla.bf16.decode' in markers
    torch.cuda.set_device(0)
    torch.manual_seed(123)
    backend=DCUMLABackend.__new__(DCUMLABackend)
    backend._hcu_deepseek_bf16_flashmla=True
    backend.kv_lora_rank,backend.kv_cache_dim,backend.qk_rope_head_dim=512,576,64
    cache=(torch.randn(4,64,576,device='cuda')*.1).bfloat16()
    query=(torch.randn(2,1,16,576,device='cuda')*.1).bfloat16()
    table=torch.tensor([[0,1],[2,3]],device='cuda',dtype=torch.int32)
    backend.forward_metadata=NS(block_kv_indices=table,flashmla_metadata='unused FP8 metadata')
    layer=NS(tp_q_head_num=16,head_dim=576,scaling=192**-.5)
    rounds=[]
    for lengths in ((73,31),(97,63)):
        seqlens=torch.tensor(lengths,device='cuda',dtype=torch.int32)
        actual=backend._call_decode(query,cache,seqlens,2,layer)
        expected=[]
        for batch,length in enumerate(lengths):
            tokens=torch.cat([cache[page] for page in table[batch].tolist()],dim=0)[:length].float()
            scores=query[batch,0].float()@tokens.T*layer.scaling
            expected.append(torch.softmax(scores,dim=-1)@tokens[:,:512])
        expected=torch.stack(expected).unsqueeze(1)
        torch.testing.assert_close(actual.float(),expected,rtol=.01,atol=.001)
        rounds.append(dict(lengths=list(lengths),max_abs_error=float((actual.float()-expected).abs().max())))
    torch.cuda.synchronize()
    result=dict(status='passed',test='real_gpu_bf16_flashmla_vs_torch_fp32',rounds=rounds,
                weights_written=False,full_model_forward=False)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
