"""Numerically compare the real v1 Bridge stream with the GLM AWEX patch.

Run in the existing DCU Megatron environment:
python -m hcu_areal.validation.test_glm5_v1_conversion_parity --report result.json
This calls v1 conversion functions in memory, never a v1 controller.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist

from .glm5_numeric_common import values, mcore_full, v1_reference, runtime_expected, assert_equal, provenance
from hcu_areal.features_manager.weight_update.glm5_awex_layout import glm5_convert_gathered, glm5_wire_tensor


def rejected(actual,expected):
    try:assert_equal(actual,expected)
    except AssertionError:return True
    raise AssertionError('Negative control was not detected')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args()
    os.environ['SGLANG_USE_AITER']='0'
    torch.set_num_threads(2)
    reports=[]
    with tempfile.TemporaryDirectory(prefix='glm5-parity-') as tmp:
        dist.init_process_group('gloo',init_method='file://'+str(Path(tmp)/'store'),rank=0,world_size=1)
        try:
            for version in (1,2):
                hf=values(version)
                reference=v1_reference(hf)
                full,_=mcore_full(hf)
                actual={}
                for name,tensor in full.items():
                    for key,value in glm5_convert_gathered('module.module.'+name,tensor):
                        assert key not in actual,key
                        actual[key]=glm5_wire_tensor(key,value)
                expected=runtime_expected(reference)
                assert_equal(actual,expected)
                raw_dtype_diffs=[n for n in actual if actual[n].dtype!=reference[n].dtype]
                assert len(raw_dtype_diffs)==8 and all('.indexer.k_norm.' in n for n in raw_dtype_diffs)
                # Failing examples ensure the comparator checks meaning, not counts.
                missing=dict(actual);missing.pop(next(iter(missing)))
                rejected(missing,expected)
                swapped=dict(actual)
                a,b='model.layers.0.mlp.gate_proj.weight','model.layers.0.mlp.up_proj.weight'
                swapped[a],swapped[b]=swapped[b],swapped[a]
                rejected(swapped,expected)
                expert=dict(actual)
                a,b='model.layers.3.mlp.experts.0.gate_proj.weight','model.layers.3.mlp.experts.64.gate_proj.weight'
                expert[a],expert[b]=expert[b],expert[a]
                rejected(expert,expected)
                wrong_dtype=dict(actual);name=raw_dtype_diffs[0];wrong_dtype[name]=wrong_dtype[name].bfloat16()
                rejected(wrong_dtype,expected)
                reports.append(dict(version=version,parameters=len(actual),max_abs_error=0.0,
                                    target_dtype_bitwise_equal=True,raw_dtype_differences=raw_dtype_diffs,
                                    negative_controls_detected=['missing weight','gate/up swap','EP expert swap','indexer norm wire dtype']))
        finally:dist.destroy_process_group()
    result=dict(status='passed',test='v1_bridge_vs_awex_conversion',rounds=reports,
                oracle='Native GLM5Bridge.stream_weights_megatron_to_hf with prepared tasks; actual mapping methods',
                sources=provenance(),weights_written=False,
                scope='All 841 four-layer/256-expert names with scaled numeric tensors; single-rank reference after gathering. Distributed TP/PP/EP is tested separately.')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
