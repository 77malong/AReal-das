"""Compare all 821 DeepSeek weights against native v1 Bridge conversion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist

from .deepseek_numeric_common import values, mcore_full, v1_reference, assert_equal, provenance
from hcu_areal.features_manager.weight_update.deepseek_awex_layout import deepseek_convert_gathered


def must_reject(actual, expected):
    try:
        assert_equal(actual,expected)
    except AssertionError:
        return
    raise AssertionError('Negative control was not detected')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report',type=Path,required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    rounds = []
    with tempfile.TemporaryDirectory(prefix='deepseek-parity-') as tmp:
        dist.init_process_group('gloo',init_method='file://'+str(Path(tmp)/'store'),rank=0,world_size=1)
        try:
            for version in (1,2):
                reference = v1_reference(values(version))
                full,_ = mcore_full(values(version))
                actual = {}
                for name,tensor in full.items():
                    for key,value in deepseek_convert_gathered('module.module.'+name,tensor):
                        assert key not in actual,key
                        actual[key] = value
                assert_equal(actual,reference)
                missing = dict(actual)
                missing.pop(next(iter(missing)))
                must_reject(missing,reference)
                for a,b in (
                    ('model.layers.0.mlp.gate_proj.weight','model.layers.0.mlp.up_proj.weight'),
                    ('model.layers.3.mlp.experts.0.gate_proj.weight','model.layers.3.mlp.experts.32.gate_proj.weight'),
                ):
                    swapped = dict(actual)
                    swapped[a],swapped[b] = swapped[b],swapped[a]
                    must_reject(swapped,reference)
                bias = 'model.layers.3.mlp.gate.e_score_correction_bias'
                rounded = dict(actual)
                rounded[bias] = rounded[bias].bfloat16().float()
                must_reject(rounded,reference)
                rounds.append(dict(version=version,parameters=821,max_abs_error=0.0,
                                   dtype_bitwise_equal=True,negative_controls_detected=[
                                       'missing weight','gate/up swap','EP expert swap','FP32 bias downcast']))
        finally:
            dist.destroy_process_group()
    result = dict(status='passed',test='deepseek_v1_bridge_vs_awex_conversion',rounds=rounds,
                  sources=provenance(),weights_written=False,
                  scope='Real 821-key schema, scaled numeric tensors, native v1 conversion methods; not full-model logits')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
