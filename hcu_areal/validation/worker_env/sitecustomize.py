"""Set physical HSA visibility before importing Torch in a v2 actor guard."""
import os
import sys

args = getattr(sys,'orig_argv',[])
if '-m' in args and args[args.index('-m')+1] == 'areal.v2.training_service.guard':
    physical = os.environ.get('CUDA_VISIBLE_DEVICES','')
    if physical.isdigit() and not os.environ.get('ROCR_VISIBLE_DEVICES'):
        os.environ['ROCR_VISIBLE_DEVICES'] = physical
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ.pop('HIP_VISIBLE_DEVICES',None)
        print(f'HCU v2 guard: physical GPU {physical}, local GPU 0',flush=True)
