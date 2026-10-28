#!/usr/bin/env python3
"""Convert a DeepSeek blockwise FP8 safetensors checkpoint to BF16 on CPU.

Conversion needs torch and safetensors; --inspect uses only the standard library.
Writes a separate directory, retaining shard names and rebuilding the index.
"""

import argparse
import collections
import json
import math
import shutil
import struct
from pathlib import Path


def numel(shape):
    size = 1
    for dimension in shape:
        size *= dimension
    return size


def inspect_checkpoint(source):
    config = json.loads((source / "config.json").read_text())
    index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    headers = {}
    for shard in sorted(set(weight_map.values())):
        with (source / shard).open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            if size > 100_000_000:
                raise ValueError(f"Invalid safetensors header: {shard}")
            header = json.loads(stream.read(size))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if weight_map.get(name) != shard or name in headers:
                raise ValueError(f"Index/header mismatch: {name}")
            headers[name] = info
    if set(headers) != set(weight_map):
        raise ValueError("Index contains missing tensors")
    quant = config.get("quantization_config")
    fp8 = {name for name, info in headers.items() if info["dtype"].startswith("F8")}
    scales = {name + "_scale_inv" for name in fp8}
    if fp8:
        if not quant or (
            quant.get("quant_method"), quant.get("fmt"), quant.get("activation_scheme")
        ) != ("fp8", "e4m3", "dynamic"):
            raise ValueError("Only DeepSeek dynamic E4M3 blockwise FP8 is supported")
        block = quant.get("weight_block_size")
        if not isinstance(block, list) or len(block) != 2 or any(
            not isinstance(size, int) or size <= 0 for size in block
        ):
            raise ValueError(f"Invalid weight_block_size: {block}")
        for name in fp8:
            info = headers[name]
            scale = headers.get(name + "_scale_inv")
            expected = [math.ceil(n / b) for n, b in zip(info["shape"], block)]
            if info["dtype"] != "F8_E4M3" or len(info["shape"]) != 2:
                raise ValueError(f"Unsupported FP8 tensor: {name}: {info}")
            if scale is None or scale["dtype"] != "F32" or scale["shape"] != expected:
                raise ValueError(f"Missing or incompatible block scale: {name}")
    dtypes = dict(collections.Counter(info["dtype"] for info in headers.values()))
    expected_bytes = sum(
        numel(info["shape"]) * 2 if name in fp8
        else info["data_offsets"][1] - info["data_offsets"][0]
        for name, info in headers.items() if name not in scales
    )
    report = {
        "source": str(source), "tensor_dtypes": dtypes,
        "fp8_weight_count": len(fp8), "quantization_config": quant,
        "output_tensor_bytes": expected_bytes,
        "output_tensor_GiB": round(expected_bytes / 2**30, 2),
    }
    print(json.dumps(report, indent=2), flush=True)
    return config, index, headers, fp8, scales


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--require-bf16", action="store_true",
                        help="Inspect and reject quantized/FP8/FP16 model weights")
    args = parser.parse_args()
    source = args.source.resolve()
    config, index, headers, fp8, scales = inspect_checkpoint(source)
    if args.require_bf16:
        if config.get("quantization_config") or fp8 or any(
            name.endswith("_scale_inv") or info["dtype"] not in ("BF16", "F32", "I64", "I32", "BOOL")
            for name, info in headers.items()
        ):
            raise ValueError("Checkpoint is not unquantized BF16/FP32")
        print("PASS: unquantized BF16/FP32 checkpoint")
        return
    if args.inspect:
        return
    if args.output is None or not fp8:
        parser.error("Conversion requires --output and an FP8 source checkpoint")
    output = args.output.resolve()
    if output == source or (output.exists() and any(output.iterdir())):
        raise ValueError("Output must be a separate, empty or nonexistent directory")
    # Import after inspection so header checks also work on a machine without torch.
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    block_m, block_n = config["quantization_config"]["weight_block_size"]
    output.mkdir(parents=True, exist_ok=True)
    new_map, total_size = {}, 0
    for shard in sorted(set(index["weight_map"].values())):
        converted = {}
        tensor = None
        with safe_open(str(source / shard), framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if name in scales:
                    continue
                tensor = reader.get_tensor(name)
                if name in fp8:
                    scale_name = name + "_scale_inv"
                    scale_shard = index["weight_map"][scale_name]
                    with safe_open(str(source / scale_shard), framework="pt", device="cpu") as scale_reader:
                        scale = scale_reader.get_tensor(scale_name).float()
                    rows, cols = tensor.shape
                    restored = torch.empty((rows, cols), dtype=torch.bfloat16)
                    # DeepSeek weight = float(FP8 bytes) * weight_scale_inv.
                    # Work one block row at a time, including partial edge blocks.
                    for start in range(0, rows, block_m):
                        end = min(start + block_m, rows)
                        row_scale = scale[start // block_m].repeat_interleave(block_n)[:cols]
                        restored[start:end] = (tensor[start:end].float() * row_scale).bfloat16()
                    tensor = restored
                converted[name] = tensor.contiguous()
                new_map[name] = shard
                total_size += tensor.numel() * tensor.element_size()
            if converted:
                temporary = output / (shard + ".partial")
                save_file(converted, str(temporary), metadata={"format": "pt"})
                temporary.rename(output / shard)
                print(f"Saved {shard}", flush=True)
            else:
                print(f"Skipped scale-only shard {shard}", flush=True)
        del converted, tensor
    if total_size != sum(
        numel(info["shape"]) * 2 if name in fp8
        else info["data_offsets"][1] - info["data_offsets"][0]
        for name, info in headers.items() if name not in scales
    ):
        raise ValueError("Converted byte count mismatch")
    # Copy tokenizer, model code and metadata, excluding old checkpoint metadata.
    excluded = {"config.json", "model.safetensors.index.json", "tensor_dtypes.json"}
    for path in source.iterdir():
        if (path.is_file() and path.name not in excluded
                and (path.suffix in (".json", ".py", ".model", ".txt", ".md")
                     or path.name in ("LICENSE", "NOTICE"))):
            shutil.copy2(path, output / path.name)
    config.pop("quantization_config", None)
    config["torch_dtype"] = "bfloat16"
    if "dtype" in config:
        config["dtype"] = "bfloat16"
    new_index = {"metadata": {"total_size": total_size}, "weight_map": new_map}
    (output / "model.safetensors.index.json").write_text(json.dumps(new_index, indent=2) + "\n")
    # Write config last: an interrupted conversion is not a loadable model.
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    inspect_checkpoint(output)
    print(f"Done: {output}", flush=True)


if __name__ == "__main__":
    main()
