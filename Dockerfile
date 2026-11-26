# SPDX-License-Identifier: Apache-2.0

# Select an Ubuntu 22.04 DTK 26.10 development image with Python 3.12.
ARG BASE_IMAGE=harbor.sourcefind.cn:5443/hcu/admin/base/dtk@sha256:5be259fdc0c16043c311f5f6dfbaad18d436e1a1374e6312a4286a62b12dc4e8
FROM ${BASE_IMAGE}

ARG SGLANG_REF=453edca64008bdaf162bb42dcebda14bd1b4927a
ARG MAX_JOBS=8

ENV VIRTUAL_ENV=/opt/areal-venv \
    PATH=/opt/areal-venv/bin:$PATH \
    PYTHONPATH="" \
    PYTHONNOUSERSITE=1 \
    ROCM_HOME=/opt/dtk \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/areal-build
RUN grep -qx '26.10' /opt/dtk/.info/version-dev \
    && command -v hipcc \
    && python3.12 -m venv "$VIRTUAL_ENV" \
    && python -m pip install --no-cache-dir \
        build ninja setuptools setuptools-rust setuptools-scm wheel

COPY docker/requirements-dtk2610-torch213.txt ./requirements-hcu.txt
RUN python -m pip install --no-cache-dir --no-deps -r requirements-hcu.txt \
    && python -m pip list --format=freeze > constraints.txt

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/dtk/.hyhal/rocm_smi/lib

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --timeout 60 pip==26.2.1
RUN sed '/^ninja==/d' constraints.txt > runtime-constraints.txt \
    && python -m pip install --no-cache-dir --timeout 60 --resume-retries 5 -c runtime-constraints.txt -r requirements.txt \
        aiofiles anthropic apache-tvm-ffi==0.1.11 blobfile colorlog compressed-tensors==0.18.0 \
        diffusers distro easydict filelock gguf h5py interegular ipython json5 \
        lark 'llguidance>=1.7.6,<2' mathruler==0.1.0 'mistral-common>=1.11.5' \
        modelscope networkx ninja==1.11.1 numba==0.65.1 nvdlfw-inspect \
        nvidia-ml-py nvidia-resiliency-ext openai==2.6.1 openai-harmony==0.0.4 \
        outlines==0.1.11 partial-json-parser pillow prettytable py-spy pyarrow==25.0.1 pybind11 \
        pytest python-multipart pyzmq qwen-vl-utils redis scipy soundfile sympy \
        tiktoken timm tokenizers==0.22.2 watchfiles word2number xgrammar==0.2.1 \
        xxhash zstandard

RUN python -c 'from importlib.metadata import version; import torch; assert version("torch") == "2.13.0+dtk2610.2610261432.gefb4e1" and torch.version.hip'
RUN python -c 'from datasets import Dataset; assert Dataset.from_dict({"value": [1]})[0]["value"] == 1'

RUN git init /opt/sglang-das \
    && git -C /opt/sglang-das fetch --depth=1 https://github.com/HYGON-AI/sglang-das.git "$SGLANG_REF" \
    && git -C /opt/sglang-das checkout --detach FETCH_HEAD \
    && git -C /opt/sglang-das submodule update --init --recursive --depth=1

WORKDIR /opt/sglang-das/python/sglang/kernels/aot
# pos_enc.cu launches up to 512 threads. DTK defaults to 256 without explicit bounds.
RUN sed -i 's/^__global__ void rotary_embedding_kernel(/__global__ __launch_bounds__(512) void rotary_embedding_kernel(/' csrc/elementwise/pos_enc.cu

WORKDIR /opt/sglang-das/python
ARG SGLANG_VERSION=0.5.18
# Install the runtime dependencies above to select the HCU builds of native packages.
RUN short_ref="$(git rev-parse --short=7 HEAD)" \
    && SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="${SGLANG_VERSION}+g${short_ref}" \
        SGLANG_BUILD_RUST_EXTS=none python -m build --wheel --no-isolation -Cfeatures=all_hip \
    && python -m pip install --no-cache-dir --no-deps dist/*.whl \
    && rm -rf build dist

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/dtk/.hyhal/lib

# Vendor Triton uses AICC clang for JIT compilation on HCU.
ARG AICC_PACKAGE_URL=https://resource.flagos.net/repository/flagos-filestore/hygon/dtk_llvm_aicc-2.0.0+dtk2610.2611032005.ge358cd-x86_64.run
ARG AICC_PACKAGE_SHA256=2eca83675ce0c17a185628a54608c213c7e1954e5b660435e7b8ca42b0d6d11b
RUN curl --fail --location --retry 5 --retry-delay 2 "$AICC_PACKAGE_URL" -o /tmp/aicc.run \
    && echo "$AICC_PACKAGE_SHA256  /tmp/aicc.run" | sha256sum -c - \
    && bash /tmp/aicc.run --dtk_dir /opt/dtk \
    && /opt/dtk/aillvm/bin/clang-22 --version >/dev/null \
    && rm /tmp/aicc.run

# Megatron reads NVRx's version at import time. Build it without the CUDA profiler.
ARG NVRX_REF=300d9c03c967c101c227b203d5b89e59189bca68
RUN STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD=1 python -m pip install --no-cache-dir --no-deps \
        "nvidia-resiliency-ext @ git+https://github.com/NVIDIA/nvidia-resiliency-ext.git@$NVRX_REF" \
    && python -c 'import nvidia_resiliency_ext; print(nvidia_resiliency_ext.__version__)'

# Keep mbridge compatible with the pinned Megatron and Transformers versions.
RUN mbridge_core="$VIRTUAL_ENV/lib/python3.12/site-packages/mbridge/core" \
    && sed -i 's/ModelType\.encoder_and_decoder/getattr(ModelType, "encoder_and_decoder", None)/g' "$mbridge_core/util.py" \
    && sed -i 's/rotary_base=self\.hf_config\.rope_theta,/rotary_base=(getattr(self.hf_config, "rope_theta", None) or self.hf_config.rope_parameters["rope_theta"]),/' "$mbridge_core/llm_bridge.py"

# AReaL v2 uses LiteLLM in the data proxy and the Agents SDK in the math workflow.
RUN python -m pip install --no-cache-dir --timeout 60 --resume-retries 5 \
        litellm==1.103.0 openai==2.33.0 openai-agents==0.14.8 \
    && python -c 'from agents import Agent, Runner, SQLiteSession; import litellm.llms.anthropic.experimental_pass_through.adapters.transformation'

# Megatron's GatedDeltaNet uses the Triton kernels from FLA.
RUN python -m pip install --no-cache-dir --timeout 60 --resume-retries 5 \
        -c /opt/areal-build/runtime-constraints.txt 'flash-linear-attention[rocm]==0.5.1' \
    && python -c 'from fla.ops.gated_delta_rule import chunk_gated_delta_rule'

COPY docker/requirements-dtk2610-torch213-dsa.txt /opt/areal-build/requirements-dsa.txt
# SGLang's HIP DSA indexer imports the TileLang activation quantizer.
RUN python -m pip list --format=freeze > /opt/areal-build/dsa-constraints.txt \
    && python -m pip install --no-cache-dir --timeout 60 --resume-retries 5 \
        -c /opt/areal-build/dsa-constraints.txt \
        -r /opt/areal-build/requirements-dsa.txt \
    && python -c 'import tilelang.language'

# The pinned fast Hadamard wheel provides a native kernel for gfx938.
# Use SGLang's Gemma RMSNorm when the vLLM op leaves its output unwritten on HCU.
COPY hcu_areal/third_party/sglang/dsa-hadamard.patch hcu_areal/third_party/sglang/gemma-rmsnorm.patch /opt/areal-build/
RUN for patch_file in /opt/areal-build/dsa-hadamard.patch /opt/areal-build/gemma-rmsnorm.patch; do \
        patch --batch -p1 -d /opt/sglang-das/python < "$patch_file" \
            && patch --batch -p1 -d "$VIRTUAL_ENV/lib/python3.12/site-packages" < "$patch_file" \
            || exit 1; \
    done

# AReaL's Geometry3K reward parses LaTeX answers.
RUN python -m pip install --no-cache-dir --timeout 60 --resume-retries 5 pylatexenc==2.11 \
    && python -c 'from pylatexenc.latex2text import LatexNodes2Text'

# Keep the target argument here so all builds share the installed runtime layers.
WORKDIR /opt/sglang-das/python/sglang/kernels/aot
ARG HCU_TARGET=dual
RUN case "$HCU_TARGET" in \
        dual) export PYTORCH_ROCM_ARCH="gfx936;gfx938" ;; \
        bw1000) export PYTORCH_ROCM_ARCH=gfx936 ;; \
        bw1100) export PYTORCH_ROCM_ARCH=gfx938 ;; \
        *) echo "Unsupported HCU_TARGET: $HCU_TARGET (expected dual, bw1000, or bw1100)" >&2; exit 1 ;; \
    esac \
    && echo "Building SGLang kernels for $HCU_TARGET ($PYTORCH_ROCM_ARCH)" \
    && MAX_JOBS="$MAX_JOBS" python setup_hip.py bdist_wheel \
    && python -m pip install --no-cache-dir --no-deps dist/*.whl \
    && rm -rf build dist \
    && python -c 'from importlib.metadata import version; print({name: version(name) for name in ("torch", "sglang", "sglang-kernel", "awex")})'

WORKDIR /workspace/AReal-das
LABEL io.hygon.hcu.target=${HCU_TARGET}
CMD ["/bin/bash"]
