#!/usr/bin/env python3
import os
import sys
import traceback
from pathlib import Path
import math
import torch

# run from gpu_mode/012
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
# allow importing utils from parent dir as notebook does
sys.path.insert(0, str(ROOT.parent))

try:
    from utils import load_cuda, get_sig
except Exception as e:
    print("Failed to import utils from repo:", e)
    traceback.print_exc()
    # try importing from parent of parent
    sys.path.insert(0, str(ROOT.parent.parent))
    try:
        from utils import load_cuda, get_sig
    except Exception as e2:
        print("Still failed to import utils:", e2)
        traceback.print_exc()
        raise SystemExit(1)

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

def get_loaded_cuda_module(fname, verbose=False):
    cuda_src_path = ROOT / f"{fname}.cu"
    torch_src_path = ROOT / "torch_extension_template.cu"
    if not cuda_src_path.exists():
        raise FileNotFoundError(f"{cuda_src_path} not found")
    if not torch_src_path.exists():
        raise FileNotFoundError(f"{torch_src_path} not found")
    cuda_src = cuda_src_path.read_text()
    cuda_src += torch_src_path.read_text()
    cuda_src = cuda_src.replace("your_function_name", fname)
    cpp_src = get_sig(fname, cuda_src)
    return load_cuda(cuda_src, cpp_src, [fname], verbose=verbose)


def get_test_tensors(N_inp, N_out, d):
    Q = torch.randn(N_out, d).contiguous().to("cuda")
    K = torch.randn(N_inp, d).contiguous().to("cuda")
    V = torch.randn(N_inp, d).contiguous().to("cuda")
    scaling = 1.0 / math.sqrt(d)
    O_expected = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    S = (Q @ K.T) * scaling
    L_expected = torch.logsumexp(S, dim=-1)
    return Q, K, V, scaling, O_expected, L_expected


def smoke_test(fname, N_inp=8, N_out=8, d=128):
    print(f"\n=== Smoke test for module: {fname} ===")
    try:
        module = get_loaded_cuda_module(fname, verbose=True)
    except Exception:
        print("Module load failed:")
        traceback.print_exc()
        return
    print("Module loaded. Preparing small tensors...")
    Q, K, V, scaling, Oexp, Lexp = get_test_tensors(N_inp, N_out, d)
    try:
        res = getattr(module, fname)(Q, K, V)
        torch.cuda.synchronize()
        print("Module call returned:", type(res))
        # if res is tuple or tensors, try to print max diff vs expected
        try:
            if isinstance(res, tuple) and len(res) >= 1:
                O = res[0]
            else:
                O = res
            diff = (O - Oexp).abs().max().item()
            print(f"Max abs diff vs torch sdpa: {diff}")
        except Exception as e:
            print("Could not compare outputs:", e)
    except Exception:
        print("Exception during module call:")
        traceback.print_exc()


if __name__ == '__main__':
    # run two modules: main and spilling variant
    for name in ("flash_attention", "flash_attention_spilling_from_registers"):
        smoke_test(name)
    print("Smoke tests finished.")
