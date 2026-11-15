"""GH200 worker for selective placement with stock PyTorch. Run via the thesis runner."""

import argparse
import os
from pathlib import Path
import sys

THESIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS / "scripts"))
from torch_host_malloc_common import TORCH_COMMIT, write_json


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(library, result):
    require(os.environ.get("THESIS_HOST_MALLOC") in (None, "0"), "global host mode is active")
    import torch

    sys.path.insert(0, str(THESIS / "pytorch_custom_alloc"))
    from grace_pool import GracePool

    torch_version = torch.__version__
    result.update(torch_version=torch_version, torch_commit=torch.version.git_version,
                  torch_cuda=torch.version.cuda, python=sys.executable)
    require(torch_version.split("+")[0] == "2.11.0" and torch.version.git_version == TORCH_COMMIT,
            "expected the pinned PyTorch 2.11.0 stock build from step 2")
    require(torch.cuda.device_count() == 1, "expected one visible GPU")
    torch.cuda.set_device(0)
    result["gpu"] = torch.cuda.get_device_name(0)
    require("GH200" in result["gpu"], "expected a GH200 GPU")
    require(torch.cuda.get_allocator_backend() == "native", "native CUDA allocator required")

    byte_count = 32 * 1024 * 1024
    shape = (byte_count // 4,)
    pages = byte_count // os.sysconf("SC_PAGE_SIZE")
    result.update(tensor_bytes=byte_count, expected_grace_pages=pages)
    placement = result["placement"] = {}

    def allocate_hbm(label):
        before = torch.cuda.mem_get_info(0)[0]
        tensor = torch.empty(shape, dtype=torch.float32, device="cuda:0")
        delta = before - torch.cuda.mem_get_info(0)[0]
        observation = placement[label] = {
            "ptr": hex(tensor.data_ptr()),
            "hbm_delta_bytes": delta,
            "device_pointer": pool.is_hbm(tensor),
            "grace_pages": pool.verified_pages(tensor),
        }
        require(observation["device_pointer"] and observation["grace_pages"] == 0
                and delta == byte_count, f"{label}: expected a new 32 MiB HBM allocation")
        return tensor

    def all_equal(tensor, value):
        return bool(torch.all(tensor == value).item())

    pool = GracePool(library)
    with torch.inference_mode():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        default = allocate_hbm("hbm_input")

        before = torch.cuda.mem_get_info(0)[0]
        grace = pool.empty_on_grace(shape, dtype=torch.float32)
        delta = before - torch.cuda.mem_get_info(0)[0]
        require(default.is_cuda and grace.is_cuda and default.device == grace.device,
                "expected both inputs to be CUDA tensors on the same device")
        host = placement["grace_input"] = {
            "ptr": hex(grace.data_ptr()),
            "hbm_delta_bytes": delta,
            "grace_pages_before": pool.verified_pages(grace),
            "cpu_numa_node": 0,
        }
        require(host["grace_pages_before"] == pages and delta == 0,
                "expected all Grace tensor pages on node 0 and no HBM growth")

        output = allocate_hbm("hbm_output_after_pool_scope")
        print("PASS placement and default allocator after pool scope", flush=True)

        default.fill_(2).mul_(2).add_(1)
        grace.fill_(3).mul_(2).add_(1)
        torch.add(default, grace, out=output)
        torch.cuda.synchronize()
        result["gpu_checks"] = {
            "hbm_all_5": all_equal(default, 5),
            "grace_all_7": all_equal(grace, 7),
            "mixed_sum_all_12": all_equal(output, 12),
        }
        require(all(result["gpu_checks"].values()), "GPU output mismatch")
        host["grace_pages_after"] = pool.verified_pages(grace)
        require(host["grace_pages_after"] == pages, "Grace pages changed after GPU work")
        print("PASS GPU results and post-kernel placement", flush=True)

        release = result["release"] = {"live_bytes_before": pool.live_bytes()}
        require(release["live_bytes_before"] >= byte_count, "missing live Grace allocation")
        torch.cuda.synchronize()
        del grace
        pool.close()
        release["live_bytes_after"] = pool.live_bytes()
        require(release["live_bytes_after"] == 0, "Grace segments retained after pool teardown")
        print("PASS custom free and zero retained bytes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    result = {"status": "FAIL", "scope": "single GH200, stock PyTorch eager"}
    try:
        run(args.library, result)
    except Exception as exc:
        result["error"] = str(exc)
        raise
    else:
        result["status"] = "PASS"
    finally:
        write_json(args.result, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
