#!/usr/bin/env python3
"""
GPU 可用性快速检查脚本
运行方式：
    python tools/check_cuda.py

脚本会输出 PyTorch 版本、CUDA 是否可用、GPU 数量以及每块 GPU 的名称。
"""

import sys

try:
    import torch
except ImportError as exc:
    sys.exit(f"[ERROR] 未找到 PyTorch，请先安装。详情: {exc}")


def main() -> None:
    print("===== PyTorch CUDA 检查 =====")
    print(f"PyTorch 版本: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 可用: {cuda_available}")

    if not cuda_available:
        print("提示: 若期望使用 GPU，请确认 CUDA 驱动和对应的 PyTorch 版本已正确安装。")
        return

    device_count = torch.cuda.device_count()
    print(f"GPU 数量: {device_count}")
    for idx in range(device_count):
        print(f"  - GPU {idx}: {torch.cuda.get_device_name(idx)}")

    current_idx = torch.cuda.current_device()
    print(f"当前默认 GPU: {current_idx} ({torch.cuda.get_device_name(current_idx)})")


if __name__ == "__main__":
    main()

