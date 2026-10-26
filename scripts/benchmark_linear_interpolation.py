"""Stable microbenchmark for one batched linear-interpolation output frame."""

import argparse
import statistics
import time

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--total-interp-steps", type=int, default=20)
    return parser.parse_args()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main():
    args = parse_args()
    if args.repetitions <= 0 or args.block_size <= 0:
        raise ValueError("repetitions and block-size must be positive")
    if args.repetitions % args.block_size:
        raise ValueError("repetitions must be divisible by block-size")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.device == "cpu":
        torch.set_num_threads(args.num_threads)

    device = torch.device(args.device)
    shape = (args.batch_size, 1, args.resolution, args.resolution)
    generator = torch.Generator(device=device).manual_seed(0)
    condition_start = torch.randn(shape, device=device, generator=generator)
    condition_end = torch.randn(shape, device=device, generator=generator)
    weight = args.step / (args.total_interp_steps + 1.0)

    def interpolate():
        return (1.0 - weight) * condition_start + weight * condition_end

    with torch.inference_mode():
        for _ in range(args.warmup):
            output = interpolate()
        synchronize(device)

        seconds_per_call = []
        blocks = args.repetitions // args.block_size
        for _ in range(blocks):
            synchronize(device)
            start_ns = time.perf_counter_ns()
            for _ in range(args.block_size):
                output = interpolate()
            synchronize(device)
            elapsed_ns = time.perf_counter_ns() - start_ns
            seconds_per_call.append(elapsed_ns * 1e-9 / args.block_size)

    # Keep the final operation observable until after timing.
    _ = output[0, 0, 0, 0].item()
    median_s = statistics.median(seconds_per_call)
    print(f"device: {device}")
    print(f"torch_num_threads: {torch.get_num_threads()}")
    print(f"shape: {shape}")
    print(f"warmup_calls: {args.warmup}")
    print(f"measured_calls: {args.repetitions}")
    print(f"block_size: {args.block_size}")
    print(f"median_ms_per_frame_per_batch: {median_s * 1e3:.6f}")
    print(f"p25_ms_per_frame_per_batch: {percentile(seconds_per_call, 25) * 1e3:.6f}")
    print(f"p75_ms_per_frame_per_batch: {percentile(seconds_per_call, 75) * 1e3:.6f}")
    print(f"median_ms_per_frame_per_sample: {median_s * 1e3 / args.batch_size:.6f}")


if __name__ == "__main__":
    main()
