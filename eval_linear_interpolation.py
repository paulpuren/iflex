"""Evaluate the endpoint linear-interpolation baseline.

This uses the same datasets, deterministic subsets, metrics, and output format as
the learned-model evaluation scripts.  The prediction at intermediate step i is

    (1 - i / (T + 1)) * x_start + i / (T + 1) * x_end.

For SST, interpolation is performed in the normalized representation used to
train FLEX; RFNE and Pearson correlation are then computed in physical units,
matching ``eval_sst.py``.
"""

import os
import time

import numpy as np
import scipy.stats
import torch
from torch.utils.data import DataLoader, Subset

import src.super_slomo as slomo_model
from eval_super_slomo import (
    DATA_RANGE_BY_DATASET,
    RE_EVAL_LIST,
    build_eval_dataset,
    get_eval_args_with_yaml,
    inverse_z_score_sst,
)
from src.utilities import cal_ssim_like_shanghai, save_metrics


def deterministic_eval_subset(dataset, dataset_name, seed=0):
    """Apply the same default sample caps as the dataset-specific evaluators."""
    env_name = {
        "nskt": "NSKT_EVAL_MAX_SAMPLES",
        "sea_temp": "SST_EVAL_MAX_SAMPLES",
    }.get(dataset_name)
    default = {"nskt": "256", "sea_temp": "2048"}.get(dataset_name)
    if env_name is None:
        return dataset

    # SST may already have been subset by build_eval_dataset.
    max_samples = int(os.getenv(env_name, default))
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), size=max_samples, replace=False)).tolist()
    print(f"Using {dataset_name} eval subset: {len(indices)}/{len(dataset)} samples")
    return Subset(dataset, indices)


def output_paths(args, dataset_name):
    run_name = f"LinearInterpolation_T{args.total_interp_steps}"
    if dataset_name == "nskt":
        re_value = RE_EVAL_LIST[args.re_id]
        stem = f"{run_name}_RE{re_value}_seed{args.test_seed}"
        metrics_path = f"./assets/eval_re{re_value}_seed{args.test_seed}_{run_name}.txt"
    elif dataset_name == "shanghai":
        stem = f"{run_name}_Shanghai"
        metrics_path = f"./assets/eval_shanghai_{run_name}.txt"
    else:
        stem = f"{run_name}_SST"
        metrics_path = f"./assets/eval_sst_{run_name}.txt"
    return run_name, f"./samples/{stem}", metrics_path, stem


def main():
    args = get_eval_args_with_yaml()
    args.data_name = slomo_model.normalize_dataset_name(args.data_name)
    seed = 0
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)

    print("Loading the test dataset...")
    test_set, dataset_name = build_eval_dataset(args)
    test_set = deterministic_eval_subset(test_set, dataset_name, seed=seed)
    testloader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        pin_memory=False,
        shuffle=False,
        num_workers=int(os.getenv("EVAL_NUM_WORKERS", "8")),
    )
    print(f"Number of batches: {len(testloader)}")

    _, sample_path, metrics_path, header = output_paths(args, dataset_name)
    rfne_error, r2s, ssim_values, inference_time = [], [], [], []

    with torch.no_grad():
        for batch_index, (inputs, targets, cond_params) in enumerate(testloader):
            print(batch_index)
            condition_start, condition_end = inputs
            norm_stats = None
            if dataset_name == "sea_temp" and len(cond_params) > 2:
                norm_stats = cond_params[2].numpy()

            predictions = []
            batch_times = []
            for step in range(1, len(targets) + 1):
                start_time = time.perf_counter()
                weight = step / (len(targets) + 1.0)
                prediction = (1.0 - weight) * condition_start + weight * condition_end
                batch_times.append(time.perf_counter() - start_time)
                predictions.append(prediction.numpy())
            inference_time.append(float(np.mean(batch_times)))

            for sample_index in range(condition_start.shape[0]):
                sample_rfne, sample_r2, sample_ssim = [], [], []
                for step_index, target_tensor in enumerate(targets):
                    target = target_tensor.numpy()
                    prediction = predictions[step_index]
                    if dataset_name == "sea_temp":
                        metric_target = inverse_z_score_sst(target, norm_stats, step_index + 1)
                        metric_prediction = inverse_z_score_sst(
                            prediction, norm_stats, step_index + 1
                        )
                    else:
                        metric_target = target
                        metric_prediction = prediction

                    target_frame = metric_target[sample_index, 0]
                    prediction_frame = metric_prediction[sample_index, 0]
                    sample_rfne.append(
                        np.linalg.norm(prediction_frame - target_frame)
                        / np.linalg.norm(target_frame)
                    )
                    sample_r2.append(
                        scipy.stats.pearsonr(
                            prediction_frame.ravel(), target_frame.ravel()
                        )[0]
                    )

                    ssim_prediction = prediction[sample_index, 0]
                    ssim_target = target[sample_index, 0]
                    if dataset_name == "shanghai":
                        # Match eval_shanghai_update.py's algebra exactly.
                        ssim_prediction = ssim_prediction * 90.0
                        ssim_target = ssim_target * 90.0
                        data_range = 90.0
                    else:
                        data_range = DATA_RANGE_BY_DATASET[dataset_name]
                    sample_ssim.append(
                        cal_ssim_like_shanghai(
                            ssim_prediction, ssim_target, data_range=data_range
                        )
                    )
                rfne_error.append(sample_rfne)
                r2s.append(sample_r2)
                ssim_values.append(sample_ssim)

            if batch_index == 0:
                samples = {
                    "conditioning_snapshots": condition_start.numpy(),
                    "targets": targets,
                    "predictions": predictions,
                }
                if norm_stats is not None:
                    samples["norm_stats"] = norm_stats
                os.makedirs("./samples", exist_ok=True)
                np.save(sample_path + ".npy", samples)
                print("Generated samples saved...")

    avg_rfne = np.mean(np.vstack(rfne_error), axis=0)
    avg_r2 = np.mean(np.vstack(r2s), axis=0)
    avg_ssim = np.mean(np.vstack(ssim_values), axis=0)
    avg_infer_time = float(np.mean(inference_time))
    print(f"Average RFNE={repr(avg_rfne)}")
    print(f"Average Pearson correlation coefficients={repr(avg_r2)}")
    print(f"Average SSIM value={repr(avg_ssim)}")
    print(f"Average Inference Time={repr(avg_infer_time)}")

    metrics = {
        "avg_rfne_steps": avg_rfne,
        "avg_r2_steps": avg_r2,
        "avg_ssim_steps": avg_ssim,
        "avg_rfne_value": float(np.mean(avg_rfne)),
        "avg_r2_value": float(np.mean(avg_r2)),
        "avg_ssim_value": float(np.mean(avg_ssim)),
        "avg_run_time": avg_infer_time,
    }
    save_metrics(metrics=metrics, save_path=metrics_path, header=header)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
