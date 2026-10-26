# FLEX: Generative Refinement for Temporal Interpolation

PyTorch research code for temporal interpolation of scientific fields. FLEX combines endpoint conditioning with diffusion refinement to predict intermediate states between two observed frames. The repository also includes UNet, SuperSloMo, and linear interpolation baselines, along with NSKT forecasting and interpolation ablations.

## Supported datasets

| Dataset | Task | Data expected by the loaders |
| --- | --- | --- |
| NSKT | Turbulent flow interpolation and forecasting | HDF5 trajectories with a `w` array shaped `(time, height, width)` |
| Shanghai | Radar frame interpolation | An HDF5 file containing `train` and `test` groups with `all_len` and sample sequences |
| SST | Sea surface temperature interpolation | Regional NumPy fields, latitude/longitude arrays, and cached sequence indices |

Datasets and trained checkpoints are not distributed with this repository. Loader-specific preprocessing and file naming must match the supplied code.

## Installation

Create and activate a Python virtual environment, then install the dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Training entry points require NVIDIA GPUs and a CUDA-enabled PyTorch installation. Install a PyTorch/torchvision combination appropriate for your machine. The requirements list declares direct dependencies; it is not a tested, pinned environment lockfile.

Training uses Weights & Biases. Authenticate using `wandb login` or set `WANDB_API_KEY` in your environment. For local runs without uploading logs:

```bash
export WANDB_MODE=offline
```

## Data preparation

Store large datasets outside the repository and pass their location through `--scratch_dir`.

- **NSKT:** the current interpolation loader uses filenames such as `2000_2048_2048_seed_2150.h5`. Training/validation Reynolds numbers are 2000, 4000, 8000, 16000, and 32000, using seed 2150. Evaluation uses seed 3407 by default and supports additional Reynolds numbers. See `datasets/data_nskt_updated.py` for the complete lists and sampling logic.
- **Shanghai:** pass the path to the HDF5 file. See `datasets/data_shanghai.py` for group and sample layout. The current loader maps validation to the `test` group; account for this when interpreting validation-selected results.
- **SST:** pass the regional data root. See `datasets/data_sea_temp.py` for directory names, date parsing, temporal splits, and index-cache construction. Existing pickle caches should come from a trusted source.

## Training

Run commands from the repository root. Start with a base configuration and override the data path:

```bash
python train.py --config config/nskt/flex.yaml --scratch_dir /path/to/nskt
python train.py --config config/shanghai/flex.yaml --scratch_dir /path/to/shanghai.h5
python train.py --config config/sst/flex.yaml --scratch_dir /path/to/sst
```

The training entry points spawn one worker per visible GPU. Use `CUDA_VISIBLE_DEVICES` to select GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config config/nskt/flex.yaml \
  --scratch_dir /path/to/nskt \
  --batch_size 4
```

UNet and SuperSloMo use separate entry points:

```bash
python train_unet.py --config config/nskt/unet.yaml --scratch_dir /path/to/nskt
python train_super_slomo.py --config config/nskt/super_slomo.yaml --scratch_dir /path/to/nskt
```

For these training commands, precedence is **explicit command-line options > YAML values > parser defaults**. Post-training configurations may require an existing checkpoint; use the base configurations above for training from scratch. FLEX uses `--resume_training true` to retain the checkpoint epoch and optimizer state; the default finetuning path starts a fresh epoch schedule.

The shell scripts under `scripts/` are experiment templates for the original HPC environment. They contain cluster module commands, private Conda paths, data paths, and checkpoint defaults. Adapt those before use; the direct Python commands above avoid their environment setup.

## Evaluation

For NSKT FLEX evaluation, provide a checkpoint and matching model configuration:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --config config/nskt/flex.yaml \
  --scratch_dir /path/to/nskt \
  --checkpoint_path /path/to/model_best.pt \
  --re_id 5 \
  --total_interp_steps 20 \
  --batch_size 4
```

Here `re_id=5` selects Re=12000 in the current evaluation list. `eval.py` supports YAML configurations and explicit CLI overrides, including a fallback from `total_interp_steps_train` to `total_interp_steps`.

Other evaluation entry points include `eval_shanghai_update.py`, `eval_sst.py`, `eval_super_slomo.py`, and `eval_linear_interpolation.py`. Their interfaces differ; consult `--help` and the corresponding scripts under `scripts/shanghai/`, `scripts/sst/`, and `scripts/nskt/`. Checkpoint architecture and conditioning settings must match the model being evaluated.

Metrics include relative Frobenius norm error (RFNE), R², and dataset-specific structural or radar scores. Available outputs depend on the entry point; metric files are written under `assets/`, and training writes checkpoints under `checkpoints/` and visualizations under `samples/`.

## Forecasting, ablations, and analysis

- `forecasting/` contains NSKT forecasting code, with configurations in `config/forecasting/` and launchers in `scripts/forecasting/`.
- `src/ablation/`, `config/ablation/`, and `scripts/ablation/` contain interpolation variants.
- `analysis/` contains notebooks and figure-generation scripts. Some retain local experiment paths and require predictions or checkpoints from completed runs.
- `scripts/benchmark_inference_time.py` provides inference benchmarking.

**EDEN integration:** EDEN launchers and analysis hooks are included, but the `EDEN/` implementation and its checkpoints are absent from this checkout. Those workflows require a compatible external EDEN checkout at that location and its own dependencies. The root requirements file covers this repository's code, not the absent EDEN environment.

## Repository layout

```text
├── train.py / train_unet.py / train_super_slomo.py
├── eval*.py              # Dataset and baseline evaluation entry points
├── src/                  # Models, diffusion wrappers, metrics, and utilities
├── datasets/             # Loaders and preprocessing
├── forecasting/          # NSKT forecasting implementation
├── config/               # YAML experiment settings
├── scripts/              # Experiment launchers and benchmarks
├── analysis/             # Notebooks and publication figures
├── tests/                # Lightweight regression checks
├── requirements.txt
└── LICENSE
```

## Validation and reproducibility

Run the lightweight configuration regression suite with:

```bash
python -m unittest discover -s tests -v
```

GPU training and evaluation require the matching datasets and checkpoints. Record package versions, GPU hardware, random seeds, data splits, and the exact configuration when reporting results. See [REVIEW.md](REVIEW.md) for the publication review and remaining limitations.

## License and acknowledgments

This repository is licensed under [GNU GPL v3](LICENSE). Retain applicable upstream attribution and licenses when distributing third-party code. The Shanghai loader cites DiffCast, and model components reference prior interpolation and diffusion implementations in their source files.
