# GitHub publication review

This review covers the current source tree, direct Python imports, launch scripts, documentation, and Git publication hygiene. It is a static review with lightweight regression checks, not a reproduction of model results.

## Fixed

- Removed embedded W&B credentials from all three interpolation training entry points. Authentication now uses the user's W&B environment or login configuration.
- Corrected YAML/CLI merging in FLEX, UNet, and SuperSloMo training and NSKT FLEX evaluation. Explicit CLI options now override YAML even when their value equals the parser default. Added regression coverage.
- Restored UNet training mode at each training epoch after validation/sample generation switches the model to evaluation mode.
- Reset FLEX finetuning epochs when `resume_training` is false. A source checkpoint's epoch no longer shortens or skips a fresh finetuning schedule.
- Load shared FLEX and baseline training checkpoints onto CPU first, avoiding dependence on the GPU ordinal used to save them. Missing training checkpoints now raise `FileNotFoundError` directly.
- Added an explicit error when interpolation training sees no CUDA GPUs.
- Added missing direct dependencies: `Pillow`, `py7zr`, and `cmweather`.
- Added ignore rules for checkpoints, raw HDF5 data, generated run directories, credentials, local agent state, and unused local third-party checkouts.
- Rewrote the README with actual entry points, data requirements, authentication, configuration precedence, and environment limitations.

## Actions before publishing

1. **Revoke or rotate the W&B API key.** The hard-coded login is also present in the current Git `HEAD` versions of the three training files. Editing the working tree does not remove it from history. Publish from a fresh, reviewed source export/new Git history, or scrub the affected history before pushing. No history was rewritten during this review.
2. **Review the final staged file list.** The tree already had extensive staged and unstaged work. No files were staged, unstaged, committed, or deleted by this review. Ignore rules do not remove already tracked files.
3. **Use an explicit experiment environment.** Requirements are unpinned and no tested environment lockfile was available. Record a working package snapshot after running on the intended GPU system. The local review environment has no PyTorch, so package installation, imports, training, and numerical evaluation remain unverified.
4. **Adapt cluster templates.** Many shell scripts, YAML configs, legacy evaluation files, notebooks, and preprocessing utilities retain institution-specific paths. The README supplies direct Python examples with path overrides. Existing launchers still require adaptation.
5. **Supply EDEN separately if advertising that baseline.** Its implementation is absent; existing EDEN scripts and analysis hooks cannot run from this checkout alone. Its environment is outside the root dependency audit.

## Scientific and implementation caveats

- **Shanghai validation uses the test group.** `datasets/data_shanghai.py` maps `val` to `test`, and training uses that loader for checkpoint selection. Establish a separate validation split before claiming an untouched test evaluation. This behavior was documented rather than changed because it affects comparability of existing experiments.
- **NSKT train/validation windows can overlap in time.** `datasets/data_nskt_updated.py` splits logical sample indices, then maps them to frame indices with integer division by 17; interpolation windows around the boundary can share frames. Use independent trajectories or a temporal gap for strictly disjoint validation. Existing sampling was preserved.
- **NSKT training defaults to one Reynolds number.** The loader's default `re_id=3` selects Re=16000. A list of available training files does not imply all Reynolds numbers are sampled. Review the intended experiment before reporting multi-Re training.
- **Distributed validation reporting needs care.** Main FLEX training averages each rank's local batch losses, while rank zero uses its local mean for logging and checkpoint selection. There is no cross-rank metric reduction in that path. Results can depend on GPU count and shard assignment.
- **SST sequence caches use fixed filenames.** Cache names contain `_10_10` regardless of requested sequence length. Reusing a data directory with changed temporal settings can reuse incompatible indices; regenerate the caches when changing those settings.
- **Other evaluation interfaces differ.** The shared configuration fix covers `eval.py`; dataset-specific and baseline evaluators retain their own parsers and checkpoint-loading behavior.
- **Local dependency checkouts contain nested Git repositories.** `src/clip/` and `src/taming-transformers/` are not imported by the first-party Python code audited here. They are ignored to avoid accidentally staging incomplete embedded repositories. If later required, distribute them deliberately with their upstream licenses and installation instructions.

## Validation performed

- Parsed 53 first-party Python files with Python's AST parser: passed.
- Checked 59 shell scripts with `bash -n`: passed.
- Ran configuration regression tests without GPU dependencies: passed.
- Scanned current training sources for hard-coded W&B login calls: none remain; confirmed their presence in `HEAD` without printing credential values.
- Checked whitespace with `git diff --check` after cleanup.

No GPU or end-to-end accuracy claims were validated. The dependency review covers first-party Python source imports; it does not certify arbitrary notebook environments or the absent EDEN implementation.
