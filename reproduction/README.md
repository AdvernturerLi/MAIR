# Reproduction source and offline experience workflow

This folder versions the source used for baseline analysis and controlled validation exploration. It contains no images, model weights, runtime credentials, or generated experience/results.

## Layout

- `analysis/`: paired baseline metrics, sequence/prior analysis, rollback trees, input baselines and source-image diagnostics. Uses the existing shared workspace results; set `IMAGE_RECONSTRUCT_ROOT` when that workspace cannot be located automatically.
- `experience_learning/`: cached-prefix tool exploration, independent DepictQA and seven-IQA evaluation, matched tool/source-HQ evidence, separate text-only distillation, offline weight preparation, and completion validation.
- `agenticir_changes.patch` plus `agenticir_additions/`: changes to the separate upstream AgenticIR checkout. The patch base revision is recorded below; copy the additions into that checkout after applying the patch. These files are archived in the user's MAIR repository rather than pushed to the upstream author's repository.

## Current experiment contract

Ten validation HQ sources and seven existing double-degradation combinations give 70 inputs, 3140 tool-prefix calls and 2540 leaves. RIDCP is explicitly excluded because its runtime deformable-convolution extension is unavailable. The completed two-source pilot contributes 178 reusable nodes. The separate conditional noise+blur+SR experiment is not started.

Two experience methods share generated images, but use separate objectives: AgenticIR uses residual degradation failure rates; MAIR uses seven signed within-image standardized IQA metrics. MAIR distills only within-stage ordering. Source-HQ clusters, fixed task/tool assignments and leave-one-HQ-out results are retained. Test images/statistics are not experience inputs.

## Completion gate

After exploration and evaluation finish, `validate_pairwise_run.py --root RUN` verifies counts, hashes/readability, task/tool profiles, all metrics and matched order evidence. Only a successful check with `--approve` creates `review_approved.json`. The DeepSeek text distillation queue checks the protected input hashes before sending aggregate statistics. No validation images are sent for distillation. Review experience text before any held-out evaluation.

Server launcher scripts preserve the deployed Linux paths and experiment identifiers for auditability; inspect these before reusing them on another server. Runtime API config stays on the server and is never bundled. Weight downloads are only official public model files with provenance checks.

Run workflow tests from this folder with `python -m pytest experience_learning/tests`. MAIR inference and runner tests remain in the repository's top-level `tests/` folder.

AgenticIR patch base: `9640a291480dee3ba8f2974125d4ee9e3440f3d6`.

## Interruption protection

`export_completed_snapshot.py` exports committed successful nodes and exact validation inputs without credentials or model weights. `backup_exploration_local.py` downloads incremental archives, verifies archive/file SHA-256 and image readability, preserves history, and writes a verified local index only after the complete restorable snapshot passes. An uncompleted node and results newer than the last verified backup may need to be rerun. A replacement server still requires its tool environments and weights.
