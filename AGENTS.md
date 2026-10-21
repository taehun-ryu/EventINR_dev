# AGENTS.md — EventINR

## 1. Purpose and Scope

This file defines the global instructions for Codex operating in the `EventINR-main` repository.

These rules apply to all tasks, including code investigation, implementation, debugging, refactoring, training, evaluation, and experiment analysis.

EventINR is a research project for reconstructing intensity images from event camera data, with a particular interest in Implicit Neural Representations (INRs).

The project may evolve as research progresses. Do not treat any particular architecture, loss function, optimizer, or experimental configuration as a permanent design requirement.

**The objective is scientifically valid, reproducible research, not merely producing executable code or improving benchmark scores.**

## 2. Mandatory Reading Order

Before starting any task, read the following files in order:

1. `AGENTS.md`
2. `CURRENT_STATE.md`
3. The explicitly selected task file under `codex/tasks/active/`

Then inspect the repository files relevant to the selected task.

Rules:

- Treat `AGENTS.md` as the source of global working rules.
- Treat `CURRENT_STATE.md` as the reference for documented project status.
- Treat the selected task file as the authoritative specification for the current task.
- Treat the actual source code as the source of truth for current implementation behavior.
- If documentation and source code disagree, investigate the discrepancy rather than assuming either is correct.
- Do not execute unrelated tasks.
- Do not automatically select another task after completing the current one.
- If no task file is specified, do not arbitrarily choose a task.

For initial repository setup, a bootstrap task may be used before the normal task structure is fully established.

## 3. Repository Organization

Use the following task-management structure:

```text
EventINR-main/
├── AGENTS.md
├── CURRENT_STATE.md
└── codex/
    └── tasks/
        ├── active/
        └── archive/
```

Rules:

- Keep global instructions in the root `AGENTS.md`.
- Keep project progress and verified implementation status in the root `CURRENT_STATE.md`.
- Store active task instructions in `codex/tasks/active/`.
- Move completed task instructions to `codex/tasks/archive/`.
- Use descriptive, purpose-centered task filenames.
- Do not create additional root-level Markdown files for task management.
- Store task-specific investigation reports and documentation under `codex/` when appropriate.
- Preserve the existing source-code organization unless the selected task explicitly requires structural changes.

## 4. Task Scope and Execution

Each task must have a clearly defined objective and completion criteria.

### 4.1 Scope Control

- Execute only the selected task.
- Prefer focused, incremental changes over large rewrites.
- Do not introduce unrelated features, optimizations, or architectural changes.
- Do not perform speculative refactoring.
- Do not modify working components solely to make the code stylistically cleaner.
- Preserve backward compatibility when reasonably possible.

### 4.2 Task Size

Tasks should generally be scoped so that they can be completed within approximately five hours of focused work.

If a task is substantially larger:

- Identify independently executable subtasks.
- Complete only the feasible portion when partial completion is appropriate.
- Document the remaining work.
- Do not silently expand the scope.

### 4.3 Long-Running Experiments

- Avoid combining multiple large GPU experiments into one task.
- Prefer separate tasks for large sweeps across datasets, sequences, architectures, or hyperparameters.
- Do not initiate expensive experiments unless explicitly requested.
- Estimate the expected computational cost before launching long-running jobs.
- Use lightweight checks before committing substantial GPU resources.

## 5. Code Investigation and Modification

### 5.1 Inspect Before Modifying

Before changing code:

1. Identify the actual execution entry points.
2. Trace the relevant data and model paths.
3. Identify existing implementations that may already solve the problem.
4. Examine related configurations, scripts, and checkpoints.
5. Determine the expected effect of the proposed modification.

Do not infer runtime behavior from filenames, comments, or unused implementations alone.

### 5.2 Minimal Changes

- Modify only files necessary for the selected task.
- Reuse existing abstractions and utilities where appropriate.
- Avoid unnecessary dependencies.
- Do not duplicate existing functionality.
- Do not change public interfaces without a clear reason.
- Do not silently change default behavior.
- Clearly document intentional changes to existing behavior.

### 5.3 Preserve Existing Work

- Never overwrite datasets, checkpoints, logs, or experiment results unintentionally.
- Do not delete previous experiments without explicit authorization.
- Do not reset or discard unrelated changes.
- Inspect the working tree before modifying files.
- Keep unrelated user modifications intact.
- Do not create commits, rewrite Git history, or push changes unless explicitly requested.

## 6. Scientific and Mathematical Correctness

Research code must reflect the intended mathematical formulation.

### 6.1 Mathematical Consistency

- Preserve the mathematical meaning of variables and operations.
- Verify tensor shapes, coordinate conventions, indexing, units, and normalization.
- Distinguish intensity from log-intensity.
- Distinguish absolute timestamps from normalized or relative timestamps.
- Distinguish continuous-time representations from discretized observations.
- Ensure implementation assumptions are consistent with the mathematical model.

### 6.2 No Unsupported Assumptions

- Do not assume a method is mathematically valid because it produces plausible outputs.
- Do not introduce approximations without identifying them.
- Document assumptions and numerical approximations that materially affect correctness.
- Clearly distinguish theoretical properties from experimentally demonstrated behavior.

### 6.3 Research Hypotheses

When implementing an experimental method:

- State the hypothesis being tested.
- Identify the component being changed.
- Define the expected measurable effect.
- Propose an appropriate comparison or ablation.
- Do not interpret numerical improvements as proof of a theoretical explanation without supporting evidence.

## 7. Event Camera and INR Specific Requirements

### 7.1 Event Data

Pay particular attention to:

- Event timestamps and ordering.
- Event polarity conventions.
- Spatial coordinate conventions.
- Event-to-frame alignment.
- Temporal binning and voxelization.
- Sensor-dependent preprocessing.
- Handling of empty event intervals.
- Causal versus non-causal information usage.

Do not assume different datasets share identical timestamp units, polarity encodings, or event packet definitions.

### 7.2 Intensity Reconstruction

- Explicitly track whether model outputs represent intensity, log-intensity, or another transformed quantity.
- Apply exponential, sigmoid, clamping, normalization, and tone mapping only when mathematically appropriate.
- Do not silently change the interpretation of model outputs.
- Keep raw reconstruction results distinguishable from post-processed results.

### 7.3 Continuous Representations

- Distinguish coordinate-query capability from demonstrated continuous reconstruction accuracy.
- Do not claim arbitrary-time reconstruction quality without corresponding evaluation.
- Verify the temporal support of features and query timestamps.
- Identify interpolation, extrapolation, and future-information dependencies.
- Do not assume arbitrary spatial queries imply validated super-resolution performance.

## 8. Experiment Design and Reproducibility

### 8.1 Controlled Comparisons

When evaluating a proposed improvement:

- Change one major experimental factor at a time when feasible.
- Keep unrelated configurations fixed.
- Record all relevant experimental settings.
- Use consistent datasets, preprocessing, and evaluation procedures for direct comparisons.
- Clearly identify comparisons that are not fully controlled.

### 8.2 Reproducibility

Record, where applicable:

- Model architecture and configuration.
- Dataset and sequence selection.
- Data preprocessing and augmentation.
- Random seed.
- Optimizer and learning-rate settings.
- Loss functions and weights.
- Training duration and checkpoint selection.
- Evaluation configuration.
- Code version or commit identifier.
- Hardware and software environment when relevant.

### 8.3 Experiment Integrity

- Never fabricate experiment results.
- Never present expected performance as measured performance.
- Distinguish completed, failed, interrupted, and pending experiments.
- Do not silently select favorable results.
- Report meaningful failures and negative results.
- Preserve original experimental artifacts whenever possible.

## 9. Evaluation Protocol

Evaluation consistency is critical.

Before reporting or comparing metrics, verify:

- Dataset and evaluated sequences.
- Ground-truth alignment.
- Image resolution.
- Intensity range and normalization.
- Output post-processing.
- Metric definitions and implementations.
- Frame selection and aggregation rules.
- Checkpoint selection.
- Training-data comparability.

For image reconstruction, metrics may include MSE, PSNR, SSIM, and LPIPS.

Rules:

- Do not compare metrics calculated under incompatible protocols without explicitly identifying the difference.
- Report whether CLAHE, tone mapping, clipping, or other transformations were applied.
- Distinguish raw-output evaluation from post-processed evaluation.
- Do not assume improved perceptual metrics imply improved intensity accuracy.
- Include qualitative comparisons where they provide useful diagnostic evidence.
- Preserve per-sequence results when available.

## 10. Verification Requirements

Every implementation task should include appropriate verification.

### 10.1 Lightweight Verification

Use applicable checks such as:

- Syntax validation.
- Import checks.
- Tensor shape assertions.
- Small synthetic-input tests.
- Forward-pass tests.
- Backward-pass and gradient checks.
- Numerical sanity checks.
- Boundary and edge-case tests.

### 10.2 Functional Verification

When relevant:

- Verify that the modified execution path is actually used.
- Test compatibility with existing configurations.
- Check that losses and outputs are finite.
- Confirm tensor device and dtype consistency.
- Verify saved outputs and checkpoints.
- Inspect whether existing behavior remains intact.

### 10.3 Verification Reporting

Clearly distinguish:

- Implemented but not tested.
- Passed static checks.
- Passed synthetic tests.
- Passed real-data functional tests.
- Passed full training or benchmark validation.

Do not claim successful training or benchmark reproduction based solely on syntax checks or synthetic forward passes.

If verification cannot be performed, state the reason and document the remaining checks.

## 11. Resource Management

- Avoid unnecessary GPU memory allocation.
- Avoid repeatedly loading large datasets or checkpoints.
- Prefer small-scale experiments before full-scale training.
- Monitor numerical instability and memory consumption.
- Avoid launching duplicate long-running jobs.
- Do not terminate unrelated processes.
- Do not modify shared environment settings unnecessarily.
- Record resource-related failures when they affect experiment validity.

When a resource limit prevents completion, document the limitation rather than silently reducing the experiment scope in a way that changes its scientific meaning.

## 12. Documentation and Current State

`CURRENT_STATE.md` must reflect verified project progress.

After completing a task, update the relevant sections with:

- Completed changes.
- Files modified.
- Verification performed.
- Experimental results, if any.
- Known issues or limitations.
- Remaining work.
- Relevant artifact or checkpoint paths when applicable.

Rules:

- Keep entries concise and factual.
- Do not duplicate extensive implementation details already documented elsewhere.
- Do not report proposed changes as implemented.
- Do not report unverified behavior as confirmed.
- Keep historical context when needed to understand existing results.
- Maintain consistency between task status and `CURRENT_STATE.md`.

Do not put frequently changing project-state information into `AGENTS.md`.

## 13. Task Completion and Archiving

A task is complete only when:

1. The requested implementation or investigation has been performed.
2. Applicable verification has been executed.
3. Results and limitations have been documented.
4. `CURRENT_STATE.md` has been updated.
5. The task file has been moved from `codex/tasks/active/` to `codex/tasks/archive/`.

If the task cannot be completed:

- Keep it in `active/`, unless explicitly instructed otherwise.
- Record completed work and remaining blockers.
- Do not falsely mark the task as complete.

A completed task may still contain documented limitations, provided its defined scope and acceptance criteria have been satisfied.

## 14. Communication and Reporting

At the end of each task, provide a concise summary containing:

1. What was investigated or implemented.
2. Which files were changed.
3. What verification was performed.
4. What results were observed.
5. What remains unresolved.

Reporting rules:

- Be precise and evidence-based.
- Do not exaggerate improvements.
- Do not conceal failures or uncertainty.
- Separate observations from interpretations.
- Explicitly identify unverified claims.
- Avoid unnecessary explanations unrelated to the selected task.

## 15. Priority and Conflict Resolution

When instructions conflict, follow this order:

1. Explicit user instructions and task-specific requirements, provided they do not violate higher-priority constraints.
2. Global safety and preservation requirements in this file.
3. Existing verified repository behavior and documented project constraints.
4. Implementation preferences and conventions.

A task-specific instruction may intentionally override a default workflow, but must not silently bypass data preservation, scientific integrity, or truthful verification requirements.

If a conflict materially affects the task outcome, report it rather than making an unsupported assumption.

---

**Core Principle**

Make minimal, scientifically justified, reproducible changes. Verify what was actually implemented, preserve existing research artifacts, and document the distinction between intended behavior and demonstrated results.