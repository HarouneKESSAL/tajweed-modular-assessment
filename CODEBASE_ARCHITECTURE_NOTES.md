# Codebase Architecture Notes

## Short Answer

Yes. The codebase needs some structural cleanup, but the issue is concentrated in `scripts/`, not in `src/`.

## What Is Fine

The `src/tajweed_assessment/` package already has a reasonable shape:

- `alignment`
- `data`
- `features`
- `inference`
- `models`
- `training`
- `utils`

That is a sensible application/package layout for the core implementation.

## What Is Not Fine

The problem is that `scripts/` has become too large and is now mixing several different responsibilities:

- manifest building
- training
- evaluation
- analysis
- tuning
- prediction / inspection
- one-off research utilities

This is normal during rapid iteration, but it now creates friction:

- it is harder to discover the right entrypoint
- related logic is repeated across scripts
- experimental scripts and baseline scripts sit side by side without a clear boundary
- checkpoint-selection logic is spread across multiple CLI files

## Main Structural Smell

The main smell is that `scripts/` is acting both as:

1. a lightweight CLI layer
2. a research workspace
3. an orchestration layer

Those roles should be more separated.

## Recommended Refactor Direction

### 1. Split `scripts/` into clearer groups

Recommended grouping:

- `scripts/train/`
- `scripts/eval/`
- `scripts/analyze/`
- `scripts/build/`
- `scripts/inspect/`

This would immediately reduce cognitive load.

### 2. Move repeated evaluator logic into `src/`

A lot of CLI scripts currently reimplement:

- checkpoint loading
- split handling
- localized model loading
- decoder config loading
- metric formatting

Those should be moved into reusable service modules under something like:

- `src/tajweed_assessment/evaluation/`
- `src/tajweed_assessment/decoding/`
- `src/tajweed_assessment/checkpoints/`

The scripts should mostly become thin wrappers.

### 3. Separate baseline paths from experiments

Right now, promoted and experimental artifacts are distinguishable, but the code path still requires care when reading.

A cleaner structure would be:

- `baseline` logic in one place
- `experimental` comparison scripts in another place

This matters because it reduces the risk of accidentally evaluating an experimental path as if it were the official baseline.

### 4. Add a single baseline config registry

Checkpoint selection is currently handled in several scripts through helper functions like “preferred checkpoint” or “approved checkpoint”.

That should be centralized into one file, for example:

- `src/tajweed_assessment/baselines.py`

That file would define the official current checkpoints and decoder configs for:

- duration
- transition
- burst
- content

Then every evaluator and inference script would import from one place.

### 5. Add a top-level orchestration entrypoint

The project now has enough stable moving parts that one orchestration CLI would help.

For example:

- `python -m tajweed_assessment.run suite`
- `python -m tajweed_assessment.run inference`
- `python -m tajweed_assessment.run analysis content`

This is not required immediately, but it is the natural next cleanup step once the baseline is stable.

## What Not To Refactor Right Now

Do not refactor everything at once.

The right order is:

1. centralize baseline checkpoint selection
2. group scripts by function
3. extract repeated loading/evaluation helpers into `src`
4. only then consider bigger CLI restructuring

That gives architectural improvement without destabilizing the current working baseline.

## Practical Conclusion

So yes, the codebase needs a bit more architecture. But the correct interpretation is:

- the core modular design is good
- the baseline system is working
- the main cleanup target is the tooling/orchestration layer around it

This is a normal stage for a project that moved from research iteration into a more stable baseline phase.
