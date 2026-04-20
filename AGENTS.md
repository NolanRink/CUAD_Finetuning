# AGENTS.md

## Working style
- Think before editing.
- Prefer the smallest correct change.
- Keep repo structure clean.
- Do not add dependencies unless clearly justified.
- After making changes, explain what changed and how to verify it.

## Communication
- Summarize assumptions clearly.
- Call out blockers early.
- Suggest follow-up improvements only after the current milestone is stable.

## Project objective
Build a reproducible single-GPU FABRIC pipeline for CUAD legal-contract LLM fine-tuning.

Primary framing:
- Fine-tune an open instruction-tuned LLM on CUAD for structured legal clause extraction.
- Compare it against one simple extractive QA baseline.

Primary output schema:
```json
{
  "category": "<CUAD category>",
  "found": true,
  "normalized_answer": "<short normalized answer or null>",
  "evidence_text": "<supporting contract text or null>"
}
```

## File policy
Keep the repo minimal.

Core repo files:
- AGENTS.md
- cuad_finetune.ipynb
- train_cuad.py
- requirements.txt
- README.md only if clearly useful

Do not create helper modules, extra notebooks, utility scripts, benchmark scripts, or config files unless there is a strong execution or reproducibility reason.

## Workflow split
Use a hybrid workflow.

Notebook responsibilities:
- environment inspection
- CUAD dataset inspection
- preprocessing design and validation
- prompt and output-schema experiments
- zero-shot baseline experiments on small samples
- metrics debugging
- final plots, tables, and qualitative examples

Script responsibilities:
- reproducible preprocessing path
- training pipeline
- evaluation pipeline
- checkpointing
- saving prediction artifacts

## Modeling defaults
Primary task:
- structured generation for CUAD clause and category extraction

Baseline:
- one simple extractive QA baseline for comparison

Preferred model strategy:
- prefer a safe non-gated fallback first
- allow Qwen as a preferred option if access is available
- default to parameter-efficient fine-tuning
- prefer LoRA or QLoRA on a single GPU

## Environment assumptions
Target runtime is a FABRIC Linux VM with one RTX 6000 GPU.

Store large artifacts under mounted project storage:
- /mnt/project/data
- /mnt/project/checkpoints
- /mnt/project/cache/hf
- /mnt/project/artifacts

Do not place large model caches or checkpoints in the home directory.

## Early environment checks
Before real training, verify:
- GPU name and VRAM
- NVIDIA driver
- CUDA availability
- Python version
- writable mounted storage
- free disk space
- Hugging Face cache path
- whether model downloads work from the VM

## Data decisions
- Inspect CUAD formats in the notebook first.
- Decide the working source format before locking in preprocessing.
- Keep train, validation, and test splits at the contract level, not the chunk level.

## Scope rules
Version 1:
- no retrieval pipeline
- no experiment tracker
- no multi-node or distributed training
- no unnecessary file expansion

Leave clean hooks for later retrieval support, but do not implement retrieval in v1.

## Execution rules
For each task, follow:
- Goal
- Context
- Constraints
- Done when

Prefer small scoped diffs.
Validate after each milestone.
If validation fails, fix it before moving on.
Do not silently expand scope.

## Validation expectations
After environment and bootstrap changes:
- notebook runs environment inspection cells successfully
- train_cuad.py runs with --help

After dataset inspection changes:
- notebook shows available CUAD files, schema notes, and example records

After preprocessing changes:
- notebook demonstrates a few transformed examples, including no-answer cases

After training changes:
- run a small smoke test before any full run

After evaluation changes:
- produce sample predictions and category-level metrics
