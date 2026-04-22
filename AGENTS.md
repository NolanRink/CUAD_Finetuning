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

## Upgrade plan
Treat the next phase as moving from "pipeline exists" to "project tells a strong experimental story."

Graduate-project quality means:
- strong baselines
- clear experiments
- clear conclusions
- not just a runnable pipeline

Recommended execution order:
1. lock a real baseline
2. complete one real end-to-end fine-tuned run
3. strengthen evaluation and reporting
4. run a small focused ablation set
5. add qualitative error analysis and representative examples
6. polish reproducibility and presentation
7. consider stretch goals only after the core story is complete

Milestone 1: Real baseline comparison
- Keep the simple extractive QA baseline as the main comparison baseline for the core project result.
- Add a small zero-shot or few-shot prompted baseline only as a lightweight additional comparison, not as a new major baseline track.
- Keep all baseline comparisons simple, reproducible, and aligned to the same structured output schema when possible.
- Prefer baseline runs that are cheap enough to validate on a small but meaningful evaluation slice before scaling up.
- Acceptance criteria:
- the extractive QA baseline is documented and evaluated against the same split as the main model
- any zero-shot or few-shot baseline remains small in scope and directly comparable
- baseline results are included in the project narrative

Milestone 2: First complete fine-tuned run
- Complete at least one real end-to-end fine-tuned run, not just a smoke run or dry run.
- Save the checkpoint, prediction artifacts, metrics, and run summary needed to support later analysis.
- Prefer one clean completed run over multiple partial or inconsistent runs.
- Acceptance criteria:
- one full train/evaluate path has run successfully on the intended task
- artifacts are saved and inspectable
- the run can be described clearly in the notebook and supporting documentation

Milestone 3: Stronger evaluation
- Strengthen evaluation beyond simple exact-match summaries.
- Include found / not-found performance.
- Include normalized answer quality.
- Include no-answer handling quality.
- Include per-category results.
- Include stronger overlap or F1-style metrics where appropriate for answer text and evidence spans.
- Keep metrics interpretable and tied to the structured extraction objective.
- Acceptance criteria:
- evaluation captures both detection quality and extraction quality
- no-answer behavior is explicitly measured
- category-level strengths and weaknesses are visible

Milestone 4: Small focused ablation plan
- Add a minimal ablation plan that tests a few high-value design choices without exploding scope.
- Focus the ablations on:
- chunk size and stride
- prompt format
- negative example handling
- model size or model choice if practical within the available compute budget
- Keep the ablation matrix small and intentional.
- Acceptance criteria:
- each ablation has a clear motivation
- the number of runs stays manageable
- results support a concrete conclusion rather than a large unfocused sweep

Milestone 5: Qualitative analysis
- Add qualitative error analysis and representative success/failure examples.
- Cover examples such as:
- correct found predictions
- false positives
- false negatives
- weak normalized answers
- difficult no-answer cases
- Use this section to explain why the model succeeds or fails, not just to display outputs.
- Acceptance criteria:
- the notebook includes representative examples
- examples support the claims made in the conclusions

Milestone 6: Reproducibility and presentation polish
- Improve the repo so another collaborator can understand and rerun the project with minimal friction.
- Add a README only if it is concretely useful.
- Include exact run commands for the main paths.
- Describe the main artifact files and what each one contains.
- Document smoke-run guidance for quick validation before expensive runs.
- Keep the repo minimal while making the execution path easy to follow.
- Acceptance criteria:
- a new collaborator can identify the main commands quickly
- artifact meanings are clear
- smoke-run and full-run expectations are documented

Milestone 7: Scope discipline
- Keep retrieval out of scope for the core version unless the main experiments, comparisons, and analysis are already complete.
- Treat retrieval as a stretch goal, not a requirement for the main result.
- Acceptance criteria:
- the main reportable result stands on its own without retrieval
- any retrieval work is clearly labeled as optional follow-on work

Future Codex priority:
- focus on completing the core experimental story before adding stretch features
- prefer clear comparisons and conclusions over additional infrastructure
