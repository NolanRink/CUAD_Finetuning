# Additional CUAD Visualizations

These figures were generated from approved final CUAD artifacts only. They do not rerun training or evaluation.

## Generated Visualizations

### `cuad_method_metric_scorecard_heatmap.png`

- Chart type: Heatmap scorecard
- Source files: `artifacts/cuad_final_results_summary.json`, `artifacts/cuad_eval_metrics_full_report.json`, `artifacts/cuad_extractive_eval_metrics_final_report.json`, `artifacts/cuad_zero_shot_eval_metrics_main_report.json`
- Recommended use: both
- Caption: Headline CUAD evaluation metrics for the fine-tuned structured model, extractive QA baseline, and zero-shot generation baseline. Whole-set metrics are the primary comparison because invalid JSON counts as task failure; positive-only normalized F1 is included as a stricter answer-present extraction diagnostic.
- Interpretation note: The fine-tuned structured model has the strongest whole-set found accuracy and normalized token F1, while zero-shot generation is limited by low parse reliability.
- Caveat: Positive-only F1 answers a different diagnostic question than whole-set F1.

### `cuad_finetuned_vs_extractive_found_accuracy_lift_by_category.png`

- Chart type: Sorted horizontal delta bar chart
- Source files: `artifacts/cuad_final_results_summary.json`
- Recommended use: both
- Caption: Category-level difference in whole-set found accuracy between fine-tuned structured generation and extractive QA. Positive values mean the fine-tuned structured model scored higher for that category.
- Interpretation note: The plot identifies categories where fine-tuning adds the most value and the few categories where the extractive baseline is competitive or higher.
- Caveat: Category counts vary, so this is a category-level diagnostic rather than a statistical significance claim.

### `cuad_finetuned_vs_extractive_norm_f1_lift_by_category.png`

- Chart type: Sorted horizontal delta bar chart
- Source files: `artifacts/cuad_final_results_summary.json`
- Recommended use: both
- Caption: Category-level difference in whole-set normalized answer token F1 between fine-tuned structured generation and extractive QA. Positive values mean the fine-tuned structured model scored higher for that category.
- Interpretation note: The plot identifies categories where fine-tuning adds the most value and the few categories where the extractive baseline is competitive or higher.
- Caveat: Category counts vary, so this is a category-level diagnostic rather than a statistical significance claim.

### `cuad_detection_outcome_matrices_2x3.png`

- Chart type: Small-multiple 2x3 heatmaps
- Source files: `artifacts/cuad_eval_predictions_full.jsonl`, `artifacts/cuad_extractive_eval_predictions_final.jsonl`, `artifacts/cuad_zero_shot_eval_predictions_main.jsonl`
- Recommended use: both
- Caption: Found-detection outcomes by method, separating invalid or unparseable outputs from predicted found and predicted not-found decisions. This view is more appropriate than a category confusion matrix because the CUAD category is supplied as part of each input.
- Interpretation note: Fine-tuned generation reduces invalid structured outputs compared with zero-shot generation, while extractive QA has no JSON parsing failure mode.
- Caveat: Invalid JSON is shown separately instead of being forced into false positives or false negatives.

### `cuad_category_present_recall_vs_no_answer_accuracy.png`

- Chart type: Small-multiple scatter plot
- Source files: `artifacts/cuad_final_results_summary.json`
- Recommended use: both
- Caption: Category-level answer-present recall versus no-answer accuracy for each method. High no-answer accuracy alone does not mean extraction is solved because a model can still miss many present clauses.
- Interpretation note: The scatter view separates abstention behavior from answer-present detection and shows that categories differ in detection behavior.
- Caveat: Categories with few answer-present examples should be interpreted cautiously.

### `cuad_invalid_json_rate_by_category.png`

- Chart type: Grouped horizontal bar chart
- Source files: `artifacts/cuad_final_results_summary.json`
- Recommended use: presentation
- Caption: Invalid JSON rate by CUAD category for the two generative methods. Zero-shot generation often fails by format rather than by producing a valid structured answer, and invalid JSON counts as task failure for structured generation.
- Interpretation note: The plot explains why zero-shot generation has weak whole-set performance and shows where fine-tuned generation still has residual parse failures.
- Caveat: Extractive QA is omitted because it does not produce free-form JSON generations.

### `cuad_eval_runtime_vs_performance.png`

- Chart type: Bubble scatter plot
- Source files: `artifacts/cuad_eval_metrics_full_report.json`, `artifacts/cuad_extractive_eval_metrics_final_report.json`, `artifacts/cuad_zero_shot_eval_metrics_main_report.json`
- Recommended use: presentation
- Caption: Evaluation runtime compared with whole-set normalized token F1. Marker size reflects parse rate, so the plot gives compute context without treating the methods as a perfectly controlled efficiency benchmark.
- Interpretation note: Extractive QA evaluates much faster, while zero-shot generation is slow and weak because many outputs are not valid structured JSON.
- Caveat: Runtime is compute context, not a fair architecture-normalized efficiency benchmark.

### `cuad_whole_set_vs_positive_only_norm_f1.png`

- Chart type: Grouped bar chart
- Source files: `artifacts/cuad_eval_metrics_full_report.json`, `artifacts/cuad_extractive_eval_metrics_final_report.json`, `artifacts/cuad_zero_shot_eval_metrics_main_report.json`
- Recommended use: paper
- Caption: Whole-set normalized answer token F1 compared with positive-only normalized token F1. Whole-set F1 reflects no-answer behavior as well as extraction; positive-only F1 isolates answer-present extraction quality.
- Interpretation note: Positive-only extraction remains harder than the whole-set score suggests, including for the fine-tuned structured model.
- Caveat: Whole-set and positive-only F1 should not be read as interchangeable metrics.

### `cuad_category_difficulty_scatter_finetuned.png`

- Chart type: Scatter plot
- Source files: `artifacts/cuad_final_results_summary.json`
- Recommended use: paper
- Caption: Fine-tuned structured generation category diagnostic comparing answer-present recall with normalized answer token F1. Marker size reflects category record count.
- Interpretation note: The plot identifies categories that are difficult because the model both misses present clauses and has weaker normalized answer overlap.
- Caveat: This is a category-level diagnostic; categories with few positives need cautious interpretation.

### `cuad_training_eval_loss_with_best_checkpoints.png`

- Chart type: Line chart with checkpoint markers
- Source files: `artifacts/cuad_train_history_full.jsonl`, `artifacts/cuad_extractive_train_history_final.jsonl`, `artifacts/cuad_train_summary_full.json`, `artifacts/cuad_extractive_train_summary_final.json`
- Recommended use: paper
- Caption: Training and evaluation loss traces for the fine-tuned structured model and the extractive QA baseline, with the best evaluation checkpoint marked for each method.
- Interpretation note: The figure provides training diagnostic context and shows where early stopping selected the saved checkpoint.
- Caveat: Training loss is not the main evidence for extraction quality; final test metrics remain primary.

## Skipped Visualizations

- None. All requested highest-priority and medium-priority visualizations were generated.

## Scope Notes

- Whole-set metrics are primary because invalid JSON counts as task failure for structured generation.
- Positive-only metrics are diagnostic and isolate answer-present extraction quality.
- Category-level plots are diagnostics, not statistical significance tests.
- These figures do not claim retrieval, Qwen, model-sweep, long-context, or production legal-system results.
- Forbidden dryrun, v1, deprecated local prediction, ignored local preprocessing, retrieval, Qwen, and model-sweep artifacts were not used.
