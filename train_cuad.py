"""First runnable CUAD preprocessing, training, and evaluation CLI."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_DATA_ROOT = "/mnt/project/data"
DEFAULT_CHECKPOINT_ROOT = "/mnt/project/checkpoints"
DEFAULT_CACHE_ROOT = "/mnt/project/cache/hf"
DEFAULT_ARTIFACT_ROOT = "/mnt/project/artifacts"

LOCAL_DATA_ROOT = PROJECT_ROOT / "data"
LOCAL_CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
LOCAL_CACHE_ROOT = PROJECT_ROOT / ".hf_cache"
LOCAL_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"

DEFAULT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_QWEN_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_CHECKPOINT_NAME = "cuad_structured_generation"

CUAD_DATASET_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
CUAD_RAW_DIR_NAME = "cuad_qa_raw"
CUAD_PREPROCESSED_DIR_NAME = "cuad_preprocessed"
CUAD_ARCHIVE_NAME = "data.zip"
CUAD_TRAIN_FILENAME = "train_separate_questions.json"
CUAD_TEST_FILENAME = "test.json"

DEFAULT_VALIDATION_FRACTION = 0.1
DEFAULT_SPLIT_SEED = 13
DEFAULT_CHUNK_SIZE_CHARS = 2200
DEFAULT_CHUNK_STRIDE_CHARS = 1600
DEFAULT_MAX_NEGATIVE_CHUNKS_PER_CATEGORY = 1

DEFAULT_MAX_SOURCE_LENGTH = 1024
DEFAULT_MAX_TARGET_LENGTH = 192
DEFAULT_MAX_NEW_TOKENS = 192
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_NUM_TRAIN_EPOCHS = 1.0

DEFAULT_SMOKE_MAX_TRAIN_CONTRACTS = 6
DEFAULT_SMOKE_MAX_TEST_CONTRACTS = 4
DEFAULT_SMOKE_MAX_TRAIN_RECORDS = 32
DEFAULT_SMOKE_MAX_VALIDATION_RECORDS = 16
DEFAULT_SMOKE_MAX_EVAL_RECORDS = 16
DEFAULT_SMOKE_MAX_STEPS = 2

CATEGORY_PATTERN = re.compile(r'related to "(?P<category>.+?)"')
JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
WHITESPACE_PATTERN = re.compile(r"\s+")

FABRIC_FIRST_RUN_GUIDE = """FABRIC first run:
  1. Verify mounted project storage before creating the environment:
     ls -ld /mnt/project
     mkdir -p /mnt/project/data /mnt/project/checkpoints /mnt/project/cache/hf /mnt/project/artifacts
     ls -ld /mnt/project/data /mnt/project/checkpoints /mnt/project/cache/hf /mnt/project/artifacts
     test -w /mnt/project/data && test -w /mnt/project/checkpoints && test -w /mnt/project/cache/hf && test -w /mnt/project/artifacts

  2. Set and verify Hugging Face cache paths:
     export HF_HOME=/mnt/project/cache/hf
     export TRANSFORMERS_CACHE=/mnt/project/cache/hf
     echo $HF_HOME
     echo $TRANSFORMERS_CACHE

  3. Verify GPU, CUDA, and free space:
     nvidia-smi
     python -c "import shutil; print('free_gb=', round(shutil.disk_usage('/mnt/project').free / (1024 ** 3), 2))"

  4. Create the environment and install dependencies:
     python -m venv .venv
     source .venv/bin/activate
     pip install -r requirements.txt
     python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

  5. Run the first smoke path with mounted storage defaults:
     python train_cuad.py preprocess --smoke --output-name cuad_preprocessed_smoke
     python train_cuad.py train --preprocessed-name cuad_preprocessed_smoke --smoke --max-train-steps 2 --summary-name cuad_train_summary_smoke.json
     python train_cuad.py evaluate --preprocessed-name cuad_preprocessed_smoke --smoke --checkpoint-name cuad_structured_generation --prediction-name cuad_eval_predictions_smoke.jsonl --sample-prediction-name cuad_eval_prediction_samples_smoke.jsonl --metrics-name cuad_eval_metrics_smoke.json
"""


class ListDataset:
    def __init__(self, features: list[dict[str, Any]]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.features[index]


class SupervisedDataCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []

        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def should_preserve_fabric_default_root(
    configured_value: str,
    default_value: str,
) -> bool:
    return (
        configured_value == default_value
        and default_value.startswith("/mnt/project/")
        and Path("/mnt/project").exists()
    )


def resolve_default_root(
    configured_value: str,
    default_value: str,
    local_fallback: Path,
) -> str:
    configured_path = Path(configured_value)
    if should_preserve_fabric_default_root(configured_value, default_value):
        return str(configured_path)
    if configured_value == default_value and not configured_path.exists():
        return str(local_fallback)
    return str(configured_path)


def resolve_roots(args: argparse.Namespace) -> dict[str, str]:
    return {
        "data_root": resolve_default_root(args.data_root, DEFAULT_DATA_ROOT, LOCAL_DATA_ROOT),
        "checkpoint_root": resolve_default_root(
            args.checkpoint_root,
            DEFAULT_CHECKPOINT_ROOT,
            LOCAL_CHECKPOINT_ROOT,
        ),
        "cache_root": resolve_default_root(args.cache_root, DEFAULT_CACHE_ROOT, LOCAL_CACHE_ROOT),
        "artifact_root": resolve_default_root(
            args.artifact_root,
            DEFAULT_ARTIFACT_ROOT,
            LOCAL_ARTIFACT_ROOT,
        ),
    }


def configure_hf_cache(cache_root: str) -> None:
    os.environ["HF_HOME"] = cache_root
    os.environ["TRANSFORMERS_CACHE"] = cache_root
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def ensure_runtime_roots(
    roots: dict[str, str],
) -> None:
    for name, root_value in roots.items():
        root_path = Path(root_value)
        if root_value.startswith("/mnt/project/"):
            if not Path("/mnt/project").exists():
                raise FileNotFoundError(
                    f"Expected mounted FABRIC project storage at /mnt/project before using {name} "
                    f"root {root_path}. Mount project storage first or override the path explicitly."
                )
            ensure_directory(root_path)
        else:
            ensure_directory(root_path)


def resolve_model_name(args: argparse.Namespace) -> str:
    if args.model_name:
        return args.model_name
    if getattr(args, "use_qwen", False):
        return DEFAULT_QWEN_MODEL_NAME
    return DEFAULT_MODEL_NAME


def normalize_answer_texts(answer_texts: list[str]) -> str | None:
    seen: set[str] = set()
    deduped: list[str] = []
    for answer_text in answer_texts:
        cleaned = answer_text.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    if not deduped:
        return None
    return "; ".join(deduped)


def build_evidence_text(evidence_spans: list[str]) -> str | None:
    deduped = normalize_answer_texts(evidence_spans)
    if deduped is None:
        return None
    return deduped


def normalize_metric_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = WHITESPACE_PATTERN.sub(" ", str(value).strip()).lower()
    return normalized or None


def build_structured_target(
    category: str,
    found: bool,
    normalized_answer: str | None,
    evidence_text: str | None,
) -> dict[str, Any]:
    if not isinstance(found, bool):
        raise TypeError("found must be a JSON boolean.")
    if not found and (normalized_answer is not None or evidence_text is not None):
        raise ValueError("No-answer examples must use null answer and evidence fields.")
    if found and (not isinstance(evidence_text, str) or not evidence_text.strip()):
        raise ValueError("Positive examples must include non-empty evidence text.")

    return {
        "category": category,
        "found": found,
        "normalized_answer": normalized_answer if found else None,
        "evidence_text": evidence_text if found else None,
    }


def build_instruction_text(category: str) -> str:
    return (
        "Read the contract chunk and return JSON only. Use exactly these keys: "
        "category, found, normalized_answer, evidence_text. For the CUAD "
        f"category '{category}', if the category is absent set found=false, "
        "normalized_answer=null, and evidence_text=null."
    )


def truncate_preview_text(text: str | None, max_chars: int = 400) -> str | None:
    if text is None or len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def render_training_prompt(record: dict[str, Any]) -> str:
    return (
        f"{record['instruction']}\n\n"
        f"Contract chunk:\n{record['input_text']}\n\n"
        "JSON:\n"
    )


def extract_category(question: str, question_id: str) -> str:
    match = CATEGORY_PATTERN.search(question)
    if match:
        return match.group("category").strip()
    if "__" in question_id:
        return question_id.split("__", 1)[1].rsplit("_", 1)[0].strip()
    raise ValueError(f"Could not infer CUAD category from id={question_id!r}")


def derive_contract_id(title: str, question_id: str) -> str:
    cleaned_title = title.strip()
    if cleaned_title:
        return cleaned_title
    return question_id.split("__", 1)[0].strip()


def find_cuad_raw_files(dataset_root: Path) -> dict[str, Path] | None:
    expected = {
        "train": dataset_root / CUAD_TRAIN_FILENAME,
        "test": dataset_root / CUAD_TEST_FILENAME,
    }
    if all(path.exists() for path in expected.values()):
        return expected

    matches: dict[str, Path] = {}
    for path in dataset_root.rglob("*.json"):
        if path.name == CUAD_TRAIN_FILENAME:
            matches["train"] = path
        elif path.name == CUAD_TEST_FILENAME:
            matches["test"] = path
    if {"train", "test"} <= set(matches):
        return matches
    return None


def download_and_extract_cuad_dataset(dataset_root: Path) -> dict[str, Path]:
    ensure_directory(dataset_root)
    archive_path = dataset_root / CUAD_ARCHIVE_NAME
    urllib.request.urlretrieve(CUAD_DATASET_URL, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(dataset_root)
    raw_files = find_cuad_raw_files(dataset_root)
    if raw_files is None:
        raise FileNotFoundError(
            f"Downloaded CUAD archive but could not find {CUAD_TRAIN_FILENAME} and "
            f"{CUAD_TEST_FILENAME} under {dataset_root}."
        )
    return raw_files


def resolve_answer_span(
    contract_text: str,
    answer_text: str,
    answer_start: int,
) -> dict[str, Any] | None:
    cleaned_answer_text = answer_text.strip()
    if not cleaned_answer_text:
        return None

    if 0 <= answer_start < len(contract_text):
        candidate_end = answer_start + len(cleaned_answer_text)
        if contract_text[answer_start:candidate_end] == cleaned_answer_text:
            return {
                "text": cleaned_answer_text,
                "start": answer_start,
                "end": candidate_end,
            }

    fallback_start = contract_text.find(cleaned_answer_text)
    if fallback_start >= 0:
        return {
            "text": cleaned_answer_text,
            "start": fallback_start,
            "end": fallback_start + len(cleaned_answer_text),
        }
    return None


def load_cuad_contracts(filepath: Path) -> list[dict[str, Any]]:
    payload = json.loads(filepath.read_text(encoding="utf-8"))
    contracts: list[dict[str, Any]] = []

    for document in payload.get("data", []):
        title = str(document.get("title", "")).strip()
        paragraphs = document.get("paragraphs", []) or []
        if not paragraphs:
            continue

        contract_parts: list[str] = []
        paragraph_offsets: list[int] = []
        offset = 0
        for paragraph_index, paragraph in enumerate(paragraphs):
            if paragraph_index > 0:
                contract_parts.append("\n\n")
                offset += 2
            paragraph_offsets.append(offset)
            context = str(paragraph.get("context", ""))
            contract_parts.append(context)
            offset += len(context)

        contract_text = "".join(contract_parts)
        first_question_id = ""
        first_qas = paragraphs[0].get("qas", []) or []
        if first_qas:
            first_question_id = str(first_qas[0].get("id", "")).strip()
        contract_id = derive_contract_id(title, first_question_id)

        annotations: list[dict[str, Any]] = []
        for paragraph_index, paragraph in enumerate(paragraphs):
            paragraph_offset = paragraph_offsets[paragraph_index]
            for qa in paragraph.get("qas", []) or []:
                question = str(qa.get("question", "")).strip()
                question_id = str(qa.get("id", "")).strip()
                answer_texts: list[str] = []
                answer_spans: list[dict[str, Any]] = []
                for answer in qa.get("answers", []) or []:
                    answer_text = str(answer.get("text", "")).strip()
                    if not answer_text:
                        continue
                    answer_texts.append(answer_text)
                    try:
                        local_start = int(answer.get("answer_start", -1))
                    except (TypeError, ValueError):
                        local_start = -1
                    global_start = paragraph_offset + local_start if local_start >= 0 else -1
                    resolved_span = resolve_answer_span(contract_text, answer_text, global_start)
                    if resolved_span is not None:
                        answer_spans.append(resolved_span)

                annotations.append(
                    {
                        "question_id": question_id,
                        "question": question,
                        "category": extract_category(question, question_id),
                        "answer_texts": answer_texts,
                        "answer_spans": answer_spans,
                    }
                )

        contracts.append(
            {
                "title": title,
                "contract_id": contract_id,
                "contract_text": contract_text,
                "num_paragraphs": len(paragraphs),
                "annotations": annotations,
            }
        )

    return contracts


def chunk_contract_text(
    contract_id: str,
    contract_text: str,
    chunk_size_chars: int,
    chunk_stride_chars: int,
) -> list[dict[str, Any]]:
    if chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars must be positive.")
    if chunk_stride_chars <= 0 or chunk_stride_chars > chunk_size_chars:
        raise ValueError("chunk_stride_chars must be in the range [1, chunk_size_chars].")

    if not contract_text:
        return [
            {
                "contract_id": contract_id,
                "chunk_id": f"{contract_id}_chunk_0000",
                "chunk_index": 0,
                "chunk_char_start": 0,
                "chunk_char_end": 0,
                "chunk_text": "",
            }
        ]

    chunks: list[dict[str, Any]] = []
    text_length = len(contract_text)
    start = 0
    chunk_index = 0

    while start < text_length:
        end = min(start + chunk_size_chars, text_length)
        chunks.append(
            {
                "contract_id": contract_id,
                "chunk_id": f"{contract_id}_chunk_{chunk_index:04d}",
                "chunk_index": chunk_index,
                "chunk_char_start": start,
                "chunk_char_end": end,
                "chunk_text": contract_text[start:end],
            }
        )
        if end >= text_length:
            break
        start += chunk_stride_chars
        chunk_index += 1

    return chunks


def select_evenly_spaced_indices(total_count: int, max_count: int) -> list[int]:
    if total_count <= 0 or max_count <= 0:
        return []
    if total_count <= max_count:
        return list(range(total_count))
    if max_count == 1:
        return [0]
    return sorted(
        {
            round(index * (total_count - 1) / (max_count - 1))
            for index in range(max_count)
        }
    )


def build_chunk_annotations_for_contract(
    contract: dict[str, Any],
    chunk_size_chars: int,
    chunk_stride_chars: int,
    max_negative_chunks_per_category: int,
) -> list[dict[str, Any]]:
    chunks = chunk_contract_text(
        contract_id=contract["contract_id"],
        contract_text=contract["contract_text"],
        chunk_size_chars=chunk_size_chars,
        chunk_stride_chars=chunk_stride_chars,
    )
    chunk_annotations: list[dict[str, Any]] = []

    for annotation in contract["annotations"]:
        positive_rows: list[dict[str, Any]] = []
        negative_candidates: list[dict[str, Any]] = []
        for chunk in chunks:
            visible_span_records = [
                span
                for span in annotation["answer_spans"]
                if chunk["chunk_char_start"] <= span["start"] and span["end"] <= chunk["chunk_char_end"]
            ]
            visible_answers = [
                span["text"]
                for span in visible_span_records
            ]
            visible_evidence_spans = [
                contract["contract_text"][span["start"] : span["end"]]
                for span in visible_span_records
            ]
            if not visible_answers and annotation["answer_texts"]:
                visible_answers = [
                    answer_text.strip()
                    for answer_text in annotation["answer_texts"]
                    if answer_text.strip() and answer_text.strip() in chunk["chunk_text"]
                ]
                visible_evidence_spans = visible_answers

            if visible_answers:
                normalized_answer = normalize_answer_texts(visible_answers)
                evidence_text = build_evidence_text(visible_evidence_spans)
                positive_rows.append(
                    {
                        **chunk,
                        "source_id": annotation["question_id"],
                        "source_question": annotation["question"],
                        "source_title": contract["title"],
                        "category": annotation["category"],
                        "found": True,
                        "normalized_answer": normalized_answer,
                        "evidence_text": evidence_text,
                        "num_visible_answers": len(visible_answers),
                    }
                )
            else:
                negative_candidates.append(chunk)

        if positive_rows:
            chunk_annotations.extend(positive_rows)
        for negative_index in select_evenly_spaced_indices(
            len(negative_candidates),
            max_negative_chunks_per_category,
        ):
            chunk = negative_candidates[negative_index]
            chunk_annotations.append(
                {
                    **chunk,
                    "source_id": annotation["question_id"],
                    "source_question": annotation["question"],
                    "source_title": contract["title"],
                    "category": annotation["category"],
                    "found": False,
                    "normalized_answer": None,
                    "evidence_text": None,
                    "num_visible_answers": 0,
                }
            )

    return chunk_annotations


def transform_chunk_annotation_to_record(
    *,
    contract_id: str,
    chunk_id: str,
    chunk_index: int,
    category: str,
    found: bool,
    chunk_text: str,
    normalized_answer: str | None,
    evidence_text: str | None,
) -> dict[str, Any]:
    target = build_structured_target(
        category=category,
        found=found,
        normalized_answer=normalized_answer,
        evidence_text=evidence_text,
    )
    return {
        "contract_id": contract_id,
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "category": category,
        "instruction": build_instruction_text(category),
        "input_text": chunk_text,
        "target_json": json.dumps(target, ensure_ascii=True),
    }


def build_training_examples_from_chunk_annotations(
    chunk_annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for annotation in chunk_annotations:
        record = transform_chunk_annotation_to_record(
            contract_id=annotation["contract_id"],
            chunk_id=annotation["chunk_id"],
            chunk_index=annotation["chunk_index"],
            category=annotation["category"],
            found=annotation["found"],
            chunk_text=annotation["chunk_text"],
            normalized_answer=annotation["normalized_answer"],
            evidence_text=annotation["evidence_text"],
        )
        record["chunk_char_start"] = annotation["chunk_char_start"]
        record["chunk_char_end"] = annotation["chunk_char_end"]
        record["source_id"] = annotation["source_id"]
        record["source_question"] = annotation["source_question"]
        record["source_title"] = annotation["source_title"]
        record["num_visible_answers"] = annotation["num_visible_answers"]
        records.append(record)
    return records


def build_records_for_contracts(
    contracts: list[dict[str, Any]],
    chunk_size_chars: int,
    chunk_stride_chars: int,
    max_negative_chunks_per_category: int,
) -> list[dict[str, Any]]:
    chunk_annotations: list[dict[str, Any]] = []
    for contract in contracts:
        chunk_annotations.extend(
            build_chunk_annotations_for_contract(
                contract=contract,
                chunk_size_chars=chunk_size_chars,
                chunk_stride_chars=chunk_stride_chars,
                max_negative_chunks_per_category=max_negative_chunks_per_category,
            )
        )
    return build_training_examples_from_chunk_annotations(chunk_annotations)


def select_validation_contract_ids(
    contract_ids: list[str],
    validation_fraction: float,
    split_seed: int,
) -> set[str]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in the range [0.0, 1.0).")

    unique_contract_ids = sorted(set(contract_ids))
    if len(unique_contract_ids) <= 1 or validation_fraction == 0.0:
        return set()

    rng = random.Random(split_seed)
    rng.shuffle(unique_contract_ids)
    validation_count = int(round(len(unique_contract_ids) * validation_fraction))
    validation_count = max(1, min(validation_count, len(unique_contract_ids) - 1))
    return set(unique_contract_ids[:validation_count])


def limit_contracts_for_smoke(
    contracts: list[dict[str, Any]],
    max_contracts: int,
) -> list[dict[str, Any]]:
    if max_contracts <= 0:
        raise ValueError("max_contracts must be positive when smoke mode is enabled.")
    return contracts[:max_contracts]


def build_preprocessed_splits(
    train_contracts: list[dict[str, Any]],
    test_contracts: list[dict[str, Any]],
    validation_fraction: float,
    split_seed: int,
    chunk_size_chars: int,
    chunk_stride_chars: int,
    max_negative_chunks_per_category: int,
) -> dict[str, list[dict[str, Any]]]:
    validation_contract_ids = select_validation_contract_ids(
        [contract["contract_id"] for contract in train_contracts],
        validation_fraction=validation_fraction,
        split_seed=split_seed,
    )

    train_split_contracts: list[dict[str, Any]] = []
    validation_split_contracts: list[dict[str, Any]] = []
    for contract in train_contracts:
        if contract["contract_id"] in validation_contract_ids:
            validation_split_contracts.append(contract)
        else:
            train_split_contracts.append(contract)

    return {
        "train": build_records_for_contracts(
            train_split_contracts,
            chunk_size_chars=chunk_size_chars,
            chunk_stride_chars=chunk_stride_chars,
            max_negative_chunks_per_category=max_negative_chunks_per_category,
        ),
        "validation": build_records_for_contracts(
            validation_split_contracts,
            chunk_size_chars=chunk_size_chars,
            chunk_stride_chars=chunk_stride_chars,
            max_negative_chunks_per_category=max_negative_chunks_per_category,
        ),
        "test": build_records_for_contracts(
            test_contracts,
            chunk_size_chars=chunk_size_chars,
            chunk_stride_chars=chunk_stride_chars,
            max_negative_chunks_per_category=max_negative_chunks_per_category,
        ),
    }


def write_jsonl(records: list[dict[str, Any]], filepath: Path) -> None:
    with filepath.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_json(payload: dict[str, Any], filepath: Path) -> None:
    filepath.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_run_summary(
    artifact_root: Path,
    filename: str,
    payload: dict[str, Any],
) -> Path:
    summary_path = artifact_root / filename
    write_json(payload, summary_path)
    return summary_path


def load_jsonl(filepath: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with filepath.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def validate_preprocessed_record(record: dict[str, Any]) -> None:
    required_keys = {
        "contract_id",
        "chunk_id",
        "chunk_index",
        "category",
        "instruction",
        "input_text",
        "target_json",
    }
    missing_keys = required_keys - set(record)
    if missing_keys:
        raise ValueError(f"Preprocessed record is missing keys: {sorted(missing_keys)}")

    target = json.loads(record["target_json"])
    build_structured_target(
        category=target["category"],
        found=target["found"],
        normalized_answer=target["normalized_answer"],
        evidence_text=target["evidence_text"],
    )


def summarize_split(records: list[dict[str, Any]]) -> dict[str, Any]:
    positive_records = 0
    categories: dict[str, int] = {}
    contract_ids: set[str] = set()
    chunk_ids: set[str] = set()
    for record in records:
        target = json.loads(record["target_json"])
        if target["found"]:
            positive_records += 1
        categories[record["category"]] = categories.get(record["category"], 0) + 1
        contract_ids.add(record["contract_id"])
        chunk_ids.add(record["chunk_id"])

    sample_categories = dict(sorted(categories.items())[:5])
    return {
        "num_records": len(records),
        "num_contracts": len(contract_ids),
        "num_chunks": len(chunk_ids),
        "num_positive": positive_records,
        "num_no_answer": len(records) - positive_records,
        "num_categories": len(categories),
        "sample_categories": sample_categories,
    }


def sample_records(
    records: list[dict[str, Any]],
    max_records: int | None,
    split_seed: int,
) -> list[dict[str, Any]]:
    if max_records is None or len(records) <= max_records:
        return records
    rng = random.Random(split_seed)
    sampled_records = list(records)
    rng.shuffle(sampled_records)
    return sampled_records[:max_records]


def load_preprocessed_split(preprocessed_dir: Path, split_name: str) -> list[dict[str, Any]]:
    split_path = preprocessed_dir / f"{split_name}.jsonl"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Expected preprocessed split at {split_path}. Run preprocess first."
        )
    records = load_jsonl(split_path)
    for record in records[:5]:
        validate_preprocessed_record(record)
    return records


def parse_prediction_text(
    prediction_text: str,
    fallback_category: str,
) -> tuple[dict[str, Any], bool, str | None]:
    match = JSON_BLOCK_PATTERN.search(prediction_text)
    json_text = match.group(0) if match else prediction_text
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return (
            build_structured_target(
                category=fallback_category,
                found=False,
                normalized_answer=None,
                evidence_text=None,
            ),
            False,
            str(exc),
        )

    if not isinstance(parsed, dict):
        return (
            build_structured_target(
                category=fallback_category,
                found=False,
                normalized_answer=None,
                evidence_text=None,
            ),
            False,
            "Prediction JSON must decode to an object.",
        )

    found = parsed.get("found")
    if not isinstance(found, bool):
        return (
            build_structured_target(
                category=fallback_category,
                found=False,
                normalized_answer=None,
                evidence_text=None,
            ),
            False,
            "Prediction JSON must include a boolean 'found' field.",
        )
    normalized_answer = parsed.get("normalized_answer")
    evidence_text = parsed.get("evidence_text")
    category = str(parsed.get("category", fallback_category)).strip() or fallback_category
    try:
        return (
            build_structured_target(
                category=category,
                found=found,
                normalized_answer=normalized_answer,
                evidence_text=evidence_text,
            ),
            True,
            None,
        )
    except (TypeError, ValueError) as exc:
        return (
            build_structured_target(
                category=fallback_category,
                found=False,
                normalized_answer=None,
                evidence_text=None,
            ),
            False,
            str(exc),
        )


def build_generation_features(
    records: list[dict[str, Any]],
    tokenizer: Any,
    max_source_length: int,
    max_target_length: int,
) -> list[dict[str, Any]]:
    eos_text = tokenizer.eos_token or ""
    features: list[dict[str, Any]] = []
    for record in records:
        prompt_text = render_training_prompt(record)
        target_text = record["target_json"] + eos_text

        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_source_length,
        )["input_ids"]
        target_ids = tokenizer(
            target_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_target_length,
        )["input_ids"]
        if not target_ids:
            raise ValueError("Tokenized target is empty; increase max_target_length.")

        input_ids = prompt_ids + target_ids
        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": [-100] * len(prompt_ids) + target_ids,
            }
        )
    return features


def import_training_stack(
    *,
    include_trainer: bool,
) -> tuple[Any, Any, Any | None, Any | None]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Training requires torch and transformers from requirements.txt."
        ) from exc

    Trainer = None
    TrainingArguments = None
    if include_trainer:
        try:
            from transformers import Trainer, TrainingArguments
        except ImportError as exc:
            raise RuntimeError(
                "Training requires Trainer and TrainingArguments from transformers."
            ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def choose_torch_dtype(torch: Any) -> Any | None:
    if not torch.cuda.is_available():
        return None
    if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        return torch.bfloat16
    return torch.float16


def infer_lora_target_modules(model: Any, torch: Any) -> list[str]:
    common_names = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "c_attn",
        "c_proj",
        "c_fc",
    }
    found = {
        name.split(".")[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in common_names
    }
    return sorted(found)


def prepare_tokenizer(tokenizer: Any, model: Any | None = None) -> Any:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer must have either a pad token or eos token.")
    if model is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer


def load_model_for_training(
    model_name: str,
    cache_root: str,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> tuple[Any, Any, dict[str, Any]]:
    torch, AutoModelForCausalLM, AutoTokenizer, _, _ = import_training_stack(
        include_trainer=False,
    )
    model_kwargs: dict[str, Any] = {"cache_dir": cache_root}
    torch_dtype = choose_torch_dtype(torch)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
        model_kwargs["low_cpu_mem_usage"] = True

    tokenizer = prepare_tokenizer(
        AutoTokenizer.from_pretrained(model_name, cache_dir=cache_root),
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    prepare_tokenizer(tokenizer, model=model)

    lora_summary: dict[str, Any] = {"enabled": False, "target_modules": []}
    if use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise RuntimeError("LoRA requested, but peft is not installed.") from exc

        target_modules = infer_lora_target_modules(model, torch)
        if target_modules:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=target_modules,
            )
            model = get_peft_model(model, peft_config)
            lora_summary = {"enabled": True, "target_modules": target_modules}

    return model, tokenizer, lora_summary


def checkpoint_contains_model_artifacts(checkpoint_path: Path) -> bool:
    has_adapter_config = (checkpoint_path / "adapter_config.json").exists()
    has_adapter_weights = any(
        (checkpoint_path / filename).exists()
        for filename in ("adapter_model.safetensors", "adapter_model.bin")
    )
    if has_adapter_config and has_adapter_weights:
        return True

    has_model_config = (checkpoint_path / "config.json").exists()
    has_model_weights = any(
        (checkpoint_path / filename).exists()
        for filename in ("model.safetensors", "pytorch_model.bin")
    )
    has_tokenizer_config = (checkpoint_path / "tokenizer_config.json").exists()
    has_tokenizer_files = any(
        (checkpoint_path / filename).exists()
        for filename in (
            "tokenizer.json",
            "tokenizer.model",
            "sentencepiece.bpe.model",
            "vocab.json",
        )
    )
    return (
        has_model_config
        and has_model_weights
        and has_tokenizer_config
        and has_tokenizer_files
    )


def resolve_checkpoint_source(
    args: argparse.Namespace,
    roots: dict[str, str],
) -> Path:
    if args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Explicit checkpoint path does not exist: {checkpoint_path}"
            )
        if not checkpoint_contains_model_artifacts(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint path exists but does not contain model artifacts: {checkpoint_path}"
            )
        return checkpoint_path
    candidate = Path(roots["checkpoint_root"]) / args.checkpoint_name
    if candidate.exists() and checkpoint_contains_model_artifacts(candidate):
        return candidate
    raise FileNotFoundError(
        f"No checkpoint found at {candidate}. Run training first, pass --checkpoint-path, "
        "or use --dry-run for reference-only evaluation."
    )


def load_model_for_inference(
    model_name: str,
    checkpoint_source: Path | None,
    cache_root: str,
) -> tuple[Any, Any, str]:
    torch, AutoModelForCausalLM, AutoTokenizer, _, _ = import_training_stack(
        include_trainer=False,
    )
    model_kwargs: dict[str, Any] = {"cache_dir": cache_root}
    torch_dtype = choose_torch_dtype(torch)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
        model_kwargs["low_cpu_mem_usage"] = True

    if checkpoint_source is not None and (checkpoint_source / "adapter_config.json").exists():
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Adapter checkpoint found, but peft is not installed.") from exc

        tokenizer_source = (
            str(checkpoint_source)
            if (checkpoint_source / "tokenizer_config.json").exists()
            else model_name
        )
        tokenizer = prepare_tokenizer(
            AutoTokenizer.from_pretrained(tokenizer_source, cache_dir=cache_root),
        )
        base_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model = PeftModel.from_pretrained(base_model, str(checkpoint_source))
        prepare_tokenizer(tokenizer, model=model)
        return model, tokenizer, str(checkpoint_source)

    model_source = str(checkpoint_source) if checkpoint_source is not None else model_name
    tokenizer_source = (
        str(checkpoint_source)
        if checkpoint_source is not None and (checkpoint_source / "tokenizer_config.json").exists()
        else model_source
    )
    tokenizer = prepare_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_source, cache_dir=cache_root),
    )
    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    prepare_tokenizer(tokenizer, model=model)
    return model, tokenizer, model_source


def build_training_arguments(
    TrainingArguments: Any,
    checkpoint_dir: Path,
    args: argparse.Namespace,
    *,
    has_validation: bool,
    use_bf16: bool,
    use_fp16: bool,
    dataloader_pin_memory: bool,
) -> Any:
    """Adapt Trainer args across transformers releases with small API drift."""
    supported_args = set(inspect.signature(TrainingArguments.__init__).parameters)
    training_kwargs: dict[str, Any] = {
        "output_dir": str(checkpoint_dir),
        "overwrite_output_dir": True,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_train_steps,
        "logging_steps": 1,
        "eval_steps": 1 if has_validation else None,
        "save_strategy": "no",
        "report_to": [],
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "bf16": use_bf16,
        "fp16": use_fp16,
        "dataloader_pin_memory": dataloader_pin_memory,
    }
    if "evaluation_strategy" in supported_args:
        training_kwargs["evaluation_strategy"] = "steps" if has_validation else "no"
    elif "eval_strategy" in supported_args:
        training_kwargs["eval_strategy"] = "steps" if has_validation else "no"

    compatible_kwargs = {
        key: value
        for key, value in training_kwargs.items()
        if key in supported_args and value is not None
    }
    return TrainingArguments(**compatible_kwargs)


def generate_prediction(
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    max_source_length: int,
    max_new_tokens: int,
) -> str:
    import torch

    prompt_text = render_training_prompt(record)
    model_device = next(model.parameters()).device
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_source_length,
    )
    inputs = {key: value.to(model_device) for key, value in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    generated_tokens = generated[0][prompt_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def build_prediction_summary(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not prediction_rows:
        return {
            "num_examples": 0,
            "parsed_json_rate": 0.0,
            "found_accuracy": 0.0,
            "normalized_answer_exact_match": 0.0,
            "evidence_exact_match": 0.0,
            "per_category_found_accuracy": {},
        }

    parsed_count = 0
    found_correct = 0
    normalized_correct = 0
    evidence_correct = 0
    per_category: dict[str, dict[str, int]] = {}

    for row in prediction_rows:
        reference = row["reference_target"]
        prediction = row["prediction_structured"]
        category = reference["category"]
        if row["parsed_json"]:
            parsed_count += 1
        if prediction["found"] == reference["found"]:
            found_correct += 1
        if normalize_metric_text(prediction["normalized_answer"]) == normalize_metric_text(
            reference["normalized_answer"]
        ):
            normalized_correct += 1
        if normalize_metric_text(prediction["evidence_text"]) == normalize_metric_text(
            reference["evidence_text"]
        ):
            evidence_correct += 1

        category_summary = per_category.setdefault(
            category,
            {"num_examples": 0, "found_correct": 0},
        )
        category_summary["num_examples"] += 1
        if prediction["found"] == reference["found"]:
            category_summary["found_correct"] += 1

    per_category_accuracy = {
        category: round(
            counts["found_correct"] / counts["num_examples"],
            4,
        )
        for category, counts in sorted(per_category.items())
    }
    total = len(prediction_rows)
    return {
        "num_examples": total,
        "parsed_json_rate": round(parsed_count / total, 4),
        "found_accuracy": round(found_correct / total, 4),
        "normalized_answer_exact_match": round(normalized_correct / total, 4),
        "evidence_exact_match": round(evidence_correct / total, 4),
        "per_category_found_accuracy": per_category_accuracy,
    }


def add_shared_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="Root directory for CUAD data and derived local files.",
    )
    parser.add_argument(
        "--checkpoint-root",
        default=DEFAULT_CHECKPOINT_ROOT,
        help="Root directory for model checkpoints.",
    )
    parser.add_argument(
        "--cache-root",
        default=DEFAULT_CACHE_ROOT,
        help="Root directory for Hugging Face cache data.",
    )
    parser.add_argument(
        "--artifact-root",
        default=DEFAULT_ARTIFACT_ROOT,
        help="Root directory for metrics, predictions, and reports.",
    )


def add_shared_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Explicit model name or path. If omitted, the script uses the safe "
            "non-gated fallback, unless --use-qwen is set."
        ),
    )
    parser.add_argument(
        "--use-qwen",
        action="store_true",
        help="Use the Qwen instruct model family instead of the default fallback.",
    )


def add_chunking_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--chunk-size-chars",
        type=int,
        default=DEFAULT_CHUNK_SIZE_CHARS,
        help="Fixed chunk size in characters for contract windowing.",
    )
    parser.add_argument(
        "--chunk-stride-chars",
        type=int,
        default=DEFAULT_CHUNK_STRIDE_CHARS,
        help="Stride in characters between chunk windows.",
    )
    parser.add_argument(
        "--max-negative-chunks-per-category",
        type=int,
        default=DEFAULT_MAX_NEGATIVE_CHUNKS_PER_CATEGORY,
        help="Number of sampled negative chunks to keep when a category is absent.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Runnable CUAD preprocessing, smoke-training, and evaluation CLI.",
        epilog=FABRIC_FIRST_RUN_GUIDE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Load raw CUAD, build chunked long-form examples, and write JSONL splits.",
    )
    add_shared_paths(preprocess_parser)
    add_chunking_args(preprocess_parser)
    preprocess_parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional explicit CUAD raw-dataset directory. Defaults to data-root/cuad_qa_raw.",
    )
    preprocess_parser.add_argument(
        "--output-name",
        default=CUAD_PREPROCESSED_DIR_NAME,
        help="Name of the derived preprocessing directory under data-root.",
    )
    preprocess_parser.add_argument(
        "--validation-contract-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
        help="Fraction of train contracts to hold out as validation.",
    )
    preprocess_parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Random seed for contract-level train/validation assignment.",
    )
    preprocess_parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not attempt a download if the raw CUAD JSON files are missing.",
    )
    preprocess_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Restrict preprocessing to a tiny contract subset for a quick smoke pass.",
    )
    preprocess_parser.add_argument(
        "--smoke-max-train-contracts",
        type=int,
        default=DEFAULT_SMOKE_MAX_TRAIN_CONTRACTS,
        help="Train-contract cap used with --smoke.",
    )
    preprocess_parser.add_argument(
        "--smoke-max-test-contracts",
        type=int,
        default=DEFAULT_SMOKE_MAX_TEST_CONTRACTS,
        help="Test-contract cap used with --smoke.",
    )
    preprocess_parser.set_defaults(func=run_preprocess)

    train_parser = subparsers.add_parser(
        "train",
        help="Run the first structured-generation training path on preprocessed CUAD data.",
    )
    add_shared_paths(train_parser)
    add_shared_model_args(train_parser)
    train_parser.add_argument(
        "--preprocessed-name",
        default=CUAD_PREPROCESSED_DIR_NAME,
        help="Derived preprocessing directory name under data-root.",
    )
    train_parser.add_argument(
        "--train-split",
        default="train",
        help="Training split name after preprocessing.",
    )
    train_parser.add_argument(
        "--validation-split",
        default="validation",
        help="Validation split name after preprocessing.",
    )
    train_parser.add_argument(
        "--output-name",
        default=DEFAULT_CHECKPOINT_NAME,
        help="Checkpoint folder name under checkpoint-root.",
    )
    train_parser.add_argument(
        "--max-source-length",
        type=int,
        default=DEFAULT_MAX_SOURCE_LENGTH,
        help="Maximum prompt length after tokenization.",
    )
    train_parser.add_argument(
        "--max-target-length",
        type=int,
        default=DEFAULT_MAX_TARGET_LENGTH,
        help="Maximum target JSON length after tokenization.",
    )
    train_parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Per-device training batch size.",
    )
    train_parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=1,
        help="Per-device evaluation batch size.",
    )
    train_parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps for the Trainer loop.",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate for AdamW.",
    )
    train_parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=DEFAULT_NUM_TRAIN_EPOCHS,
        help="Number of training epochs when max steps is not specified.",
    )
    train_parser.add_argument(
        "--max-train-steps",
        type=int,
        default=-1,
        help="Optional max-step override. Use a positive value for a very small smoke train.",
    )
    train_parser.add_argument(
        "--disable-lora",
        action="store_true",
        help="Train the base model directly instead of adding LoRA adapters.",
    )
    train_parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank.",
    )
    train_parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha.",
    )
    train_parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout.",
    )
    train_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a tiny sample and tiny step budget.",
    )
    train_parser.add_argument(
        "--smoke-max-train-records",
        type=int,
        default=DEFAULT_SMOKE_MAX_TRAIN_RECORDS,
        help="Training-record cap used with --smoke.",
    )
    train_parser.add_argument(
        "--smoke-max-validation-records",
        type=int,
        default=DEFAULT_SMOKE_MAX_VALIDATION_RECORDS,
        help="Validation-record cap used with --smoke.",
    )
    train_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and print a training summary without downloading or training a model.",
    )
    train_parser.add_argument(
        "--summary-name",
        default="cuad_train_summary.json",
        help="Training summary artifact filename under artifact-root.",
    )
    train_parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Deterministic sampling seed for smoke mode.",
    )
    train_parser.set_defaults(func=run_train)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run generation on a preprocessed split and export JSONL predictions.",
    )
    add_shared_paths(evaluate_parser)
    add_shared_model_args(evaluate_parser)
    evaluate_parser.add_argument(
        "--preprocessed-name",
        default=CUAD_PREPROCESSED_DIR_NAME,
        help="Derived preprocessing directory name under data-root.",
    )
    evaluate_parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate.",
    )
    evaluate_parser.add_argument(
        "--checkpoint-name",
        default=DEFAULT_CHECKPOINT_NAME,
        help="Checkpoint directory name under checkpoint-root.",
    )
    evaluate_parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional explicit checkpoint path.",
    )
    evaluate_parser.add_argument(
        "--prediction-name",
        default="cuad_eval_predictions.jsonl",
        help="Prediction artifact filename under artifact-root.",
    )
    evaluate_parser.add_argument(
        "--metrics-name",
        default="cuad_eval_metrics.json",
        help="Summary metrics artifact filename under artifact-root.",
    )
    evaluate_parser.add_argument(
        "--sample-prediction-name",
        default="cuad_eval_prediction_samples.jsonl",
        help="Sample prediction artifact filename under artifact-root.",
    )
    evaluate_parser.add_argument(
        "--num-sample-predictions",
        type=int,
        default=25,
        help="Number of prediction rows to export in the sample artifact.",
    )
    evaluate_parser.add_argument(
        "--max-source-length",
        type=int,
        default=DEFAULT_MAX_SOURCE_LENGTH,
        help="Maximum prompt length after tokenization.",
    )
    evaluate_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Generation budget for prediction JSON.",
    )
    evaluate_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Evaluate only a tiny subset.",
    )
    evaluate_parser.add_argument(
        "--smoke-max-eval-records",
        type=int,
        default=DEFAULT_SMOKE_MAX_EVAL_RECORDS,
        help="Evaluation-record cap used with --smoke.",
    )
    evaluate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write reference-copy predictions without loading a model.",
    )
    evaluate_parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Deterministic sampling seed for smoke mode.",
    )
    evaluate_parser.set_defaults(func=run_evaluate)

    return parser


def run_preprocess(args: argparse.Namespace) -> int:
    roots = resolve_roots(args)
    ensure_runtime_roots(roots)
    data_root = ensure_directory(Path(roots["data_root"]))
    dataset_root = (
        Path(args.dataset_path)
        if args.dataset_path is not None
        else data_root / CUAD_RAW_DIR_NAME
    )

    raw_files = find_cuad_raw_files(dataset_root)
    if raw_files is None:
        if args.skip_download:
            raise FileNotFoundError(
                f"Expected CUAD JSON files under {dataset_root}, but none were found."
            )
        raw_files = download_and_extract_cuad_dataset(dataset_root)

    train_contracts = load_cuad_contracts(raw_files["train"])
    test_contracts = load_cuad_contracts(raw_files["test"])
    if args.smoke:
        train_contracts = limit_contracts_for_smoke(
            train_contracts,
            args.smoke_max_train_contracts,
        )
        test_contracts = limit_contracts_for_smoke(
            test_contracts,
            args.smoke_max_test_contracts,
        )

    splits = build_preprocessed_splits(
        train_contracts=train_contracts,
        test_contracts=test_contracts,
        validation_fraction=args.validation_contract_fraction,
        split_seed=args.split_seed,
        chunk_size_chars=args.chunk_size_chars,
        chunk_stride_chars=args.chunk_stride_chars,
        max_negative_chunks_per_category=args.max_negative_chunks_per_category,
    )

    output_dir = ensure_directory(data_root / args.output_name)
    for split_name, records in splits.items():
        for record in records[:5]:
            validate_preprocessed_record(record)
        write_jsonl(records, output_dir / f"{split_name}.jsonl")

    summary = {
        "dataset_root": str(dataset_root),
        "raw_files": {name: str(path) for name, path in raw_files.items()},
        "output_dir": str(output_dir),
        "smoke": args.smoke,
        "chunk_size_chars": args.chunk_size_chars,
        "chunk_stride_chars": args.chunk_stride_chars,
        "max_negative_chunks_per_category": args.max_negative_chunks_per_category,
        "split_seed": args.split_seed,
        "validation_contract_fraction": args.validation_contract_fraction,
        "raw_train_contracts": len(train_contracts),
        "raw_test_contracts": len(test_contracts),
        "splits": {
            split_name: summarize_split(records) for split_name, records in splits.items()
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("CUAD preprocessing complete.")
    print(json.dumps(summary, indent=2))
    return 0


def run_train(args: argparse.Namespace) -> int:
    roots = resolve_roots(args)
    ensure_runtime_roots(roots)
    model_name = resolve_model_name(args)
    preprocessed_dir = Path(roots["data_root"]) / args.preprocessed_name
    train_records = load_preprocessed_split(preprocessed_dir, args.train_split)
    validation_records = load_preprocessed_split(preprocessed_dir, args.validation_split)
    artifact_root = ensure_directory(Path(roots["artifact_root"]))

    if args.smoke:
        train_records = sample_records(
            train_records,
            args.smoke_max_train_records,
            args.split_seed,
        )
        validation_records = sample_records(
            validation_records,
            args.smoke_max_validation_records,
            args.split_seed,
        )

    summary: dict[str, Any] = {
        "preprocessed_dir": str(preprocessed_dir),
        "model_name": model_name,
        "smoke": args.smoke,
        "dry_run": args.dry_run,
        "train_split": args.train_split,
        "validation_split": args.validation_split,
        "train_summary": summarize_split(train_records),
        "validation_summary": summarize_split(validation_records),
        "prompt_preview": (
            truncate_preview_text(render_training_prompt(train_records[0]))
            if train_records
            else None
        ),
        "target_preview": (
            truncate_preview_text(train_records[0]["target_json"])
            if train_records
            else None
        ),
    }
    if args.dry_run:
        summary["summary_artifact"] = str(artifact_root / args.summary_name)
        write_run_summary(artifact_root, args.summary_name, summary)
        print("Training dry run complete.")
        print(json.dumps(summary, indent=2))
        return 0

    configure_hf_cache(roots["cache_root"])
    torch, _, _, Trainer, TrainingArguments = import_training_stack(
        include_trainer=True,
    )
    checkpoint_dir = ensure_directory(Path(roots["checkpoint_root"]) / args.output_name)
    model, tokenizer, lora_summary = load_model_for_training(
        model_name=model_name,
        cache_root=roots["cache_root"],
        use_lora=not args.disable_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model.config.use_cache = False

    train_features = build_generation_features(
        train_records,
        tokenizer=tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    validation_features = build_generation_features(
        validation_records,
        tokenizer=tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )

    max_train_steps = args.max_train_steps
    if args.smoke and max_train_steps <= 0:
        max_train_steps = DEFAULT_SMOKE_MAX_STEPS

    use_bf16 = torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    args.max_train_steps = max_train_steps
    training_args = build_training_arguments(
        TrainingArguments,
        checkpoint_dir,
        args,
        has_validation=bool(validation_features),
        use_bf16=use_bf16,
        use_fp16=use_fp16,
        dataloader_pin_memory=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ListDataset(train_features),
        eval_dataset=ListDataset(validation_features) if validation_features else None,
        data_collator=SupervisedDataCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(checkpoint_dir)

    summary.update(
        {
            "checkpoint_dir": str(checkpoint_dir),
            "max_train_steps": max_train_steps,
            "max_source_length": args.max_source_length,
            "max_target_length": args.max_target_length,
            "lora": lora_summary,
            "train_feature_count": len(train_features),
            "validation_feature_count": len(validation_features),
        }
    )
    summary["summary_artifact"] = str(artifact_root / args.summary_name)
    write_run_summary(artifact_root, args.summary_name, summary)
    print("Training complete.")
    print(json.dumps(summary, indent=2))
    return 0


def run_evaluate(args: argparse.Namespace) -> int:
    roots = resolve_roots(args)
    ensure_runtime_roots(roots)
    model_name = resolve_model_name(args)
    preprocessed_dir = Path(roots["data_root"]) / args.preprocessed_name
    records = load_preprocessed_split(preprocessed_dir, args.split)
    if args.smoke:
        records = sample_records(records, args.smoke_max_eval_records, args.split_seed)

    artifact_root = ensure_directory(Path(roots["artifact_root"]))
    prediction_path = artifact_root / args.prediction_name
    metrics_path = artifact_root / args.metrics_name
    sample_prediction_path = artifact_root / args.sample_prediction_name
    prediction_rows: list[dict[str, Any]] = []

    if args.dry_run:
        model_source = "reference_copy_dry_run"
        for record in records:
            reference_target = json.loads(record["target_json"])
            prediction_text = record["target_json"]
            prediction_structured, parsed_json, parse_error = parse_prediction_text(
                prediction_text=prediction_text,
                fallback_category=record["category"],
            )
            prediction_rows.append(
                {
                    "contract_id": record["contract_id"],
                    "chunk_id": record["chunk_id"],
                    "category": record["category"],
                    "source_id": record.get("source_id"),
                    "prediction_text": prediction_text,
                    "prediction_structured": prediction_structured,
                    "reference_target": reference_target,
                    "parsed_json": parsed_json,
                    "parse_error": parse_error,
                }
            )
    else:
        configure_hf_cache(roots["cache_root"])
        checkpoint_source = resolve_checkpoint_source(args, roots)
        model, tokenizer, model_source = load_model_for_inference(
            model_name=model_name,
            checkpoint_source=checkpoint_source,
            cache_root=roots["cache_root"],
        )
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        for record in records:
            prediction_text = generate_prediction(
                model=model,
                tokenizer=tokenizer,
                record=record,
                max_source_length=args.max_source_length,
                max_new_tokens=args.max_new_tokens,
            )
            prediction_structured, parsed_json, parse_error = parse_prediction_text(
                prediction_text=prediction_text,
                fallback_category=record["category"],
            )
            prediction_rows.append(
                {
                    "contract_id": record["contract_id"],
                    "chunk_id": record["chunk_id"],
                    "category": record["category"],
                    "source_id": record.get("source_id"),
                    "prediction_text": prediction_text,
                    "prediction_structured": prediction_structured,
                    "reference_target": json.loads(record["target_json"]),
                    "parsed_json": parsed_json,
                    "parse_error": parse_error,
                }
            )

    write_jsonl(prediction_rows, prediction_path)
    write_jsonl(
        prediction_rows[: max(0, args.num_sample_predictions)],
        sample_prediction_path,
    )
    summary = {
        "preprocessed_dir": str(preprocessed_dir),
        "evaluated_split": args.split,
        "smoke": args.smoke,
        "dry_run": args.dry_run,
        "model_source": model_source,
        "prediction_artifact": str(prediction_path),
        "sample_prediction_artifact": str(sample_prediction_path),
        "num_sample_predictions": min(len(prediction_rows), max(0, args.num_sample_predictions)),
        "metrics_artifact": str(metrics_path),
        "metrics": build_prediction_summary(prediction_rows),
    }
    write_json(summary, metrics_path)
    print("Evaluation complete.")
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
