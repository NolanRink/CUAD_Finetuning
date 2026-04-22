"""First runnable CUAD preprocessing, training, and evaluation CLI."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import re
import signal
import time
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
DEFAULT_EXTRACTIVE_QA_MODEL_NAME = "deepset/bert-base-cased-squad2"
DEFAULT_CHECKPOINT_NAME = "cuad_structured_generation"
DEFAULT_EXTRACTIVE_QA_CHECKPOINT_NAME = "cuad_extractive_qa"

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
DEFAULT_EXTRACTIVE_MAX_SEQ_LENGTH = 384
DEFAULT_EXTRACTIVE_MAX_ANSWER_LENGTH = 64
DEFAULT_EXTRACTIVE_NO_ANSWER_THRESHOLD = 0.0
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_NUM_TRAIN_EPOCHS = 1.0

DEFAULT_SMOKE_MAX_TRAIN_CONTRACTS = 6
DEFAULT_SMOKE_MAX_TEST_CONTRACTS = 4
DEFAULT_SMOKE_MAX_TRAIN_RECORDS = 32
DEFAULT_SMOKE_MAX_VALIDATION_RECORDS = 16
DEFAULT_SMOKE_MAX_EVAL_RECORDS = 16
DEFAULT_SMOKE_MAX_STEPS = 2
MODEL_MAX_LENGTH_SENTINEL = 1_000_000

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
     export HF_HUB_CACHE=/mnt/project/cache/hf/hub
     export HF_XET_CACHE=/mnt/project/cache/hf/xet
     export HF_DATASETS_CACHE=/mnt/project/cache/hf/datasets
     export HF_HUB_DISABLE_XET=1
     echo $HF_HOME
     echo $HF_HUB_CACHE
     echo $HF_DATASETS_CACHE

  3. Verify GPU, CUDA, and free space:
     nvidia-smi
     python -c "import shutil; print('free_gb=', round(shutil.disk_usage('/mnt/project').free / (1024 ** 3), 2))"

  4. Create the environment and install dependencies:
     python -m venv .venv
     source .venv/bin/activate
     python -m pip install --upgrade pip setuptools wheel
     python -m pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
     python -m pip install -r requirements.txt
     python -c "import torch; print('torch=', torch.__version__); print('torch_cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

  5. Run the first smoke path with mounted storage defaults:
     python train_cuad.py preprocess --smoke --output-name cuad_preprocessed_smoke
     python train_cuad.py train --preprocessed-name cuad_preprocessed_smoke --smoke --max-train-steps 2 --summary-name cuad_train_summary_smoke.json
     python train_cuad.py evaluate --preprocessed-name cuad_preprocessed_smoke --smoke --checkpoint-name cuad_structured_generation --prediction-name cuad_eval_predictions_smoke.jsonl --sample-prediction-name cuad_eval_prediction_samples_smoke.jsonl --metrics-name cuad_eval_metrics_smoke.json --max-new-tokens 192

  6. Inspect training observability artifacts:
     cat /mnt/project/artifacts/cuad_train_summary.json
     head -n 5 /mnt/project/artifacts/cuad_train_history.jsonl
     ls -l /mnt/project/artifacts/cuad_train_loss_curve.png
     press Ctrl+C once during training to request a graceful stop with artifact save
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
    os.environ["HF_HUB_CACHE"] = str(Path(cache_root) / "hub")
    os.environ["HF_XET_CACHE"] = str(Path(cache_root) / "xet")
    os.environ["HF_DATASETS_CACHE"] = str(Path(cache_root) / "datasets")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.pop("TRANSFORMERS_CACHE", None)
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


def render_extractive_question(record: dict[str, Any]) -> str:
    category = record["category"]
    return (
        f"What is the exact contract span for the CUAD category '{category}'? "
        "If the category is absent in this chunk, answer no span."
    )


def split_answer_candidates(text: str | None) -> list[str]:
    if not isinstance(text, str):
        return []
    candidates: list[str] = []
    for piece in [text, *text.split(";")]:
        cleaned = piece.strip()
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)
    return candidates


def locate_extractive_answer_span(
    context_text: str,
    target: dict[str, Any],
) -> tuple[str, int, int] | None:
    if not target["found"]:
        return None

    candidate_sources = [
        target.get("evidence_text"),
        target.get("normalized_answer"),
    ]
    for source in candidate_sources:
        for candidate in split_answer_candidates(source):
            start_char = context_text.find(candidate)
            if start_char != -1:
                end_char = start_char + len(candidate)
                return candidate, start_char, end_char
    return None


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


def append_jsonl_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    handle.flush()


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


def write_history_artifact(
    artifact_root: Path,
    filename: str,
    rows: list[dict[str, Any]],
) -> Path:
    history_path = artifact_root / filename
    write_jsonl(rows, history_path)
    return history_path


def write_loss_curve_artifact(
    artifact_root: Path,
    filename: str,
    history_rows: list[dict[str, Any]],
) -> Path | None:
    train_steps: list[int] = []
    train_losses: list[float] = []
    eval_steps: list[int] = []
    eval_losses: list[float] = []

    for row in history_rows:
        step = row.get("step")
        if not isinstance(step, int):
            continue
        loss = row.get("loss")
        if row.get("event") == "log" and isinstance(loss, (int, float)):
            train_steps.append(step)
            train_losses.append(float(loss))
        eval_loss = row.get("eval_loss")
        if row.get("event") == "eval" and isinstance(eval_loss, (int, float)):
            eval_steps.append(step)
            eval_losses.append(float(eval_loss))

    if not train_steps and not eval_steps:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    if train_steps:
        axis.plot(train_steps, train_losses, marker="o", linewidth=1.5, label="train_loss")
    if eval_steps:
        axis.plot(eval_steps, eval_losses, marker="s", linewidth=1.5, label="eval_loss")
    axis.set_xlabel("Step")
    axis.set_ylabel("Loss")
    axis.set_title("Training Loss Curve")
    axis.grid(True, alpha=0.3)
    if train_steps or eval_steps:
        axis.legend()

    curve_path = artifact_root / filename
    figure.tight_layout()
    figure.savefig(curve_path, dpi=150)
    plt.close(figure)
    return curve_path


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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
    model_context_window: int,
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
        if len(target_ids) >= model_context_window:
            target_ids = target_ids[: model_context_window - 1]

        max_prompt_tokens = model_context_window - len(target_ids)
        if max_prompt_tokens <= 0:
            raise ValueError(
                "Target token budget exhausts the model context window after truncation. "
                "Reduce max_target_length or choose a larger-context model."
            )
        if len(prompt_ids) > max_prompt_tokens:
            prompt_ids = prompt_ids[:max_prompt_tokens]
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
) -> tuple[Any, Any, Any | None, Any | None, Any | None]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Training requires torch and transformers from requirements.txt."
        ) from exc

    Trainer = None
    TrainingArguments = None
    TrainerCallback = None
    if include_trainer:
        try:
            from transformers import Trainer, TrainerCallback, TrainingArguments
        except ImportError as exc:
            raise RuntimeError(
                "Training requires Trainer and TrainingArguments from transformers."
            ) from exc
    return (
        torch,
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        TrainerCallback,
    )


def import_extractive_qa_stack(
    *,
    include_trainer: bool,
) -> tuple[Any, Any, Any, Any | None, Any | None, Any | None, Any | None]:
    try:
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Extractive QA baseline requires torch and transformers from requirements.txt."
        ) from exc

    Trainer = None
    TrainingArguments = None
    TrainerCallback = None
    DataCollatorWithPadding = None
    if include_trainer:
        try:
            from transformers import (
                DataCollatorWithPadding,
                Trainer,
                TrainerCallback,
                TrainingArguments,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Extractive QA baseline requires Trainer and TrainingArguments from transformers."
            ) from exc
    return (
        torch,
        AutoModelForQuestionAnswering,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        TrainerCallback,
        DataCollatorWithPadding,
    )


def choose_torch_dtype(torch: Any) -> Any | None:
    if not torch.cuda.is_available():
        return None
    if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        return torch.bfloat16
    return torch.float16


def collect_model_length_candidates(tokenizer: Any, model: Any | None = None) -> list[int]:
    candidates: list[int] = []
    tokenizer_candidate = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_candidate, int) and 0 < tokenizer_candidate < MODEL_MAX_LENGTH_SENTINEL:
        candidates.append(tokenizer_candidate)

    config_candidates = (
        "max_position_embeddings",
        "n_positions",
        "max_seq_len",
        "max_sequence_length",
        "seq_length",
        "model_max_length",
    )
    pending_configs: list[Any] = []
    if model is not None and getattr(model, "config", None) is not None:
        pending_configs.append(model.config)

    visited: set[int] = set()
    while pending_configs:
        config = pending_configs.pop()
        config_id = id(config)
        if config_id in visited:
            continue
        visited.add(config_id)

        for attribute in config_candidates:
            candidate = getattr(config, attribute, None)
            if isinstance(candidate, int) and 0 < candidate < MODEL_MAX_LENGTH_SENTINEL:
                candidates.append(candidate)

        nested_config = getattr(config, "text_config", None)
        if nested_config is not None:
            pending_configs.append(nested_config)

    deduped = sorted(set(candidates))
    if not deduped:
        raise RuntimeError(
            "Could not infer a usable model context window from the tokenizer/model config. "
            "Use a model with a defined max sequence length before training or evaluation."
        )
    return deduped


def resolve_model_context_window(tokenizer: Any, model: Any) -> int:
    return min(collect_model_length_candidates(tokenizer, model=model))


def resolve_training_token_budgets(
    tokenizer: Any,
    model: Any,
    requested_max_source_length: int,
    requested_max_target_length: int,
) -> dict[str, int]:
    context_window = resolve_model_context_window(tokenizer, model)
    if requested_max_source_length <= 0:
        raise ValueError("max_source_length must be positive.")
    if requested_max_target_length <= 0:
        raise ValueError("max_target_length must be positive.")
    if context_window < 2:
        raise ValueError(f"Model context window is too small for training: {context_window}")

    effective_max_target_length = min(requested_max_target_length, context_window - 1)
    effective_max_source_length = min(
        requested_max_source_length,
        context_window - effective_max_target_length,
    )
    if effective_max_source_length <= 0:
        raise ValueError(
            "Requested target budget leaves no room for prompt tokens inside the model context "
            f"window ({context_window}). Lower max_target_length or choose a larger-context model."
        )
    return {
        "context_window": context_window,
        "max_source_length": effective_max_source_length,
        "max_target_length": effective_max_target_length,
    }


def resolve_eval_token_budgets(
    tokenizer: Any,
    model: Any,
    requested_max_source_length: int,
    requested_max_new_tokens: int,
) -> dict[str, int]:
    context_window = resolve_model_context_window(tokenizer, model)
    if requested_max_source_length <= 0:
        raise ValueError("max_source_length must be positive.")
    if requested_max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    if context_window < 2:
        raise ValueError(f"Model context window is too small for evaluation: {context_window}")

    effective_max_source_length = min(requested_max_source_length, context_window - 1)
    if effective_max_source_length <= 0:
        raise ValueError(
            f"Model context window ({context_window}) leaves no room for evaluation prompts."
        )
    return {
        "context_window": context_window,
        "max_source_length": effective_max_source_length,
        "max_new_tokens": requested_max_new_tokens,
    }


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
    torch, AutoModelForCausalLM, AutoTokenizer, _, _, _ = import_training_stack(
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


def resolve_adapter_base_model_name(checkpoint_source: Path) -> str:
    adapter_config_path = checkpoint_source / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Adapter config not found at {adapter_config_path}.")
    payload = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    for key in ("base_model_name_or_path", "base_model_name"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise RuntimeError(
        f"Adapter checkpoint at {checkpoint_source} does not record a base model name. "
        "Recreate the checkpoint with adapter metadata or evaluate with a full checkpoint."
    )


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
    torch, AutoModelForCausalLM, AutoTokenizer, _, _, _ = import_training_stack(
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

        base_model_name = resolve_adapter_base_model_name(checkpoint_source)

        tokenizer_source = (
            str(checkpoint_source)
            if (checkpoint_source / "tokenizer_config.json").exists()
            else base_model_name
        )
        tokenizer = prepare_tokenizer(
            AutoTokenizer.from_pretrained(tokenizer_source, cache_dir=cache_root),
        )
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
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


def load_extractive_qa_model(
    model_name: str,
    cache_root: str,
    *,
    checkpoint_source: Path | None = None,
) -> tuple[Any, Any, str]:
    (
        torch,
        AutoModelForQuestionAnswering,
        AutoTokenizer,
        _,
        _,
        _,
        _,
    ) = import_extractive_qa_stack(include_trainer=False)
    model_kwargs: dict[str, Any] = {"cache_dir": cache_root}
    torch_dtype = choose_torch_dtype(torch)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
        model_kwargs["low_cpu_mem_usage"] = True

    model_source = str(checkpoint_source) if checkpoint_source is not None else model_name
    tokenizer_source = (
        str(checkpoint_source)
        if checkpoint_source is not None and (checkpoint_source / "tokenizer_config.json").exists()
        else model_source
    )
    tokenizer = prepare_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_source, cache_dir=cache_root),
    )
    model = AutoModelForQuestionAnswering.from_pretrained(model_source, **model_kwargs)
    prepare_tokenizer(tokenizer, model=model)
    return model, tokenizer, model_source


def build_extractive_qa_features(
    records: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_seq_length: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    features: list[dict[str, Any]] = []
    positive_examples = 0
    aligned_positive_examples = 0
    positive_without_span = 0
    no_answer_examples = 0

    for record in records:
        target = json.loads(record["target_json"])
        question = render_extractive_question(record)
        context_text = record["input_text"]
        tokenized = tokenizer(
            question,
            context_text,
            truncation="only_second",
            max_length=max_seq_length,
            return_offsets_mapping=True,
        )
        offsets = tokenized.pop("offset_mapping")
        sequence_ids = tokenized.sequence_ids()
        cls_token_id = tokenizer.cls_token_id
        cls_index = tokenized["input_ids"].index(cls_token_id) if cls_token_id in tokenized["input_ids"] else 0

        answer_span = locate_extractive_answer_span(context_text, target)
        if target["found"]:
            positive_examples += 1
        else:
            no_answer_examples += 1

        start_position = cls_index
        end_position = cls_index
        if answer_span is not None:
            _, answer_start_char, answer_end_char = answer_span
            context_token_indexes = [index for index, sid in enumerate(sequence_ids) if sid == 1]
            if context_token_indexes:
                context_start = context_token_indexes[0]
                context_end = context_token_indexes[-1]
                if (
                    offsets[context_start][0] <= answer_start_char
                    and offsets[context_end][1] >= answer_end_char
                ):
                    start_position = context_start
                    while (
                        start_position <= context_end
                        and offsets[start_position][0] <= answer_start_char
                    ):
                        start_position += 1
                    start_position -= 1

                    end_position = context_end
                    while (
                        end_position >= context_start
                        and offsets[end_position][1] >= answer_end_char
                    ):
                        end_position -= 1
                    end_position += 1
                    aligned_positive_examples += 1
                else:
                    positive_without_span += 1
            else:
                positive_without_span += 1
        elif target["found"]:
            positive_without_span += 1

        feature = {
            key: value
            for key, value in tokenized.items()
            if key != "offset_mapping"
        }
        feature["start_positions"] = start_position
        feature["end_positions"] = end_position
        features.append(feature)

    stats = {
        "positive_examples": positive_examples,
        "aligned_positive_examples": aligned_positive_examples,
        "positive_without_span": positive_without_span,
        "no_answer_examples": no_answer_examples,
    }
    return features, stats


def extract_best_qa_span(
    start_logits: list[float],
    end_logits: list[float],
    offsets: list[Any],
    sequence_ids: list[Any],
    context_text: str,
    *,
    cls_index: int,
    max_answer_length: int,
    no_answer_threshold: float,
) -> tuple[bool, str | None, float, float]:
    top_k = min(20, len(start_logits))
    start_indexes = sorted(range(len(start_logits)), key=lambda idx: start_logits[idx], reverse=True)[:top_k]
    end_indexes = sorted(range(len(end_logits)), key=lambda idx: end_logits[idx], reverse=True)[:top_k]
    null_score = float(start_logits[cls_index] + end_logits[cls_index])

    best_text: str | None = None
    best_score: float | None = None
    for start_index in start_indexes:
        for end_index in end_indexes:
            if sequence_ids[start_index] != 1 or sequence_ids[end_index] != 1:
                continue
            if end_index < start_index:
                continue
            if end_index - start_index + 1 > max_answer_length:
                continue
            start_offset = offsets[start_index]
            end_offset = offsets[end_index]
            if not isinstance(start_offset, (list, tuple)) or not isinstance(end_offset, (list, tuple)):
                continue
            if len(start_offset) != 2 or len(end_offset) != 2:
                continue
            start_char, _ = start_offset
            _, end_char = end_offset
            if end_char <= start_char:
                continue
            candidate_text = context_text[start_char:end_char].strip()
            if not candidate_text:
                continue
            score = float(start_logits[start_index] + end_logits[end_index])
            if best_score is None or score > best_score:
                best_score = score
                best_text = candidate_text

    if best_text is None or best_score is None or best_score <= null_score + no_answer_threshold:
        return False, None, null_score, null_score
    return True, best_text, best_score, null_score

def build_training_arguments(
    TrainingArguments: Any,
    checkpoint_dir: Path,
    args: argparse.Namespace,
    *,
    has_validation: bool,
    use_bf16: bool,
    use_fp16: bool,
    dataloader_pin_memory: bool,
    label_names: list[str] | None = None,
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
        "label_names": label_names if label_names is not None else ["labels"],
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


def install_safe_stop_handler() -> tuple[dict[str, Any], Any]:
    state = {
        "requested": False,
        "signal_count": 0,
        "requested_at_step": None,
    }
    previous_handler = signal.getsignal(signal.SIGINT)

    def handle_sigint(signum: int, frame: Any) -> None:
        del signum, frame
        state["signal_count"] += 1
        if state["signal_count"] == 1:
            state["requested"] = True
            print(
                "\nSafe stop requested. Training will stop after the current step or evaluation, "
                "save the checkpoint, and write artifacts. Press Ctrl+C again to abort immediately.",
                flush=True,
            )
            return
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_sigint)
    return state, previous_handler


def restore_safe_stop_handler(previous_handler: Any) -> None:
    signal.signal(signal.SIGINT, previous_handler)


def build_training_history_entry(
    *,
    event: str,
    state: Any,
    started_at: float,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "event": event,
        "step": getattr(state, "global_step", None),
        "epoch": getattr(state, "epoch", None),
        "wall_clock_seconds": round(time.monotonic() - started_at, 4),
    }
    if payload:
        entry.update(make_json_safe(payload))
    return entry


def resolve_best_eval_loss(history_rows: list[dict[str, Any]]) -> float | None:
    eval_losses = [
        row.get("eval_loss")
        for row in history_rows
        if row.get("event") == "eval" and isinstance(row.get("eval_loss"), (int, float))
    ]
    if not eval_losses:
        return None
    return float(min(eval_losses))


def count_completed_eval_events(history_rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in history_rows if row.get("event") == "eval")


def build_training_control_callback(
    TrainerCallback: Any,
    *,
    safe_stop_state: dict[str, Any],
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    checkpoint_dir: Path,
    tokenizer: Any,
) -> Any:
    class TrainingControlCallback(TrainerCallback):
        def __init__(self) -> None:
            self.started_at = time.monotonic()
            self.history_rows: list[dict[str, Any]] = []
            self.best_eval_loss: float | None = None
            self.best_eval_step: int | None = None
            self.evals_without_improvement = 0
            self.stop_reason: str | None = None
            self.manual_stop_requested = False
            self.early_stopped = False
            self.best_checkpoint_saved = False

        def _save_best_checkpoint(self, model: Any) -> None:
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            self.best_checkpoint_saved = True

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            self.history_rows.append(
                build_training_history_entry(
                    event="train_begin",
                    state=state,
                    started_at=self.started_at,
                    payload={
                        "early_stopping_patience": early_stopping_patience,
                        "early_stopping_min_delta": early_stopping_min_delta,
                    },
                )
            )
            return control

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            if safe_stop_state["requested"]:
                self.manual_stop_requested = True
                if safe_stop_state["requested_at_step"] is None:
                    safe_stop_state["requested_at_step"] = getattr(state, "global_step", None)
                if self.stop_reason is None:
                    self.stop_reason = "safe_manual_stop_requested"
                control.should_training_stop = True
            return control

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            del args, kwargs
            self.history_rows.append(
                build_training_history_entry(
                    event="log",
                    state=state,
                    started_at=self.started_at,
                    payload=logs or {},
                )
            )
            return control

        def on_evaluate(
            self,
            args: Any,
            state: Any,
            control: Any,
            metrics: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            del args
            payload = metrics or {}
            self.history_rows.append(
                build_training_history_entry(
                    event="eval",
                    state=state,
                    started_at=self.started_at,
                    payload=payload,
                )
            )
            eval_loss = payload.get("eval_loss")
            if isinstance(eval_loss, (int, float)):
                eval_loss = float(eval_loss)
                improvement_threshold = (
                    self.best_eval_loss - early_stopping_min_delta
                    if self.best_eval_loss is not None
                    else None
                )
                if self.best_eval_loss is None or eval_loss < improvement_threshold:
                    self.best_eval_loss = eval_loss
                    self.best_eval_step = getattr(state, "global_step", None)
                    self.evals_without_improvement = 0
                    model = kwargs.get("model")
                    if model is not None:
                        self._save_best_checkpoint(model)
                else:
                    self.evals_without_improvement += 1

                if (
                    early_stopping_patience > 0
                    and self.evals_without_improvement >= early_stopping_patience
                ):
                    self.early_stopped = True
                    self.stop_reason = "early_stopping_no_improvement"
                    control.should_training_stop = True
            return control

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            self.history_rows.append(
                build_training_history_entry(
                    event="train_end",
                    state=state,
                    started_at=self.started_at,
                    payload={
                        "stop_reason": self.stop_reason or "completed_budget",
                        "best_eval_loss": self.best_eval_loss,
                        "best_eval_step": self.best_eval_step,
                        "manual_stop_requested": self.manual_stop_requested,
                        "early_stopped": self.early_stopped,
                        "best_checkpoint_saved": self.best_checkpoint_saved,
                    },
                )
            )
            return control

    return TrainingControlCallback()


def generate_prediction(
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    max_source_length: int,
    max_new_tokens: int,
    model_context_window: int,
) -> str:
    import torch

    prompt_text = render_training_prompt(record)
    model_device = next(model.parameters()).device
    prompt_token_limit = min(max_source_length, model_context_window - 1)
    if prompt_token_limit <= 0:
        raise ValueError(
            f"Model context window ({model_context_window}) leaves no room for prompt tokens."
        )
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=prompt_token_limit,
    )
    inputs = {key: value.to(model_device) for key, value in inputs.items()}
    prompt_length = inputs["input_ids"].shape[1]
    available_generation_tokens = model_context_window - prompt_length
    if available_generation_tokens <= 0:
        raise ValueError(
            "Evaluation prompt exhausts the model context window after truncation. "
            "Lower max_source_length or choose a larger-context model."
        )
    effective_max_new_tokens = min(max_new_tokens, available_generation_tokens)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=effective_max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = generated[0][prompt_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def generate_predictions_batch(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    max_source_length: int,
    max_new_tokens: int,
    model_context_window: int,
) -> list[str]:
    import torch

    if not records:
        return []

    prompt_token_limit = min(max_source_length, model_context_window - 1)
    if prompt_token_limit <= 0:
        raise ValueError(
            f"Model context window ({model_context_window}) leaves no room for prompt tokens."
        )

    prompt_texts = [render_training_prompt(record) for record in records]
    model_device = next(model.parameters()).device
    original_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(
            prompt_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=prompt_token_limit,
        )
    finally:
        tokenizer.padding_side = original_padding_side

    inputs = {key: value.to(model_device) for key, value in inputs.items()}
    prompt_lengths = inputs["attention_mask"].sum(dim=1)
    available_generation_tokens = model_context_window - prompt_lengths
    effective_max_new_tokens = min(
        max_new_tokens,
        int(available_generation_tokens.min().item()),
    )
    if effective_max_new_tokens <= 0:
        raise ValueError(
            "Evaluation prompt batch exhausts the model context window after truncation. "
            "Lower max_source_length or choose a larger-context model."
        )

    input_width = inputs["input_ids"].shape[1]
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=effective_max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prediction_texts: list[str] = []
    for generated_row in generated:
        generated_tokens = generated_row[input_width:]
        prediction_texts.append(
            tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    )
    return prediction_texts


def predict_extractive_answers_batch(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    *,
    max_seq_length: int,
    max_answer_length: int,
    no_answer_threshold: float,
) -> list[dict[str, Any]]:
    import torch

    if not records:
        return []

    questions = [render_extractive_question(record) for record in records]
    contexts = [record["input_text"] for record in records]
    model_device = next(model.parameters()).device
    encodings = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        padding=True,
        max_length=max_seq_length,
        return_offsets_mapping=True,
    )
    offset_mappings = encodings.pop("offset_mapping")
    inputs = {
        key: torch.tensor(value, dtype=torch.long, device=model_device)
        for key, value in encodings.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits.detach().cpu().tolist()
    end_logits = outputs.end_logits.detach().cpu().tolist()

    predictions: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        sequence_ids = encodings.sequence_ids(index)
        input_ids = encodings["input_ids"][index]
        cls_token_id = tokenizer.cls_token_id
        cls_index = input_ids.index(cls_token_id) if cls_token_id in input_ids else 0
        found, answer_text, best_score, null_score = extract_best_qa_span(
            start_logits[index],
            end_logits[index],
            offset_mappings[index],
            sequence_ids,
            record["input_text"],
            cls_index=cls_index,
            max_answer_length=max_answer_length,
            no_answer_threshold=no_answer_threshold,
        )
        prediction_structured = build_structured_target(
            category=record["category"],
            found=found,
            normalized_answer=answer_text if found else None,
            evidence_text=answer_text if found else None,
        )
        predictions.append(
            {
                "prediction_structured": prediction_structured,
                "qa_best_span_score": best_score,
                "qa_null_score": null_score,
            }
        )
    return predictions


def build_prediction_summary(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not prediction_rows:
        return {
            "num_examples": 0,
            "num_parsed_examples": 0,
            "num_invalid_json_examples": 0,
            "parsed_json_rate": 0.0,
            "structured_metrics_scope": "parsed_examples_only",
            "found_accuracy": None,
            "normalized_answer_exact_match": None,
            "evidence_exact_match": None,
            "per_category_found_accuracy": {},
        }

    parsed_count = 0
    found_correct = 0
    normalized_correct = 0
    evidence_correct = 0
    per_category: dict[str, dict[str, int]] = {}

    for row in prediction_rows:
        if not row["parsed_json"]:
            continue
        reference = row["reference_target"]
        prediction = row["prediction_structured"]
        category = reference["category"]
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
    if parsed_count == 0:
        return {
            "num_examples": total,
            "num_parsed_examples": 0,
            "num_invalid_json_examples": total,
            "parsed_json_rate": 0.0,
            "structured_metrics_scope": "parsed_examples_only",
            "found_accuracy": None,
            "normalized_answer_exact_match": None,
            "evidence_exact_match": None,
            "per_category_found_accuracy": {},
        }
    return {
        "num_examples": total,
        "num_parsed_examples": parsed_count,
        "num_invalid_json_examples": total - parsed_count,
        "parsed_json_rate": round(parsed_count / total, 4),
        "structured_metrics_scope": "parsed_examples_only",
        "found_accuracy": round(found_correct / parsed_count, 4),
        "normalized_answer_exact_match": round(normalized_correct / parsed_count, 4),
        "evidence_exact_match": round(evidence_correct / parsed_count, 4),
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
        help="Optional max-step override for a bounded run that stops cleanly before full epochs.",
    )
    train_parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help=(
            "Stop early after this many evals without eval-loss improvement. "
            "Set to 0 to disable early stopping."
        ),
    )
    train_parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum eval-loss improvement required to reset early-stopping patience.",
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
        "--history-name",
        default="cuad_train_history.jsonl",
        help="Training log-history artifact filename under artifact-root.",
    )
    train_parser.add_argument(
        "--loss-curve-name",
        default="cuad_train_loss_curve.png",
        help="Training loss-curve PNG filename under artifact-root.",
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
        "--progress-every",
        type=int,
        default=25,
        help="Print evaluation progress after this many records. Use 0 to disable periodic progress lines.",
    )
    evaluate_parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=4,
        help="Batch size for generation during evaluation. Lower it if you hit GPU memory limits.",
    )
    evaluate_parser.add_argument(
        "--max-source-length",
        type=int,
        default=DEFAULT_MAX_SOURCE_LENGTH,
        help="Maximum prompt length after tokenization.",
    )
    evaluate_parser.add_argument(
        "--max-new-tokens",
        "--max-target-length",
        dest="max_new_tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=(
            "Generation budget for prediction JSON. `--max-target-length` is accepted "
            "as a compatibility alias during evaluation."
        ),
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

    train_extractive_parser = subparsers.add_parser(
        "train-extractive",
        help="Train a simple extractive QA baseline on preprocessed CUAD data.",
    )
    add_shared_paths(train_extractive_parser)
    train_extractive_parser.add_argument(
        "--model-name",
        default=DEFAULT_EXTRACTIVE_QA_MODEL_NAME,
        help="Extractive QA baseline model to fine-tune.",
    )
    train_extractive_parser.add_argument(
        "--preprocessed-name",
        default=CUAD_PREPROCESSED_DIR_NAME,
        help="Derived preprocessing directory name under data-root.",
    )
    train_extractive_parser.add_argument(
        "--train-split",
        default="train",
        help="Training split name after preprocessing.",
    )
    train_extractive_parser.add_argument(
        "--validation-split",
        default="validation",
        help="Validation split name after preprocessing.",
    )
    train_extractive_parser.add_argument(
        "--output-name",
        default=DEFAULT_EXTRACTIVE_QA_CHECKPOINT_NAME,
        help="Checkpoint folder name under checkpoint-root.",
    )
    train_extractive_parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_EXTRACTIVE_MAX_SEQ_LENGTH,
        help="Maximum tokenized question+context length for extractive QA.",
    )
    train_extractive_parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=8,
        help="Per-device training batch size.",
    )
    train_extractive_parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=8,
        help="Per-device evaluation batch size.",
    )
    train_extractive_parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps for the Trainer loop.",
    )
    train_extractive_parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
        help="Learning rate for AdamW.",
    )
    train_extractive_parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=2.0,
        help="Number of training epochs when max steps is not specified.",
    )
    train_extractive_parser.add_argument(
        "--max-train-steps",
        type=int,
        default=-1,
        help="Optional max-step override for a bounded baseline run.",
    )
    train_extractive_parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop early after this many evals without eval-loss improvement. Set to 0 to disable early stopping.",
    )
    train_extractive_parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum eval-loss improvement required to reset early-stopping patience.",
    )
    train_extractive_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a tiny sample and tiny step budget.",
    )
    train_extractive_parser.add_argument(
        "--smoke-max-train-records",
        type=int,
        default=DEFAULT_SMOKE_MAX_TRAIN_RECORDS,
        help="Training-record cap used with --smoke.",
    )
    train_extractive_parser.add_argument(
        "--smoke-max-validation-records",
        type=int,
        default=DEFAULT_SMOKE_MAX_VALIDATION_RECORDS,
        help="Validation-record cap used with --smoke.",
    )
    train_extractive_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and print a training summary without downloading or training a model.",
    )
    train_extractive_parser.add_argument(
        "--summary-name",
        default="cuad_extractive_train_summary.json",
        help="Training summary artifact filename under artifact-root.",
    )
    train_extractive_parser.add_argument(
        "--history-name",
        default="cuad_extractive_train_history.jsonl",
        help="Training log-history artifact filename under artifact-root.",
    )
    train_extractive_parser.add_argument(
        "--loss-curve-name",
        default="cuad_extractive_train_loss_curve.png",
        help="Training loss-curve PNG filename under artifact-root.",
    )
    train_extractive_parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Deterministic sampling seed for smoke mode.",
    )
    train_extractive_parser.set_defaults(func=run_train_extractive)

    evaluate_extractive_parser = subparsers.add_parser(
        "evaluate-extractive",
        help="Evaluate the extractive QA baseline on a preprocessed CUAD split.",
    )
    add_shared_paths(evaluate_extractive_parser)
    evaluate_extractive_parser.add_argument(
        "--model-name",
        default=DEFAULT_EXTRACTIVE_QA_MODEL_NAME,
        help="Extractive QA baseline model to load when no checkpoint is provided.",
    )
    evaluate_extractive_parser.add_argument(
        "--preprocessed-name",
        default=CUAD_PREPROCESSED_DIR_NAME,
        help="Derived preprocessing directory name under data-root.",
    )
    evaluate_extractive_parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate.",
    )
    evaluate_extractive_parser.add_argument(
        "--checkpoint-name",
        default=DEFAULT_EXTRACTIVE_QA_CHECKPOINT_NAME,
        help="Checkpoint directory name under checkpoint-root.",
    )
    evaluate_extractive_parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional explicit checkpoint path.",
    )
    evaluate_extractive_parser.add_argument(
        "--prediction-name",
        default="cuad_extractive_eval_predictions.jsonl",
        help="Prediction artifact filename under artifact-root.",
    )
    evaluate_extractive_parser.add_argument(
        "--metrics-name",
        default="cuad_extractive_eval_metrics.json",
        help="Summary metrics artifact filename under artifact-root.",
    )
    evaluate_extractive_parser.add_argument(
        "--sample-prediction-name",
        default="cuad_extractive_eval_prediction_samples.jsonl",
        help="Sample prediction artifact filename under artifact-root.",
    )
    evaluate_extractive_parser.add_argument(
        "--num-sample-predictions",
        type=int,
        default=25,
        help="Number of prediction rows to export in the sample artifact.",
    )
    evaluate_extractive_parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print evaluation progress after this many records. Use 0 to disable periodic progress lines.",
    )
    evaluate_extractive_parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=16,
        help="Batch size for extractive QA inference. Lower it if you hit GPU memory limits.",
    )
    evaluate_extractive_parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_EXTRACTIVE_MAX_SEQ_LENGTH,
        help="Maximum tokenized question+context length for extractive QA.",
    )
    evaluate_extractive_parser.add_argument(
        "--max-answer-length",
        type=int,
        default=DEFAULT_EXTRACTIVE_MAX_ANSWER_LENGTH,
        help="Maximum answer span length in tokens during inference.",
    )
    evaluate_extractive_parser.add_argument(
        "--no-answer-threshold",
        type=float,
        default=DEFAULT_EXTRACTIVE_NO_ANSWER_THRESHOLD,
        help="Extra score margin the best span must beat the null score by to predict found=true.",
    )
    evaluate_extractive_parser.add_argument(
        "--smoke",
        action="store_true",
        help="Evaluate only a tiny subset.",
    )
    evaluate_extractive_parser.add_argument(
        "--smoke-max-eval-records",
        type=int,
        default=DEFAULT_SMOKE_MAX_EVAL_RECORDS,
        help="Evaluation-record cap used with --smoke.",
    )
    evaluate_extractive_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write reference-copy predictions without loading a model.",
    )
    evaluate_extractive_parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Deterministic sampling seed for smoke mode.",
    )
    evaluate_extractive_parser.set_defaults(func=run_evaluate_extractive)

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
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
    }
    if args.dry_run:
        summary["summary_artifact"] = str(artifact_root / args.summary_name)
        summary["history_artifact"] = str(artifact_root / args.history_name)
        summary["loss_curve_artifact"] = str(artifact_root / args.loss_curve_name)
        write_run_summary(artifact_root, args.summary_name, summary)
        print("Training dry run complete.")
        print(json.dumps(summary, indent=2))
        return 0

    configure_hf_cache(roots["cache_root"])
    torch, _, _, Trainer, TrainingArguments, TrainerCallback = import_training_stack(
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
    training_budgets = resolve_training_token_budgets(
        tokenizer,
        model,
        requested_max_source_length=args.max_source_length,
        requested_max_target_length=args.max_target_length,
    )

    train_features = build_generation_features(
        train_records,
        tokenizer=tokenizer,
        max_source_length=training_budgets["max_source_length"],
        max_target_length=training_budgets["max_target_length"],
        model_context_window=training_budgets["context_window"],
    )
    validation_features = build_generation_features(
        validation_records,
        tokenizer=tokenizer,
        max_source_length=training_budgets["max_source_length"],
        max_target_length=training_budgets["max_target_length"],
        model_context_window=training_budgets["context_window"],
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
        label_names=["labels"],
    )
    safe_stop_state, previous_sigint_handler = install_safe_stop_handler()
    training_control = build_training_control_callback(
        TrainerCallback,
        safe_stop_state=safe_stop_state,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        checkpoint_dir=checkpoint_dir,
        tokenizer=tokenizer,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ListDataset(train_features),
        eval_dataset=ListDataset(validation_features) if validation_features else None,
        data_collator=SupervisedDataCollator(tokenizer.pad_token_id),
        callbacks=[training_control],
    )
    train_result = None
    stop_reason = "completed_budget"
    interrupted = False
    try:
        train_result = trainer.train()
    except KeyboardInterrupt:
        interrupted = True
        stop_reason = training_control.stop_reason or "keyboard_interrupt"
    finally:
        restore_safe_stop_handler(previous_sigint_handler)

    if not interrupted and not training_control.best_checkpoint_saved:
        trainer.save_model()
        tokenizer.save_pretrained(checkpoint_dir)

    completed_steps = int(getattr(trainer.state, "global_step", 0) or 0)
    completed_epoch = getattr(trainer.state, "epoch", None)
    world_size = max(1, int(getattr(training_args, "world_size", 1)))
    estimated_batches_seen = completed_steps * args.gradient_accumulation_steps * world_size
    estimated_examples_seen = estimated_batches_seen * args.per_device_train_batch_size
    history_artifact_path = write_history_artifact(
        artifact_root,
        args.history_name,
        training_control.history_rows,
    )
    loss_curve_artifact_path = write_loss_curve_artifact(
        artifact_root,
        args.loss_curve_name,
        training_control.history_rows,
    )
    best_eval_loss = (
        training_control.best_eval_loss
        if training_control.best_eval_loss is not None
        else resolve_best_eval_loss(training_control.history_rows)
    )
    if training_control.stop_reason is not None:
        stop_reason = training_control.stop_reason
    summary.update(
        {
            "checkpoint_dir": str(checkpoint_dir),
            "model_context_window": training_budgets["context_window"],
            "requested_max_source_length": args.max_source_length,
            "requested_max_target_length": args.max_target_length,
            "max_train_steps": max_train_steps,
            "max_source_length": training_budgets["max_source_length"],
            "max_target_length": training_budgets["max_target_length"],
            "lora": lora_summary,
            "train_feature_count": len(train_features),
            "validation_feature_count": len(validation_features),
            "history_artifact": str(history_artifact_path),
            "loss_curve_artifact": (
                str(loss_curve_artifact_path) if loss_curve_artifact_path is not None else None
            ),
            "completed_steps": completed_steps,
            "effective_epoch_reached": completed_epoch,
            "estimated_batches_seen": estimated_batches_seen,
            "estimated_examples_seen": estimated_examples_seen,
            "wall_clock_runtime_seconds": round(time.monotonic() - training_control.started_at, 4),
            "train_runtime_metrics": make_json_safe(train_result.metrics) if train_result else None,
            "completed_eval_events": count_completed_eval_events(training_control.history_rows),
            "early_stopping_enabled": args.early_stopping_patience > 0 and bool(validation_features),
            "best_eval_loss": best_eval_loss,
            "best_eval_step": training_control.best_eval_step,
            "best_checkpoint_dir": str(checkpoint_dir) if training_control.best_checkpoint_saved else None,
            "best_checkpoint_saved": training_control.best_checkpoint_saved,
            "saved_checkpoint_selection": (
                "best_eval" if training_control.best_checkpoint_saved else "final_state"
            ),
            "manual_stop_requested": training_control.manual_stop_requested,
            "manual_stop_signal_count": safe_stop_state["signal_count"],
            "manual_stop_requested_at_step": safe_stop_state["requested_at_step"],
            "early_stopped": training_control.early_stopped,
            "stop_reason": stop_reason,
            "checkpoint_saved": not interrupted,
        }
    )
    summary["summary_artifact"] = str(artifact_root / args.summary_name)
    write_run_summary(artifact_root, args.summary_name, summary)
    if interrupted:
        print("Training interrupted before a graceful stop completed.")
    else:
        print("Training complete.")
    print(json.dumps(summary, indent=2))
    return 130 if interrupted else 0


def run_train_extractive(args: argparse.Namespace) -> int:
    roots = resolve_roots(args)
    ensure_runtime_roots(roots)
    preprocessed_dir = Path(roots["data_root"]) / args.preprocessed_name
    train_records = load_preprocessed_split(preprocessed_dir, args.train_split)
    validation_records = load_preprocessed_split(preprocessed_dir, args.validation_split)
    artifact_root = ensure_directory(Path(roots["artifact_root"]))

    if args.smoke:
        train_records = sample_records(train_records, args.smoke_max_train_records, args.split_seed)
        validation_records = sample_records(
            validation_records,
            args.smoke_max_validation_records,
            args.split_seed,
        )

    summary: dict[str, Any] = {
        "baseline_type": "extractive_qa",
        "preprocessed_dir": str(preprocessed_dir),
        "model_name": args.model_name,
        "smoke": args.smoke,
        "dry_run": args.dry_run,
        "train_split": args.train_split,
        "validation_split": args.validation_split,
        "train_summary": summarize_split(train_records),
        "validation_summary": summarize_split(validation_records),
        "question_preview": render_extractive_question(train_records[0]) if train_records else None,
        "context_preview": truncate_preview_text(train_records[0]["input_text"]) if train_records else None,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "max_seq_length": args.max_seq_length,
    }
    if args.dry_run:
        summary["summary_artifact"] = str(artifact_root / args.summary_name)
        summary["history_artifact"] = str(artifact_root / args.history_name)
        summary["loss_curve_artifact"] = str(artifact_root / args.loss_curve_name)
        write_run_summary(artifact_root, args.summary_name, summary)
        print("Extractive QA training dry run complete.")
        print(json.dumps(summary, indent=2))
        return 0

    configure_hf_cache(roots["cache_root"])
    (
        torch,
        _,
        _,
        Trainer,
        TrainingArguments,
        TrainerCallback,
        DataCollatorWithPadding,
    ) = import_extractive_qa_stack(include_trainer=True)
    checkpoint_dir = ensure_directory(Path(roots["checkpoint_root"]) / args.output_name)
    model, tokenizer, model_source = load_extractive_qa_model(
        model_name=args.model_name,
        cache_root=roots["cache_root"],
    )
    model_context_window = resolve_model_context_window(tokenizer, model)
    max_seq_length = min(args.max_seq_length, model_context_window)

    train_features, train_alignment_stats = build_extractive_qa_features(
        train_records,
        tokenizer,
        max_seq_length=max_seq_length,
    )
    validation_features, validation_alignment_stats = build_extractive_qa_features(
        validation_records,
        tokenizer,
        max_seq_length=max_seq_length,
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
        label_names=["start_positions", "end_positions"],
    )
    safe_stop_state, previous_sigint_handler = install_safe_stop_handler()
    training_control = build_training_control_callback(
        TrainerCallback,
        safe_stop_state=safe_stop_state,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        checkpoint_dir=checkpoint_dir,
        tokenizer=tokenizer,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ListDataset(train_features),
        eval_dataset=ListDataset(validation_features) if validation_features else None,
        data_collator=data_collator,
        callbacks=[training_control],
    )
    train_result = None
    stop_reason = "completed_budget"
    interrupted = False
    try:
        train_result = trainer.train()
    except KeyboardInterrupt:
        interrupted = True
        stop_reason = training_control.stop_reason or "keyboard_interrupt"
    finally:
        restore_safe_stop_handler(previous_sigint_handler)

    if not interrupted and not training_control.best_checkpoint_saved:
        trainer.save_model()
        tokenizer.save_pretrained(checkpoint_dir)

    completed_steps = int(getattr(trainer.state, "global_step", 0) or 0)
    completed_epoch = getattr(trainer.state, "epoch", None)
    world_size = max(1, int(getattr(training_args, "world_size", 1)))
    estimated_batches_seen = completed_steps * args.gradient_accumulation_steps * world_size
    estimated_examples_seen = estimated_batches_seen * args.per_device_train_batch_size
    history_artifact_path = write_history_artifact(
        artifact_root,
        args.history_name,
        training_control.history_rows,
    )
    loss_curve_artifact_path = write_loss_curve_artifact(
        artifact_root,
        args.loss_curve_name,
        training_control.history_rows,
    )
    best_eval_loss = (
        training_control.best_eval_loss
        if training_control.best_eval_loss is not None
        else resolve_best_eval_loss(training_control.history_rows)
    )
    if training_control.stop_reason is not None:
        stop_reason = training_control.stop_reason

    summary.update(
        {
            "model_source": model_source,
            "checkpoint_dir": str(checkpoint_dir),
            "model_context_window": model_context_window,
            "effective_max_seq_length": max_seq_length,
            "max_train_steps": max_train_steps,
            "train_feature_count": len(train_features),
            "validation_feature_count": len(validation_features),
            "train_alignment_stats": train_alignment_stats,
            "validation_alignment_stats": validation_alignment_stats,
            "history_artifact": str(history_artifact_path),
            "loss_curve_artifact": (
                str(loss_curve_artifact_path) if loss_curve_artifact_path is not None else None
            ),
            "completed_steps": completed_steps,
            "effective_epoch_reached": completed_epoch,
            "estimated_batches_seen": estimated_batches_seen,
            "estimated_examples_seen": estimated_examples_seen,
            "wall_clock_runtime_seconds": round(time.monotonic() - training_control.started_at, 4),
            "train_runtime_metrics": make_json_safe(train_result.metrics) if train_result else None,
            "completed_eval_events": count_completed_eval_events(training_control.history_rows),
            "early_stopping_enabled": args.early_stopping_patience > 0 and bool(validation_features),
            "best_eval_loss": best_eval_loss,
            "best_eval_step": training_control.best_eval_step,
            "best_checkpoint_dir": str(checkpoint_dir) if training_control.best_checkpoint_saved else None,
            "best_checkpoint_saved": training_control.best_checkpoint_saved,
            "saved_checkpoint_selection": (
                "best_eval" if training_control.best_checkpoint_saved else "final_state"
            ),
            "manual_stop_requested": training_control.manual_stop_requested,
            "manual_stop_signal_count": safe_stop_state["signal_count"],
            "manual_stop_requested_at_step": safe_stop_state["requested_at_step"],
            "early_stopped": training_control.early_stopped,
            "stop_reason": stop_reason,
            "checkpoint_saved": not interrupted,
        }
    )
    summary["summary_artifact"] = str(artifact_root / args.summary_name)
    write_run_summary(artifact_root, args.summary_name, summary)
    if interrupted:
        print("Extractive QA training interrupted before a graceful stop completed.")
    else:
        print("Extractive QA training complete.")
    print(json.dumps(summary, indent=2))
    return 130 if interrupted else 0


def run_evaluate_extractive(args: argparse.Namespace) -> int:
    roots = resolve_roots(args)
    ensure_runtime_roots(roots)
    preprocessed_dir = Path(roots["data_root"]) / args.preprocessed_name
    records = load_preprocessed_split(preprocessed_dir, args.split)
    if args.smoke:
        records = sample_records(records, args.smoke_max_eval_records, args.split_seed)

    artifact_root = ensure_directory(Path(roots["artifact_root"]))
    prediction_path = artifact_root / args.prediction_name
    metrics_path = artifact_root / args.metrics_name
    sample_prediction_path = artifact_root / args.sample_prediction_name
    prediction_rows: list[dict[str, Any]] = []
    total_records = len(records)
    evaluation_started_at = time.monotonic()

    if args.dry_run:
        model_source = "reference_copy_dry_run"
        print(f"Extractive QA evaluation starting on {total_records} records using reference-copy dry run.", flush=True)
        with prediction_path.open("w", encoding="utf-8") as prediction_handle:
            for index, record in enumerate(records, start=1):
                reference_target = json.loads(record["target_json"])
                row = {
                    "contract_id": record["contract_id"],
                    "chunk_id": record["chunk_id"],
                    "category": record["category"],
                    "source_id": record.get("source_id"),
                    "prediction_text": reference_target["evidence_text"],
                    "prediction_structured": reference_target,
                    "reference_target": reference_target,
                    "parsed_json": True,
                    "parse_error": None,
                    "qa_best_span_score": None,
                    "qa_null_score": None,
                }
                prediction_rows.append(row)
                append_jsonl_record(prediction_handle, row)
                if (
                    total_records > 0
                    and (
                        index == 1
                        or index == total_records
                        or (args.progress_every > 0 and index % args.progress_every == 0)
                    )
                ):
                    elapsed_seconds = max(time.monotonic() - evaluation_started_at, 1e-9)
                    records_per_second = index / elapsed_seconds
                    remaining_records = total_records - index
                    eta_minutes = (remaining_records / records_per_second) / 60.0
                    print(
                        f"Evaluated {index}/{total_records} extractive QA records "
                        f"({records_per_second:.2f} records/s, eta_minutes={eta_minutes:.1f}).",
                        flush=True,
                    )
    else:
        configure_hf_cache(roots["cache_root"])
        checkpoint_source = resolve_checkpoint_source(args, roots)
        model, tokenizer, model_source = load_extractive_qa_model(
            model_name=args.model_name,
            checkpoint_source=checkpoint_source,
            cache_root=roots["cache_root"],
        )
        model_context_window = resolve_model_context_window(tokenizer, model)
        max_seq_length = min(args.max_seq_length, model_context_window)
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        print(
            f"Extractive QA evaluation starting on {total_records} records with device={device} "
            f"from model_source={model_source}.",
            flush=True,
        )

        batch_size = max(1, args.per_device_eval_batch_size)
        with prediction_path.open("w", encoding="utf-8") as prediction_handle:
            for batch_start in range(0, total_records, batch_size):
                batch_records = records[batch_start : batch_start + batch_size]
                batch_predictions = predict_extractive_answers_batch(
                    model=model,
                    tokenizer=tokenizer,
                    records=batch_records,
                    max_seq_length=max_seq_length,
                    max_answer_length=args.max_answer_length,
                    no_answer_threshold=args.no_answer_threshold,
                )
                for record, prediction in zip(batch_records, batch_predictions):
                    reference_target = json.loads(record["target_json"])
                    prediction_text = prediction["prediction_structured"]["evidence_text"]
                    row = {
                        "contract_id": record["contract_id"],
                        "chunk_id": record["chunk_id"],
                        "category": record["category"],
                        "source_id": record.get("source_id"),
                        "prediction_text": prediction_text,
                        "prediction_structured": prediction["prediction_structured"],
                        "reference_target": reference_target,
                        "parsed_json": True,
                        "parse_error": None,
                        "qa_best_span_score": prediction["qa_best_span_score"],
                        "qa_null_score": prediction["qa_null_score"],
                    }
                    prediction_rows.append(row)
                    append_jsonl_record(prediction_handle, row)

                index = len(prediction_rows)
                if (
                    total_records > 0
                    and (
                        index <= batch_size
                        or index == total_records
                        or (args.progress_every > 0 and index % args.progress_every == 0)
                    )
                ):
                    elapsed_seconds = max(time.monotonic() - evaluation_started_at, 1e-9)
                    records_per_second = index / elapsed_seconds
                    remaining_records = total_records - index
                    eta_minutes = (remaining_records / records_per_second) / 60.0
                    print(
                        f"Evaluated {index}/{total_records} extractive QA records with batch_size={batch_size} "
                        f"({records_per_second:.2f} records/s, eta_minutes={eta_minutes:.1f}).",
                        flush=True,
                    )

    write_jsonl(
        prediction_rows[: max(0, args.num_sample_predictions)],
        sample_prediction_path,
    )
    summary = {
        "baseline_type": "extractive_qa",
        "preprocessed_dir": str(preprocessed_dir),
        "evaluated_split": args.split,
        "smoke": args.smoke,
        "dry_run": args.dry_run,
        "model_source": model_source,
        "model_context_window": (model_context_window if not args.dry_run else None),
        "max_seq_length": args.max_seq_length,
        "effective_max_seq_length": (max_seq_length if not args.dry_run else None),
        "max_answer_length": args.max_answer_length if not args.dry_run else None,
        "no_answer_threshold": args.no_answer_threshold if not args.dry_run else None,
        "per_device_eval_batch_size": args.per_device_eval_batch_size if not args.dry_run else None,
        "prediction_artifact": str(prediction_path),
        "sample_prediction_artifact": str(sample_prediction_path),
        "num_sample_predictions": min(len(prediction_rows), max(0, args.num_sample_predictions)),
        "metrics_artifact": str(metrics_path),
        "evaluation_runtime_seconds": round(time.monotonic() - evaluation_started_at, 4),
        "comparison_notes": (
            "found_accuracy is directly comparable to the structured-generation path. "
            "normalized_answer_exact_match and evidence_exact_match compare the single extracted span "
            "against the structured-generation references."
        ),
        "metrics": build_prediction_summary(prediction_rows),
    }
    write_json(summary, metrics_path)
    print("Extractive QA evaluation complete.")
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
    total_records = len(records)
    evaluation_started_at = time.monotonic()

    if args.dry_run:
        model_source = "reference_copy_dry_run"
        print(f"Evaluation starting on {total_records} records using reference-copy dry run.", flush=True)
        with prediction_path.open("w", encoding="utf-8") as prediction_handle:
            for index, record in enumerate(records, start=1):
                reference_target = json.loads(record["target_json"])
                prediction_text = record["target_json"]
                prediction_structured, parsed_json, parse_error = parse_prediction_text(
                    prediction_text=prediction_text,
                    fallback_category=record["category"],
                )
                row = {
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
                prediction_rows.append(row)
                append_jsonl_record(prediction_handle, row)
                if (
                    total_records > 0
                    and (
                        index == 1
                        or index == total_records
                        or (args.progress_every > 0 and index % args.progress_every == 0)
                    )
                ):
                    elapsed_seconds = max(time.monotonic() - evaluation_started_at, 1e-9)
                    records_per_second = index / elapsed_seconds
                    remaining_records = total_records - index
                    eta_minutes = (remaining_records / records_per_second) / 60.0
                    print(
                        f"Evaluated {index}/{total_records} records "
                        f"({records_per_second:.2f} records/s, eta_minutes={eta_minutes:.1f}).",
                        flush=True,
                    )
    else:
        configure_hf_cache(roots["cache_root"])
        checkpoint_source = resolve_checkpoint_source(args, roots)
        model, tokenizer, model_source = load_model_for_inference(
            model_name=model_name,
            checkpoint_source=checkpoint_source,
            cache_root=roots["cache_root"],
        )
        eval_budgets = resolve_eval_token_budgets(
            tokenizer,
            model,
            requested_max_source_length=args.max_source_length,
            requested_max_new_tokens=args.max_new_tokens,
        )
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        print(
            f"Evaluation starting on {total_records} records with device={device} "
            f"from model_source={model_source}.",
            flush=True,
        )

        batch_size = max(1, args.per_device_eval_batch_size)
        with prediction_path.open("w", encoding="utf-8") as prediction_handle:
            for batch_start in range(0, total_records, batch_size):
                batch_records = records[batch_start : batch_start + batch_size]
                prediction_texts = generate_predictions_batch(
                    model=model,
                    tokenizer=tokenizer,
                    records=batch_records,
                    max_source_length=eval_budgets["max_source_length"],
                    max_new_tokens=eval_budgets["max_new_tokens"],
                    model_context_window=eval_budgets["context_window"],
                )
                for record, prediction_text in zip(batch_records, prediction_texts):
                    prediction_structured, parsed_json, parse_error = parse_prediction_text(
                        prediction_text=prediction_text,
                        fallback_category=record["category"],
                    )
                    row = {
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
                    prediction_rows.append(row)
                    append_jsonl_record(prediction_handle, row)

                index = len(prediction_rows)
                if (
                    total_records > 0
                    and (
                        index <= batch_size
                        or index == total_records
                        or (args.progress_every > 0 and index % args.progress_every == 0)
                    )
                ):
                    elapsed_seconds = max(time.monotonic() - evaluation_started_at, 1e-9)
                    records_per_second = index / elapsed_seconds
                    remaining_records = total_records - index
                    eta_minutes = (remaining_records / records_per_second) / 60.0
                    print(
                        f"Evaluated {index}/{total_records} records "
                        f"with batch_size={batch_size} "
                        f"({records_per_second:.2f} records/s, eta_minutes={eta_minutes:.1f}).",
                        flush=True,
                    )

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
        "model_context_window": eval_budgets["context_window"] if not args.dry_run else None,
        "per_device_eval_batch_size": args.per_device_eval_batch_size if not args.dry_run else None,
        "requested_max_source_length": args.max_source_length,
        "requested_max_new_tokens": args.max_new_tokens,
        "effective_max_source_length": (
            eval_budgets["max_source_length"] if not args.dry_run else None
        ),
        "prediction_artifact": str(prediction_path),
        "sample_prediction_artifact": str(sample_prediction_path),
        "num_sample_predictions": min(len(prediction_rows), max(0, args.num_sample_predictions)),
        "metrics_artifact": str(metrics_path),
        "evaluation_runtime_seconds": round(time.monotonic() - evaluation_started_at, 4),
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
