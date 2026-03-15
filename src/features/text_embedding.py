"""Helpers for computing text embeddings with DistilClinicalBERT and ClinicalBERT."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from transformers import AutoModel, AutoTokenizer


def load_distilclinicalbert(
    model_name: str = "nlpie/distil-clinicalbert",
    device: Optional[str] = None,
) -> Tuple[AutoTokenizer, AutoModel]:
    """Load DistilClinicalBERT tokenizer and model; optional device (e.g. 'cuda')."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    if device is not None:
        # Preserve simple, explicit device handling; callers are
        # responsible for passing a valid device string.
        model = model.to(device)
    return tokenizer, model


def load_clinicalbert(
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    device: Optional[str] = None,
) -> Tuple[AutoTokenizer, AutoModel]:
    """Load Bio_ClinicalBERT tokenizer and model (non-distilled variant)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    if device is not None:
        model = model.to(device)
    return tokenizer, model


def tokenize_and_block(text: str, tokenizer, block_size: int = 510) -> List[List[int]]:
    """Tokenize text and split into fixed-length blocks (no truncation); pad last block."""
    tokenized = tokenizer(text, padding=False, truncation=False, return_tensors="pt")
    input_ids = tokenized["input_ids"].squeeze().tolist()
    total_length = len(input_ids)

    blocks: List[List[int]] = []
    for start in range(0, total_length, block_size):
        end = min(start + block_size, total_length)
        block = input_ids[start:end]
        if len(block) < block_size:
            block.extend([tokenizer.pad_token_id] * (block_size - len(block)))
        blocks.append(block)

    return blocks


def add_special_tokens_to_blocks(blocks: Sequence[Sequence[int]], tokenizer) -> List[List[int]]:
    """Wrap each block as [CLS] + block + [SEP]."""
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    wrapped: List[List[int]] = []
    for block in blocks:
        wrapped.append([cls_id] + list(block) + [sep_id])
    return wrapped


def extract_block_embeddings(blocks: Sequence[Sequence[int]], model, device: Optional[str] = None) -> np.ndarray:
    """Run model over each block, return (num_blocks, seq_len, hidden_size) last hidden states."""
    import torch

    all_embeddings: List[np.ndarray] = []
    for block in blocks:
        block_tensor = torch.tensor([block])
        if device is not None:
            block_tensor = block_tensor.to(device)
        with torch.no_grad():
            outputs = model(block_tensor)
            hidden = outputs.last_hidden_state  # [batch, seq_len, hidden]
            all_embeddings.append(hidden.squeeze().cpu().numpy())
    return np.array(all_embeddings)


def reduce_block_embeddings_to_document_vector(block_embeddings: np.ndarray) -> np.ndarray:
    """Mean over tokens per block, then mean over blocks -> single (hidden_size,) vector."""
    chunk_vectors = np.mean(block_embeddings, axis=1)
    doc_vector = np.mean(chunk_vectors, axis=0)
    return doc_vector


def encode_document(
    text: str,
    tokenizer,
    model,
    block_size: int = 510,
    device: Optional[str] = None,
) -> np.ndarray:
    """End-to-end: tokenize+block -> add CLS/SEP -> extract embeddings -> mean to one vector."""
    blocks = tokenize_and_block(text, tokenizer, block_size=block_size)
    blocks = add_special_tokens_to_blocks(blocks, tokenizer)
    block_embeddings = extract_block_embeddings(blocks, model, device=device)
    return reduce_block_embeddings_to_document_vector(block_embeddings)


def encode_document_clinicalbert_truncated(
    text: str,
    tokenizer,
    model,
    *,
    max_length: int = 512,
    device: Optional[str] = None,
) -> np.ndarray:
    """ClinicalBERT truncated: first max_length tokens, one forward pass, mean over tokens -> doc vector."""
    import torch

    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", None)

    if device is not None:
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [batch, seq_len, hidden]

    doc_vec = hidden.mean(dim=1).squeeze(0).cpu().numpy()
    return doc_vec


def encode_document_clinicalbert_sliding_window(
    text: str,
    tokenizer,
    model,
    *,
    window_size: int = 256,
    stride: int = 128,
    device: Optional[str] = None,
) -> np.ndarray:
    """ClinicalBERT sliding window: overlapping token windows, [CLS]/[SEP], mean over tokens then windows."""
    import torch
    encoded = tokenizer(text, padding=False, truncation=False, return_tensors="pt")
    input_ids = encoded["input_ids"].squeeze().tolist()
    if not input_ids:
        raise ValueError("Cannot encode empty text with ClinicalBERT.")

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    windows: List[List[int]] = []
    effective_window = max(window_size - 2, 1)
    n_tokens = len(input_ids)

    start = 0
    while start < n_tokens:
        end = min(start + effective_window, n_tokens)
        chunk = input_ids[start:end]
        if len(chunk) < effective_window:
            chunk = chunk + [pad_id] * (effective_window - len(chunk))
        window_ids = [cls_id] + chunk + [sep_id]
        windows.append(window_ids)
        start += stride

    window_tensor = torch.tensor(windows)
    attention_mask = (window_tensor != pad_id).long()

    if device is not None:
        window_tensor = window_tensor.to(device)
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=window_tensor, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [num_windows, seq_len, hidden]

    window_vecs = hidden.mean(dim=1)
    doc_vec = window_vecs.mean(dim=0).cpu().numpy()
    return doc_vec


