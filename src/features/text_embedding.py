"""
Text embedding utilities based on BioClinical DistilBERT (or similar models).

This module encapsulates logic that previously lived across several
notebooks (P4, CP4, CP4-1) so that text embeddings can be computed in a
consistent and reusable way.

Only low-risk, behavior-preserving helpers are implemented here as part
of the pilot migration. The higher-level orchestration remains in the
notebooks for now.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from transformers import AutoModel, AutoTokenizer


def load_distilclinicalbert(
    model_name: str = "nlpie/distil-clinicalbert",
    device: Optional[str] = None,
) -> Tuple[AutoTokenizer, AutoModel]:
    """
    Load the DistilClinicalBERT tokenizer and model.

    This helper mirrors the pattern used in the notebooks and keeps
    model selection and device handling explicit.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier. Defaults to the value used in the
        notebooks.
    device:
        Optional device specifier (e.g. ``\"cpu\"``, ``\"cuda\"``). If
        provided, the model will be moved to that device. No additional
        device management is performed.

    Returns
    -------
    tokenizer, model
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    if device is not None:
        # Preserve simple, explicit device handling; callers are
        # responsible for passing a valid device string.
        model = model.to(device)
    return tokenizer, model


def tokenize_and_block(text: str, tokenizer, block_size: int = 510) -> List[List[int]]:
    """
    Tokenize a single document and split it into fixed-length blocks.

    This is a direct extraction of the `tokenize_and_block` helper used
    in the text-embedding notebooks. The behavior is intentionally kept
    identical:

    - no truncation is applied at the tokenizer level
    - the token ids are split into contiguous blocks of length
      ``block_size``
    - the final block is padded with ``tokenizer.pad_token_id`` if it is
      shorter than ``block_size``

    Parameters
    ----------
    text:
        Raw input text to tokenize.
    tokenizer:
        A HuggingFace tokenizer with a ``pad_token_id`` attribute.
    block_size:
        The desired length of each block, excluding special tokens.

    Returns
    -------
    list of list of int
        A list of token-id blocks, each of length ``block_size``.
    """

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
    """
    Add CLS and SEP tokens around each block of token ids.

    This helper is equivalent to the notebooks' `add_special_tokens_to_blocks`:

    - each block is wrapped as: ``[CLS] + block + [SEP]``

    Parameters
    ----------
    blocks:
        Iterable of token-id sequences.
    tokenizer:
        Tokenizer providing ``cls_token_id`` and ``sep_token_id``.

    Returns
    -------
    list of list of int
        New list of blocks with special tokens added.
    """

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    wrapped: List[List[int]] = []
    for block in blocks:
        # Copy into a new list to avoid mutating caller data.
        wrapped.append([cls_id] + list(block) + [sep_id])
    return wrapped


def extract_block_embeddings(blocks: Sequence[Sequence[int]], model, device: Optional[str] = None) -> np.ndarray:
    """
    Run DistilBERT over each block and collect last hidden states.

    This is the modular equivalent of the notebooks' `extract_features_from_blocks`
    function:

    - for each block, create a batch of size 1
    - run the model under ``torch.no_grad()``
    - append ``outputs.last_hidden_state.squeeze().numpy()`` to a list
    - finally return ``np.array(all_embeddings)``

    Parameters
    ----------
    blocks:
        Sequence of token-id sequences (usually output of
        :func:`tokenize_and_block` followed by
        :func:`add_special_tokens_to_blocks`).
    model:
        HuggingFace model with a ``last_hidden_state`` output.
    device:
        Optional device string; if provided, block tensors are moved to
        that device before the forward pass. The model is assumed to
        already reside on the same device if needed.

    Returns
    -------
    np.ndarray
        A NumPy array of shape approximately
        ``(num_blocks, seq_len, hidden_size)``.
    """

    # Local import to avoid forcing torch as a hard dependency of the
    # module in contexts where it is not needed.
    import torch

    all_embeddings: List[np.ndarray] = []
    for block in blocks:
        block_tensor = torch.tensor([block])  # add batch dimension
        if device is not None:
            block_tensor = block_tensor.to(device)
        with torch.no_grad():
            outputs = model(block_tensor)
            hidden = outputs.last_hidden_state  # [batch, seq_len, hidden]
            all_embeddings.append(hidden.squeeze().cpu().numpy())
    return np.array(all_embeddings)


def reduce_block_embeddings_to_document_vector(block_embeddings: np.ndarray) -> np.ndarray:
    """
    Reduce a stack of block embeddings to a single document vector.

    This function mirrors the original `reduce_to_768` implementation:

    - first average over the token dimension for each block
    - then average across blocks to obtain a single 768-dim vector

    Parameters
    ----------
    block_embeddings:
        A NumPy array of shape approximately
        ``(num_blocks, seq_len, hidden_size)``. The function assumes the
        final dimension is the embedding dimension.

    Returns
    -------
    np.ndarray
        A one-dimensional vector of length ``hidden_size``.
    """

    # Step 1: mean over tokens: (num_blocks, seq_len, hidden) -> (num_blocks, hidden)
    chunk_vectors = np.mean(block_embeddings, axis=1)
    # Step 2: mean over chunks: (num_blocks, hidden) -> (hidden,)
    doc_vector = np.mean(chunk_vectors, axis=0)
    return doc_vector


def encode_document(
    text: str,
    tokenizer,
    model,
    block_size: int = 510,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Encode a single document into a fixed-size embedding vector.

    This function wires together:

    - :func:`tokenize_and_block`
    - :func:`add_special_tokens_to_blocks`
    - :func:`extract_block_embeddings`
    - :func:`reduce_block_embeddings_to_document_vector`

    to reproduce the end-to-end behavior used in the notebooks.

    Parameters
    ----------
    text:
        Raw input document.
    tokenizer, model:
        Components of the text encoder (e.g. DistilClinicalBERT).
    block_size:
        Block size prior to adding CLS/SEP tokens.
    device:
        Optional device string; forwarded to
        :func:`extract_block_embeddings`.

    Returns
    -------
    np.ndarray
        Document-level embedding vector of length equal to the model's
        hidden size (typically 768).
    """

    blocks = tokenize_and_block(text, tokenizer, block_size=block_size)
    blocks = add_special_tokens_to_blocks(blocks, tokenizer)
    block_embeddings = extract_block_embeddings(blocks, model, device=device)
    return reduce_block_embeddings_to_document_vector(block_embeddings)

