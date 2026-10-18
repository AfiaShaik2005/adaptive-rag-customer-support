# Adaptive RAG for Customer-Support Question Answering

## Overview

This project evaluates retrieval strategies and adaptive retrieval gating for customer-support question answering using a quantized Qwen2.5-1.5B-Instruct language model.

The study compares retrieval configurations and investigates whether lightweight gating signals can identify when retrieval is likely to be beneficial.

## Methods

The project investigates the following configurations:

- No-retrieval baseline
- Sparse retrieval using BM25
- Dense retrieval using sentence embeddings
- Hybrid retrieval using Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- A learned router for retrieval selection
- Confidence-based and TARG-inspired adaptive gating

## Dataset

The experiments use the [Tobi-Bueck/customer-support-tickets dataset on Hugging Face](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets).

Refer to the dataset card for its license and usage conditions.

## Implementation

The project uses Python and a Google Colab-based experimental workflow. The implementation uses Transformers, PyTorch, PEFT, sentence-transformers, and other supporting libraries.

The repository includes the experimental notebook and selected supporting results.

## Execution and limitations

The notebook was developed in a Google Colab environment and references files stored in Google Drive. Some adapter-dependent cells require a separately available adapter directory and associated model files. These dependencies are not included in this repository, so the notebook may require additional configuration and assets to execute.

The repository is provided to document the implementation associated with the research paper. Its contents should not be interpreted as a turnkey reproduction package.

## Research scope

The reported evaluation uses a held-out test set of 350 tickets. Performance is assessed primarily using ROUGE metrics. The findings are subject to the dataset, evaluation protocol, and limitations described in the accompanying paper.
