#!/usr/bin/env python3
"""Create small ONNX test models for onnx-ruby tests."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "test", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


class SimpleModel(nn.Module):
    """Linear model: 4 inputs -> 3 outputs."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)


class EmbeddingModel(nn.Module):
    """Simulates a token embedding model.
    Takes input_ids [batch, seq_len] and attention_mask [batch, seq_len].
    Returns embeddings [batch, embed_dim] via mean pooling.
    """

    def __init__(self, vocab_size=100, embed_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, input_ids, attention_mask):
        embeds = self.embedding(input_ids)  # [batch, seq, dim]
        # Mean pooling with attention mask
        mask = attention_mask.unsqueeze(-1).float()  # [batch, seq, 1]
        pooled = (embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.linear(pooled)  # [batch, dim]


class ClassifierModel(nn.Module):
    """Simple classifier: float input [batch, features] -> logits [batch, num_classes]."""

    def __init__(self, features=8, num_classes=4):
        super().__init__()
        self.linear = nn.Linear(features, num_classes)

    def forward(self, features):
        return self.linear(features)


class RerankerModel(nn.Module):
    """Cross-encoder reranker.
    Takes input_ids [batch, seq_len] and attention_mask [batch, seq_len].
    Returns relevance scores [batch, 1].
    """

    def __init__(self, vocab_size=100, embed_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, input_ids, attention_mask):
        embeds = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.linear(pooled)  # [batch, 1]


def export_simple():
    model = SimpleModel()
    model.eval()
    dummy = torch.randn(1, 4)
    path = os.path.join(MODELS_DIR, "simple.onnx")
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Exported: {path}")


def export_embedding():
    model = EmbeddingModel(vocab_size=100, embed_dim=8)
    model.eval()
    input_ids = torch.randint(0, 100, (1, 6))
    attention_mask = torch.ones(1, 6, dtype=torch.long)
    path = os.path.join(MODELS_DIR, "embedding.onnx")
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        path,
        input_names=["input_ids", "attention_mask"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "embeddings": {0: "batch"},
        },
    )
    print(f"Exported: {path}")


def export_classifier():
    model = ClassifierModel(features=8, num_classes=4)
    model.eval()
    dummy = torch.randn(1, 8)
    path = os.path.join(MODELS_DIR, "classifier.onnx")
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={
            "features": {0: "batch"},
            "logits": {0: "batch"},
        },
    )
    print(f"Exported: {path}")


def export_reranker():
    model = RerankerModel(vocab_size=100, embed_dim=8)
    model.eval()
    input_ids = torch.randint(0, 100, (1, 10))
    attention_mask = torch.ones(1, 10, dtype=torch.long)
    path = os.path.join(MODELS_DIR, "reranker.onnx")
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        path,
        input_names=["input_ids", "attention_mask"],
        output_names=["scores"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "scores": {0: "batch"},
        },
    )
    print(f"Exported: {path}")


if __name__ == "__main__":
    export_simple()
    export_embedding()
    export_classifier()
    export_reranker()
    print("Done.")
