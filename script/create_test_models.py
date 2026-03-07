#!/usr/bin/env python3
"""Create small ONNX test models for onnx-ruby tests."""

import os
import torch
import torch.nn as nn

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "test", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


class SimpleModel(nn.Module):
    """Linear model: 4 inputs -> 3 outputs."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)


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


if __name__ == "__main__":
    export_simple()
    print("Done.")
