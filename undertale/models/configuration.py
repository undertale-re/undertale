"""Shared encoder configuration."""

from typing import Any, Dict


class InstructionTraceTransformerEncoderConfiguration:
    """Model size configurations for InstructionTraceTransformerEncoder.

    To make use of this class, simply pass the model size dictionary to model
    initialization as kwargs.
    """

    regularization: Dict[str, Any] = {
        "dropout": 0.1,
        "eps": 1e-12,
    }

    medium: Dict[str, Any] = {
        "sequence_length": 512,
        "depth": 12,
        "heads": 12,
        "hidden_dimensions": 768,
        "intermediate_dimensions": 3072,
        **regularization,
    }

    options = {"medium": medium}
