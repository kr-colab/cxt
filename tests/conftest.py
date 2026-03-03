"""Shared fixtures for cxt tests."""

import pytest
import numpy as np
import torch


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Ensure reproducibility across tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    yield


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"
