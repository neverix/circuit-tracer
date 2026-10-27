"""The bisection path of find_threshold, used for tensors torch.sort cannot handle."""

import pytest
import torch

from circuit_tracer.graph import _find_threshold_bisect, find_threshold


@pytest.mark.parametrize("threshold", [0.0, 0.5, 0.9, 0.99, 1.0])
def test_bisect_matches_sort(threshold):
    torch.manual_seed(0)
    scores = torch.rand(10_000) * (torch.rand(10_000) < 0.3)  # 70% exact zeros
    # small chunks so that the chunked sums are exercised
    cut = _find_threshold_bisect(scores, threshold, chunk_size=1_000)
    assert cut > 0  # never keeps the zero scores
    if threshold < 1.0:
        assert cut == find_threshold(scores, threshold)
    else:
        assert cut == scores[scores > 0].min()


def test_bisect_all_zero():
    assert _find_threshold_bisect(torch.zeros(100), 0.9) == 0
