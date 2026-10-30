"""The bisection path of find_threshold, used for tensors torch.sort cannot handle."""

import pytest
import torch

from circuit_tracer import graph
from circuit_tracer.graph import _find_threshold_bisect, find_threshold

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLDS = torch.linspace(0, 1, 21).tolist()


def sparse_scores(numel: int, seed: int = 0) -> torch.Tensor:
    """Non-negative scores, 70% exact zeros (as in pruned graphs)."""
    generator = torch.Generator(DEVICE).manual_seed(seed)
    scores = torch.rand(numel, device=DEVICE, generator=generator)
    return scores * (torch.rand(numel, device=DEVICE, generator=generator) < 0.3)


def kept_fraction(scores: torch.Tensor, cut: torch.Tensor) -> float:
    """The fraction of the total score kept at a cut-off, summed in float64."""
    kept = total = 0.0
    for chunk in scores.split(2**26):
        chunk = chunk.double()
        kept += torch.where(chunk >= cut, chunk, 0).sum().item()
        total += chunk.sum().item()
    return kept / total


@pytest.mark.parametrize("numel", [10, 1_000, 100_000, 10_000_000])
def test_bisect_matches_sort(numel):
    """Both paths keep the fraction that a float64 sort keeps.

    The float32 sums of the two paths round differently, so the cut-offs can differ
    by one score; the check is on the kept fraction, not on the exact cut-off.
    """
    scores = sparse_scores(numel)
    for threshold in THRESHOLDS:
        expected = kept_fraction(scores, find_threshold(scores.double(), threshold))
        sort_cut = find_threshold(scores, threshold)
        # chunks smaller than the tensor, so that the chunked sums are exercised
        bisect_cut = _find_threshold_bisect(scores, threshold, chunk_size=max(numel // 7, 1))
        for cut in sort_cut, bisect_cut:
            assert cut > 0  # never keeps the zero scores
            assert cut in scores
            assert kept_fraction(scores, cut) == pytest.approx(expected, abs=1e-4), threshold


def test_bisect_all_zero():
    assert _find_threshold_bisect(torch.zeros(100), 0.9) == 0


@pytest.mark.parametrize("max_sort_numel", [0, 999, 1_000, 10**9])
def test_switch_point_does_not_change_cut(monkeypatch, max_sort_numel):
    """find_threshold gives the same cut on either side of the sort-size limit."""
    scores = sparse_scores(1_000)
    expected = [find_threshold(scores, threshold) for threshold in THRESHOLDS]
    monkeypatch.setattr(graph, "_MAX_SORT_NUMEL", max_sort_numel)
    assert [find_threshold(scores, threshold) for threshold in THRESHOLDS] == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_beyond_sort_limit():
    """A tensor torch.sort cannot handle: no OOM, bounded extra memory, the right cut.

    The scores are one small tensor repeated, so the kept fraction of each score is
    the same as in the small tensor, where a float64 sort gives the reference.
    """
    base = sparse_scores(2**20)
    repeats = graph._MAX_SORT_NUMEL // base.numel() + 1
    free, _ = torch.cuda.mem_get_info()
    if free < 2 * repeats * base.numel() * base.element_size():
        pytest.skip("not enough free GPU memory")
    scores = base.repeat(repeats)
    assert scores.numel() > graph._MAX_SORT_NUMEL

    for threshold in [0.1, 0.5, 0.9, 0.99]:
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.memory_allocated()
        cut = find_threshold(scores, threshold)
        extra_gib = (torch.cuda.max_memory_allocated() - start) / 2**30
        assert extra_gib < 2, extra_gib

        expected = kept_fraction(base, find_threshold(base.double(), threshold))
        assert kept_fraction(scores, cut) == pytest.approx(expected, abs=1e-4)
