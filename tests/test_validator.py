from __future__ import annotations
import pytest
from src.cases.validator import _normalise, _normalise_list, CaseValidationResult


def test_normalise_test_names():
    assert _normalise("fbc") == "CBC"
    assert _normalise("FBC") == "CBC"
    assert _normalise("u&e") == "BMP"
    assert _normalise("cxr") == "CXR"
    assert _normalise("chest x-ray") == "CXR"
    assert _normalise("ua") == "Urinalysis POCT"
    assert _normalise("ecg") == "ECG (immediate)"
    assert _normalise("troponin") == "Troponin POCT"


def test_agreement_rate_perfect():
    actual = ["CBC", "BMP", "CRP"]
    proposed = ["CBC", "BMP", "CRP"]
    actual_n = _normalise_list(actual)
    proposed_n = _normalise_list(proposed)
    matched = list(set(actual_n) & set(proposed_n))
    rate = len(matched) / len(set(actual_n))
    assert rate == 1.0


def test_agreement_rate_partial():
    actual = ["CBC", "BMP", "CRP", "ECG (immediate)", "Troponin POCT"]
    proposed = ["CBC", "BMP", "CRP"]
    actual_n = set(_normalise_list(actual))
    proposed_n = set(_normalise_list(proposed))
    matched = actual_n & proposed_n
    rate = len(matched) / len(actual_n)
    assert abs(rate - 0.6) < 0.01


def test_missing_tests_identified():
    actual = ["cbc", "bmp", "crp"]
    proposed = ["BMP", "CRP"]
    actual_n = set(_normalise_list(actual))
    proposed_n = set(_normalise_list(proposed))
    missed = actual_n - proposed_n
    assert "CBC" in missed


def test_empty_cases_file():
    from src.cases.validator import load_case_list
    result = load_case_list("nonexistent_file_xyz.json")
    assert result == []


def test_normalise_list_preserves_length():
    tests = ["ua", "cbc", "troponin", "ecg", "cxr"]
    result = _normalise_list(tests)
    assert len(result) == len(tests)


def test_unknown_test_passthrough():
    assert _normalise("some_custom_test") == "some_custom_test"
