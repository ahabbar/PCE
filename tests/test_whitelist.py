from __future__ import annotations

import pytest

from src.core.whitelist import ProtocolWhitelist


@pytest.fixture(scope="module")
def wl():
    return ProtocolWhitelist("protocols/whitelist.yaml")


def test_loads_6_protocols(wl):
    assert len(wl.all_protocols()) == 6


def test_match_uti_english(wl):
    assert wl.match("dysuria and flank pain") == "uti_pyelonephritis"


def test_match_uti_arabic(wl):
    assert wl.match("حرقة بولية وألم خاصرة") == "uti_pyelonephritis"


def test_match_chest(wl):
    assert wl.match("chest pain and diaphoresis") == "chest_pain"


def test_match_altered(wl):
    assert wl.match("confusion and weakness") == "altered_consciousness"


def test_match_none(wl):
    assert wl.match("headache migraine") is None


def test_is_allowed_cbc_fever(wl):
    assert wl.is_allowed("fever_infectious", "CBC") is True


def test_is_allowed_cbc_partial(wl):
    assert wl.is_allowed("fever_infectious", "CBC with differential") is True


def test_is_allowed_ct_false(wl):
    assert wl.is_allowed("fever_infectious", "CT scan") is False


def test_ct_in_requires_doctor(wl):
    requires = wl.get_requires_doctor("fever_infectious")
    assert any("CT scan" in r for r in requires)


def test_is_allowed_wrong_protocol(wl):
    assert wl.is_allowed("NONEXISTENT", "CBC") is False


def test_auto_orders_nonempty(wl):
    orders = wl.get_auto_orders("chest_pain")
    assert len(orders) >= 4
