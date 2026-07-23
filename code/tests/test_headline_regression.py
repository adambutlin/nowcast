"""Headline output must NOT move when the pipeline is parameterized by target.
Golden values captured 2026-07-23 from the frozen model on the quant env (June-2026 nowcast).
Fetches + backtests, so this is a slow integration test — run it explicitly.
"""
import production.model as M
import targets as T


def test_headline_nowcast_byte_identical():
    nc = M.nowcast(target=T.HEADLINE)
    assert nc["nowcast_date"] == "2026-06-30"
    assert round(nc["aa"], 3) == 2.726
    assert round(nc["forecast"], 3) == 2.683
    assert nc["n_pinned"] == 9 and nc["n_live"] == 8       # gscp expected-drop
    assert nc["degraded"] is False


def test_core_prints_aa_only_for_now():
    nc = M.nowcast(target=T.CORE)
    assert nc["target"] == "uk_core_cpi"
    assert nc["nowcast_date"] == "2026-06-30"              # core CSV truncated to May -> targets June
    assert nc["n_pinned"] == 0                             # AA-only until the race ships an overlay
    assert nc["forecast"] == nc["aa"]
