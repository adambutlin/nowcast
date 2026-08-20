"""CPI target: MoM transform and the MoM -> YoY reconstruction identity."""
import sys, os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dfm import target as T


@pytest.fixture
def index_series():
    """A synthetic CPI index growing 0.2%/month for 3 years."""
    idx = pd.date_range("2020-01-31", periods=36, freq="ME")
    return pd.Series(100.0 * np.exp(np.arange(36) * 0.002), index=idx, name="cpi_index")


class TestMoM:
    def test_is_100x_log_difference(self, index_series):
        mom = T.mom_from_index(index_series)
        assert mom.iloc[1] == pytest.approx(0.2, abs=1e-9)
        assert np.isnan(mom.iloc[0])

    def test_length_matches_input(self, index_series):
        assert len(T.mom_from_index(index_series)) == len(index_series)


class TestYoYReconstruction:
    def test_identity_recovers_yoy_exactly(self, index_series):
        """YoY built from carry + modelled MoM equals YoY straight off the index."""
        mom = T.mom_from_index(index_series)
        direct = 100.0 * (index_series / index_series.shift(12) - 1.0)
        for m in index_series.index[13:]:
            got = T.yoy_from_mom(float(mom.loc[m]), index_series, m)
            assert got == pytest.approx(float(direct.loc[m]), abs=1e-9)

    def test_carry_uses_only_published_months(self, index_series):
        """The carry term must not touch the index value of month T itself."""
        m = index_series.index[20]
        truncated = index_series.copy()
        truncated.loc[m] = np.nan
        assert T.carry(truncated, m) == pytest.approx(T.carry(index_series, m))

    def test_raises_when_history_too_short(self, index_series):
        with pytest.raises(ValueError, match="12 months"):
            T.carry(index_series, index_series.index[5])


class TestRoundingFloor:
    def test_one_dp_rounding_stays_under_the_documented_floor(self, index_series):
        """Rounding the index to 1 d.p. must not move YoY by more than 0.135pp."""
        rounded = index_series.round(1)
        mom = T.mom_from_index(index_series)
        errs = []
        for m in index_series.index[13:]:
            errs.append(abs(T.yoy_from_mom(float(mom.loc[m]), rounded, m)
                            - T.yoy_from_mom(float(mom.loc[m]), index_series, m)))
        assert max(errs) < 0.135
