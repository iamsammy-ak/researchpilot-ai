import numpy as np
import pandas as pd
import pytest
from scipy import stats

from researchpilot import dataset as ds
from researchpilot.demo import load_bundled_dataset


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(0, 1, n)
    group = rng.choice(["A", "B"], n)
    frame = pd.DataFrame(
        {
            "id": [f"r{i}" for i in range(n)],
            "x": x,
            "y": 2 * x + rng.normal(0, 0.5, n),  # strong positive relationship with x
            "noise": rng.normal(0, 1, n),
            "score": np.where(group == "A", 10.0, 12.0) + rng.normal(0, 1, n),  # B is higher
            "group": group,
            "colour": rng.choice(["red", "green", "blue"], n),  # independent of everything
            "flag": rng.choice(["yes", "no"], n),
        }
    )
    frame["linked"] = np.where(frame["group"] == "A", "left", "right")  # fully dependent on group
    frame.loc[:4, "noise"] = 50.0  # five planted outliers
    frame.loc[10:19, "x"] = np.nan  # ten missing values
    return frame


def test_column_kinds_and_csv_reading(df):
    kinds = ds.profile(df).set_index("column")["kind"]
    assert kinds["id"] == "identifier" and kinds["x"] == "numeric"
    assert kinds["group"] == "categorical" and kinds["flag"] == "boolean"
    assert ds.profile(df).set_index("column").loc["x", "missing"] == 10
    frame = ds.read_csv("name;temp\ncafé;21,5\nbar;19\n".encode("latin-1"))  # sniffed delimiter, legacy encoding
    assert list(frame.columns) == ["name", "temp"] and frame.iloc[0, 0] == "café"


def test_correlations_find_the_planted_relationship(df):
    pairs = ds.correlation_pairs(df)
    top = pairs.iloc[0]
    assert {top["a"], top["b"]} == {"x", "y"}
    assert top["pearson"] > 0.9 and top["n"] == 290  # pairwise-complete rows
    assert "higher x goes with higher y" in top["readout"]
    assert ds.strength(0.05) == "no meaningful" and ds.strength(-0.75) == "very strong"


def test_iqr_outliers_flag_planted_values(df):
    out = ds.iqr_outliers(df).set_index("column")
    assert out.loc["noise", "outliers"] >= 5
    assert out.loc["noise", "examples"].startswith("50")


def test_group_comparisons_welch_and_anova(df):
    result = ds.compare_groups(df, "score", "group")
    a, b = (df.loc[df["group"] == g, "score"] for g in ("A", "B"))
    assert result.test == "Welch's t-test"
    assert result.p_value == pytest.approx(stats.ttest_ind(a, b, equal_var=False).pvalue)
    assert result.significant and result.effect_label == "large"
    assert result.detail.iloc[0]["group"] == "B"  # sorted by mean, highest first
    assert "highest in B" in result.readout
    anova = ds.compare_groups(df, "noise", "colour")  # three groups, no real difference
    assert anova.test == "One-way ANOVA" and not anova.significant
    assert anova.readout.startswith("No evidence")
    with pytest.raises(ValueError):
        ds.compare_groups(df, "score", "id")  # every group has one row


def test_chi_square_dependent_versus_independent(df):
    dependent = ds.chi_square(df, "group", "linked")
    independent = ds.chi_square(df, "group", "colour")
    assert dependent.significant and dependent.effect == pytest.approx(1.0, abs=0.01)
    assert not independent.significant and independent.readout.startswith("No evidence")
    with pytest.raises(ValueError):
        ds.chi_square(df, "group", "group")


def test_bundled_dataset_findings():
    frame = load_bundled_dataset()
    findings = ds.findings(frame)
    assert findings[0].startswith("360 rows × 12 columns")
    assert any("internal_resistance_mohm" in f for f in findings[1:2])  # most missing column
    assert any(f.startswith("Largest group difference") and "chemistry" in f for f in findings)
