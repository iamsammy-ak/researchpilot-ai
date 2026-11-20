"""Dataset insights: schema, missing values, correlations, outliers and simple significance tests,
each with a plain-language readout. Everything here is computed with pandas and SciPy."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
MAX_GROUPS = 12
MAX_ROWS = 200_000


def read_csv(data: bytes) -> pd.DataFrame:
    """Read an uploaded CSV, sniffing the delimiter and falling back to Latin-1 for legacy encodings."""
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding=encoding, nrows=MAX_ROWS)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - latin-1 decodes any byte sequence
        raise ValueError("could not decode the file")
    if df.shape[1] == 0:
        raise ValueError("the file has no columns")
    df.columns = [str(c).strip() or f"column_{i + 1}" for i, c in enumerate(df.columns)]
    return df


def column_kind(s: pd.Series) -> str:
    """numeric, categorical, boolean, datetime, identifier, text or empty."""
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric" if s.notna().any() else "empty"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    values = s.dropna().astype(str)
    if values.empty:
        return "empty"
    unique = values.nunique()
    if unique <= 2 and set(values.str.lower().unique()) <= {"yes", "no", "true", "false", "y", "n", "0", "1"}:
        return "boolean"
    if unique == len(values) and len(values) > 20:
        return "identifier"
    if unique <= max(20, int(0.05 * len(values))):
        return "categorical"
    return "text" if values.str.len().mean() > 30 else "categorical" if unique <= 50 else "identifier"


def _fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def format_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        kind = column_kind(s)
        missing = int(s.isna().sum())
        if kind == "numeric":
            summary = f"mean {_fmt(s.mean())} · sd {_fmt(s.std())} · {_fmt(s.min())} to {_fmt(s.max())}"
        elif kind in ("categorical", "boolean"):
            top = s.value_counts(dropna=True)
            summary = f"most common: {top.index[0]} ({top.iloc[0] / max(1, s.notna().sum()):.0%})" if len(top) else "—"
        elif kind == "identifier":
            summary = "a distinct value on every row"
        elif kind == "text":
            summary = f"free text, {s.dropna().astype(str).str.len().mean():.0f} characters on average"
        else:
            summary = "—"
        rows.append(
            {
                "column": col,
                "kind": kind,
                "non_null": n - missing,
                "missing": missing,
                "missing_pct": missing / n if n else 0.0,
                "unique": int(s.nunique(dropna=True)),
                "summary": summary,
            }
        )
    return pd.DataFrame(rows)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns with at least three values and some variation."""
    return [c for c in df.columns
            if column_kind(df[c]) == "numeric" and df[c].count() >= 3 and df[c].std(skipna=True) > 0]


def group_columns(df: pd.DataFrame) -> list[str]:
    """Columns that split rows into 2-12 groups: categorical, boolean, or numeric with few levels."""
    out = []
    for c in df.columns:
        kind = column_kind(df[c])
        levels = df[c].nunique(dropna=True)
        if 2 <= levels <= MAX_GROUPS and (kind in ("categorical", "boolean") or kind == "numeric" and levels <= 6):
            out.append(c)
    return out


def strength(r: float) -> str:
    a = abs(r)
    if a < 0.1:
        return "no meaningful"
    if a < 0.3:
        return "weak"
    if a < 0.5:
        return "moderate"
    if a < 0.7:
        return "strong"
    return "very strong"


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    cols = numeric_columns(df)
    return df[cols].corr(method=method) if len(cols) >= 2 else pd.DataFrame()


def correlation_pairs(df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Every pair of numeric columns with Pearson r, Spearman rho, their p-values and a readout."""
    rows = []
    for a, b in combinations(numeric_columns(df), 2):
        pair = df[[a, b]].dropna()
        if len(pair) < 3 or pair[a].nunique() < 2 or pair[b].nunique() < 2:
            continue
        r, p_r = stats.pearsonr(pair[a], pair[b])
        rho, p_rho = stats.spearmanr(pair[a], pair[b])
        r, rho = float(r), float(rho)
        direction = "higher" if (rho if abs(rho) > abs(r) else r) > 0 else "lower"
        word = strength(max(abs(r), abs(rho)))
        if word == "no meaningful":
            readout = f"No meaningful relationship between {a} and {b}."
        else:
            readout = f"{word.capitalize()} relationship: higher {a} goes with {direction} {b}."
            if abs(rho) - abs(r) > 0.1:
                readout += " Spearman is clearly larger than Pearson, so it is monotonic but not linear."
            if min(p_r, p_rho) >= ALPHA:
                readout += f" Not significant at the 5% level ({format_p(min(p_r, p_rho))}), so treat it as noise."
        rows.append({"a": a, "b": b, "n": len(pair), "pearson": r, "p_pearson": float(p_r), "spearman": rho,
                     "p_spearman": float(p_rho), "strength": word, "readout": readout})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rank_key"] = out[["pearson", "spearman"]].abs().max(axis=1)
    return out.sort_values("rank_key", ascending=False, ignore_index=True).drop(columns="rank_key").head(top)


def iqr_outliers(df: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    """Tukey fences: values below Q1 - k*IQR or above Q3 + k*IQR."""
    rows = []
    for col in numeric_columns(df):
        s = df[col].dropna()
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        low, high = q1 - k * iqr, q3 + k * iqr
        flagged = s[(s < low) | (s > high)]
        extremes = flagged.reindex(flagged.sub(s.median()).abs().sort_values(ascending=False).index)
        rows.append(
            {
                "column": col,
                "q1": q1,
                "q3": q3,
                "low_fence": low,
                "high_fence": high,
                "outliers": int(len(flagged)),
                "share": len(flagged) / len(s) if len(s) else 0.0,
                "examples": ", ".join(_fmt(float(v)) for v in extremes.head(3)),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("outliers", ascending=False, ignore_index=True) if not out.empty else out


@dataclass(frozen=True)
class TestResult:
    test: str
    statistic: float
    p_value: float
    effect_name: str
    effect: float
    effect_label: str
    n: int
    readout: str
    detail: pd.DataFrame
    caveats: tuple[str, ...] = field(default_factory=tuple)

    @property
    def significant(self) -> bool:
        return self.p_value < ALPHA


def _effect_label(value: float, bands: tuple[float, float, float]) -> str:
    small, medium, large = bands
    a = abs(value)
    if a < small:
        return "negligible"
    if a < medium:
        return "small"
    if a < large:
        return "medium"
    return "large"


def compare_groups(df: pd.DataFrame, value: str, group: str) -> TestResult:
    """Welch's t-test for two groups, one-way ANOVA for more, with a rank-based test as a cross-check."""
    data = df[[value, group]].dropna()
    groups = {str(k): g[value].astype(float).to_numpy() for k, g in data.groupby(group, sort=True) if len(g) >= 2}
    if len(groups) < 2:
        raise ValueError(f"{group} needs at least two groups with two or more values each")
    if len(groups) > MAX_GROUPS:
        raise ValueError(f"{group} has more than {MAX_GROUPS} groups")
    detail = pd.DataFrame(
        [{"group": k, "n": len(v), "mean": v.mean(), "median": float(np.median(v)), "sd": v.std(ddof=1)}
         for k, v in groups.items()]
    ).sort_values("mean", ascending=False, ignore_index=True)
    samples = list(groups.values())
    n = int(sum(len(v) for v in samples))
    caveats = []
    if len(samples) == 2:
        a, b = samples
        stat, p = stats.ttest_ind(a, b, equal_var=False)
        pooled = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        effect = (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0
        test, effect_name, bands = "Welch's t-test", "Cohen's d", (0.2, 0.5, 0.8)
        p_rank = float(stats.mannwhitneyu(a, b).pvalue)
        rank_name = "Mann-Whitney U"
    else:
        stat, p = stats.f_oneway(*samples)
        everything = np.concatenate(samples)
        ss_between = sum(len(v) * (v.mean() - everything.mean()) ** 2 for v in samples)
        ss_total = ((everything - everything.mean()) ** 2).sum()
        effect = ss_between / ss_total if ss_total > 0 else 0.0
        test, effect_name, bands = "One-way ANOVA", "eta squared", (0.01, 0.06, 0.14)
        p_rank = float(stats.kruskal(*samples).pvalue)
        rank_name = "Kruskal-Wallis"
    stat, p = float(stat), float(p)
    if math.isnan(p):
        raise ValueError("the test is undefined for these values (no variation within groups)")
    label = _effect_label(effect, bands)
    if (p < ALPHA) != (p_rank < ALPHA):
        caveats.append(f"The rank-based {rank_name} test disagrees ({format_p(p_rank)}): check for outliers or skew.")
    if detail["n"].min() < 10:
        caveats.append("At least one group has fewer than 10 values, so the test has little power.")
    hi, lo = detail.iloc[0], detail.iloc[-1]
    stat_name = "t" if len(samples) == 2 else "F"
    numbers = f"{test}: {stat_name} = {stat:.2f}, {format_p(p)}, {effect_name} = {effect:.2f} ({label} effect)"
    if p < ALPHA:
        readout = (f"{value} differs between {group} groups: highest in {hi['group']} (mean {_fmt(hi['mean'])}), "
                   f"lowest in {lo['group']} (mean {_fmt(lo['mean'])}). A gap this size is unlikely to be chance "
                   f"at the 5% level. {numbers}.")
    else:
        readout = (f"No evidence that {value} differs between {group} groups: the gap between {hi['group']} and "
                   f"{lo['group']} could be chance. {numbers}.")
    return TestResult(test, stat, p, effect_name, float(effect), label, n, readout, detail, tuple(caveats))


def chi_square(df: pd.DataFrame, a: str, b: str) -> TestResult:
    """Chi-square test of independence between two categorical columns, with Cramér's V."""
    if a == b:
        raise ValueError("pick two different columns")
    table = pd.crosstab(df[a], df[b])
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("both columns need at least two categories")
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = int(table.to_numpy().sum())
    v = math.sqrt(chi2 / (n * (min(table.shape) - 1))) if n else 0.0
    label = _effect_label(v, (0.1, 0.3, 0.5))
    caveats = []
    low_share = float((expected < 5).mean())
    if low_share > 0.2:
        caveats.append(f"{low_share:.0%} of cells expect fewer than 5 rows, so the p-value is approximate.")
    numbers = f"χ² = {chi2:.1f}, df = {dof}, {format_p(p)}, Cramér's V = {v:.2f} ({label} association)"
    if p < ALPHA:
        readout = f"{a} and {b} are related: the mix of {b} changes with {a}. {numbers}."
    else:
        readout = f"No evidence that {a} and {b} are related: they look independent. {numbers}."
    return TestResult("Chi-square test of independence", float(chi2), float(p), "Cramér's V", v, label, n, readout,
                      table, tuple(caveats))


def findings(df: pd.DataFrame) -> list[str]:
    """Plain-language headline findings, all computed."""
    prof = profile(df)
    kinds = prof["kind"].value_counts()
    out = [
        f"{len(df):,} rows × {df.shape[1]} columns: "
        + ", ".join(f"{count} {kind}" for kind, count in kinds.items())
        + "."
    ]
    missing = prof[prof["missing"] > 0].sort_values("missing_pct", ascending=False)
    if missing.empty:
        out.append("No missing values.")
    else:
        parts = [f"{r.column} ({r.missing_pct:.1%})" for r in missing.head(3).itertuples()]
        out.append("Missing values: " + ", ".join(parts) + ".")
    pairs = correlation_pairs(df, top=1)
    if not pairs.empty:
        top = pairs.iloc[0]
        out.append(f"Strongest correlation: {top['a']} and {top['b']} (r = {top['pearson']:.2f}, "
                   f"ρ = {top['spearman']:.2f}). {top['readout']}")
    outliers = iqr_outliers(df)
    if not outliers.empty and outliers.iloc[0]["outliers"] > 0:
        top = outliers.iloc[0]
        out.append(f"Most outliers: {top['column']} has {top['outliers']} values outside the 1.5 × IQR fences "
                   f"(e.g. {top['examples']}).")
    best = strongest_group_effect(df)
    if best is not None:
        value, group, result = best
        out.append(f"Largest group difference: {value} by {group} ({result.effect_name} = {result.effect:.2f}, "
                   f"{format_p(result.p_value)}).")
    return out


def strongest_group_effect(df: pd.DataFrame) -> tuple[str, str, TestResult] | None:
    """The significant numeric-by-group comparison with the largest effect size (ANOVA eta squared or |d|).
    Only text categories are used as groups: yes/no flags are often outcomes derived from another column
    (here, end of life), and numeric levels are already covered by the correlations."""
    best = None
    for group in [g for g in group_columns(df) if column_kind(df[g]) == "categorical"][:10]:
        for value in numeric_columns(df)[:15]:
            if value == group:
                continue
            try:
                result = compare_groups(df, value, group)
            except ValueError:
                continue
            if not result.significant:
                continue
            # put d and eta squared on one scale: eta^2 = d^2 / (d^2 + 4)
            size = result.effect**2 / (result.effect**2 + 4) if result.effect_name == "Cohen's d" else result.effect
            if best is None or size > best[0]:
                best = (size, value, group, result)
    return None if best is None else best[1:]
