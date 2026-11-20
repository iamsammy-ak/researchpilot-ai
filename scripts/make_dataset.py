"""Generate the bundled SYNTHETIC battery ageing dataset (demo/datasets/battery_ageing_synthetic.csv).

No real measurements are involved. Each row is one simulated lithium-ion cell after a cycling test,
produced by a seeded random generator so the file is reproducible. The ageing model is a deliberately
simple, textbook-style empirical form chosen so the dataset has realistic *directions* of effect:

    capacity fade (%) = A_chem * (cycles / 1000) ** 0.55          square-root-like calendar/cycle ageing
                        * exp(0.035 * (temperature - 25))           Arrhenius-style: hotter ages faster
                        * (1 + 0.25 * (C-rate - 1))                 faster charging ages faster
                        * (depth of discharge / 80) ** 0.8          deeper cycles age faster
                        + noise ~ N(0, 1.2)

    A_chem: NMC 12, NCA 14, LFP 7 (LFP fades slowest, as commonly reported)
    capacity retention (state of health, %) = 100 - fade, clipped to [55, 100]
    internal resistance (mOhm) = R0_chem * (1 + 0.018 * fade) + noise ~ N(0, 1.0); R0: NMC 22, NCA 25, LFP 18
    max cell temperature (C) = temperature + 2 + 3 * C-rate + noise ~ N(0, 1.5)
    form factor depends on chemistry (LFP is mostly prismatic), lab does not depend on anything
    end_of_life = "yes" when capacity retention is below 80%

Data-quality issues are injected on purpose so the dataset tab has something to find: about 4% of
resistance values and 2% of max temperatures are missing, and five resistance readings are multiplied
by three to mimic sensor faults.

    python scripts/make_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "demo" / "datasets" / "battery_ageing_synthetic.csv"
SEED = 42
N_CELLS = 360

FADE_SCALE = {"NMC": 12.0, "NCA": 14.0, "LFP": 7.0}
R0 = {"NMC": 22.0, "NCA": 25.0, "LFP": 18.0}
FORM_FACTOR_P = {  # cylindrical, pouch, prismatic
    "NMC": (0.45, 0.40, 0.15),
    "NCA": (0.80, 0.15, 0.05),
    "LFP": (0.15, 0.20, 0.65),
}


def generate(n: int = N_CELLS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    chemistry = rng.choice(["NMC", "LFP", "NCA"], size=n, p=[0.45, 0.35, 0.20])
    form = np.array([rng.choice(["cylindrical", "pouch", "prismatic"], p=FORM_FACTOR_P[c]) for c in chemistry])
    lab = rng.choice(["Lab A", "Lab B", "Lab C"], size=n)
    temperature = rng.choice([15, 25, 35, 45], size=n, p=[0.2, 0.35, 0.3, 0.15])
    c_rate = np.round(rng.uniform(0.3, 2.0, size=n), 1)
    dod = rng.choice([40, 60, 80, 100], size=n)
    cycles = rng.integers(200, 2001, size=n)

    scale = np.array([FADE_SCALE[c] for c in chemistry])
    fade = (
        scale
        * (cycles / 1000) ** 0.55
        * np.exp(0.035 * (temperature - 25))
        * (1 + 0.25 * (c_rate - 1))
        * (dod / 80) ** 0.8
        + rng.normal(0, 1.2, size=n)
    )
    retention = np.clip(100 - fade, 55, 100)
    resistance = np.array([R0[c] for c in chemistry]) * (1 + 0.018 * (100 - retention)) + rng.normal(0, 1.0, n)
    max_temp = temperature + 2 + 3 * c_rate + rng.normal(0, 1.5, size=n)

    df = pd.DataFrame(
        {
            "cell_id": [f"C{i:03d}" for i in range(1, n + 1)],
            "lab": lab,
            "chemistry": chemistry,
            "form_factor": form,
            "ambient_temp_c": temperature,
            "charge_c_rate": c_rate,
            "depth_of_discharge_pct": dod,
            "cycles": cycles,
            "capacity_retention_pct": np.round(retention, 2),
            "internal_resistance_mohm": np.round(resistance, 2),
            "max_cell_temp_c": np.round(max_temp, 1),
            "end_of_life": np.where(retention < 80, "yes", "no"),
        }
    )
    faults = rng.choice(n, size=5, replace=False)
    df.loc[faults, "internal_resistance_mohm"] = (df.loc[faults, "internal_resistance_mohm"] * 3).round(2)
    df.loc[rng.random(n) < 0.04, "internal_resistance_mohm"] = np.nan
    df.loc[rng.random(n) < 0.02, "max_cell_temp_c"] = np.nan
    return df


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT.name}: {len(df)} rows x {df.shape[1]} columns (synthetic, seed {SEED})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
