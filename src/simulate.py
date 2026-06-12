"""
Stochastic risk layer (proposal section 3.5): two-state Markov chain rainfall
occurrence + Monte Carlo simulation of planting-window outcomes.

  1. Each dekad is classified wet/dry (>= / < DRY_DEKAD_MM).
  2. Month-specific transition probabilities P(wet|wet), P(wet|dry) are
     estimated from the historical record (training years only, to avoid
     leakage into evaluation).
  3. Wet- and dry-dekad rainfall amounts are sampled from month-specific
     empirical pools (non-parametric equivalent of the proposal's Gamma
     option - more robust for the smoothed district-average product).
  4. Dekadal tmax is sampled from dekad-of-year empirical pools
     (1961-2021 gridded record).
  5. Monte Carlo: N simulated crop-cycle sequences per candidate planting
     window. Each sequence is scored against the crop requirements,
     producing the proposal's risk features:
        p_rain_sufficient   P(cycle rainfall >= crop minimum)
        p_dry_spell         P(harmful establishment dry spell)
        p_temp_stress       P(cycle-mean tmax warm anomaly)
        risk_score          0.5*(1-p_rain_sufficient) + 0.3*p_dry_spell
                            + 0.2*p_temp_stress
  6. The chain is conditioned on the observed state at planting time:
     the wet/dry state of the preceding dekad sets the initial state, and
     a recent-rainfall anomaly (z-score of the last 3 dekads) nudges the
     wet probability of the first simulated dekads (rainfall persistence),
     decaying geometrically.
"""
import numpy as np
import pandas as pd

from crops import (CROPS, DRY_DEKAD_MM, DRY_SPELL_DEKADS, ESTABLISHMENT_DEKADS,
                   RISK_WEIGHTS)

N_SIMS = 1000
PERSISTENCE = 0.15       # weight of the recent-anomaly nudge on p(wet)
PERSISTENCE_DECAY = 0.6  # geometric decay of the nudge per simulated dekad
TEMP_PERSISTENCE = 0.7   # carry-over of the observed May-Aug tmax anomaly
                         # into the simulated season (warm years persist)


def _month_of_dekad(dekad_of_year: int) -> int:
    return (int(dekad_of_year) - 1) % 36 // 3 + 1


class WeatherGenerator:
    """Markov-chain + empirical-amount stochastic weather generator."""

    def fit(self, df: pd.DataFrame, tmax_pools: dict, year_max: int | None = None):
        """Estimate transition probabilities and amount pools.
        `year_max` limits fitting to years <= year_max (leakage control)."""
        d = df if year_max is None else df[df.year <= year_max]
        d = d.sort_values("abs_dekad")
        wet = (d.rainfall_mm >= DRY_DEKAD_MM).to_numpy()
        month = d.month.to_numpy()
        consecutive = np.diff(d.abs_dekad.to_numpy()) == 1

        self.p_wet_given_wet, self.p_wet_given_dry, self.p_wet_uncond = {}, {}, {}
        self.pool_wet, self.pool_dry = {}, {}
        for m in range(1, 13):
            idx = np.where((month[1:] == m) & consecutive)[0] + 1
            prev_wet = wet[idx - 1]
            now_wet = wet[idx]
            self.p_wet_given_wet[m] = float(now_wet[prev_wet].mean()) if prev_wet.any() else 0.5
            self.p_wet_given_dry[m] = float(now_wet[~prev_wet].mean()) if (~prev_wet).any() else 0.5
            in_m = d[d.month == m].rainfall_mm.to_numpy()
            self.p_wet_uncond[m] = float((in_m >= DRY_DEKAD_MM).mean())
            self.pool_wet[m] = in_m[in_m >= DRY_DEKAD_MM]
            self.pool_dry[m] = in_m[in_m < DRY_DEKAD_MM]
            if len(self.pool_wet[m]) == 0:
                self.pool_wet[m] = np.array([DRY_DEKAD_MM])
            if len(self.pool_dry[m]) == 0:
                self.pool_dry[m] = np.array([0.0])

        self.tmax_pools = tmax_pools
        self.tmax_clim_sond = float(np.concatenate(
            [tmax_pools[d_] for d_ in range(25, 37) if d_ in tmax_pools]).mean())
        return self

    def simulate(self, start_dekad: int, n_dekads: int, n_sims: int = N_SIMS,
                 init_wet: bool | None = None, anom_z: float = 0.0,
                 tmax_anom: float = 0.0, seed: int = 0):
        """Simulate rainfall (n_sims x n_dekads) and cycle-mean tmax (n_sims,)."""
        rng = np.random.default_rng(seed)
        rain = np.empty((n_sims, n_dekads))
        wet_states = np.empty((n_sims, n_dekads), dtype=bool)

        if init_wet is None:
            m0 = _month_of_dekad(start_dekad - 1)
            state = rng.random(n_sims) < self.p_wet_uncond[m0]
        else:
            state = np.full(n_sims, bool(init_wet))

        tmax_sum = np.zeros(n_sims)
        for k in range(n_dekads):
            dk = start_dekad + k
            m = _month_of_dekad(dk)
            p = np.where(state, self.p_wet_given_wet[m], self.p_wet_given_dry[m])
            p = np.clip(p + PERSISTENCE * anom_z * PERSISTENCE_DECAY ** k, 0.02, 0.98)
            state = rng.random(n_sims) < p
            wet_states[:, k] = state
            amounts = np.where(
                state,
                rng.choice(self.pool_wet[m], size=n_sims),
                rng.choice(self.pool_dry[m], size=n_sims))
            rain[:, k] = amounts
            dk_wrapped = (dk - 1) % 36 + 1
            pool_t = self.tmax_pools.get(dk_wrapped)
            t = rng.choice(pool_t, size=n_sims) if pool_t is not None \
                else self.tmax_clim_sond
            tmax_sum += t + TEMP_PERSISTENCE * tmax_anom
        return rain, wet_states, tmax_sum / n_dekads

    def risk_features(self, crop: str, window_dekad: int,
                      init_wet: bool | None = None, anom_z: float = 0.0,
                      tmax_anom: float = 0.0, n_sims: int = N_SIMS,
                      seed: int = 0) -> dict:
        """Monte Carlo risk features for one (crop, planting window)."""
        c = CROPS[crop]
        rain, wet, tmax_mean = self.simulate(
            window_dekad, c["cycle_dekads"], n_sims, init_wet, anom_z,
            tmax_anom, seed)

        cycle_total = rain.sum(axis=1)
        p_rain_sufficient = float((cycle_total >= c["min_cycle_mm"]).mean())

        estab_dry = ~wet[:, :ESTABLISHMENT_DEKADS]
        run = np.zeros(n_sims)
        worst = np.zeros(n_sims)
        for k in range(estab_dry.shape[1]):
            run = np.where(estab_dry[:, k], run + 1, 0)
            worst = np.maximum(worst, run)
        low_estab = rain[:, :ESTABLISHMENT_DEKADS].sum(axis=1) < c["min_establishment_mm"]
        p_dry_spell = float(((worst >= DRY_SPELL_DEKADS) | low_estab).mean())

        p_temp_stress = float(
            (tmax_mean > self.tmax_clim_sond + c["tmax_stress_anom_c"]).mean())

        risk_score = (RISK_WEIGHTS["rain_deficit"] * (1 - p_rain_sufficient)
                      + RISK_WEIGHTS["dry_spell"] * p_dry_spell
                      + RISK_WEIGHTS["temp_stress"] * p_temp_stress)
        med = float(np.median(cycle_total))
        p10 = float(np.percentile(cycle_total, 10))
        return {
            "p_rain_sufficient": round(p_rain_sufficient, 3),
            "p_dry_spell": round(p_dry_spell, 3),
            "p_temp_stress": round(p_temp_stress, 3),
            "risk_score": round(float(risk_score), 3),
            "sim_cycle_rain_p10": round(p10, 1),
            "sim_cycle_rain_median": round(med, 1),
            "sim_cycle_rain_p90": round(float(np.percentile(cycle_total, 90)), 1),
            # margins vs the crop's minimum requirement: how much headroom
            # the simulated season gives this crop in this window
            "sim_margin_median": round(med - c["min_cycle_mm"], 1),
            "sim_margin_p10": round(p10 - c["min_cycle_mm"], 1),
        }


if __name__ == "__main__":
    import json
    from data_loader import load, temperature_pools
    df = load()
    gen = WeatherGenerator().fit(df, temperature_pools(df))
    for crop in ("maize", "beans"):
        for w in (25, 29, 33):
            print(crop, w, json.dumps(gen.risk_features(crop, w)))
