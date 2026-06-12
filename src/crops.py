"""
Crop requirements and agronomic thresholds (proposal Table 4: crop_requirements).

Sources: FAO crop water guidance (maize 350-450+ mm for short-cycle varieties,
bush beans 200-300 mm over a 60-75 day cycle) adapted to the dekadal,
district-averaged ENACTS product. The merged station-satellite series is a
spatial average over the Nyagatare box, so absolute point-scale thresholds
(e.g. daily tmax > 32 C) are not observable; temperature stress is therefore
defined as a warm anomaly above the local Sep-Dec climatology. All thresholds
are documented here because they determine the suitability labels
(proposal objective 4) - validate with RAB Nyagatare before any pilot.
"""

# A dekad is "dry" below this total (~2 mm/day at dekad scale for the
# smoothed district-average product). Same threshold defines the Markov
# chain wet/dry states.
DRY_DEKAD_MM = 20.0

# Effective rainfall onset: first dekad at or above this total from 1 Sep.
ONSET_MM = 25.0

# A harmful dry spell = this many consecutive dry dekads (~20+ days)
# during the establishment phase (first 3 dekads after planting).
DRY_SPELL_DEKADS = 2
ESTABLISHMENT_DEKADS = 3

CROPS = {
    "maize": {
        "cycle_dekads": 12,          # ~120-day Season A variety
        "min_cycle_mm": 330.0,       # below this, grain fill is unreliable
        "ideal_cycle_mm": 450.0,
        "min_establishment_mm": 60.0,
        "tmax_stress_anom_c": 1.2,   # cycle-mean tmax > clim + 1.2 C
        "source_note": "FAO crop water needs (maize 350-500 mm short-cycle); "
                       "establishment ~50-70 mm/30d; thresholds calibrated to "
                       "the ENACTS district-average product (see notebook s.1); "
                       "heat stress = warm anomaly proxy vs Sep-Dec climatology "
                       "(maize optimum ~25 C) - validate with RAB Nyagatare",
    },
    "beans": {
        "cycle_dekads": 7,           # ~70-day bush bean (Nyagatare-dominant)
        "min_cycle_mm": 200.0,
        "ideal_cycle_mm": 300.0,
        "min_establishment_mm": 50.0,
        "tmax_stress_anom_c": 0.8,   # beans are the more heat-sensitive crop
        "source_note": "FAO crop water needs (bush bean 200-300 mm / 60-75 d); "
                       "establishment ~40-60 mm/30d; beans more heat-sensitive "
                       "(stress above ~25 C mean) so tighter anomaly threshold; "
                       "calibrated to ENACTS district average - validate with "
                       "RAB Nyagatare",
    },
}

# Candidate Season A planting windows (dekads of year, 1 Sep - 30 Nov).
PLANTING_WINDOWS = list(range(25, 34))

DEKAD_LABEL = {
    25: "1-10 Sep", 26: "11-20 Sep", 27: "21-30 Sep",
    28: "1-10 Oct", 29: "11-20 Oct", 30: "21-31 Oct",
    31: "1-10 Nov", 32: "11-20 Nov", 33: "21-30 Nov",
}

# Suitability labels (proposal objective 4). DELAY_FACTOR scales min_cycle_mm
# for the "delay planting" cut-off.
LABELS = ["suitable", "risky", "delay"]
DELAY_FACTOR = 0.75

# Interpretable risk score weights (proposal section 3.5):
# 0.5 x rainfall-deficit + 0.3 x dry-spell + 0.2 x temperature-stress.
RISK_WEIGHTS = {"rain_deficit": 0.5, "dry_spell": 0.3, "temp_stress": 0.2}

RISK_BANDS = [(0.30, "low"), (0.60, "medium"), (1.01, "high")]


def risk_band(score: float) -> str:
    for ceiling, band in RISK_BANDS:
        if score <= ceiling:
            return band
    return "high"
