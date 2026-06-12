"""
Recommendation layer: translate model output + risk features into the
deployed response promised by the proposal (objective 6): recommended crop,
planting window, risk label, risk probability, confidence, explanation.
Outputs are decision support, not guaranteed outcomes (ethics, section 3.12).
"""
import numpy as np

from crops import DEKAD_LABEL, LABELS, risk_band

DISCLAIMER = ("Decision support only - not a guaranteed farming outcome. "
              "Confirm with your cooperative extension officer or RAB.")


def explain(crop: str, window: int, label: str, risk: dict) -> str:
    win = DEKAD_LABEL.get(window, f"dekad {window}")
    pr = int(round(risk["p_rain_sufficient"] * 100))
    pd_ = int(round(risk["p_dry_spell"] * 100))
    pt = int(round(risk["p_temp_stress"] * 100))
    head = {
        "suitable": f"Planting {crop} in {win} looks suitable.",
        "risky": f"Planting {crop} in {win} is possible but risky.",
        "delay": f"Delay planting {crop}: {win} looks unfavourable.",
    }[label]
    return (f"{head} Simulations give a {pr}% chance of sufficient rainfall "
            f"over the {crop} cycle (median {risk['sim_cycle_rain_median']:.0f} mm), "
            f"a {pd_}% chance of a harmful early dry spell, and a {pt}% chance "
            f"of unusual heat. Overall risk score "
            f"{risk['risk_score']:.2f} ({risk_band(risk['risk_score'])} risk).")


def package(crop: str, window: int, proba: np.ndarray, risk: dict) -> dict:
    """One crop-window option, fully described."""
    label = LABELS[int(np.argmax(proba))]
    return {
        "crop": crop,
        "planting_window": DEKAD_LABEL.get(window, f"dekad {window}"),
        "window_start_dekad": window,
        "risk_label": label,
        "class_probabilities": {l: round(float(p), 3)
                                for l, p in zip(LABELS, proba)},
        "confidence": round(float(np.max(proba)), 3),
        "risk_score": risk["risk_score"],
        "risk_level": risk_band(risk["risk_score"]),
        "p_rain_sufficient": risk["p_rain_sufficient"],
        "p_dry_spell": risk["p_dry_spell"],
        "p_temp_stress": risk["p_temp_stress"],
        "simulated_cycle_rain_mm": {
            "p10": risk["sim_cycle_rain_p10"],
            "median": risk["sim_cycle_rain_median"],
            "p90": risk["sim_cycle_rain_p90"]},
        "explanation": explain(crop, window, label, risk),
    }


def pick_best(options: list[dict]) -> dict:
    """Rank options: prefer 'suitable' with highest confidence, then lowest
    risk score."""
    order = {l: i for i, l in enumerate(LABELS)}
    ranked = sorted(options, key=lambda o: (order[o["risk_label"]],
                                            o["risk_score"],
                                            -o["confidence"]))
    best = ranked[0]
    alternatives = ranked[1:4]
    return {"recommendation": best,
            "alternatives": alternatives,
            "disclaimer": DISCLAIMER}
