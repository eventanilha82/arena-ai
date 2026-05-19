from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arena_ai.worldcup_model import MatchDrivers, WorldCupModel  # noqa: E402


def main() -> None:
    model = WorldCupModel()
    package = model.package
    models = package.get("models", {})
    if "draw_xgb" in models:
        raise AssertionError("draw_xgb must not be present in the active model package")
    policy = package.get("simulation_policy", {})
    classifier_weight = float(policy.get("classifier_weight", -1.0))
    poisson_weight = float(policy.get("poisson_weight", 1.0 - classifier_weight))
    draw_floor = float(policy.get("draw_floor", -1.0))
    draw_ceiling = float(policy.get("draw_ceiling", -1.0))
    if not 0.50 <= classifier_weight <= 0.88:
        raise AssertionError(f"classifier_weight outside accepted calibrated range: {classifier_weight}")
    if poisson_weight < 0.12:
        raise AssertionError(f"poisson_weight too low for football variance: {poisson_weight}")
    if not 0.02 <= draw_floor <= 0.16 or not 0.30 <= draw_ceiling <= 0.50 or draw_floor >= draw_ceiling:
        raise AssertionError(f"draw guardrails outside accepted calibrated range: {draw_floor}-{draw_ceiling}")
    selected_by = str(policy.get("selected_by", ""))
    if selected_by != "strict_nested_temporal_component_ablation_no_leakage_no_draw_xgb":
        raise AssertionError(f"unexpected policy selector: {selected_by}")
    weights = package.get("manual_blend_weights", {})
    if not weights or abs(sum(float(value) for value in weights.values()) - 1.0) > 0.001:
        raise AssertionError(f"invalid manual blend weights: {weights}")
    teams = model.profiles()
    if len(teams) < 2:
        raise AssertionError("model smoke needs at least two teams")
    home = next((team for team in teams if team.code == "BRA"), teams[0])
    away = next((team for team in teams if team.code == "FRA" and team.key != home.key), teams[1])
    analysis = model.analyze_match(home, away, seed=2026)
    if not isinstance(analysis.drivers, MatchDrivers):
        raise AssertionError(f"match drivers must use the typed contract, got {type(analysis.drivers)!r}")
    driver_values = [driver.value for driver in analysis.drivers.rows()]
    if len(driver_values) != 9 or any(not math.isfinite(value) for value in driver_values):
        raise AssertionError(f"invalid typed driver values: {analysis.drivers!r}")
    pred = analysis.prediction
    probabilities = (pred.home, pred.draw, pred.away)
    if any((not math.isfinite(value)) or value < 0.0 or value > 1.0 for value in probabilities):
        raise AssertionError(f"invalid 1X2 probabilities: {probabilities}")
    if abs(sum(probabilities) - 1.0) > 0.02:
        raise AssertionError(f"1X2 probabilities do not sum to 1: {probabilities}")
    if not pred.top_scores:
        raise AssertionError("prediction did not produce scoreline candidates")
    if pred.score_home is None or pred.score_away is None:
        raise AssertionError("hybrid score was not sampled")
    if pred.score_probability <= 0.0 or pred.outcome_probability <= 0.0:
        raise AssertionError("hybrid score metadata is invalid")
    print(
        "smoke ok: "
        f"{home.code} x {away.code} "
        f"1X2=({pred.home:.1%}, {pred.draw:.1%}, {pred.away:.1%}) "
        f"placar={pred.score_home}x{pred.score_away}"
    )


if __name__ == "__main__":
    main()
