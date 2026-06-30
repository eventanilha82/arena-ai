# Auditoria estatística SOTA/KISS

Este relatório executa os checks estatísticos ativos e a ablação completa dos subconjuntos do blend.

## Model Card Resumido

Esta seção preserva no relatório canônico o resumo que antes ficava em `model_card.md`.

```json
{
  "version": "worldcup_2026_sota_v4",
  "simulation_policy": {
    "name": "hybrid_classifier_poisson",
    "classifier_weight": 0.88,
    "poisson_weight": 0.12,
    "draw_floor": 0.04,
    "draw_ceiling": 0.3,
    "score_engine": "Poisson/Dixon-Coles",
    "selected_by": "strict_nested_temporal_component_ablation_no_leakage_no_draw_xgb",
    "manual_blend_weights": {
      "xgb": 0.606061,
      "competitive": 0.272727,
      "logistic": 0.121212,
      "elo": 0.0,
      "poisson": 0.0,
      "count_poisson": 0.0
    },
    "backtest_rows": 3499,
    "candidate_count": 690,
    "backtest_metrics": {
      "classifier_weight": 0.88,
      "poisson_weight": 0.12,
      "draw_floor": 0.04,
      "draw_ceiling": 0.3,
      "objective": 0.971502,
      "log_loss": 0.882776,
      "rps": 0.171089,
      "brier": 0.522544,
      "ece": 0.038787,
      "draw_recall": 0.0,
      "draw_expected_rate": 0.229762,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.019452,
      "entropy": 0.788103,
      "randomness_penalty": 0.0
    },
    "holdout_best_metrics": {
      "classifier_weight": 0.88,
      "poisson_weight": 0.12,
      "draw_floor": 0.1,
      "draw_ceiling": 0.34,
      "objective": 0.96796,
      "log_loss": 0.883131,
      "rps": 0.171092,
      "brier": 0.522553,
      "ece": 0.034096,
      "draw_recall": 0.001147,
      "draw_expected_rate": 0.23815,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.011064,
      "entropy": 0.795813,
      "randomness_penalty": 0.0
    },
    "reference_metrics": {
      "reference_policy_0_62": {
        "classifier_weight": 0.62,
        "poisson_weight": 0.38,
        "draw_floor": 0.08,
        "draw_ceiling": 0.38,
        "objective": 0.970098,
        "log_loss": 0.883739,
        "rps": 0.171364,
        "brier": 0.523101,
        "ece": 0.028849,
        "draw_recall": 0.0,
        "draw_expected_rate": 0.23377,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.015444,
        "entropy": 0.792349,
        "randomness_penalty": 0.0
      },
      "reference_policy_previous_0_80": {
        "classifier_weight": 0.8,
        "poisson_weight": 0.2,
        "draw_floor": 0.08,
        "draw_ceiling": 0.38,
        "objective": 0.968493,
        "log_loss": 0.882948,
        "rps": 0.171134,
        "brier": 0.522602,
        "ece": 0.032148,
        "draw_recall": 0.004587,
        "draw_expected_rate": 0.236113,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.013101,
        "entropy": 0.792927,
        "randomness_penalty": 0.0
      }
    },
    "candidate_metrics": [
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.1,
        "draw_ceiling": 0.34,
        "objective": 0.96796,
        "log_loss": 0.883131,
        "rps": 0.171092,
        "brier": 0.522553,
        "ece": 0.034096,
        "draw_recall": 0.001147,
        "draw_expected_rate": 0.23815,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.011064,
        "entropy": 0.795813,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.08,
        "draw_ceiling": 0.34,
        "objective": 0.967962,
        "log_loss": 0.882678,
        "rps": 0.171058,
        "brier": 0.522412,
        "ece": 0.032911,
        "draw_recall": 0.001147,
        "draw_expected_rate": 0.236802,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.012412,
        "entropy": 0.793038,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.967963,
        "log_loss": 0.883219,
        "rps": 0.171098,
        "brier": 0.522607,
        "ece": 0.034827,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.238505,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.010709,
        "entropy": 0.795809,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.1,
        "draw_ceiling": 0.46,
        "objective": 0.967963,
        "log_loss": 0.883219,
        "rps": 0.171098,
        "brier": 0.522607,
        "ece": 0.034827,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.238505,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.010709,
        "entropy": 0.795809,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.08,
        "draw_ceiling": 0.42,
        "objective": 0.967965,
        "log_loss": 0.882767,
        "rps": 0.171064,
        "brier": 0.522465,
        "ece": 0.033642,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.237157,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.012057,
        "entropy": 0.793034,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.967965,
        "log_loss": 0.882767,
        "rps": 0.171064,
        "brier": 0.522465,
        "ece": 0.033642,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.237157,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.012057,
        "entropy": 0.793034,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.1,
        "draw_ceiling": 0.38,
        "objective": 0.967969,
        "log_loss": 0.883224,
        "rps": 0.171099,
        "brier": 0.52261,
        "ece": 0.034826,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.238503,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.010711,
        "entropy": 0.795809,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.08,
        "draw_ceiling": 0.38,
        "objective": 0.96797,
        "log_loss": 0.882771,
        "rps": 0.171065,
        "brier": 0.522469,
        "ece": 0.033641,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.237155,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.012059,
        "entropy": 0.793034,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.968041,
        "log_loss": 0.883235,
        "rps": 0.171112,
        "brier": 0.522628,
        "ece": 0.034009,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.238214,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.011,
        "entropy": 0.795738,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.1,
        "draw_ceiling": 0.46,
        "objective": 0.968041,
        "log_loss": 0.883235,
        "rps": 0.171112,
        "brier": 0.522628,
        "ece": 0.034009,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.238214,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.011,
        "entropy": 0.795738,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.1,
        "draw_ceiling": 0.38,
        "objective": 0.968047,
        "log_loss": 0.88324,
        "rps": 0.171113,
        "brier": 0.522632,
        "ece": 0.034008,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.238212,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.011002,
        "entropy": 0.795738,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.1,
        "draw_ceiling": 0.34,
        "objective": 0.968062,
        "log_loss": 0.88315,
        "rps": 0.171106,
        "brier": 0.522577,
        "ece": 0.033567,
        "draw_recall": 0.0,
        "draw_expected_rate": 0.237867,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.011347,
        "entropy": 0.79574,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.08,
        "draw_ceiling": 0.42,
        "objective": 0.968079,
        "log_loss": 0.882801,
        "rps": 0.17108,
        "brier": 0.522494,
        "ece": 0.033199,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.236896,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.012318,
        "entropy": 0.793016,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.968079,
        "log_loss": 0.882801,
        "rps": 0.17108,
        "brier": 0.522494,
        "ece": 0.033199,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.236896,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.012318,
        "entropy": 0.793016,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.08,
        "draw_ceiling": 0.38,
        "objective": 0.968084,
        "log_loss": 0.882806,
        "rps": 0.171081,
        "brier": 0.522497,
        "ece": 0.033198,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.236895,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.012319,
        "entropy": 0.793016,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.08,
        "draw_ceiling": 0.34,
        "objective": 0.9681,
        "log_loss": 0.882716,
        "rps": 0.171073,
        "brier": 0.522443,
        "ece": 0.032756,
        "draw_recall": 0.0,
        "draw_expected_rate": 0.23655,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.012664,
        "entropy": 0.793018,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.968103,
        "log_loss": 0.883258,
        "rps": 0.171127,
        "brier": 0.522654,
        "ece": 0.032902,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.237923,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.011291,
        "entropy": 0.79566,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.1,
        "draw_ceiling": 0.46,
        "objective": 0.968103,
        "log_loss": 0.883258,
        "rps": 0.171127,
        "brier": 0.522654,
        "ece": 0.032902,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.237923,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.011291,
        "entropy": 0.79566,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.1,
        "draw_ceiling": 0.38,
        "objective": 0.968109,
        "log_loss": 0.883263,
        "rps": 0.171128,
        "brier": 0.522657,
        "ece": 0.0329,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.237921,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.011293,
        "entropy": 0.79566,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.08,
        "draw_ceiling": 0.42,
        "objective": 0.968141,
        "log_loss": 0.882842,
        "rps": 0.171096,
        "brier": 0.522525,
        "ece": 0.032009,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.236636,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.012578,
        "entropy": 0.792992,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.968141,
        "log_loss": 0.882842,
        "rps": 0.171096,
        "brier": 0.522525,
        "ece": 0.032009,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.236636,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.012578,
        "entropy": 0.792992,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.08,
        "draw_ceiling": 0.38,
        "objective": 0.968146,
        "log_loss": 0.882846,
        "rps": 0.171097,
        "brier": 0.522529,
        "ece": 0.032008,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.236634,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.01258,
        "entropy": 0.792992,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.06,
        "draw_ceiling": 0.34,
        "objective": 0.968163,
        "log_loss": 0.882503,
        "rps": 0.171049,
        "brier": 0.522369,
        "ece": 0.032253,
        "draw_recall": 0.001147,
        "draw_expected_rate": 0.23577,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.013444,
        "entropy": 0.790647,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.06,
        "draw_ceiling": 0.42,
        "objective": 0.968166,
        "log_loss": 0.882592,
        "rps": 0.171055,
        "brier": 0.522422,
        "ece": 0.032984,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.236124,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.01309,
        "entropy": 0.790644,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.06,
        "draw_ceiling": 0.46,
        "objective": 0.968166,
        "log_loss": 0.882592,
        "rps": 0.171055,
        "brier": 0.522422,
        "ece": 0.032984,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.236124,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.01309,
        "entropy": 0.790644,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.06,
        "draw_ceiling": 0.38,
        "objective": 0.968171,
        "log_loss": 0.882596,
        "rps": 0.171056,
        "brier": 0.522426,
        "ece": 0.032983,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.236123,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.013091,
        "entropy": 0.790644,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.1,
        "draw_ceiling": 0.34,
        "objective": 0.968171,
        "log_loss": 0.883177,
        "rps": 0.171121,
        "brier": 0.522605,
        "ece": 0.033034,
        "draw_recall": 0.0,
        "draw_expected_rate": 0.237584,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.01163,
        "entropy": 0.795661,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.08,
        "draw_ceiling": 0.34,
        "objective": 0.968208,
        "log_loss": 0.882761,
        "rps": 0.17109,
        "brier": 0.522477,
        "ece": 0.032142,
        "draw_recall": 0.0,
        "draw_expected_rate": 0.236297,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.012917,
        "entropy": 0.792992,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.82,
        "poisson_weight": 0.18,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.968251,
        "log_loss": 0.883288,
        "rps": 0.171144,
        "brier": 0.522683,
        "ece": 0.032767,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.237632,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.011582,
        "entropy": 0.795577,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.82,
        "poisson_weight": 0.18,
        "draw_floor": 0.1,
        "draw_ceiling": 0.46,
        "objective": 0.968251,
        "log_loss": 0.883288,
        "rps": 0.171144,
        "brier": 0.522683,
        "ece": 0.032767,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.237632,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.011582,
        "entropy": 0.795577,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.06,
        "draw_ceiling": 0.42,
        "objective": 0.968256,
        "log_loss": 0.882637,
        "rps": 0.171071,
        "brier": 0.522453,
        "ece": 0.032222,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.235887,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.013327,
        "entropy": 0.790674,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.06,
        "draw_ceiling": 0.46,
        "objective": 0.968256,
        "log_loss": 0.882637,
        "rps": 0.171071,
        "brier": 0.522453,
        "ece": 0.032222,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.235887,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.013327,
        "entropy": 0.790674,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.82,
        "poisson_weight": 0.18,
        "draw_floor": 0.1,
        "draw_ceiling": 0.38,
        "objective": 0.968256,
        "log_loss": 0.883292,
        "rps": 0.171144,
        "brier": 0.522686,
        "ece": 0.032766,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.23763,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.011584,
        "entropy": 0.795576,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.06,
        "draw_ceiling": 0.38,
        "objective": 0.968262,
        "log_loss": 0.882642,
        "rps": 0.171072,
        "brier": 0.522457,
        "ece": 0.032221,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.235886,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.013328,
        "entropy": 0.790674,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.06,
        "draw_ceiling": 0.34,
        "objective": 0.968277,
        "log_loss": 0.882552,
        "rps": 0.171065,
        "brier": 0.522402,
        "ece": 0.031779,
        "draw_recall": 0.0,
        "draw_expected_rate": 0.235541,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.013673,
        "entropy": 0.790676,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.82,
        "poisson_weight": 0.18,
        "draw_floor": 0.08,
        "draw_ceiling": 0.42,
        "objective": 0.968297,
        "log_loss": 0.88289,
        "rps": 0.171114,
        "brier": 0.522561,
        "ece": 0.031907,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.236375,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.012839,
        "entropy": 0.792962,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.82,
        "poisson_weight": 0.18,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.968297,
        "log_loss": 0.88289,
        "rps": 0.171114,
        "brier": 0.522561,
        "ece": 0.031907,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.236375,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.012839,
        "entropy": 0.792962,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.82,
        "poisson_weight": 0.18,
        "draw_floor": 0.08,
        "draw_ceiling": 0.38,
        "objective": 0.968302,
        "log_loss": 0.882894,
        "rps": 0.171115,
        "brier": 0.522564,
        "ece": 0.031905,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.236374,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.01284,
        "entropy": 0.792962,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.06,
        "draw_ceiling": 0.42,
        "objective": 0.968322,
        "log_loss": 0.882689,
        "rps": 0.171088,
        "brier": 0.522488,
        "ece": 0.031055,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.23565,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.013564,
        "entropy": 0.790699,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.06,
        "draw_ceiling": 0.46,
        "objective": 0.968322,
        "log_loss": 0.882689,
        "rps": 0.171088,
        "brier": 0.522488,
        "ece": 0.031055,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.23565,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.013564,
        "entropy": 0.790699,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.06,
        "draw_ceiling": 0.38,
        "objective": 0.968327,
        "log_loss": 0.882694,
        "rps": 0.171089,
        "brier": 0.522491,
        "ece": 0.031054,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.235649,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.013565,
        "entropy": 0.790699,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.12,
        "draw_ceiling": 0.34,
        "objective": 0.968332,
        "log_loss": 0.88414,
        "rps": 0.17118,
        "brier": 0.522922,
        "ece": 0.035431,
        "draw_recall": 0.001147,
        "draw_expected_rate": 0.240028,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.009186,
        "entropy": 0.799259,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.12,
        "draw_ceiling": 0.42,
        "objective": 0.968335,
        "log_loss": 0.884228,
        "rps": 0.171187,
        "brier": 0.522975,
        "ece": 0.036163,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.240382,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.008832,
        "entropy": 0.799256,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.12,
        "draw_ceiling": 0.46,
        "objective": 0.968335,
        "log_loss": 0.884228,
        "rps": 0.171187,
        "brier": 0.522975,
        "ece": 0.036163,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.240382,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.008832,
        "entropy": 0.799256,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.12,
        "draw_ceiling": 0.38,
        "objective": 0.968341,
        "log_loss": 0.884233,
        "rps": 0.171188,
        "brier": 0.522979,
        "ece": 0.036161,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.240381,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.008833,
        "entropy": 0.799256,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.82,
        "poisson_weight": 0.18,
        "draw_floor": 0.1,
        "draw_ceiling": 0.34,
        "objective": 0.968342,
        "log_loss": 0.88321,
        "rps": 0.171138,
        "brier": 0.522636,
        "ece": 0.033188,
        "draw_recall": 0.0,
        "draw_expected_rate": 0.237301,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.011913,
        "entropy": 0.795575,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.82,
        "poisson_weight": 0.18,
        "draw_floor": 0.08,
        "draw_ceiling": 0.34,
        "objective": 0.968387,
        "log_loss": 0.882812,
        "rps": 0.171109,
        "brier": 0.522514,
        "ece": 0.032327,
        "draw_recall": 0.0,
        "draw_expected_rate": 0.236045,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.013169,
        "entropy": 0.792961,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.06,
        "draw_ceiling": 0.34,
        "objective": 0.968389,
        "log_loss": 0.882608,
        "rps": 0.171082,
        "brier": 0.522439,
        "ece": 0.031188,
        "draw_recall": 0.0,
        "draw_expected_rate": 0.235312,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.013902,
        "entropy": 0.790699,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.12,
        "draw_ceiling": 0.42,
        "objective": 0.968417,
        "log_loss": 0.884212,
        "rps": 0.171197,
        "brier": 0.522983,
        "ece": 0.035586,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.240049,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.009165,
        "entropy": 0.79912,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.12,
        "draw_ceiling": 0.46,
        "objective": 0.968417,
        "log_loss": 0.884212,
        "rps": 0.171197,
        "brier": 0.522983,
        "ece": 0.035586,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.240049,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.009165,
        "entropy": 0.79912,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.12,
        "draw_ceiling": 0.38,
        "objective": 0.968422,
        "log_loss": 0.884217,
        "rps": 0.171198,
        "brier": 0.522987,
        "ece": 0.035585,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.240047,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.009167,
        "entropy": 0.79912,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.12,
        "draw_ceiling": 0.34,
        "objective": 0.968438,
        "log_loss": 0.884127,
        "rps": 0.171191,
        "brier": 0.522932,
        "ece": 0.035143,
        "draw_recall": 0.0,
        "draw_expected_rate": 0.239702,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.009512,
        "entropy": 0.799122,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.8,
        "poisson_weight": 0.2,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.968461,
        "log_loss": 0.883324,
        "rps": 0.171161,
        "brier": 0.522715,
        "ece": 0.033322,
        "draw_recall": 0.004587,
        "draw_expected_rate": 0.237341,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.011874,
        "entropy": 0.795487,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.8,
        "poisson_weight": 0.2,
        "draw_floor": 0.1,
        "draw_ceiling": 0.46,
        "objective": 0.968461,
        "log_loss": 0.883324,
        "rps": 0.171161,
        "brier": 0.522715,
        "ece": 0.033322,
        "draw_recall": 0.004587,
        "draw_expected_rate": 0.237341,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.011874,
        "entropy": 0.795487,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.8,
        "poisson_weight": 0.2,
        "draw_floor": 0.1,
        "draw_ceiling": 0.38,
        "objective": 0.968466,
        "log_loss": 0.883328,
        "rps": 0.171162,
        "brier": 0.522719,
        "ece": 0.033321,
        "draw_recall": 0.004587,
        "draw_expected_rate": 0.237339,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.011875,
        "entropy": 0.795487,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.12,
        "draw_ceiling": 0.42,
        "objective": 0.968469,
        "log_loss": 0.884203,
        "rps": 0.171209,
        "brier": 0.522995,
        "ece": 0.034552,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.239715,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.009499,
        "entropy": 0.798978,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.12,
        "draw_ceiling": 0.46,
        "objective": 0.968469,
        "log_loss": 0.884203,
        "rps": 0.171209,
        "brier": 0.522995,
        "ece": 0.034552,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.239715,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.009499,
        "entropy": 0.798978,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.12,
        "draw_ceiling": 0.38,
        "objective": 0.968475,
        "log_loss": 0.884208,
        "rps": 0.17121,
        "brier": 0.522998,
        "ece": 0.034551,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.239713,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.009501,
        "entropy": 0.798977,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.82,
        "poisson_weight": 0.18,
        "draw_floor": 0.06,
        "draw_ceiling": 0.42,
        "objective": 0.968481,
        "log_loss": 0.882748,
        "rps": 0.171107,
        "brier": 0.522525,
        "ece": 0.030975,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.235413,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.013801,
        "entropy": 0.790718,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.82,
        "poisson_weight": 0.18,
        "draw_floor": 0.06,
        "draw_ceiling": 0.46,
        "objective": 0.968481,
        "log_loss": 0.882748,
        "rps": 0.171107,
        "brier": 0.522525,
        "ece": 0.030975,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.235413,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.013801,
        "entropy": 0.790718,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.82,
        "poisson_weight": 0.18,
        "draw_floor": 0.06,
        "draw_ceiling": 0.38,
        "objective": 0.968486,
        "log_loss": 0.882752,
        "rps": 0.171108,
        "brier": 0.522529,
        "ece": 0.030974,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.235412,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.013802,
        "entropy": 0.790718,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.8,
        "poisson_weight": 0.2,
        "draw_floor": 0.08,
        "draw_ceiling": 0.42,
        "objective": 0.968487,
        "log_loss": 0.882944,
        "rps": 0.171133,
        "brier": 0.522599,
        "ece": 0.032149,
        "draw_recall": 0.004587,
        "draw_expected_rate": 0.236115,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.013099,
        "entropy": 0.792927,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.8,
        "poisson_weight": 0.2,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.968487,
        "log_loss": 0.882944,
        "rps": 0.171133,
        "brier": 0.522599,
        "ece": 0.032149,
        "draw_recall": 0.004587,
        "draw_expected_rate": 0.236115,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.013099,
        "entropy": 0.792927,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.8,
        "poisson_weight": 0.2,
        "draw_floor": 0.08,
        "draw_ceiling": 0.38,
        "objective": 0.968493,
        "log_loss": 0.882948,
        "rps": 0.171134,
        "brier": 0.522602,
        "ece": 0.032148,
        "draw_recall": 0.004587,
        "draw_expected_rate": 0.236113,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.013101,
        "entropy": 0.792927,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.8,
        "poisson_weight": 0.2,
        "draw_floor": 0.1,
        "draw_ceiling": 0.34,
        "objective": 0.968506,
        "log_loss": 0.88325,
        "rps": 0.171156,
        "brier": 0.522671,
        "ece": 0.033173,
        "draw_recall": 0.0,
        "draw_expected_rate": 0.237018,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.012196,
        "entropy": 0.795484,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.8,
        "poisson_weight": 0.2,
        "draw_floor": 0.08,
        "draw_ceiling": 0.34,
        "objective": 0.968533,
        "log_loss": 0.88287,
        "rps": 0.171128,
        "brier": 0.522555,
        "ece": 0.032,
        "draw_recall": 0.0,
        "draw_expected_rate": 0.235793,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.013421,
        "entropy": 0.792925,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.12,
        "draw_ceiling": 0.34,
        "objective": 0.968537,
        "log_loss": 0.884122,
        "rps": 0.171203,
        "brier": 0.522946,
        "ece": 0.034685,
        "draw_recall": 0.0,
        "draw_expected_rate": 0.239377,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.009837,
        "entropy": 0.798978,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.04,
        "draw_ceiling": 0.34,
        "objective": 0.968547,
        "log_loss": 0.882562,
        "rps": 0.17105,
        "brier": 0.522376,
        "ece": 0.032526,
        "draw_recall": 0.001147,
        "draw_expected_rate": 0.235049,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.014165,
        "entropy": 0.78875,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.04,
        "draw_ceiling": 0.42,
        "objective": 0.96855,
        "log_loss": 0.88265,
        "rps": 0.171057,
        "brier": 0.52243,
        "ece": 0.033258,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.235404,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.01381,
        "entropy": 0.788746,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.04,
        "draw_ceiling": 0.46,
        "objective": 0.96855,
        "log_loss": 0.88265,
        "rps": 0.171057,
        "brier": 0.52243,
        "ece": 0.033258,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.235404,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.01381,
        "entropy": 0.788746,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.04,
        "draw_ceiling": 0.38,
        "objective": 0.968556,
        "log_loss": 0.882655,
        "rps": 0.171058,
        "brier": 0.522433,
        "ece": 0.033257,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.235402,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.013812,
        "entropy": 0.788746,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.82,
        "poisson_weight": 0.18,
        "draw_floor": 0.06,
        "draw_ceiling": 0.34,
        "objective": 0.968571,
        "log_loss": 0.88267,
        "rps": 0.171101,
        "brier": 0.522479,
        "ece": 0.031396,
        "draw_recall": 0.0,
        "draw_expected_rate": 0.235083,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.014131,
        "entropy": 0.790717,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.82,
        "poisson_weight": 0.18,
        "draw_floor": 0.12,
        "draw_ceiling": 0.42,
        "objective": 0.968598,
        "log_loss": 0.884201,
        "rps": 0.171222,
        "brier": 0.52301,
        "ece": 0.034379,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.239381,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.009833,
        "entropy": 0.798829,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.82,
        "poisson_weight": 0.18,
        "draw_floor": 0.12,
        "draw_ceiling": 0.46,
        "objective": 0.968598,
        "log_loss": 0.884201,
        "rps": 0.171222,
        "brier": 0.52301,
        "ece": 0.034379,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.239381,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.009833,
        "entropy": 0.798829,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.82,
        "poisson_weight": 0.18,
        "draw_floor": 0.12,
        "draw_ceiling": 0.38,
        "objective": 0.968604,
        "log_loss": 0.884205,
        "rps": 0.171223,
        "brier": 0.523014,
        "ece": 0.034378,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.23938,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.009834,
        "entropy": 0.798828,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.04,
        "draw_ceiling": 0.42,
        "objective": 0.968631,
        "log_loss": 0.882701,
        "rps": 0.171073,
        "brier": 0.522461,
        "ece": 0.032401,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.235183,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.014031,
        "entropy": 0.788817,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.04,
        "draw_ceiling": 0.46,
        "objective": 0.968631,
        "log_loss": 0.882701,
        "rps": 0.171073,
        "brier": 0.522461,
        "ece": 0.032401,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.235183,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.014031,
        "entropy": 0.788817,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.04,
        "draw_ceiling": 0.38,
        "objective": 0.968637,
        "log_loss": 0.882705,
        "rps": 0.171074,
        "brier": 0.522464,
        "ece": 0.0324,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.235181,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.014033,
        "entropy": 0.788817,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.04,
        "draw_ceiling": 0.34,
        "objective": 0.968652,
        "log_loss": 0.882616,
        "rps": 0.171067,
        "brier": 0.52241,
        "ece": 0.031959,
        "draw_recall": 0.0,
        "draw_expected_rate": 0.234837,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.014378,
        "entropy": 0.788819,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.78,
        "poisson_weight": 0.22,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.96866,
        "log_loss": 0.883367,
        "rps": 0.17118,
        "brier": 0.522751,
        "ece": 0.033655,
        "draw_recall": 0.00344,
        "draw_expected_rate": 0.237049,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.012165,
        "entropy": 0.795391,
        "randomness_penalty": 0.0
      }
    ],
    "best_by_classifier_weight": [
      {
        "classifier_weight": 0.5,
        "poisson_weight": 0.5,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.971508,
        "log_loss": 0.88472,
        "rps": 0.171584,
        "brier": 0.523628,
        "ece": 0.028891,
        "draw_recall": 0.0,
        "draw_expected_rate": 0.232974,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.01624,
        "entropy": 0.793404,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.52,
        "poisson_weight": 0.48,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.971246,
        "log_loss": 0.884575,
        "rps": 0.171546,
        "brier": 0.523542,
        "ece": 0.029144,
        "draw_recall": 0.0,
        "draw_expected_rate": 0.233266,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.015949,
        "entropy": 0.793587,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.54,
        "poisson_weight": 0.46,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.970914,
        "log_loss": 0.884437,
        "rps": 0.17151,
        "brier": 0.52346,
        "ece": 0.028425,
        "draw_recall": 0.0,
        "draw_expected_rate": 0.233557,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.015657,
        "entropy": 0.793763,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.56,
        "poisson_weight": 0.44,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.970703,
        "log_loss": 0.884308,
        "rps": 0.171476,
        "brier": 0.523382,
        "ece": 0.02912,
        "draw_recall": 0.0,
        "draw_expected_rate": 0.233848,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.015366,
        "entropy": 0.793933,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.58,
        "poisson_weight": 0.42,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.97053,
        "log_loss": 0.884186,
        "rps": 0.171442,
        "brier": 0.523307,
        "ece": 0.030176,
        "draw_recall": 0.0,
        "draw_expected_rate": 0.234139,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.015075,
        "entropy": 0.794096,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.6,
        "poisson_weight": 0.4,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.970167,
        "log_loss": 0.884071,
        "rps": 0.17141,
        "brier": 0.523235,
        "ece": 0.02877,
        "draw_recall": 0.0,
        "draw_expected_rate": 0.23443,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.014784,
        "entropy": 0.794254,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.62,
        "poisson_weight": 0.38,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.970005,
        "log_loss": 0.883965,
        "rps": 0.171379,
        "brier": 0.523167,
        "ece": 0.029764,
        "draw_recall": 0.0,
        "draw_expected_rate": 0.234721,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.014493,
        "entropy": 0.794405,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.64,
        "poisson_weight": 0.36,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.969841,
        "log_loss": 0.883865,
        "rps": 0.17135,
        "brier": 0.523103,
        "ece": 0.030639,
        "draw_recall": 0.0,
        "draw_expected_rate": 0.235012,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.014202,
        "entropy": 0.79455,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.66,
        "poisson_weight": 0.34,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.969648,
        "log_loss": 0.883773,
        "rps": 0.171322,
        "brier": 0.523042,
        "ece": 0.031054,
        "draw_recall": 0.0,
        "draw_expected_rate": 0.235303,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.013911,
        "entropy": 0.794688,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.68,
        "poisson_weight": 0.32,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.969413,
        "log_loss": 0.883687,
        "rps": 0.171295,
        "brier": 0.522985,
        "ece": 0.030844,
        "draw_recall": 0.0,
        "draw_expected_rate": 0.235594,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.01362,
        "entropy": 0.794821,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.7,
        "poisson_weight": 0.3,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.969231,
        "log_loss": 0.883609,
        "rps": 0.171269,
        "brier": 0.522931,
        "ece": 0.031205,
        "draw_recall": 0.001147,
        "draw_expected_rate": 0.235885,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.013329,
        "entropy": 0.794947,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.72,
        "poisson_weight": 0.28,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.968988,
        "log_loss": 0.883538,
        "rps": 0.171245,
        "brier": 0.522881,
        "ece": 0.030707,
        "draw_recall": 0.001147,
        "draw_expected_rate": 0.236176,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.013038,
        "entropy": 0.795067,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.74,
        "poisson_weight": 0.26,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.968867,
        "log_loss": 0.883474,
        "rps": 0.171222,
        "brier": 0.522834,
        "ece": 0.031634,
        "draw_recall": 0.002294,
        "draw_expected_rate": 0.236467,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.012747,
        "entropy": 0.795181,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.76,
        "poisson_weight": 0.24,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.968808,
        "log_loss": 0.883417,
        "rps": 0.1712,
        "brier": 0.522791,
        "ece": 0.033243,
        "draw_recall": 0.00344,
        "draw_expected_rate": 0.236758,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.012456,
        "entropy": 0.795289,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.78,
        "poisson_weight": 0.22,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.96866,
        "log_loss": 0.883367,
        "rps": 0.17118,
        "brier": 0.522751,
        "ece": 0.033655,
        "draw_recall": 0.00344,
        "draw_expected_rate": 0.237049,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.012165,
        "entropy": 0.795391,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.8,
        "poisson_weight": 0.2,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.968461,
        "log_loss": 0.883324,
        "rps": 0.171161,
        "brier": 0.522715,
        "ece": 0.033322,
        "draw_recall": 0.004587,
        "draw_expected_rate": 0.237341,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.011874,
        "entropy": 0.795487,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.82,
        "poisson_weight": 0.18,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.968251,
        "log_loss": 0.883288,
        "rps": 0.171144,
        "brier": 0.522683,
        "ece": 0.032767,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.237632,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.011582,
        "entropy": 0.795577,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.968103,
        "log_loss": 0.883258,
        "rps": 0.171127,
        "brier": 0.522654,
        "ece": 0.032902,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.237923,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.011291,
        "entropy": 0.79566,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.968041,
        "log_loss": 0.883235,
        "rps": 0.171112,
        "brier": 0.522628,
        "ece": 0.034009,
        "draw_recall": 0.005734,
        "draw_expected_rate": 0.238214,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.011,
        "entropy": 0.795738,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.1,
        "draw_ceiling": 0.34,
        "objective": 0.96796,
        "log_loss": 0.883131,
        "rps": 0.171092,
        "brier": 0.522553,
        "ece": 0.034096,
        "draw_recall": 0.001147,
        "draw_expected_rate": 0.23815,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.011064,
        "entropy": 0.795813,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.9,
        "poisson_weight": 0.1,
        "draw_floor": 0.08,
        "draw_ceiling": 0.34,
        "objective": 0.972835,
        "log_loss": 0.882646,
        "rps": 0.171043,
        "brier": 0.522384,
        "ece": 0.033121,
        "draw_recall": 0.001147,
        "draw_expected_rate": 0.237055,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.012159,
        "entropy": 0.793052,
        "randomness_penalty": 0.02
      },
      {
        "classifier_weight": 0.92,
        "poisson_weight": 0.08,
        "draw_floor": 0.08,
        "draw_ceiling": 0.34,
        "objective": 0.977697,
        "log_loss": 0.882621,
        "rps": 0.17103,
        "brier": 0.52236,
        "ece": 0.033097,
        "draw_recall": 0.002294,
        "draw_expected_rate": 0.237307,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.011907,
        "entropy": 0.79306,
        "randomness_penalty": 0.04
      },
      {
        "classifier_weight": 0.94,
        "poisson_weight": 0.06,
        "draw_floor": 0.08,
        "draw_ceiling": 0.34,
        "objective": 0.982598,
        "log_loss": 0.882603,
        "rps": 0.171018,
        "brier": 0.52234,
        "ece": 0.033477,
        "draw_recall": 0.002294,
        "draw_expected_rate": 0.237559,
        "draw_actual_rate": 0.249214,
        "draw_gap": 0.011655,
        "entropy": 0.793063,
        "randomness_penalty": 0.06
      }
    ],
    "nested_temporal_validation": {
      "version": "nested_temporal_policy_v4_orientation_invariant_no_leakage_no_draw_xgb",
      "description": "Each outer year trains inner models only before the internal validation window, using orientation-invariant neutral fixtures, selects blend components plus classifier/Poisson/draw policy on that later internal window, then refits on all prior data and evaluates the outer year without retuning.",
      "aggregate": {
        "outer_year": 2021.785531,
        "outer_rows": 10519,
        "outer_rho": -0.090343,
        "outer_objective": 0.961861,
        "outer_log_loss": 0.877499,
        "outer_rps": 0.170975,
        "outer_brier": 0.517136,
        "outer_ece": 0.035824,
        "outer_draw_recall": 0.0,
        "outer_draw_expected_rate": 0.234658,
        "outer_draw_actual_rate": 0.238806,
        "outer_draw_gap": 0.010351,
        "outer_entropy": 0.80894,
        "outer_randomness_penalty": 0.0,
        "folds": 8
      },
      "rows": [
        {
          "outer_year": 2018,
          "inner_train_start": "1990-01-12",
          "inner_train_end": "2015-12-28",
          "inner_train_rows": 28314,
          "inner_start": "2015-12-31",
          "inner_end": "2017-12-29",
          "inner_rows": 2240,
          "outer_rows": 1147,
          "inner_rho": -0.1,
          "outer_rho": -0.1,
          "selected_classifier_weight": 0.88,
          "selected_poisson_weight": 0.12,
          "selected_draw_floor": 0.04,
          "selected_draw_ceiling": 0.38,
          "inner_objective": 0.983832,
          "outer_objective": 1.016705,
          "outer_log_loss": 0.927939,
          "outer_rps": 0.182439,
          "outer_brier": 0.552334,
          "outer_ece": 0.040194,
          "outer_draw_recall": 0.0,
          "outer_draw_expected_rate": 0.252183,
          "outer_draw_actual_rate": 0.259808,
          "outer_draw_gap": 0.007625,
          "outer_entropy": 0.856035,
          "outer_randomness_penalty": 0.0
        },
        {
          "outer_year": 2019,
          "inner_train_start": "1990-01-12",
          "inner_train_end": "2016-12-30",
          "inner_train_rows": 29448,
          "inner_start": "2017-01-04",
          "inner_end": "2018-12-31",
          "inner_rows": 2253,
          "outer_rows": 1507,
          "inner_rho": -0.1,
          "outer_rho": -0.1,
          "selected_classifier_weight": 0.88,
          "selected_poisson_weight": 0.12,
          "selected_draw_floor": 0.06,
          "selected_draw_ceiling": 0.34,
          "inner_objective": 1.020075,
          "outer_objective": 0.955066,
          "outer_log_loss": 0.871391,
          "outer_rps": 0.172345,
          "outer_brier": 0.51074,
          "outer_ece": 0.031922,
          "outer_draw_recall": 0.0,
          "outer_draw_expected_rate": 0.232199,
          "outer_draw_actual_rate": 0.22296,
          "outer_draw_gap": 0.00924,
          "outer_entropy": 0.804238,
          "outer_randomness_penalty": 0.0
        },
        {
          "outer_year": 2020,
          "inner_train_start": "1990-01-12",
          "inner_train_end": "2017-12-18",
          "inner_train_rows": 30532,
          "inner_start": "2017-12-22",
          "inner_end": "2019-12-19",
          "inner_rows": 2676,
          "outer_rows": 417,
          "inner_rho": -0.1,
          "outer_rho": -0.08,
          "selected_classifier_weight": 0.68,
          "selected_poisson_weight": 0.32,
          "selected_draw_floor": 0.04,
          "selected_draw_ceiling": 0.34,
          "inner_objective": 0.976527,
          "outer_objective": 1.051511,
          "outer_log_loss": 0.959457,
          "outer_rps": 0.191995,
          "outer_brier": 0.570026,
          "outer_ece": 0.050584,
          "outer_draw_recall": 0.0,
          "outer_draw_expected_rate": 0.258739,
          "outer_draw_actual_rate": 0.254197,
          "outer_draw_gap": 0.004542,
          "outer_entropy": 0.867018,
          "outer_randomness_penalty": 0.0
        },
        {
          "outer_year": 2021,
          "inner_train_start": "1990-01-12",
          "inner_train_end": "2018-12-06",
          "inner_train_rows": 31660,
          "inner_start": "2018-12-11",
          "inner_end": "2020-12-09",
          "inner_rows": 1965,
          "outer_rows": 1504,
          "inner_rho": -0.1,
          "outer_rho": -0.08,
          "selected_classifier_weight": 0.62,
          "selected_poisson_weight": 0.38,
          "selected_draw_floor": 0.04,
          "selected_draw_ceiling": 0.3,
          "inner_objective": 0.975427,
          "outer_objective": 0.874915,
          "outer_log_loss": 0.802307,
          "outer_rps": 0.150094,
          "outer_brier": 0.467355,
          "outer_ece": 0.047045,
          "outer_draw_recall": 0.0,
          "outer_draw_expected_rate": 0.224997,
          "outer_draw_actual_rate": 0.226064,
          "outer_draw_gap": 0.001067,
          "outer_entropy": 0.783672,
          "outer_randomness_penalty": 0.0
        },
        {
          "outer_year": 2022,
          "inner_train_start": "1990-01-12",
          "inner_train_end": "2019-12-19",
          "inner_train_rows": 33208,
          "inner_start": "2020-01-07",
          "inner_end": "2021-12-31",
          "inner_rows": 1921,
          "outer_rows": 1375,
          "inner_rho": -0.08,
          "outer_rho": -0.1,
          "selected_classifier_weight": 0.5,
          "selected_poisson_weight": 0.5,
          "selected_draw_floor": 0.04,
          "selected_draw_ceiling": 0.34,
          "inner_objective": 0.911512,
          "outer_objective": 0.996787,
          "outer_log_loss": 0.911402,
          "outer_rps": 0.180558,
          "outer_brier": 0.536953,
          "outer_ece": 0.016143,
          "outer_draw_recall": 0.0,
          "outer_draw_expected_rate": 0.244419,
          "outer_draw_actual_rate": 0.237091,
          "outer_draw_gap": 0.007328,
          "outer_entropy": 0.833429,
          "outer_randomness_penalty": 0.0
        },
        {
          "outer_year": 2023,
          "inner_train_start": "1990-01-12",
          "inner_train_end": "2020-12-09",
          "inner_train_rows": 33625,
          "inner_start": "2021-01-12",
          "inner_end": "2022-12-30",
          "inner_rows": 2879,
          "outer_rows": 1358,
          "inner_rho": -0.08,
          "outer_rho": -0.08,
          "selected_classifier_weight": 0.5,
          "selected_poisson_weight": 0.5,
          "selected_draw_floor": 0.04,
          "selected_draw_ceiling": 0.3,
          "inner_objective": 0.931563,
          "outer_objective": 0.937564,
          "outer_log_loss": 0.859152,
          "outer_rps": 0.168521,
          "outer_brier": 0.50549,
          "outer_ece": 0.015851,
          "outer_draw_recall": 0.0,
          "outer_draw_expected_rate": 0.22515,
          "outer_draw_actual_rate": 0.228277,
          "outer_draw_gap": 0.003127,
          "outer_entropy": 0.786039,
          "outer_randomness_penalty": 0.0
        },
        {
          "outer_year": 2024,
          "inner_train_start": "1990-01-12",
          "inner_train_end": "2021-12-30",
          "inner_train_rows": 35127,
          "inner_start": "2021-12-31",
          "inner_end": "2023-12-31",
          "inner_rows": 2735,
          "outer_rows": 1800,
          "inner_rho": -0.1,
          "outer_rho": -0.08,
          "selected_classifier_weight": 0.54,
          "selected_poisson_weight": 0.46,
          "selected_draw_floor": 0.04,
          "selected_draw_ceiling": 0.3,
          "inner_objective": 0.966002,
          "outer_objective": 1.007185,
          "outer_log_loss": 0.910629,
          "outer_rps": 0.175209,
          "outer_brier": 0.539584,
          "outer_ece": 0.032838,
          "outer_draw_recall": 0.0,
          "outer_draw_expected_rate": 0.229078,
          "outer_draw_actual_rate": 0.263333,
          "outer_draw_gap": 0.034255,
          "outer_entropy": 0.800891,
          "outer_randomness_penalty": 0.0
        },
        {
          "outer_year": 2025,
          "inner_train_start": "1990-01-12",
          "inner_train_end": "2022-12-30",
          "inner_train_rows": 36504,
          "inner_start": "2023-01-02",
          "inner_end": "2024-12-31",
          "inner_rows": 3158,
          "outer_rows": 1411,
          "inner_rho": -0.08,
          "outer_rho": -0.1,
          "selected_classifier_weight": 0.88,
          "selected_poisson_weight": 0.12,
          "selected_draw_floor": 0.08,
          "selected_draw_ceiling": 0.34,
          "inner_objective": 0.974599,
          "outer_objective": 0.92225,
          "outer_log_loss": 0.8413,
          "outer_rps": 0.163859,
          "outer_brier": 0.496045,
          "outer_ece": 0.062326,
          "outer_draw_recall": 0.0,
          "outer_draw_expected_rate": 0.232979,
          "outer_draw_actual_rate": 0.228207,
          "outer_draw_gap": 0.004772,
          "outer_entropy": 0.793893,
          "outer_randomness_penalty": 0.0
        }
      ],
      "fold_winner_policy": {
        "classifier_weight": 0.54,
        "poisson_weight": 0.46,
        "draw_floor": 0.04,
        "draw_ceiling": 0.3,
        "selected_outer_rows": 1800,
        "selected_folds": 1,
        "selected_avg_inner_objective": 0.966002
      },
      "selected_policy": {
        "component_candidate": "subset__xgb+competitive+logistic",
        "active_components": [
          "xgb",
          "competitive",
          "logistic"
        ],
        "manual_blend_weights": {
          "xgb": 0.606061,
          "competitive": 0.272727,
          "logistic": 0.121212,
          "elo": 0.0,
          "poisson": 0.0,
          "count_poisson": 0.0
        },
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.04,
        "draw_ceiling": 0.3,
        "selected_outer_rows": 10519,
        "selected_folds": 8,
        "selected_avg_inner_objective": 0.967463,
        "selected_avg_inner_entropy": 0.807016,
        "outer_objective": 0.956044,
        "outer_log_loss": 0.876527,
        "outer_rps": 0.170828,
        "outer_brier": 0.516715,
        "outer_ece": 0.007282,
        "outer_draw_recall": 0.0,
        "outer_draw_expected_rate": 0.2344,
        "outer_draw_actual_rate": 0.238806,
        "outer_draw_gap": 0.004406,
        "outer_entropy": 0.800726,
        "outer_randomness_penalty": 0.0
      },
      "component_ablation": {
        "version": "nested_component_subset_ablation_v1",
        "candidate_count": 63,
        "policy_candidate_count_per_component": 690,
        "selection": "joint_nested_temporal_component_and_policy_grid",
        "selected_policy": {
          "component_candidate": "subset__xgb+competitive+logistic",
          "active_components": [
            "xgb",
            "competitive",
            "logistic"
          ],
          "manual_blend_weights": {
            "xgb": 0.606061,
            "competitive": 0.272727,
            "logistic": 0.121212,
            "elo": 0.0,
            "poisson": 0.0,
            "count_poisson": 0.0
          },
          "classifier_weight": 0.88,
          "poisson_weight": 0.12,
          "draw_floor": 0.04,
          "draw_ceiling": 0.3,
          "selected_outer_rows": 10519,
          "selected_folds": 8,
          "selected_avg_inner_objective": 0.967463,
          "selected_avg_inner_entropy": 0.807016,
          "outer_objective": 0.956044,
          "outer_log_loss": 0.876527,
          "outer_rps": 0.170828,
          "outer_brier": 0.516715,
          "outer_ece": 0.007282,
          "outer_draw_recall": 0.0,
          "outer_draw_expected_rate": 0.2344,
          "outer_draw_actual_rate": 0.238806,
          "outer_draw_gap": 0.004406,
          "outer_entropy": 0.800726,
          "outer_randomness_penalty": 0.0
        },
        "top_component_candidates": [
          {
            "component_candidate": "subset__xgb+logistic",
            "active_components": [
              "xgb",
              "logistic"
            ],
            "manual_blend_weights": {
              "xgb": 0.833333,
              "competitive": 0.0,
              "logistic": 0.166667,
              "elo": 0.0,
              "poisson": 0.0,
              "count_poisson": 0.0
            },
            "selected_outer_rows": 10519,
            "selected_folds": 8,
            "avg_inner_objective": 0.96563,
            "avg_inner_entropy": 0.810433
          },
          {
            "component_candidate": "subset__xgb+competitive+logistic",
            "active_components": [
              "xgb",
              "competitive",
              "logistic"
            ],
            "manual_blend_weights": {
              "xgb": 0.606061,
              "competitive": 0.272727,
              "logistic": 0.121212,
              "elo": 0.0,
              "poisson": 0.0,
              "count_poisson": 0.0
            },
            "selected_outer_rows": 10519,
            "selected_folds": 8,
            "avg_inner_objective": 0.965853,
            "avg_inner_entropy": 0.807401
          },
          {
            "component_candidate": "subset__xgb+logistic+count_poisson",
            "active_components": [
              "xgb",
              "logistic",
              "count_poisson"
            ],
            "manual_blend_weights": {
              "xgb": 0.689655,
              "competitive": 0.0,
              "logistic": 0.137931,
              "elo": 0.0,
              "poisson": 0.0,
              "count_poisson": 0.172414
            },
            "selected_outer_rows": 10519,
            "selected_folds": 8,
            "avg_inner_objective": 0.965943,
            "avg_inner_entropy": 0.811165
          },
          {
            "component_candidate": "subset__xgb+logistic+poisson",
            "active_components": [
              "xgb",
              "logistic",
              "poisson"
            ],
            "manual_blend_weights": {
              "xgb": 0.689655,
              "competitive": 0.0,
              "logistic": 0.137931,
              "elo": 0.0,
              "poisson": 0.172414,
              "count_poisson": 0.0
            },
            "selected_outer_rows": 10519,
            "selected_folds": 8,
            "avg_inner_objective": 0.966058,
            "avg_inner_entropy": 0.80907
          },
          {
            "component_candidate": "subset__xgb+competitive+logistic+count_poisson",
            "active_components": [
              "xgb",
              "competitive",
              "logistic",
              "count_poisson"
            ],
            "manual_blend_weights": {
              "xgb": 0.526316,
              "competitive": 0.236842,
              "logistic": 0.105263,
              "elo": 0.0,
              "poisson": 0.0,
              "count_poisson": 0.131579
            },
            "selected_outer_rows": 10519,
            "selected_folds": 8,
            "avg_inner_objective": 0.966139,
            "avg_inner_entropy": 0.808788
          },
          {
            "component_candidate": "subset__xgb+competitive+logistic+poisson",
            "active_components": [
              "xgb",
              "competitive",
              "logistic",
              "poisson"
            ],
            "manual_blend_weights": {
              "xgb": 0.526316,
              "competitive": 0.236842,
              "logistic": 0.105263,
              "elo": 0.0,
              "poisson": 0.131579,
              "count_poisson": 0.0
            },
            "selected_outer_rows": 10519,
            "selected_folds": 8,
            "avg_inner_objective": 0.966197,
            "avg_inner_entropy": 0.807113
          },
          {
            "component_candidate": "subset__xgb+logistic+poisson+count_poisson",
            "active_components": [
              "xgb",
              "logistic",
              "poisson",
              "count_poisson"
            ],
            "manual_blend_weights": {
              "xgb": 0.588235,
              "competitive": 0.0,
              "logistic": 0.117647,
              "elo": 0.0,
              "poisson": 0.147059,
              "count_poisson": 0.147059
            },
            "selected_outer_rows": 10519,
            "selected_folds": 8,
            "avg_inner_objective": 0.966329,
            "avg_inner_entropy": 0.810306
          },
          {
            "component_candidate": "subset__xgb+competitive+logistic+poisson+count_poisson",
            "active_components": [
              "xgb",
              "competitive",
              "logistic",
              "poisson",
              "count_poisson"
            ],
            "manual_blend_weights": {
              "xgb": 0.465116,
              "competitive": 0.209302,
              "logistic": 0.093023,
              "elo": 0.0,
              "poisson": 0.116279,
              "count_poisson": 0.116279
            },
            "selected_outer_rows": 10519,
            "selected_folds": 8,
            "avg_inner_objective": 0.966429,
            "avg_inner_entropy": 0.808387
          },
          {
            "component_candidate": "subset__competitive+logistic",
            "active_components": [
              "competitive",
              "logistic"
            ],
            "manual_blend_weights": {
              "xgb": 0.0,
              "competitive": 0.692308,
              "logistic": 0.307692,
              "elo": 0.0,
              "poisson": 0.0,
              "count_poisson": 0.0
            },
            "selected_outer_rows": 10519,
            "selected_folds": 8,
            "avg_inner_objective": 0.96643,
            "avg_inner_entropy": 0.808704
          },
          {
            "component_candidate": "subset__xgb",
            "active_components": [
              "xgb"
            ],
            "manual_blend_weights": {
              "xgb": 1.0,
              "competitive": 0.0,
              "logistic": 0.0,
              "elo": 0.0,
              "poisson": 0.0,
              "count_poisson": 0.0
            },
            "selected_outer_rows": 10519,
            "selected_folds": 8,
            "avg_inner_objective": 0.966569,
            "avg_inner_entropy": 0.806191
          },
          {
            "component_candidate": "subset__xgb+count_poisson",
            "active_components": [
              "xgb",
              "count_poisson"
            ],
            "manual_blend_weights": {
              "xgb": 0.8,
              "competitive": 0.0,
              "logistic": 0.0,
              "elo": 0.0,
              "poisson": 0.0,
              "count_poisson": 0.2
            },
            "selected_outer_rows": 10519,
            "selected_folds": 8,
            "avg_inner_objective": 0.96668,
            "avg_inner_entropy": 0.807823
          },
          {
            "component_candidate": "subset__competitive+logistic+count_poisson",
            "active_components": [
              "competitive",
              "logistic",
              "count_poisson"
            ],
            "manual_blend_weights": {
              "xgb": 0.0,
              "competitive": 0.5,
              "logistic": 0.222222,
              "elo": 0.0,
              "poisson": 0.0,
              "count_poisson": 0.277778
            },
            "selected_outer_rows": 10519,
            "selected_folds": 8,
            "avg_inner_objective": 0.966843,
            "avg_inner_entropy": 0.810934
          },
          {
            "component_candidate": "subset__xgb+poisson",
            "active_components": [
              "xgb",
              "poisson"
            ],
            "manual_blend_weights": {
              "xgb": 0.8,
              "competitive": 0.0,
              "logistic": 0.0,
              "elo": 0.0,
              "poisson": 0.2,
              "count_poisson": 0.0
            },
            "selected_outer_rows": 10519,
            "selected_folds": 8,
            "avg_inner_objective": 0.966876,
            "avg_inner_entropy": 0.806203
          },
          {
            "component_candidate": "subset__xgb+competitive+logistic+elo",
            "active_components": [
              "xgb",
              "competitive",
              "logistic",
              "elo"
            ],
            "manual_blend_weights": {
              "xgb": 0.5,
              "competitive": 0.225,
              "logistic": 0.1,
              "elo": 0.175,
              "poisson": 0.0,
              "count_poisson": 0.0
            },
            "selected_outer_rows": 10519,
            "selected_folds": 8,
            "avg_inner_objective": 0.966943,
            "avg_inner_entropy": 0.815521
          },
          {
            "component_candidate": "subset__xgb+competitive+logistic+elo+count_poisson",
            "active_components": [
              "xgb",
              "competitive",
              "logistic",
              "elo",
              "count_poisson"
            ],
            "manual_blend_weights": {
              "xgb": 0.444444,
              "competitive": 0.2,
              "logistic": 0.088889,
              "elo": 0.155556,
              "poisson": 0.0,
              "count_poisson": 0.111111
            },
            "selected_outer_rows": 10519,
            "selected_folds": 8,
            "avg_inner_objective": 0.966992,
            "avg_inner_entropy": 0.81545
          },
          {
            "component_candidate": "subset__xgb+poisson+count_poisson",
            "active_components": [
              "xgb",
              "poisson",
              "count_poisson"
            ],
            "manual_blend_weights": {
              "xgb": 0.666667,
              "competitive": 0.0,
              "logistic": 0.0,
              "elo": 0.0,
              "poisson": 0.166667,
              "count_poisson": 0.166667
            },
            "selected_outer_rows": 10519,
            "selected_folds": 8,
            "avg_inner_objective": 0.967011,
            "avg_inner_entropy": 0.807875
          },
          {
            "component_candidate": "subset__xgb+competitive+logistic+elo+poisson",
            "active_components": [
              "xgb",
              "competitive",
              "logistic",
              "elo",
              "poisson"
            ],
            "manual_blend_weights": {
              "xgb": 0.444444,
              "competitive": 0.2,
              "logistic": 0.088889,
              "elo": 0.155556,
              "poisson": 0.111111,
              "count_poisson": 0.0
            },
            "selected_outer_rows": 10519,
            "selected_folds": 8,
            "avg_inner_objective": 0.967012,
            "avg_inner_entropy": 0.81509
          },
          {
            "component_candidate": "subset__xgb+competitive+logistic+elo+poisson+count_poisson",
            "active_components": [
              "xgb",
              "competitive",
              "logistic",
              "elo",
              "poisson",
              "count_poisson"
            ],
            "manual_blend_weights": {
              "xgb": 0.4,
              "competitive": 0.18,
              "logistic": 0.08,
              "elo": 0.14,
              "poisson": 0.1,
              "count_poisson": 0.1
            },
            "selected_outer_rows": 10519,
            "selected_folds": 8,
            "avg_inner_objective": 0.967042,
            "avg_inner_entropy": 0.814643
          },
          {
            "component_candidate": "subset__xgb+competitive+count_poisson",
            "active_components": [
              "xgb",
              "competitive",
              "count_poisson"
            ],
            "manual_blend_weights": {
              "xgb": 0.588235,
              "competitive": 0.264706,
              "logistic": 0.0,
              "elo": 0.0,
              "poisson": 0.0,
              "count_poisson": 0.147059
            },
            "selected_outer_rows": 10519,
            "selected_folds": 8,
            "avg_inner_objective": 0.96706,
            "avg_inner_entropy": 0.80602
          },
          {
            "component_candidate": "subset__xgb+competitive",
            "active_components": [
              "xgb",
              "competitive"
            ],
            "manual_blend_weights": {
              "xgb": 0.689655,
              "competitive": 0.310345,
              "logistic": 0.0,
              "elo": 0.0,
              "poisson": 0.0,
              "count_poisson": 0.0
            },
            "selected_outer_rows": 10519,
            "selected_folds": 8,
            "avg_inner_objective": 0.967124,
            "avg_inner_entropy": 0.804006
          }
        ],
        "fold_best_rows": [
          {
            "outer_year": 2018,
            "component_candidate": "subset__xgb+count_poisson",
            "active_components": [
              "xgb",
              "count_poisson"
            ],
            "manual_blend_weights": {
              "xgb": 0.8,
              "competitive": 0.0,
              "logistic": 0.0,
              "elo": 0.0,
              "poisson": 0.0,
              "count_poisson": 0.2
            },
            "outer_rows": 1147,
            "selected_classifier_weight": 0.88,
            "selected_poisson_weight": 0.12,
            "selected_draw_floor": 0.04,
            "selected_draw_ceiling": 0.42,
            "inner_objective": 0.982051,
            "outer_objective": 1.014632,
            "outer_log_loss": 0.92658,
            "outer_rps": 0.18213,
            "outer_brier": 0.551623,
            "outer_ece": 0.027854,
            "outer_draw_recall": 0.010067,
            "outer_draw_expected_rate": 0.251229,
            "outer_draw_actual_rate": 0.259808,
            "outer_draw_gap": 0.008579,
            "outer_entropy": 0.847483,
            "outer_randomness_penalty": 0.0
          },
          {
            "outer_year": 2019,
            "component_candidate": "subset__xgb+logistic",
            "active_components": [
              "xgb",
              "logistic"
            ],
            "manual_blend_weights": {
              "xgb": 0.833333,
              "competitive": 0.0,
              "logistic": 0.166667,
              "elo": 0.0,
              "poisson": 0.0,
              "count_poisson": 0.0
            },
            "outer_rows": 1507,
            "selected_classifier_weight": 0.88,
            "selected_poisson_weight": 0.12,
            "selected_draw_floor": 0.06,
            "selected_draw_ceiling": 0.38,
            "inner_objective": 1.013017,
            "outer_objective": 0.959954,
            "outer_log_loss": 0.872688,
            "outer_rps": 0.172542,
            "outer_brier": 0.511224,
            "outer_ece": 0.0292,
            "outer_draw_recall": 0.017857,
            "outer_draw_expected_rate": 0.241069,
            "outer_draw_actual_rate": 0.22296,
            "outer_draw_gap": 0.01811,
            "outer_entropy": 0.799065,
            "outer_randomness_penalty": 0.0
          },
          {
            "outer_year": 2020,
            "component_candidate": "subset__xgb+competitive+logistic+elo+poisson",
            "active_components": [
              "xgb",
              "competitive",
              "logistic",
              "elo",
              "poisson"
            ],
            "manual_blend_weights": {
              "xgb": 0.444444,
              "competitive": 0.2,
              "logistic": 0.088889,
              "elo": 0.155556,
              "poisson": 0.111111,
              "count_poisson": 0.0
            },
            "outer_rows": 417,
            "selected_classifier_weight": 0.76,
            "selected_poisson_weight": 0.24,
            "selected_draw_floor": 0.04,
            "selected_draw_ceiling": 0.34,
            "inner_objective": 0.976338,
            "outer_objective": 1.050913,
            "outer_log_loss": 0.95888,
            "outer_rps": 0.191756,
            "outer_brier": 0.569663,
            "outer_ece": 0.047795,
            "outer_draw_recall": 0.0,
            "outer_draw_expected_rate": 0.259435,
            "outer_draw_actual_rate": 0.254197,
            "outer_draw_gap": 0.005238,
            "outer_entropy": 0.867965,
            "outer_randomness_penalty": 0.0
          },
          {
            "outer_year": 2021,
            "component_candidate": "subset__xgb+competitive+elo",
            "active_components": [
              "xgb",
              "competitive",
              "elo"
            ],
            "manual_blend_weights": {
              "xgb": 0.555556,
              "competitive": 0.25,
              "logistic": 0.0,
              "elo": 0.194444,
              "poisson": 0.0,
              "count_poisson": 0.0
            },
            "outer_rows": 1504,
            "selected_classifier_weight": 0.84,
            "selected_poisson_weight": 0.16,
            "selected_draw_floor": 0.04,
            "selected_draw_ceiling": 0.38,
            "inner_objective": 0.972922,
            "outer_objective": 0.877577,
            "outer_log_loss": 0.803841,
            "outer_rps": 0.150101,
            "outer_brier": 0.467982,
            "outer_ece": 0.039928,
            "outer_draw_recall": 0.0,
            "outer_draw_expected_rate": 0.221035,
            "outer_draw_actual_rate": 0.226064,
            "outer_draw_gap": 0.005029,
            "outer_entropy": 0.784818,
            "outer_randomness_penalty": 0.0
          },
          {
            "outer_year": 2022,
            "component_candidate": "subset__xgb+competitive+logistic",
            "active_components": [
              "xgb",
              "competitive",
              "logistic"
            ],
            "manual_blend_weights": {
              "xgb": 0.606061,
              "competitive": 0.272727,
              "logistic": 0.121212,
              "elo": 0.0,
              "poisson": 0.0,
              "count_poisson": 0.0
            },
            "outer_rows": 1375,
            "selected_classifier_weight": 0.6,
            "selected_poisson_weight": 0.4,
            "selected_draw_floor": 0.04,
            "selected_draw_ceiling": 0.3,
            "inner_objective": 0.911142,
            "outer_objective": 0.994302,
            "outer_log_loss": 0.909114,
            "outer_rps": 0.180052,
            "outer_brier": 0.535525,
            "outer_ece": 0.018501,
            "outer_draw_recall": 0.0,
            "outer_draw_expected_rate": 0.244034,
            "outer_draw_actual_rate": 0.237091,
            "outer_draw_gap": 0.006943,
            "outer_entropy": 0.829036,
            "outer_randomness_penalty": 0.0
          },
          {
            "outer_year": 2023,
            "component_candidate": "subset__xgb+competitive",
            "active_components": [
              "xgb",
              "competitive"
            ],
            "manual_blend_weights": {
              "xgb": 0.689655,
              "competitive": 0.310345,
              "logistic": 0.0,
              "elo": 0.0,
              "poisson": 0.0,
              "count_poisson": 0.0
            },
            "outer_rows": 1358,
            "selected_classifier_weight": 0.78,
            "selected_poisson_weight": 0.22,
            "selected_draw_floor": 0.06,
            "selected_draw_ceiling": 0.34,
            "inner_objective": 0.929606,
            "outer_objective": 0.941069,
            "outer_log_loss": 0.861148,
            "outer_rps": 0.168984,
            "outer_brier": 0.506017,
            "outer_ece": 0.023216,
            "outer_draw_recall": 0.0,
            "outer_draw_expected_rate": 0.223354,
            "outer_draw_actual_rate": 0.228277,
            "outer_draw_gap": 0.004923,
            "outer_entropy": 0.779092,
            "outer_randomness_penalty": 0.0
          },
          {
            "outer_year": 2024,
            "component_candidate": "subset__xgb",
            "active_components": [
              "xgb"
            ],
            "manual_blend_weights": {
              "xgb": 1.0,
              "competitive": 0.0,
              "logistic": 0.0,
              "elo": 0.0,
              "poisson": 0.0,
              "count_poisson": 0.0
            },
            "outer_rows": 1800,
            "selected_classifier_weight": 0.64,
            "selected_poisson_weight": 0.36,
            "selected_draw_floor": 0.04,
            "selected_draw_ceiling": 0.3,
            "inner_objective": 0.964305,
            "outer_objective": 1.008512,
            "outer_log_loss": 0.909828,
            "outer_rps": 0.175026,
            "outer_brier": 0.539136,
            "outer_ece": 0.039772,
            "outer_draw_recall": 0.0,
            "outer_draw_expected_rate": 0.225148,
            "outer_draw_actual_rate": 0.263333,
            "outer_draw_gap": 0.038185,
            "outer_entropy": 0.790972,
            "outer_randomness_penalty": 0.0
          },
          {
            "outer_year": 2025,
            "component_candidate": "subset__logistic+count_poisson",
            "active_components": [
              "logistic",
              "count_poisson"
            ],
            "manual_blend_weights": {
              "xgb": 0.0,
              "competitive": 0.0,
              "logistic": 0.444444,
              "elo": 0.0,
              "poisson": 0.0,
              "count_poisson": 0.555556
            },
            "outer_rows": 1411,
            "selected_classifier_weight": 0.58,
            "selected_poisson_weight": 0.42,
            "selected_draw_floor": 0.06,
            "selected_draw_ceiling": 0.34,
            "inner_objective": 0.96622,
            "outer_objective": 0.935897,
            "outer_log_loss": 0.848088,
            "outer_rps": 0.165336,
            "outer_brier": 0.499291,
            "outer_ece": 0.058342,
            "outer_draw_recall": 0.0,
            "outer_draw_expected_rate": 0.248627,
            "outer_draw_actual_rate": 0.228207,
            "outer_draw_gap": 0.02042,
            "outer_entropy": 0.7964,
            "outer_randomness_penalty": 0.0
          }
        ]
      }
    },
    "search_space": {
      "classifier_weights": [
        0.5,
        0.52,
        0.54,
        0.56,
        0.58,
        0.6,
        0.62,
        0.64,
        0.66,
        0.68,
        0.7,
        0.72,
        0.74,
        0.76,
        0.78,
        0.8,
        0.82,
        0.84,
        0.86,
        0.88,
        0.9,
        0.92,
        0.94
      ],
      "draw_floors": [
        0.04,
        0.06,
        0.08,
        0.1,
        0.12,
        0.14
      ],
      "draw_ceilings": [
        0.3,
        0.34,
        0.38,
        0.42,
        0.46
      ],
      "candidate_count": 690,
      "minimum_poisson_weight_preferred": 0.12,
      "entropy_floor_preferred": 0.68
    },
    "description": "Leakage-free nested-temporal selected classifier/Poisson policy with draw floor/ceiling guardrails; draw_xgb was removed because zero-weight inactive models are not SOTA/KISS."
  },
  "metrics": {
    "baseline_elo_1x2": {
      "accuracy": 0.5733,
      "test_rows": 3499
    },
    "baseline_fifa_rank_1x2": {
      "accuracy": 0.5513,
      "test_rows": 3499
    },
    "logistic_1x2": {
      "accuracy": 0.5639,
      "top2_accuracy": 0.8245,
      "log_loss": 0.9017,
      "draw_recall": 0.32,
      "test_rows": 3499
    },
    "xgb_1x2": {
      "accuracy": 0.5782,
      "top2_accuracy": 0.8285,
      "log_loss": 0.8847,
      "draw_recall": 0.0138,
      "test_rows": 3499
    },
    "xgb_temperature_calibrated_1x2": {
      "accuracy": 0.5782,
      "top2_accuracy": 0.8285,
      "log_loss": 0.8847,
      "draw_recall": 0.0138,
      "temperature": 1.0,
      "test_rows": 3499
    },
    "competitive_xgb_1x2": {
      "accuracy": 0.5861,
      "top2_accuracy": 0.8348,
      "log_loss": 0.8701,
      "test_rows": 2566
    },
    "poisson_goal_model_1x2": {
      "accuracy": 0.5719,
      "top2_accuracy": 0.8265,
      "log_loss": 0.8918,
      "draw_recall": 0.0,
      "home_goal_mae": 1.0055,
      "away_goal_mae": 0.8701,
      "test_rows": 3499
    },
    "xgb_count_poisson_1x2": {
      "accuracy": 0.5767,
      "top2_accuracy": 0.8297,
      "log_loss": 0.8859,
      "draw_recall": 0.0,
      "home_goal_mae": 0.995,
      "away_goal_mae": 0.8641,
      "test_rows": 3499
    },
    "stacking_meta_1x2": {
      "accuracy": 0.5573,
      "top2_accuracy": 0.8257,
      "log_loss": 0.9035,
      "draw_recall": 0.3842,
      "test_rows": 3499
    }
  },
  "accepted_experiments": {
    "pataterie_history": true,
    "external_elo_features": true,
    "transfermarkt_2026_squad_layer": true,
    "poisson_goal_models": true,
    "xgb_count_poisson_goal_models": true,
    "dixon_coles_rho_grid": true,
    "xgb_temperature_calibration": true,
    "stacking_meta_ensemble": false,
    "lchikry_ensemble": false,
    "official_2026_bracket_slots": true,
    "world_cup_walk_forward_backtest": true,
    "extra_time_penalty_layer": true,
    "fixture_rest_travel_home_context": true,
    "transfermarkt_injury_value_risk": true,
    "confederation_features": true,
    "temporal_importance_sample_weight": true,
    "neutral_orientation_invariance": true
  },
  "metric_gains_vs_previous_report": {
    "baseline_elo_1x2": {
      "accuracy": 0.0,
      "test_rows": 0.0
    },
    "baseline_fifa_rank_1x2": {
      "accuracy": 0.0,
      "test_rows": 0.0
    },
    "logistic_1x2": {
      "accuracy": 0.0,
      "top2_accuracy": 0.0,
      "log_loss": 0.0,
      "draw_recall": 0.0,
      "test_rows": 0.0
    },
    "xgb_1x2": {
      "accuracy": 0.0,
      "top2_accuracy": 0.0,
      "log_loss": 0.0,
      "draw_recall": 0.0,
      "test_rows": 0.0
    },
    "xgb_temperature_calibrated_1x2": {
      "accuracy": 0.0,
      "top2_accuracy": 0.0,
      "log_loss": 0.0,
      "draw_recall": 0.0,
      "temperature": 0.0,
      "test_rows": 0.0
    },
    "competitive_xgb_1x2": {
      "accuracy": 0.0,
      "top2_accuracy": 0.0,
      "log_loss": 0.0,
      "test_rows": 0.0
    },
    "poisson_goal_model_1x2": {
      "accuracy": 0.0,
      "top2_accuracy": 0.0,
      "log_loss": 0.0,
      "draw_recall": 0.0,
      "home_goal_mae": 0.0,
      "away_goal_mae": 0.0,
      "test_rows": 0.0
    },
    "xgb_count_poisson_1x2": {
      "accuracy": 0.0,
      "top2_accuracy": 0.0,
      "log_loss": 0.0,
      "draw_recall": 0.0,
      "home_goal_mae": 0.0,
      "away_goal_mae": 0.0,
      "test_rows": 0.0
    },
    "stacking_meta_1x2": {
      "accuracy": 0.0,
      "top2_accuracy": 0.0,
      "log_loss": 0.0,
      "draw_recall": 0.0,
      "test_rows": 0.0
    }
  },
  "metric_gains_vs_sota_v1": {
    "baseline_elo_1x2": {
      "accuracy": -0.016
    },
    "baseline_fifa_rank_1x2": {
      "accuracy": -0.0116
    },
    "logistic_1x2": {
      "accuracy": -0.0041,
      "top2_accuracy": -0.0166,
      "log_loss": 0.0175,
      "draw_recall": 0.0902
    },
    "xgb_1x2": {
      "accuracy": -0.0228,
      "top2_accuracy": -0.0076,
      "log_loss": 0.0197,
      "draw_recall": 0.0103
    },
    "competitive_xgb_1x2": {
      "accuracy": -0.0239,
      "top2_accuracy": -0.0082,
      "log_loss": 0.0173
    }
  },
  "world_cup_backtest_aggregate": {
    "baseline_elo_1x2": {
      "accuracy": 0.5393,
      "top2_accuracy": 0.7702,
      "log_loss": 1.1247,
      "brier": 0.676,
      "rps": 0.2244,
      "ece": 0.2207,
      "draw_recall": 0.1889
    },
    "hybrid_nested_policy_1x2": {
      "log_loss": 0.9662,
      "brier": 0.5696,
      "rps": 0.1957,
      "ece": 0.1002,
      "draw_recall": 0.005,
      "classifier_weight": 0.7623,
      "poisson_weight": 0.2377,
      "draw_floor": 0.0426,
      "draw_ceiling": 0.3585,
      "objective": 1.0767,
      "draw_expected_rate": 0.2555,
      "draw_actual_rate": 0.2361,
      "draw_gap": 0.0363,
      "entropy": 0.8893,
      "randomness_penalty": 0.0,
      "inner_train_rows": 17401.0451,
      "inner_rows": 2286.4565,
      "inner_rho": -0.107,
      "outer_rho": -0.107
    },
    "logistic_1x2": {
      "accuracy": 0.5247,
      "top2_accuracy": 0.7817,
      "log_loss": 0.9938,
      "brier": 0.5869,
      "rps": 0.2009,
      "ece": 0.091,
      "draw_recall": 0.1983
    },
    "poisson_goal_model_1x2": {
      "accuracy": 0.5572,
      "top2_accuracy": 0.7985,
      "log_loss": 0.9667,
      "brier": 0.5699,
      "rps": 0.1959,
      "ece": 0.113,
      "draw_recall": 0.0,
      "home_goal_mae": 0.9031,
      "away_goal_mae": 0.9017
    },
    "stacking_meta_1x2": {
      "accuracy": 0.5194,
      "top2_accuracy": 0.7849,
      "log_loss": 1.0018,
      "brier": 0.5907,
      "rps": 0.201,
      "ece": 0.0885,
      "draw_recall": 0.3104
    },
    "xgb_1x2": {
      "accuracy": 0.5593,
      "top2_accuracy": 0.7933,
      "log_loss": 0.969,
      "brier": 0.5707,
      "rps": 0.196,
      "ece": 0.0948,
      "draw_recall": 0.0244
    },
    "xgb_count_poisson_1x2": {
      "accuracy": 0.5561,
      "top2_accuracy": 0.7996,
      "log_loss": 0.9686,
      "brier": 0.5716,
      "rps": 0.1969,
      "ece": 0.0982,
      "draw_recall": 0.0,
      "home_goal_mae": 0.8937,
      "away_goal_mae": 0.8714
    },
    "xgb_temperature_calibrated_1x2": {
      "accuracy": 0.5593,
      "top2_accuracy": 0.7933,
      "log_loss": 0.98,
      "brier": 0.5745,
      "rps": 0.1977,
      "ece": 0.1142,
      "draw_recall": 0.0244,
      "temperature": 0.8218
    }
  },
  "monte_carlo_runs": 1000,
  "sample_champion_seed_2026": "Spain",
  "top_10_champion_odds": [
    {
      "team": "Spain",
      "wins": 232,
      "champion_probability": 0.232
    },
    {
      "team": "Argentina",
      "wins": 164,
      "champion_probability": 0.164
    },
    {
      "team": "France",
      "wins": 104,
      "champion_probability": 0.104
    },
    {
      "team": "England",
      "wins": 94,
      "champion_probability": 0.094
    },
    {
      "team": "Brazil",
      "wins": 66,
      "champion_probability": 0.066
    },
    {
      "team": "Netherlands",
      "wins": 48,
      "champion_probability": 0.048
    },
    {
      "team": "Portugal",
      "wins": 44,
      "champion_probability": 0.044
    },
    {
      "team": "Colombia",
      "wins": 36,
      "champion_probability": 0.036
    },
    {
      "team": "Germany",
      "wins": 34,
      "champion_probability": 0.034
    },
    {
      "team": "Croatia",
      "wins": 25,
      "champion_probability": 0.025
    }
  ]
}
```

## Veredito

- SOTA/KISS pragmático: `True`
- Carimbo acadêmico: `SOTA/KISS academico aplicado aos dados disponiveis`
- Motivo: Sem modelo de empate inativo, seleção nested temporal conjunta para componentes e política, gap de empate controlado, Poisson preservando variância de futebol e sensibilidade de rho/ablação perto da fronteira.
- Recomendações abertas: `0`

Critérios duros do carimbo:

```json
{
  "nested_temporal_no_leakage": true,
  "complete_component_ablation_63_subsets": true,
  "nested_component_and_policy_grid": true,
  "draw_xgb_removed": true,
  "runtime_draw_gap_lte_2pp": true,
  "runtime_log_loss_beats_same_window_elo": true,
  "runtime_rps_beats_same_window_elo": true,
  "runtime_near_ablation_frontier": true,
  "runtime_near_draw_policy_frontier_without_retrofit": true,
  "dixon_coles_near_rho_frontier": true,
  "monte_carlo_uncertainty_reported": true,
  "stage_uncertainty_reported": true,
  "advanced_calibration_exhausted": true,
  "team_strength_dixon_coles_exhausted": true,
  "class_calibration_reported": true,
  "block_bootstrap_reported": true,
  "runtime_adjustment_audit_reported": true,
  "runtime_adjustment_direct_1x2_removed": true,
  "runtime_adjustment_max_shift_lte_35pp": true,
  "runtime_adjustment_p95_shift_lte_18pp": true,
  "runtime_neutral_order_invariant": true,
  "raw_data_manifest_reported": true,
  "raw_data_manifest_hash_reported": true,
  "raw_data_semantic_sanity_passed": true,
  "neutral_orientation_invariant": true,
  "source_fingerprints_reported": true,
  "external_elo_parse_complete": true,
  "external_elo_current": true,
  "external_elo_qualified_coverage_complete": true,
  "mc_stability_available": true,
  "mc_stability_fresh": true,
  "mc_stability_passed": true,
  "mc_stage_bracket_stability_passed": true
}
```

Freshness dos artefatos:

```json
{
  "model_package": {
    "path": "/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/models/model_sota.pkl",
    "sha256": "8ddfc8d046fb92ad7166ddc54e4b5e0c2b10899596605c34e393c224ef87b33a",
    "size_bytes": 4091805,
    "mtime_ns": 1782789569903796297
  },
  "model_report": {
    "path": "/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_model_report.json",
    "sha256": "0a9d844b0a3af028cd4007c2427571c5ef8dc957ae246061352dfe62252cd454",
    "size_bytes": 96850,
    "mtime_ns": 1782789569908558073
  },
  "training_matches": {
    "path": "/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/data/processed/sota_training_matches.csv",
    "sha256": "bf67d740e988fa3d38310e3b97c201844dab618bcdff689dbbe251dea88e6a25",
    "size_bytes": 15868527,
    "mtime_ns": 1782789569875760646
  },
  "sota_pipeline": {
    "path": "/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/src/sota_pipeline.py",
    "sha256": "bafab33d1aa4c599e2c3c702d3fc7e134c9995213c079ed3f9c554f2abc50770",
    "size_bytes": 205834,
    "mtime_ns": 1782788278567602097
  },
  "stats_qa_script": {
    "path": "/Users/eventanilha/Projects/arena-ai/scripts/model_stats_qa.py",
    "sha256": "d50ab54e36ea946a774683e18abf217ad029ae44eb4bac98760711c2e77793ec",
    "size_bytes": 95391,
    "mtime_ns": 1782790446523927877
  }
}
```

Manifesto completo dos dados brutos:

```json
{
  "path": "/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_raw_data_manifest.json",
  "csv_path": "/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_raw_data_manifest.csv",
  "file_count": 14,
  "csv_file_count": 14,
  "total_size_bytes": 81082901,
  "manifest_sha256": "70ab1b3e3c816e37da3fff9a46c31abe5e5eb604c10300a5e6da1167cbe7e14e",
  "semantic": {
    "required_file_count": 14,
    "required_files_present": true,
    "checked_file_count": 14,
    "passed_file_count": 14,
    "cross_file_checks": {
      "passed": true,
      "checks": [
        {
          "name": "match_team_ids_exist",
          "passed": true,
          "missing_home": [],
          "missing_away": []
        },
        {
          "name": "match_city_ids_exist",
          "passed": true,
          "missing": []
        },
        {
          "name": "match_stage_ids_exist",
          "passed": true,
          "missing": []
        },
        {
          "name": "match_numbers_unique_1_104",
          "passed": true,
          "actual_count": 104
        },
        {
          "name": "group_stage_has_team_ids",
          "passed": true,
          "missing_home": 0,
          "missing_away": 0
        },
        {
          "name": "knockout_slots_are_placeholders_or_known_ids",
          "passed": true,
          "missing_home": 32,
          "missing_away": 32,
          "knockout_matches": 32
        }
      ]
    },
    "passed": true,
    "failures": []
  }
}
```

Auditoria de orientação dos jogos neutros:

```json
{
  "method": "every neutral historical fixture is paired with its exact swapped home/away counterpart before model fitting",
  "neutral_rows": 17944,
  "world_cup_neutral_rows": 996,
  "augmented_rows": 8972,
  "neutral_outcomes": {
    "home_wins": 6767,
    "draws": 4410,
    "away_wins": 6767
  },
  "world_cup_neutral_outcomes": {
    "home_wins": 380,
    "draws": 236,
    "away_wins": 380
  },
  "passed": true
}
```

Auditoria de inferência neutra em ordem invertida:

```json
{
  "path": "/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_runtime_neutral_order_audit.csv",
  "method": "every official group fixture is predicted in both nominal orders with no travel/rest context, then compared after reversing outcomes",
  "fixtures": 72,
  "max_delta_or_error": 0.0,
  "tolerance": 1e-10,
  "order_symmetrization_coverage": 1.0,
  "passed": true
}
```

## Política ativa

```json
{
  "classifier_weight": 0.88,
  "poisson_weight": 0.12,
  "draw_floor": 0.04,
  "draw_ceiling": 0.3,
  "manual_blend_weights": {
    "xgb": 0.606061,
    "competitive": 0.272727,
    "logistic": 0.121212,
    "elo": 0.0,
    "poisson": 0.0,
    "count_poisson": 0.0
  },
  "draw_xgb": "removed_zero_weight_model",
  "selected_by": "strict_nested_temporal_component_ablation_no_leakage_no_draw_xgb"
}
```

## 1. Calibração mais forte

A seleção da política continua vindo da validação nested temporal sem vazamento. A curva de calibração abaixo é um diagnóstico do pacote atual em 2024+.

```json
{
  "component_candidate": "subset__xgb+competitive+logistic",
  "active_components": [
    "xgb",
    "competitive",
    "logistic"
  ],
  "manual_blend_weights": {
    "xgb": 0.606061,
    "competitive": 0.272727,
    "logistic": 0.121212,
    "elo": 0.0,
    "poisson": 0.0,
    "count_poisson": 0.0
  },
  "classifier_weight": 0.88,
  "poisson_weight": 0.12,
  "draw_floor": 0.04,
  "draw_ceiling": 0.3,
  "selected_outer_rows": 10519,
  "selected_folds": 8,
  "selected_avg_inner_objective": 0.967463,
  "selected_avg_inner_entropy": 0.807016,
  "outer_objective": 0.956044,
  "outer_log_loss": 0.876527,
  "outer_rps": 0.170828,
  "outer_brier": 0.516715,
  "outer_ece": 0.007282,
  "outer_draw_recall": 0.0,
  "outer_draw_expected_rate": 0.2344,
  "outer_draw_actual_rate": 0.238806,
  "outer_draw_gap": 0.004406,
  "outer_entropy": 0.800726,
  "outer_randomness_penalty": 0.0
}
```

```json
{
  "classifier_weight": 0.88,
  "poisson_weight": 0.12,
  "draw_floor": 0.04,
  "draw_ceiling": 0.3,
  "objective": 0.971502,
  "log_loss": 0.882776,
  "rps": 0.171089,
  "brier": 0.522544,
  "ece": 0.038787,
  "draw_recall": 0.0,
  "draw_expected_rate": 0.229762,
  "draw_actual_rate": 0.249214,
  "draw_gap": 0.019452,
  "entropy": 0.788103,
  "randomness_penalty": 0.0
}
```

Resumo por classe:

| label | sample_count | weighted_abs_gap | max_abs_gap | mean_predicted_rate | mean_empirical_rate |
| --- | --- | --- | --- | --- | --- |
| casa | 3499 | 0.034064 | 0.083996 | 0.450115 | 0.428694 |
| fora | 3499 | 0.022013 | 0.057171 | 0.320123 | 0.322092 |
| empate | 3499 | 0.020197 | 0.050902 | 0.229763 | 0.249214 |

### Fronteira sem dataset externo

Calibradores extras e Dixon-Coles por ataque/defesa foram testados com dados já existentes. Só entrariam no runtime se vencessem materialmente sem vazar futuro.

```json
{
  "advanced_calibration": {
    "method": "2024 calibrates; 2025+ evaluates; no future leakage; candidates enter only if objective, log_loss and draw gap improve materially",
    "promoted": false,
    "identity": {
      "experiment": "advanced_calibration",
      "family": "identity",
      "candidate": "runtime_sem_calibracao_extra",
      "calibration_rows": 1800,
      "evaluation_rows": 1699,
      "cal_objective": 1.004288,
      "eval_objective": 0.938614,
      "eval_objective_delta_vs_identity": 0.0,
      "eval_log_loss": 0.856142,
      "eval_log_loss_delta_vs_identity": 0.0,
      "eval_ece": 0.064566,
      "eval_draw_gap": 0.004423,
      "eval_accuracy": 0.589759,
      "promoted": false,
      "decision": "candidate",
      "note": "referencia do runtime atual"
    },
    "best": {
      "experiment": "advanced_calibration",
      "family": "temperature",
      "candidate": "T=1.05",
      "calibration_rows": 1800,
      "evaluation_rows": 1699,
      "cal_objective": 0.999653,
      "eval_objective": 0.936606,
      "eval_objective_delta_vs_identity": -0.002008,
      "eval_log_loss": 0.856819,
      "eval_log_loss_delta_vs_identity": 0.000677,
      "eval_ece": 0.051508,
      "eval_draw_gap": 0.0005,
      "eval_accuracy": 0.589759,
      "promoted": false,
      "decision": "not_promoted",
      "note": "nao entrou: nenhum calibrador venceu o runtime em objetivo, log_loss e empate no split temporal 2024->2025+"
    }
  },
  "dixon_coles_team_strength": {
    "method": "team attack/defense strengths estimated from pre-2024 history only, with shrinkage and temporal decay; candidates enter only with material objective and log_loss gains",
    "promoted": false,
    "runtime": {
      "experiment": "dixon_coles_team_strength",
      "family": "runtime",
      "candidate": "poisson_regressor_dixon_coles",
      "calibration_rows": 37862,
      "evaluation_rows": 3499,
      "eval_objective": 0.971502,
      "eval_objective_delta_vs_runtime": 0.0,
      "eval_log_loss": 0.882776,
      "eval_log_loss_delta_vs_runtime": 0.0,
      "eval_ece": 0.038787,
      "eval_draw_gap": 0.019452,
      "eval_accuracy": 0.578737,
      "promoted": false,
      "decision": "candidate",
      "note": "referencia Poisson/DC atual do runtime"
    },
    "best": {
      "experiment": "dixon_coles_team_strength",
      "family": "team_attack_defense_dc",
      "candidate": "shrink=8_half_life=3",
      "calibration_rows": 37862,
      "evaluation_rows": 3499,
      "eval_objective": 0.968427,
      "eval_objective_delta_vs_runtime": -0.003075,
      "eval_log_loss": 0.882801,
      "eval_log_loss_delta_vs_runtime": 2.5e-05,
      "eval_ece": 0.029307,
      "eval_draw_gap": 0.014125,
      "eval_accuracy": 0.579594,
      "promoted": false,
      "decision": "not_promoted",
      "note": "nao entrou: ganho diagnostico insuficiente e/ou log_loss pior que o PoissonRegressor+Dixon-Coles atual"
    }
  },
  "frontier_experiments": {
    "path": "/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_internal_frontier_experiments.csv",
    "rows": 90,
    "promoted_candidates": 0,
    "rule": "promote only if a no-external-data candidate wins materially without hurting log_loss/draw calibration; otherwise document and keep runtime KISS"
  }
}
```

## 2. Intervalos de incerteza

Campeão:

| team | wins | probability | lower_95 | upper_95 | margin_95 |
| --- | --- | --- | --- | --- | --- |
| Spain | 232 | 0.232 | 0.205837 | 0.258163 | 0.026163 |
| Argentina | 164 | 0.164 | 0.14105 | 0.18695 | 0.02295 |
| France | 104 | 0.104 | 0.08508 | 0.12292 | 0.01892 |
| England | 94 | 0.094 | 0.075912 | 0.112088 | 0.018088 |
| Brazil | 66 | 0.066 | 0.050611 | 0.081389 | 0.015389 |
| Netherlands | 48 | 0.048 | 0.034751 | 0.061249 | 0.013249 |
| Portugal | 44 | 0.044 | 0.031288 | 0.056712 | 0.012712 |
| Colombia | 36 | 0.036 | 0.024454 | 0.047546 | 0.011546 |

Fases:

| team | stage | probability | lower_95 | upper_95 | margin_95 |
| --- | --- | --- | --- | --- | --- |
| Mexico | Group Stage | 1.0 | 1.0 | 1.0 | 0.0 |
| Mexico | Round of 32 | 0.94 | 0.92528 | 0.95472 | 0.01472 |
| Mexico | Round of 16 | 0.592 | 0.561539 | 0.622461 | 0.030461 |
| Mexico | Quarter-finals | 0.251 | 0.224126 | 0.277874 | 0.026874 |
| Mexico | Semi-finals | 0.083 | 0.065901 | 0.100099 | 0.017099 |
| Mexico | Final | 0.024 | 0.014514 | 0.033486 | 0.009486 |
| Mexico | Champion | 0.011 | 0.004535 | 0.017465 | 0.006465 |
| South Africa | Group Stage | 1.0 | 1.0 | 1.0 | 0.0 |

Bootstrap por bloco temporal/torneio:

| block_type | metric | block_count | mean | lower_95 | upper_95 | width_95 |
| --- | --- | --- | --- | --- | --- | --- |
| ano | log_loss | 3 | 0.883172 | 0.840187 | 0.914313 | 0.074126 |
| ano | rps | 3 | 0.171324 | 0.163864 | 0.176877 | 0.013013 |
| ano | brier | 3 | 0.522729 | 0.496262 | 0.541851 | 0.045589 |
| ano | ece | 3 | 0.045484 | 0.031927 | 0.075503 | 0.043576 |
| ano | draw_gap | 3 | 0.018278 | 0.003134 | 0.031983 | 0.028849 |
| torneio | log_loss | 48 | 0.885933 | 0.834244 | 0.935015 | 0.100772 |
| torneio | rps | 48 | 0.171936 | 0.161036 | 0.183638 | 0.022602 |
| torneio | brier | 48 | 0.524407 | 0.490931 | 0.556265 | 0.065334 |
| torneio | ece | 48 | 0.042703 | 0.029384 | 0.063034 | 0.03365 |
| torneio | draw_gap | 48 | 0.019318 | 0.004297 | 0.035033 | 0.030736 |

Estabilidade Monte Carlo offline:

```json
{
  "available": true,
  "fresh": true,
  "passed": true,
  "stage_bracket_passed": true,
  "path": "/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_monte_carlo_stability.json",
  "csv_path": "/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_monte_carlo_stability.csv",
  "stage_bracket_csv_path": "/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_monte_carlo_stage_bracket_stability.csv",
  "runs": [
    5000,
    10000
  ],
  "stage_bracket_runs": [
    1000,
    2000,
    5000
  ],
  "stability_gate": {
    "max_top16_abs_delta": 0.015,
    "max_top16_churn": 2,
    "leader_change_allowed": false,
    "max_stage_top16_abs_delta": 0.035,
    "max_stage_top16_churn": 4,
    "max_pair_top8_nested_z": 4.0,
    "max_pair_top8_churn": 16
  },
  "final_comparison": {
    "baseline": false,
    "previous_runs": 5000,
    "leader_changed": false,
    "comparison": "union_top16_abs_delta",
    "union_team_count": 16,
    "entered_top16": [],
    "exited_top16": [],
    "top16_churn_count": 0,
    "max_top16_abs_delta": 0.0056,
    "mean_top16_abs_delta": 0.001669
  },
  "stage_bracket_final_comparison": {
    "baseline": false,
    "previous_runs": 2000,
    "stage_top16": {
      "Round of 32": {
        "entered": [],
        "exited": [],
        "churn_count": 0,
        "max_abs_delta": 0.0075,
        "mean_abs_delta": 0.003119,
        "max_nested_z": 2.244426,
        "max_nested_z_name": "Spain"
      },
      "Round of 16": {
        "entered": [],
        "exited": [],
        "churn_count": 0,
        "max_abs_delta": 0.0145,
        "mean_abs_delta": 0.006737,
        "max_nested_z": 2.058951,
        "max_nested_z_name": "Spain"
      },
      "Quarterfinals": {
        "entered": [],
        "exited": [],
        "churn_count": 0,
        "max_abs_delta": 0.0131,
        "mean_abs_delta": 0.006138,
        "max_nested_z": 1.538119,
        "max_nested_z_name": "Spain"
      },
      "Semifinals": {
        "entered": [],
        "exited": [],
        "churn_count": 0,
        "max_abs_delta": 0.0169,
        "mean_abs_delta": 0.005262,
        "max_nested_z": 2.039846,
        "max_nested_z_name": "France"
      },
      "Final": {
        "entered": [],
        "exited": [],
        "churn_count": 0,
        "max_abs_delta": 0.0162,
        "mean_abs_delta": 0.004806,
        "max_nested_z": 1.955605,
        "max_nested_z_name": "Spain"
      }
    },
    "pair_top8": {
      "Round of 32": {
        "entered": [],
        "exited": [],
        "churn_count": 0,
        "max_abs_delta": 0.0103,
        "mean_abs_delta": 0.0052,
        "max_nested_z": 1.359486,
        "max_nested_z_name": "Argentina | Spain"
      },
      "Round of 16": {
        "entered": [
          "Colombia | Switzerland"
        ],
        "exited": [
          "Portugal | Switzerland"
        ],
        "churn_count": 2,
        "max_abs_delta": 0.0103,
        "mean_abs_delta": 0.003,
        "max_nested_z": 1.576512,
        "max_nested_z_name": "Croatia | Spain"
      },
      "Quarterfinals": {
        "entered": [],
        "exited": [],
        "churn_count": 0,
        "max_abs_delta": 0.0099,
        "mean_abs_delta": 0.004062,
        "max_nested_z": 1.853669,
        "max_nested_z_name": "Belgium | Spain"
      },
      "Semifinals": {
        "entered": [
          "Ecuador | Spain"
        ],
        "exited": [
          "Morocco | Spain"
        ],
        "churn_count": 2,
        "max_abs_delta": 0.0108,
        "mean_abs_delta": 0.004444,
        "max_nested_z": 1.941387,
        "max_nested_z_name": "France | Spain"
      },
      "Final": {
        "entered": [],
        "exited": [],
        "churn_count": 0,
        "max_abs_delta": 0.0054,
        "mean_abs_delta": 0.002987,
        "max_nested_z": 2.194924,
        "max_nested_z_name": "England | France"
      }
    },
    "finalist_top16": {
      "entered": [],
      "exited": [],
      "churn_count": 0,
      "max_abs_delta": 0.0162,
      "mean_abs_delta": 0.004806,
      "max_nested_z": 1.955605,
      "max_nested_z_name": "Spain"
    },
    "max_stage_top16_abs_delta": 0.0169,
    "max_stage_top16_churn": 0,
    "max_pair_top8_abs_delta": 0.0108,
    "max_pair_top8_churn": 2,
    "max_pair_top8_nested_z": 2.194924,
    "max_finalist_top16_abs_delta": 0.0162,
    "max_finalist_top16_churn": 0
  },
  "summary": {
    "max_runs": 10000,
    "min_runs": 5000,
    "max_stage_bracket_runs": 5000,
    "min_stage_bracket_runs": 1000,
    "leader_at_max_runs": "Spain",
    "leader_probability_at_max_runs": 0.2366,
    "csv_path": "/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_monte_carlo_stability.csv",
    "stage_bracket_csv_path": "/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_monte_carlo_stage_bracket_stability.csv"
  },
  "source_fingerprints": {
    "model_package": {
      "path": "/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/models/model_sota.pkl",
      "sha256": "8ddfc8d046fb92ad7166ddc54e4b5e0c2b10899596605c34e393c224ef87b33a",
      "size_bytes": 4091805,
      "mtime_ns": 1782789569903796297
    },
    "model_report": {
      "path": "/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_model_report.json",
      "sha256": "0a9d844b0a3af028cd4007c2427571c5ef8dc957ae246061352dfe62252cd454",
      "size_bytes": 96850,
      "mtime_ns": 1782789569908558073
    },
    "training_matches": {
      "path": "/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/data/processed/sota_training_matches.csv",
      "sha256": "bf67d740e988fa3d38310e3b97c201844dab618bcdff689dbbe251dea88e6a25",
      "size_bytes": 15868527,
      "mtime_ns": 1782789569875760646
    },
    "sota_pipeline": {
      "path": "/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/src/sota_pipeline.py",
      "sha256": "bafab33d1aa4c599e2c3c702d3fc7e134c9995213c079ed3f9c554f2abc50770",
      "size_bytes": 205834,
      "mtime_ns": 1782788278567602097
    },
    "mc_stability_script": {
      "path": "/Users/eventanilha/Projects/arena-ai/scripts/monte_carlo_stability.py",
      "sha256": "d20e93617818c863c3ac22a4bc60d53ab2636e6ae2ac7854f40d1ed3fbfcf525",
      "size_bytes": 29605,
      "mtime_ns": 1782691759529097682
    }
  }
}
```

## 3. Ablation study

A tabela inclui todos os 63 subconjuntos dos seis sinais (`xgb`, `competitive`, `logistic`, `elo`, `poisson`, `count_poisson`) e compara contra a política ativa do runtime.

| ablation | objective | log_loss | rps | draw_gap | entropy |
| --- | --- | --- | --- | --- | --- |
| subset__logistic+count_poisson | 0.964492 | 0.884531 | 0.171323 | 0.00026 | 0.814529 |
| subset__xgb+logistic+elo | 0.968031 | 0.882178 | 0.1707 | 0.01552 | 0.809682 |
| subset__xgb+logistic+elo+count_poisson | 0.968354 | 0.882292 | 0.170723 | 0.015849 | 0.808661 |
| subset__logistic+poisson+count_poisson | 0.968611 | 0.885739 | 0.171662 | 0.007021 | 0.806236 |
| subset__xgb+logistic+count_poisson | 0.968616 | 0.881743 | 0.170794 | 0.015997 | 0.794569 |
| subset__xgb+logistic | 0.968771 | 0.881661 | 0.170797 | 0.015852 | 0.792959 |
| subset__competitive+logistic+count_poisson | 0.968917 | 0.883615 | 0.171325 | 0.013499 | 0.796077 |
| subset__competitive+logistic | 0.969127 | 0.884133 | 0.171531 | 0.012548 | 0.793612 |
| subset__xgb+logistic+elo+poisson | 0.969133 | 0.882662 | 0.170859 | 0.016479 | 0.807174 |
| subset__xgb+logistic+elo+poisson+count_poisson | 0.969147 | 0.882771 | 0.170873 | 0.016669 | 0.806526 |

## 4. Draw-specific calibration

```json
{
  "candidate_count": 690,
  "current_policy_rank": 471,
  "current_policy": [
    0.88,
    0.04,
    0.3
  ],
  "current_policy_metrics": {
    "classifier_weight": 0.88,
    "poisson_weight": 0.12,
    "draw_floor": 0.04,
    "draw_ceiling": 0.3,
    "objective": 0.971502,
    "log_loss": 0.882776,
    "rps": 0.171089,
    "brier": 0.522544,
    "ece": 0.038787,
    "draw_recall": 0.0,
    "draw_expected_rate": 0.229762,
    "draw_actual_rate": 0.249214,
    "draw_gap": 0.019452,
    "entropy": 0.788103,
    "randomness_penalty": 0.0
  },
  "best_same_holdout_objective": 0.96796,
  "current_policy_objective_gap_vs_same_holdout_best": 0.003542,
  "accepted_objective_gap_without_retrofit": 0.005,
  "interpretation": "The active policy was selected by nested temporal validation, not by this later diagnostic window. The audit rejects a material regression but does not retune the deployed policy to maximize the same holdout.",
  "top_10": [
    {
      "classifier_weight": 0.88,
      "poisson_weight": 0.12,
      "draw_floor": 0.1,
      "draw_ceiling": 0.34,
      "objective": 0.96796,
      "log_loss": 0.883131,
      "rps": 0.171092,
      "brier": 0.522553,
      "ece": 0.034096,
      "draw_recall": 0.001147,
      "draw_expected_rate": 0.23815,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.011064,
      "entropy": 0.795813,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.88,
      "poisson_weight": 0.12,
      "draw_floor": 0.08,
      "draw_ceiling": 0.34,
      "objective": 0.967962,
      "log_loss": 0.882678,
      "rps": 0.171058,
      "brier": 0.522412,
      "ece": 0.032911,
      "draw_recall": 0.001147,
      "draw_expected_rate": 0.236802,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.012412,
      "entropy": 0.793038,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.88,
      "poisson_weight": 0.12,
      "draw_floor": 0.1,
      "draw_ceiling": 0.42,
      "objective": 0.967963,
      "log_loss": 0.883219,
      "rps": 0.171098,
      "brier": 0.522607,
      "ece": 0.034827,
      "draw_recall": 0.005734,
      "draw_expected_rate": 0.238505,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.010709,
      "entropy": 0.795809,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.88,
      "poisson_weight": 0.12,
      "draw_floor": 0.1,
      "draw_ceiling": 0.46,
      "objective": 0.967963,
      "log_loss": 0.883219,
      "rps": 0.171098,
      "brier": 0.522607,
      "ece": 0.034827,
      "draw_recall": 0.005734,
      "draw_expected_rate": 0.238505,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.010709,
      "entropy": 0.795809,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.88,
      "poisson_weight": 0.12,
      "draw_floor": 0.08,
      "draw_ceiling": 0.42,
      "objective": 0.967965,
      "log_loss": 0.882767,
      "rps": 0.171064,
      "brier": 0.522465,
      "ece": 0.033642,
      "draw_recall": 0.005734,
      "draw_expected_rate": 0.237157,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.012057,
      "entropy": 0.793034,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.88,
      "poisson_weight": 0.12,
      "draw_floor": 0.08,
      "draw_ceiling": 0.46,
      "objective": 0.967965,
      "log_loss": 0.882767,
      "rps": 0.171064,
      "brier": 0.522465,
      "ece": 0.033642,
      "draw_recall": 0.005734,
      "draw_expected_rate": 0.237157,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.012057,
      "entropy": 0.793034,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.88,
      "poisson_weight": 0.12,
      "draw_floor": 0.1,
      "draw_ceiling": 0.38,
      "objective": 0.967969,
      "log_loss": 0.883224,
      "rps": 0.171099,
      "brier": 0.52261,
      "ece": 0.034826,
      "draw_recall": 0.005734,
      "draw_expected_rate": 0.238503,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.010711,
      "entropy": 0.795809,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.88,
      "poisson_weight": 0.12,
      "draw_floor": 0.08,
      "draw_ceiling": 0.38,
      "objective": 0.96797,
      "log_loss": 0.882771,
      "rps": 0.171065,
      "brier": 0.522469,
      "ece": 0.033641,
      "draw_recall": 0.005734,
      "draw_expected_rate": 0.237155,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.012059,
      "entropy": 0.793034,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.86,
      "poisson_weight": 0.14,
      "draw_floor": 0.1,
      "draw_ceiling": 0.42,
      "objective": 0.968041,
      "log_loss": 0.883235,
      "rps": 0.171112,
      "brier": 0.522628,
      "ece": 0.034009,
      "draw_recall": 0.005734,
      "draw_expected_rate": 0.238214,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.011,
      "entropy": 0.795738,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.86,
      "poisson_weight": 0.14,
      "draw_floor": 0.1,
      "draw_ceiling": 0.46,
      "objective": 0.968041,
      "log_loss": 0.883235,
      "rps": 0.171112,
      "brier": 0.522628,
      "ece": 0.034009,
      "draw_recall": 0.005734,
      "draw_expected_rate": 0.238214,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.011,
      "entropy": 0.795738,
      "randomness_penalty": 0.0
    }
  ],
  "best_by_classifier_weight": [
    {
      "classifier_weight": 0.5,
      "poisson_weight": 0.5,
      "draw_floor": 0.1,
      "draw_ceiling": 0.42,
      "objective": 0.971508,
      "log_loss": 0.88472,
      "rps": 0.171584,
      "brier": 0.523628,
      "ece": 0.028891,
      "draw_recall": 0.0,
      "draw_expected_rate": 0.232974,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.01624,
      "entropy": 0.793404,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.52,
      "poisson_weight": 0.48,
      "draw_floor": 0.1,
      "draw_ceiling": 0.42,
      "objective": 0.971246,
      "log_loss": 0.884575,
      "rps": 0.171546,
      "brier": 0.523542,
      "ece": 0.029144,
      "draw_recall": 0.0,
      "draw_expected_rate": 0.233266,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.015949,
      "entropy": 0.793587,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.54,
      "poisson_weight": 0.46,
      "draw_floor": 0.1,
      "draw_ceiling": 0.42,
      "objective": 0.970914,
      "log_loss": 0.884437,
      "rps": 0.17151,
      "brier": 0.52346,
      "ece": 0.028425,
      "draw_recall": 0.0,
      "draw_expected_rate": 0.233557,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.015657,
      "entropy": 0.793763,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.56,
      "poisson_weight": 0.44,
      "draw_floor": 0.1,
      "draw_ceiling": 0.42,
      "objective": 0.970703,
      "log_loss": 0.884308,
      "rps": 0.171476,
      "brier": 0.523382,
      "ece": 0.02912,
      "draw_recall": 0.0,
      "draw_expected_rate": 0.233848,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.015366,
      "entropy": 0.793933,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.58,
      "poisson_weight": 0.42,
      "draw_floor": 0.1,
      "draw_ceiling": 0.42,
      "objective": 0.97053,
      "log_loss": 0.884186,
      "rps": 0.171442,
      "brier": 0.523307,
      "ece": 0.030176,
      "draw_recall": 0.0,
      "draw_expected_rate": 0.234139,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.015075,
      "entropy": 0.794096,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.6,
      "poisson_weight": 0.4,
      "draw_floor": 0.1,
      "draw_ceiling": 0.42,
      "objective": 0.970167,
      "log_loss": 0.884071,
      "rps": 0.17141,
      "brier": 0.523235,
      "ece": 0.02877,
      "draw_recall": 0.0,
      "draw_expected_rate": 0.23443,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.014784,
      "entropy": 0.794254,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.62,
      "poisson_weight": 0.38,
      "draw_floor": 0.1,
      "draw_ceiling": 0.42,
      "objective": 0.970005,
      "log_loss": 0.883965,
      "rps": 0.171379,
      "brier": 0.523167,
      "ece": 0.029764,
      "draw_recall": 0.0,
      "draw_expected_rate": 0.234721,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.014493,
      "entropy": 0.794405,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.64,
      "poisson_weight": 0.36,
      "draw_floor": 0.1,
      "draw_ceiling": 0.42,
      "objective": 0.969841,
      "log_loss": 0.883865,
      "rps": 0.17135,
      "brier": 0.523103,
      "ece": 0.030639,
      "draw_recall": 0.0,
      "draw_expected_rate": 0.235012,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.014202,
      "entropy": 0.79455,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.66,
      "poisson_weight": 0.34,
      "draw_floor": 0.1,
      "draw_ceiling": 0.42,
      "objective": 0.969648,
      "log_loss": 0.883773,
      "rps": 0.171322,
      "brier": 0.523042,
      "ece": 0.031054,
      "draw_recall": 0.0,
      "draw_expected_rate": 0.235303,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.013911,
      "entropy": 0.794688,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.68,
      "poisson_weight": 0.32,
      "draw_floor": 0.1,
      "draw_ceiling": 0.42,
      "objective": 0.969413,
      "log_loss": 0.883687,
      "rps": 0.171295,
      "brier": 0.522985,
      "ece": 0.030844,
      "draw_recall": 0.0,
      "draw_expected_rate": 0.235594,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.01362,
      "entropy": 0.794821,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.7,
      "poisson_weight": 0.3,
      "draw_floor": 0.1,
      "draw_ceiling": 0.42,
      "objective": 0.969231,
      "log_loss": 0.883609,
      "rps": 0.171269,
      "brier": 0.522931,
      "ece": 0.031205,
      "draw_recall": 0.001147,
      "draw_expected_rate": 0.235885,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.013329,
      "entropy": 0.794947,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.72,
      "poisson_weight": 0.28,
      "draw_floor": 0.1,
      "draw_ceiling": 0.42,
      "objective": 0.968988,
      "log_loss": 0.883538,
      "rps": 0.171245,
      "brier": 0.522881,
      "ece": 0.030707,
      "draw_recall": 0.001147,
      "draw_expected_rate": 0.236176,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.013038,
      "entropy": 0.795067,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.74,
      "poisson_weight": 0.26,
      "draw_floor": 0.1,
      "draw_ceiling": 0.42,
      "objective": 0.968867,
      "log_loss": 0.883474,
      "rps": 0.171222,
      "brier": 0.522834,
      "ece": 0.031634,
      "draw_recall": 0.002294,
      "draw_expected_rate": 0.236467,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.012747,
      "entropy": 0.795181,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.76,
      "poisson_weight": 0.24,
      "draw_floor": 0.1,
      "draw_ceiling": 0.42,
      "objective": 0.968808,
      "log_loss": 0.883417,
      "rps": 0.1712,
      "brier": 0.522791,
      "ece": 0.033243,
      "draw_recall": 0.00344,
      "draw_expected_rate": 0.236758,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.012456,
      "entropy": 0.795289,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.78,
      "poisson_weight": 0.22,
      "draw_floor": 0.1,
      "draw_ceiling": 0.42,
      "objective": 0.96866,
      "log_loss": 0.883367,
      "rps": 0.17118,
      "brier": 0.522751,
      "ece": 0.033655,
      "draw_recall": 0.00344,
      "draw_expected_rate": 0.237049,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.012165,
      "entropy": 0.795391,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.8,
      "poisson_weight": 0.2,
      "draw_floor": 0.1,
      "draw_ceiling": 0.42,
      "objective": 0.968461,
      "log_loss": 0.883324,
      "rps": 0.171161,
      "brier": 0.522715,
      "ece": 0.033322,
      "draw_recall": 0.004587,
      "draw_expected_rate": 0.237341,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.011874,
      "entropy": 0.795487,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.82,
      "poisson_weight": 0.18,
      "draw_floor": 0.1,
      "draw_ceiling": 0.42,
      "objective": 0.968251,
      "log_loss": 0.883288,
      "rps": 0.171144,
      "brier": 0.522683,
      "ece": 0.032767,
      "draw_recall": 0.005734,
      "draw_expected_rate": 0.237632,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.011582,
      "entropy": 0.795577,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.84,
      "poisson_weight": 0.16,
      "draw_floor": 0.1,
      "draw_ceiling": 0.42,
      "objective": 0.968103,
      "log_loss": 0.883258,
      "rps": 0.171127,
      "brier": 0.522654,
      "ece": 0.032902,
      "draw_recall": 0.005734,
      "draw_expected_rate": 0.237923,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.011291,
      "entropy": 0.79566,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.86,
      "poisson_weight": 0.14,
      "draw_floor": 0.1,
      "draw_ceiling": 0.42,
      "objective": 0.968041,
      "log_loss": 0.883235,
      "rps": 0.171112,
      "brier": 0.522628,
      "ece": 0.034009,
      "draw_recall": 0.005734,
      "draw_expected_rate": 0.238214,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.011,
      "entropy": 0.795738,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.88,
      "poisson_weight": 0.12,
      "draw_floor": 0.1,
      "draw_ceiling": 0.34,
      "objective": 0.96796,
      "log_loss": 0.883131,
      "rps": 0.171092,
      "brier": 0.522553,
      "ece": 0.034096,
      "draw_recall": 0.001147,
      "draw_expected_rate": 0.23815,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.011064,
      "entropy": 0.795813,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.9,
      "poisson_weight": 0.1,
      "draw_floor": 0.08,
      "draw_ceiling": 0.34,
      "objective": 0.972835,
      "log_loss": 0.882646,
      "rps": 0.171043,
      "brier": 0.522384,
      "ece": 0.033121,
      "draw_recall": 0.001147,
      "draw_expected_rate": 0.237055,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.012159,
      "entropy": 0.793052,
      "randomness_penalty": 0.02
    },
    {
      "classifier_weight": 0.92,
      "poisson_weight": 0.08,
      "draw_floor": 0.08,
      "draw_ceiling": 0.34,
      "objective": 0.977697,
      "log_loss": 0.882621,
      "rps": 0.17103,
      "brier": 0.52236,
      "ece": 0.033097,
      "draw_recall": 0.002294,
      "draw_expected_rate": 0.237307,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.011907,
      "entropy": 0.79306,
      "randomness_penalty": 0.04
    },
    {
      "classifier_weight": 0.94,
      "poisson_weight": 0.06,
      "draw_floor": 0.08,
      "draw_ceiling": 0.34,
      "objective": 0.982598,
      "log_loss": 0.882603,
      "rps": 0.171018,
      "brier": 0.52234,
      "ece": 0.033477,
      "draw_recall": 0.002294,
      "draw_expected_rate": 0.237559,
      "draw_actual_rate": 0.249214,
      "draw_gap": 0.011655,
      "entropy": 0.793063,
      "randomness_penalty": 0.06
    }
  ]
}
```

## 5. Dixon-Coles

| rho | hybrid_objective | hybrid_log_loss | hybrid_draw_gap | poisson_only_log_loss |
| --- | --- | --- | --- | --- |
| -0.18 | 0.970172 | 0.882549 | 0.017093 | 0.890833 |
| -0.16 | 0.970429 | 0.882591 | 0.017565 | 0.890805 |
| -0.14 | 0.970709 | 0.882635 | 0.018037 | 0.890885 |
| -0.12 | 0.9709 | 0.88268 | 0.018508 | 0.891072 |
| -0.1 | 0.971194 | 0.882727 | 0.01898 | 0.891368 |
| -0.08 | 0.971502 | 0.882776 | 0.019452 | 0.891774 |
| -0.06 | 0.971684 | 0.882826 | 0.019923 | 0.892292 |
| -0.04 | 0.971882 | 0.882877 | 0.020395 | 0.892923 |

## Benchmark externo disponível

```json
{
  "status": "same_window_calibration_baseline_available",
  "market_odds_benchmark": {
    "available": false,
    "reason": "O pacote atual nao contem odds historicas limpas de casas de aposta; nao inventamos benchmark externo sem dados auditaveis."
  },
  "same_window_elo_1x2": {
    "rows": 3499,
    "accuracy": 0.57988,
    "log_loss": 0.915315,
    "rps": 0.177341,
    "brier": 0.53977,
    "ece": 0.024865,
    "draw_recall": 0.0,
    "draw_expected_rate": 0.223912,
    "draw_actual_rate": 0.249214,
    "draw_gap": 0.025302,
    "entropy": 0.851209
  },
  "package_holdout_context_only": {
    "elo_accuracy": 0.5733,
    "fifa_rank_accuracy": 0.5513,
    "xgb_calibrated_log_loss": 0.8847,
    "competitive_xgb_log_loss": 0.8701
  },
  "runtime_policy": {
    "accuracy": 0.578737,
    "log_loss": 0.882776,
    "rps": 0.171089,
    "accuracy_gain_vs_elo_pp": -0.114,
    "log_loss_gain_vs_same_window_elo": 0.032539,
    "rps_gain_vs_same_window_elo": 0.006252,
    "log_loss_gap_vs_xgb_calibrated": -0.001924,
    "log_loss_gap_vs_competitive_xgb": 0.012676
  },
  "interpretation": "O runtime e o ELO sao medidos no mesmo recorte 2024+ por log-loss e RPS; acuracia de classe fica apenas como diagnostico. Isso evita comparar recortes diferentes ou premiar uma classe majoritaria."
}
```

## Auditoria dos ajustes 2026

Os proxies 2026 de elenco e Transfermarkt permanecem na camada de xG/Poisson. Como não existem snapshots históricos equivalentes no pacote, eles não deslocam manualmente a probabilidade 1X2 calibrada; a auditoria confirma essa separação.

```json
{
  "path": "/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_runtime_adjustment_audit.csv",
  "method": "all ordered 2026 qualified-team pairs; verify squad/Transfermarkt snapshots do not directly shift calibrated 1X2 probabilities",
  "teams": 48,
  "pairs": 2256,
  "max_abs_shift_pre_draw": 0.0,
  "p95_abs_shift_pre_draw": 0.0,
  "mean_abs_shift_pre_draw": 0.0,
  "argmax_flip_rate_pre_draw": 0.0,
  "max_abs_shift_final": 0.044486,
  "p95_abs_shift_final": 0.016388,
  "decision": "direct_1x2_adjustment_removed",
  "reason": "Ajustes de elenco/Transfermarkt usam proxies 2026 sem snapshots hist\u00f3ricos equivalentes. Eles permanecem apenas na camada de xG/Poisson, n\u00e3o como deslocamento manual da probabilidade 1X2 calibrada.",
  "top_10": [
    {
      "home_team": "Algeria",
      "away_team": "Argentina",
      "base_home": 0.092248,
      "base_draw": 0.147087,
      "base_away": 0.760665,
      "adjusted_home": 0.092248,
      "adjusted_draw": 0.147087,
      "adjusted_away": 0.760665,
      "final_home": 0.092248,
      "final_draw": 0.147087,
      "final_away": 0.760665,
      "max_abs_shift_pre_draw": 0.0,
      "sum_abs_shift_pre_draw": 0.0,
      "max_abs_shift_final": 0.0,
      "base_argmax": "fora",
      "adjusted_argmax": "fora",
      "final_argmax": "fora",
      "argmax_changed_pre_draw": false,
      "argmax_changed_final": false,
      "squad_top26_diff": -6.076923,
      "tm_market_value_log_diff": -1.013485,
      "tm_caps_diff": 0.122119,
      "tm_recent_injury_days_diff": -0.201436,
      "context_shift": 0.0
    },
    {
      "home_team": "Panama",
      "away_team": "Austria",
      "base_home": 0.220892,
      "base_draw": 0.295424,
      "base_away": 0.483684,
      "adjusted_home": 0.220892,
      "adjusted_draw": 0.295424,
      "adjusted_away": 0.483684,
      "final_home": 0.220892,
      "final_draw": 0.295424,
      "final_away": 0.483684,
      "max_abs_shift_pre_draw": 0.0,
      "sum_abs_shift_pre_draw": 0.0,
      "max_abs_shift_final": 0.0,
      "base_argmax": "fora",
      "adjusted_argmax": "fora",
      "final_argmax": "fora",
      "argmax_changed_pre_draw": false,
      "argmax_changed_final": false,
      "squad_top26_diff": -10.379487,
      "tm_market_value_log_diff": -2.048614,
      "tm_caps_diff": -0.49819,
      "tm_recent_injury_days_diff": -0.139308,
      "context_shift": 0.0
    },
    {
      "home_team": "Norway",
      "away_team": "USA",
      "base_home": 0.50496,
      "base_draw": 0.281636,
      "base_away": 0.213404,
      "adjusted_home": 0.50496,
      "adjusted_draw": 0.281636,
      "adjusted_away": 0.213404,
      "final_home": 0.50496,
      "final_draw": 0.281636,
      "final_away": 0.213404,
      "max_abs_shift_pre_draw": 0.0,
      "sum_abs_shift_pre_draw": 0.0,
      "max_abs_shift_final": 0.0,
      "base_argmax": "casa",
      "adjusted_argmax": "casa",
      "final_argmax": "casa",
      "argmax_changed_pre_draw": false,
      "argmax_changed_final": false,
      "squad_top26_diff": 0.0,
      "tm_market_value_log_diff": 0.19841,
      "tm_caps_diff": -0.084687,
      "tm_recent_injury_days_diff": -0.51822,
      "context_shift": 0.0
    },
    {
      "home_team": "Norway",
      "away_team": "Uruguay",
      "base_home": 0.312649,
      "base_draw": 0.29165,
      "base_away": 0.395701,
      "adjusted_home": 0.312649,
      "adjusted_draw": 0.29165,
      "adjusted_away": 0.395701,
      "final_home": 0.312649,
      "final_draw": 0.29165,
      "final_away": 0.395701,
      "max_abs_shift_pre_draw": 0.0,
      "sum_abs_shift_pre_draw": 0.0,
      "max_abs_shift_final": 0.0,
      "base_argmax": "fora",
      "adjusted_argmax": "fora",
      "final_argmax": "fora",
      "argmax_changed_pre_draw": false,
      "argmax_changed_final": false,
      "squad_top26_diff": -1.653846,
      "tm_market_value_log_diff": 0.142057,
      "tm_caps_diff": -0.039553,
      "tm_recent_injury_days_diff": -0.258422,
      "context_shift": 0.0
    },
    {
      "home_team": "Norway",
      "away_team": "Uzbekistan",
      "base_home": 0.561169,
      "base_draw": 0.265735,
      "base_away": 0.173096,
      "adjusted_home": 0.561169,
      "adjusted_draw": 0.265735,
      "adjusted_away": 0.173096,
      "final_home": 0.561169,
      "final_draw": 0.265735,
      "final_away": 0.173096,
      "max_abs_shift_pre_draw": 0.0,
      "sum_abs_shift_pre_draw": 0.0,
      "max_abs_shift_final": 0.0,
      "base_argmax": "casa",
      "adjusted_argmax": "casa",
      "final_argmax": "casa",
      "argmax_changed_pre_draw": false,
      "argmax_changed_final": false,
      "squad_top26_diff": 7.407692,
      "tm_market_value_log_diff": 2.534147,
      "tm_caps_diff": 0.287362,
      "tm_recent_injury_days_diff": 1.328802,
      "context_shift": 0.0
    },
    {
      "home_team": "Panama",
      "away_team": "Algeria",
      "base_home": 0.345306,
      "base_draw": 0.281258,
      "base_away": 0.373436,
      "adjusted_home": 0.345306,
      "adjusted_draw": 0.281258,
      "adjusted_away": 0.373436,
      "final_home": 0.345306,
      "final_draw": 0.281258,
      "final_away": 0.373436,
      "max_abs_shift_pre_draw": 0.0,
      "sum_abs_shift_pre_draw": 0.0,
      "max_abs_shift_final": 0.0,
      "base_argmax": "fora",
      "adjusted_argmax": "fora",
      "final_argmax": "fora",
      "argmax_changed_pre_draw": false,
      "argmax_changed_final": false,
      "squad_top26_diff": -9.687179,
      "tm_market_value_log_diff": -2.334914,
      "tm_caps_diff": -0.190015,
      "tm_recent_injury_days_diff": -0.096047,
      "context_shift": 0.0
    },
    {
      "home_team": "Panama",
      "away_team": "Argentina",
      "base_home": 0.077779,
      "base_draw": 0.13726,
      "base_away": 0.78496,
      "adjusted_home": 0.077779,
      "adjusted_draw": 0.13726,
      "adjusted_away": 0.78496,
      "final_home": 0.077779,
      "final_draw": 0.13726,
      "final_away": 0.78496,
      "max_abs_shift_pre_draw": 0.0,
      "sum_abs_shift_pre_draw": 0.0,
      "max_abs_shift_final": 0.0,
      "base_argmax": "fora",
      "adjusted_argmax": "fora",
      "final_argmax": "fora",
      "argmax_changed_pre_draw": false,
      "argmax_changed_final": false,
      "squad_top26_diff": -15.764103,
      "tm_market_value_log_diff": -3.348399,
      "tm_caps_diff": -0.067896,
      "tm_recent_injury_days_diff": -0.297483,
      "context_shift": 0.0
    },
    {
      "home_team": "Panama",
      "away_team": "Australia",
      "base_home": 0.304612,
      "base_draw": 0.287912,
      "base_away": 0.407476,
      "adjusted_home": 0.304612,
      "adjusted_draw": 0.287912,
      "adjusted_away": 0.407476,
      "final_home": 0.304612,
      "final_draw": 0.287912,
      "final_away": 0.407476,
      "max_abs_shift_pre_draw": 0.0,
      "sum_abs_shift_pre_draw": 0.0,
      "max_abs_shift_final": 0.0,
      "base_argmax": "fora",
      "adjusted_argmax": "fora",
      "final_argmax": "fora",
      "argmax_changed_pre_draw": false,
      "argmax_changed_final": false,
      "squad_top26_diff": -4.494872,
      "tm_market_value_log_diff": -0.733741,
      "tm_caps_diff": -0.014837,
      "tm_recent_injury_days_diff": -0.704668,
      "context_shift": 0.0
    },
    {
      "home_team": "Panama",
      "away_team": "Belgium",
      "base_home": 0.191827,
      "base_draw": 0.290579,
      "base_away": 0.517594,
      "adjusted_home": 0.191827,
      "adjusted_draw": 0.290579,
      "adjusted_away": 0.517594,
      "final_home": 0.191827,
      "final_draw": 0.290579,
      "final_away": 0.517594,
      "max_abs_shift_pre_draw": 0.0,
      "sum_abs_shift_pre_draw": 0.0,
      "max_abs_shift_final": 0.0,
      "base_argmax": "fora",
      "adjusted_argmax": "fora",
      "final_argmax": "fora",
      "argmax_changed_pre_draw": false,
      "argmax_changed_final": false,
      "squad_top26_diff": -14.225641,
      "tm_market_value_log_diff": -2.974071,
      "tm_caps_diff": -0.397963,
      "tm_recent_injury_days_diff": -0.370484,
      "context_shift": 0.0
    },
    {
      "home_team": "Norway",
      "away_team": "Tunisia",
      "base_home": 0.573615,
      "base_draw": 0.249343,
      "base_away": 0.177041,
      "adjusted_home": 0.573615,
      "adjusted_draw": 0.249343,
      "adjusted_away": 0.177041,
      "final_home": 0.573615,
      "final_draw": 0.249343,
      "final_away": 0.177041,
      "max_abs_shift_pre_draw": 0.0,
      "sum_abs_shift_pre_draw": 0.0,
      "max_abs_shift_final": 0.0,
      "base_argmax": "casa",
      "adjusted_argmax": "casa",
      "final_argmax": "casa",
      "argmax_changed_pre_draw": false,
      "argmax_changed_final": false,
      "squad_top26_diff": 7.846154,
      "tm_market_value_log_diff": 1.736924,
      "tm_caps_diff": 0.55602,
      "tm_recent_injury_days_diff": 0.104581,
      "context_shift": 0.0
    }
  ],
  "policy_reference": {
    "classifier_weight": 0.88,
    "poisson_weight": 0.12,
    "draw_floor": 0.04,
    "draw_ceiling": 0.3
  }
}
```

## Escopo do carimbo acadêmico

Carimbo academico pragmatico: valida componente, politica, empate, Poisson/Dixon-Coles, incerteza e baselines publicos disponiveis no pacote. Nao declara superioridade contra mercado sem odds historicas externas.

Itens não promovidos por desenho:

| item | reason |
| --- | --- |
| odds de mercado/bookmakers | ausentes no pacote atual; sem fonte auditavel nao ha benchmark externo honesto |
| Dixon-Coles hierarquico completo com shrinkage bayesiano | aproximacao ataque/defesa por selecao foi testada com historico pre-2024; nao ganhou materialmente do PoissonRegressor+Dixon-Coles atual |
| isotonic/vector/Dirichlet calibration em runtime | isotonic, temperature, vector scaling e Dirichlet foram testados em 2024->2025+; nenhum venceu objetivo, log_loss e empate ao mesmo tempo |
| ajustes 2026 de elenco/Transfermarkt/contexto como modelo calibrado historicamente | permanecem apenas na camada de xG/Poisson; nao entram como deslocamento manual do 1X2 porque faltam snapshots historicos equivalentes |

## 6. Relatório estatístico

Arquivos gerados:

- `/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_statistical_report.json`
- `/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_calibration_bins.csv`
- `/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_class_calibration_summary.csv`
- `/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_block_bootstrap_intervals.csv`
- `/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_ablation_study.csv`
- `/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_uncertainty_intervals.csv`
- `/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_stage_uncertainty_intervals.csv`
- `/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_dixon_coles_rho_sensitivity.csv`
- `/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_internal_frontier_experiments.csv`
- `/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_runtime_adjustment_audit.csv`
- `/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_runtime_neutral_order_audit.csv`
- `/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_monte_carlo_stability.json`
- `/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_raw_data_manifest.json`
- `/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_raw_data_manifest.csv`
