# Auditoria estatística SOTA/KISS

Este relatório executa os checks estatísticos ativos e a ablação completa dos subconjuntos do blend.

## Model Card Resumido

Esta seção preserva no relatório canônico o resumo que antes ficava em `model_card.md`.

```json
{
  "version": "worldcup_2026_sota_v3",
  "simulation_policy": {
    "name": "hybrid_classifier_poisson",
    "classifier_weight": 0.88,
    "poisson_weight": 0.12,
    "draw_floor": 0.04,
    "draw_ceiling": 0.46,
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
    "backtest_rows": 2570,
    "candidate_count": 690,
    "backtest_metrics": {
      "classifier_weight": 0.88,
      "poisson_weight": 0.12,
      "draw_floor": 0.04,
      "draw_ceiling": 0.46,
      "objective": 0.807857,
      "log_loss": 0.740046,
      "rps": 0.136614,
      "brier": 0.443882,
      "ece": 0.017843,
      "draw_recall": 0.021036,
      "draw_expected_rate": 0.233025,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.007442,
      "entropy": 0.678402,
      "randomness_penalty": 0.0
    },
    "holdout_best_metrics": {
      "classifier_weight": 0.88,
      "poisson_weight": 0.12,
      "draw_floor": 0.06,
      "draw_ceiling": 0.46,
      "objective": 0.807422,
      "log_loss": 0.73999,
      "rps": 0.13661,
      "brier": 0.443875,
      "ece": 0.018606,
      "draw_recall": 0.021036,
      "draw_expected_rate": 0.233879,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.006588,
      "entropy": 0.680666,
      "randomness_penalty": 0.0
    },
    "reference_metrics": {
      "reference_policy_0_62": {
        "classifier_weight": 0.62,
        "poisson_weight": 0.38,
        "draw_floor": 0.08,
        "draw_ceiling": 0.38,
        "objective": 0.81941,
        "log_loss": 0.7502,
        "rps": 0.137695,
        "brier": 0.447454,
        "ece": 0.018761,
        "draw_recall": 0.008091,
        "draw_expected_rate": 0.230876,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.00959,
        "entropy": 0.703771,
        "randomness_penalty": 0.0
      },
      "reference_policy_previous_0_80": {
        "classifier_weight": 0.8,
        "poisson_weight": 0.2,
        "draw_floor": 0.08,
        "draw_ceiling": 0.38,
        "objective": 0.813852,
        "log_loss": 0.744637,
        "rps": 0.137139,
        "brier": 0.446176,
        "ece": 0.023402,
        "draw_recall": 0.003236,
        "draw_expected_rate": 0.231199,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.009268,
        "entropy": 0.690164,
        "randomness_penalty": 0.0
      }
    },
    "candidate_metrics": [
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.06,
        "draw_ceiling": 0.46,
        "objective": 0.807422,
        "log_loss": 0.73999,
        "rps": 0.13661,
        "brier": 0.443875,
        "ece": 0.018606,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.233879,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.006588,
        "entropy": 0.680666,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.807465,
        "log_loss": 0.74047,
        "rps": 0.13664,
        "brier": 0.444,
        "ece": 0.019135,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.235057,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.00541,
        "entropy": 0.683408,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.04,
        "draw_ceiling": 0.46,
        "objective": 0.807857,
        "log_loss": 0.740046,
        "rps": 0.136614,
        "brier": 0.443882,
        "ece": 0.017843,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.233025,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.007442,
        "entropy": 0.678402,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.1,
        "draw_ceiling": 0.46,
        "objective": 0.807894,
        "log_loss": 0.741477,
        "rps": 0.136725,
        "brier": 0.444333,
        "ece": 0.020035,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.236704,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.003763,
        "entropy": 0.686831,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.06,
        "draw_ceiling": 0.46,
        "objective": 0.807963,
        "log_loss": 0.740482,
        "rps": 0.136652,
        "brier": 0.443983,
        "ece": 0.018504,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.233786,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.006681,
        "entropy": 0.682647,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.807989,
        "log_loss": 0.740943,
        "rps": 0.136681,
        "brier": 0.444105,
        "ece": 0.018916,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.234937,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.00553,
        "entropy": 0.685333,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.04,
        "draw_ceiling": 0.46,
        "objective": 0.808317,
        "log_loss": 0.740543,
        "rps": 0.136655,
        "brier": 0.443991,
        "ece": 0.017762,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.232951,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.007516,
        "entropy": 0.680432,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.1,
        "draw_ceiling": 0.46,
        "objective": 0.808421,
        "log_loss": 0.741921,
        "rps": 0.136763,
        "brier": 0.444427,
        "ece": 0.02004,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.236546,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.00392,
        "entropy": 0.688687,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.12,
        "draw_ceiling": 0.46,
        "objective": 0.80852,
        "log_loss": 0.742846,
        "rps": 0.136864,
        "brier": 0.444869,
        "ece": 0.022379,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.239086,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.001381,
        "entropy": 0.691277,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.06,
        "draw_ceiling": 0.46,
        "objective": 0.808586,
        "log_loss": 0.741009,
        "rps": 0.136698,
        "brier": 0.444102,
        "ece": 0.018965,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.233692,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.006775,
        "entropy": 0.68456,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.80866,
        "log_loss": 0.741452,
        "rps": 0.136726,
        "brier": 0.44422,
        "ece": 0.02007,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.234816,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.005651,
        "entropy": 0.687189,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.04,
        "draw_ceiling": 0.46,
        "objective": 0.808933,
        "log_loss": 0.741076,
        "rps": 0.136701,
        "brier": 0.44411,
        "ece": 0.018166,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.232877,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.00759,
        "entropy": 0.682394,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.12,
        "draw_ceiling": 0.46,
        "objective": 0.809025,
        "log_loss": 0.743253,
        "rps": 0.136898,
        "brier": 0.444946,
        "ece": 0.022331,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.238874,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.001593,
        "entropy": 0.693046,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.1,
        "draw_ceiling": 0.46,
        "objective": 0.809111,
        "log_loss": 0.7424,
        "rps": 0.136806,
        "brier": 0.444532,
        "ece": 0.021623,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.236389,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.004078,
        "entropy": 0.690474,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.82,
        "poisson_weight": 0.18,
        "draw_floor": 0.06,
        "draw_ceiling": 0.46,
        "objective": 0.809329,
        "log_loss": 0.741567,
        "rps": 0.136748,
        "brier": 0.444232,
        "ece": 0.020512,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.233599,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.006868,
        "entropy": 0.686411,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.82,
        "poisson_weight": 0.18,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.809357,
        "log_loss": 0.741992,
        "rps": 0.136775,
        "brier": 0.444346,
        "ece": 0.021143,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.234696,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.005771,
        "entropy": 0.688982,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.06,
        "draw_ceiling": 0.42,
        "objective": 0.809473,
        "log_loss": 0.740922,
        "rps": 0.136765,
        "brier": 0.444659,
        "ece": 0.02357,
        "draw_recall": 0.014563,
        "draw_expected_rate": 0.232364,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.008103,
        "entropy": 0.680575,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.08,
        "draw_ceiling": 0.42,
        "objective": 0.809516,
        "log_loss": 0.741402,
        "rps": 0.136795,
        "brier": 0.444785,
        "ece": 0.024099,
        "draw_recall": 0.014563,
        "draw_expected_rate": 0.233542,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.006925,
        "entropy": 0.683317,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.82,
        "poisson_weight": 0.18,
        "draw_floor": 0.04,
        "draw_ceiling": 0.46,
        "objective": 0.809674,
        "log_loss": 0.741639,
        "rps": 0.136752,
        "brier": 0.44424,
        "ece": 0.019733,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.232803,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.007664,
        "entropy": 0.684293,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.12,
        "draw_ceiling": 0.46,
        "objective": 0.809695,
        "log_loss": 0.743695,
        "rps": 0.136936,
        "brier": 0.445034,
        "ece": 0.02386,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.238662,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.001805,
        "entropy": 0.694745,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.82,
        "poisson_weight": 0.18,
        "draw_floor": 0.1,
        "draw_ceiling": 0.46,
        "objective": 0.809791,
        "log_loss": 0.742911,
        "rps": 0.136853,
        "brier": 0.444648,
        "ece": 0.022659,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.236231,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.004236,
        "entropy": 0.692198,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.06,
        "draw_ceiling": 0.42,
        "objective": 0.809854,
        "log_loss": 0.741396,
        "rps": 0.1368,
        "brier": 0.444749,
        "ece": 0.021908,
        "draw_recall": 0.014563,
        "draw_expected_rate": 0.232305,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.008162,
        "entropy": 0.682554,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.08,
        "draw_ceiling": 0.42,
        "objective": 0.80988,
        "log_loss": 0.741857,
        "rps": 0.136829,
        "brier": 0.44487,
        "ece": 0.02232,
        "draw_recall": 0.014563,
        "draw_expected_rate": 0.233456,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.007011,
        "entropy": 0.68524,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.04,
        "draw_ceiling": 0.42,
        "objective": 0.809913,
        "log_loss": 0.740978,
        "rps": 0.136769,
        "brier": 0.444667,
        "ece": 0.022806,
        "draw_recall": 0.014563,
        "draw_expected_rate": 0.23151,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.008957,
        "entropy": 0.678311,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.809945,
        "log_loss": 0.742408,
        "rps": 0.13688,
        "brier": 0.445118,
        "ece": 0.024999,
        "draw_recall": 0.014563,
        "draw_expected_rate": 0.235189,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.005278,
        "entropy": 0.68674,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.8,
        "poisson_weight": 0.2,
        "draw_floor": 0.06,
        "draw_ceiling": 0.46,
        "objective": 0.810028,
        "log_loss": 0.742152,
        "rps": 0.136803,
        "brier": 0.444372,
        "ece": 0.021155,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.233505,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.006962,
        "entropy": 0.688203,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.8,
        "poisson_weight": 0.2,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.810036,
        "log_loss": 0.742559,
        "rps": 0.136829,
        "brier": 0.444482,
        "ece": 0.021623,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.234576,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.005891,
        "entropy": 0.690717,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.04,
        "draw_ceiling": 0.42,
        "objective": 0.810208,
        "log_loss": 0.741457,
        "rps": 0.136803,
        "brier": 0.444757,
        "ece": 0.021166,
        "draw_recall": 0.014563,
        "draw_expected_rate": 0.23147,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.008996,
        "entropy": 0.680339,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.810312,
        "log_loss": 0.742835,
        "rps": 0.136911,
        "brier": 0.445193,
        "ece": 0.023444,
        "draw_recall": 0.014563,
        "draw_expected_rate": 0.235066,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.005401,
        "entropy": 0.688594,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.82,
        "poisson_weight": 0.18,
        "draw_floor": 0.12,
        "draw_ceiling": 0.46,
        "objective": 0.810353,
        "log_loss": 0.744168,
        "rps": 0.136979,
        "brier": 0.445133,
        "ece": 0.024843,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.23845,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.002017,
        "entropy": 0.696382,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.8,
        "poisson_weight": 0.2,
        "draw_floor": 0.04,
        "draw_ceiling": 0.46,
        "objective": 0.810378,
        "log_loss": 0.742229,
        "rps": 0.136806,
        "brier": 0.444381,
        "ece": 0.020474,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.232729,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.007738,
        "entropy": 0.686134,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.8,
        "poisson_weight": 0.2,
        "draw_floor": 0.1,
        "draw_ceiling": 0.46,
        "objective": 0.810427,
        "log_loss": 0.743449,
        "rps": 0.136904,
        "brier": 0.444774,
        "ece": 0.022789,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.236073,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.004394,
        "entropy": 0.693864,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.06,
        "draw_ceiling": 0.42,
        "objective": 0.810451,
        "log_loss": 0.741905,
        "rps": 0.136839,
        "brier": 0.444848,
        "ece": 0.02249,
        "draw_recall": 0.014563,
        "draw_expected_rate": 0.232246,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.008221,
        "entropy": 0.684466,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.08,
        "draw_ceiling": 0.42,
        "objective": 0.810524,
        "log_loss": 0.742348,
        "rps": 0.136867,
        "brier": 0.444966,
        "ece": 0.023595,
        "draw_recall": 0.014563,
        "draw_expected_rate": 0.23337,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.007097,
        "entropy": 0.687095,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.78,
        "poisson_weight": 0.22,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.810559,
        "log_loss": 0.743151,
        "rps": 0.136888,
        "brier": 0.444629,
        "ece": 0.019822,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.234456,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.006011,
        "entropy": 0.692398,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.78,
        "poisson_weight": 0.22,
        "draw_floor": 0.06,
        "draw_ceiling": 0.46,
        "objective": 0.810567,
        "log_loss": 0.742762,
        "rps": 0.136863,
        "brier": 0.444523,
        "ece": 0.019464,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.233412,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.007055,
        "entropy": 0.689941,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.12,
        "draw_ceiling": 0.42,
        "objective": 0.810571,
        "log_loss": 0.743778,
        "rps": 0.137019,
        "brier": 0.445654,
        "ece": 0.027343,
        "draw_recall": 0.014563,
        "draw_expected_rate": 0.23757,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.002896,
        "entropy": 0.691186,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.82,
        "poisson_weight": 0.18,
        "draw_floor": 0.06,
        "draw_ceiling": 0.42,
        "objective": 0.810744,
        "log_loss": 0.742444,
        "rps": 0.136882,
        "brier": 0.444959,
        "ece": 0.018868,
        "draw_recall": 0.016181,
        "draw_expected_rate": 0.232187,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.00828,
        "entropy": 0.686316,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.82,
        "poisson_weight": 0.18,
        "draw_floor": 0.08,
        "draw_ceiling": 0.42,
        "objective": 0.810772,
        "log_loss": 0.742869,
        "rps": 0.136909,
        "brier": 0.445073,
        "ece": 0.019499,
        "draw_recall": 0.016181,
        "draw_expected_rate": 0.233284,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.007183,
        "entropy": 0.688888,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.14,
        "draw_ceiling": 0.46,
        "objective": 0.810797,
        "log_loss": 0.74455,
        "rps": 0.137054,
        "brier": 0.445616,
        "ece": 0.025608,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.242376,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.001909,
        "entropy": 0.696836,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.04,
        "draw_ceiling": 0.42,
        "objective": 0.810798,
        "log_loss": 0.741972,
        "rps": 0.136842,
        "brier": 0.444857,
        "ece": 0.021691,
        "draw_recall": 0.014563,
        "draw_expected_rate": 0.231431,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.009036,
        "entropy": 0.6823,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.12,
        "draw_ceiling": 0.42,
        "objective": 0.810916,
        "log_loss": 0.744167,
        "rps": 0.137045,
        "brier": 0.445712,
        "ece": 0.025735,
        "draw_recall": 0.014563,
        "draw_expected_rate": 0.237393,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.003074,
        "entropy": 0.692953,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.78,
        "poisson_weight": 0.22,
        "draw_floor": 0.1,
        "draw_ceiling": 0.46,
        "objective": 0.810922,
        "log_loss": 0.744011,
        "rps": 0.13696,
        "brier": 0.444912,
        "ece": 0.020813,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.235916,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.004551,
        "entropy": 0.695475,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.78,
        "poisson_weight": 0.22,
        "draw_floor": 0.04,
        "draw_ceiling": 0.46,
        "objective": 0.810927,
        "log_loss": 0.742843,
        "rps": 0.136866,
        "brier": 0.444532,
        "ece": 0.018958,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.232655,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.007812,
        "entropy": 0.687921,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.8,
        "poisson_weight": 0.2,
        "draw_floor": 0.12,
        "draw_ceiling": 0.46,
        "objective": 0.810968,
        "log_loss": 0.744669,
        "rps": 0.137026,
        "brier": 0.445243,
        "ece": 0.02492,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.238238,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.002229,
        "entropy": 0.697959,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.810976,
        "log_loss": 0.743296,
        "rps": 0.136947,
        "brier": 0.445278,
        "ece": 0.025148,
        "draw_recall": 0.014563,
        "draw_expected_rate": 0.234942,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.005525,
        "entropy": 0.69038,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.14,
        "draw_ceiling": 0.46,
        "objective": 0.811037,
        "log_loss": 0.744909,
        "rps": 0.137081,
        "brier": 0.445668,
        "ece": 0.025486,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.242089,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.001622,
        "entropy": 0.698498,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.82,
        "poisson_weight": 0.18,
        "draw_floor": 0.04,
        "draw_ceiling": 0.42,
        "objective": 0.811089,
        "log_loss": 0.742516,
        "rps": 0.136885,
        "brier": 0.444967,
        "ece": 0.018088,
        "draw_recall": 0.016181,
        "draw_expected_rate": 0.231391,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.009076,
        "entropy": 0.684199,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.76,
        "poisson_weight": 0.24,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.811199,
        "log_loss": 0.743764,
        "rps": 0.136951,
        "brier": 0.444787,
        "ece": 0.019189,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.234335,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.006132,
        "entropy": 0.694028,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.76,
        "poisson_weight": 0.24,
        "draw_floor": 0.06,
        "draw_ceiling": 0.46,
        "objective": 0.811205,
        "log_loss": 0.743393,
        "rps": 0.136926,
        "brier": 0.444685,
        "ece": 0.018705,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.233318,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.007149,
        "entropy": 0.691627,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.82,
        "poisson_weight": 0.18,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.811206,
        "log_loss": 0.743788,
        "rps": 0.136987,
        "brier": 0.445375,
        "ece": 0.021015,
        "draw_recall": 0.016181,
        "draw_expected_rate": 0.234819,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.005648,
        "entropy": 0.692104,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.8,
        "poisson_weight": 0.2,
        "draw_floor": 0.06,
        "draw_ceiling": 0.42,
        "objective": 0.811384,
        "log_loss": 0.743011,
        "rps": 0.13693,
        "brier": 0.44508,
        "ece": 0.019219,
        "draw_recall": 0.016181,
        "draw_expected_rate": 0.232128,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.008339,
        "entropy": 0.688108,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.8,
        "poisson_weight": 0.2,
        "draw_floor": 0.08,
        "draw_ceiling": 0.42,
        "objective": 0.811391,
        "log_loss": 0.743417,
        "rps": 0.136956,
        "brier": 0.44519,
        "ece": 0.019686,
        "draw_recall": 0.016181,
        "draw_expected_rate": 0.233198,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.007269,
        "entropy": 0.690623,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.14,
        "draw_ceiling": 0.46,
        "objective": 0.811441,
        "log_loss": 0.745304,
        "rps": 0.137113,
        "brier": 0.44573,
        "ece": 0.026942,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.241802,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.001335,
        "entropy": 0.700091,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.78,
        "poisson_weight": 0.22,
        "draw_floor": 0.12,
        "draw_ceiling": 0.46,
        "objective": 0.811442,
        "log_loss": 0.745194,
        "rps": 0.137078,
        "brier": 0.445364,
        "ece": 0.022891,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.238027,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.00244,
        "entropy": 0.699481,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.76,
        "poisson_weight": 0.24,
        "draw_floor": 0.1,
        "draw_ceiling": 0.46,
        "objective": 0.811546,
        "log_loss": 0.744596,
        "rps": 0.137021,
        "brier": 0.44506,
        "ece": 0.020161,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.235758,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.004709,
        "entropy": 0.697035,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.12,
        "draw_ceiling": 0.42,
        "objective": 0.811559,
        "log_loss": 0.744591,
        "rps": 0.137077,
        "brier": 0.44578,
        "ece": 0.027385,
        "draw_recall": 0.014563,
        "draw_expected_rate": 0.237216,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.003251,
        "entropy": 0.694651,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.76,
        "poisson_weight": 0.24,
        "draw_floor": 0.04,
        "draw_ceiling": 0.46,
        "objective": 0.811569,
        "log_loss": 0.74348,
        "rps": 0.13693,
        "brier": 0.444694,
        "ece": 0.018301,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.232581,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.007886,
        "entropy": 0.689657,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.14,
        "draw_ceiling": 0.42,
        "objective": 0.811575,
        "log_loss": 0.745482,
        "rps": 0.137209,
        "brier": 0.446401,
        "ece": 0.030572,
        "draw_recall": 0.014563,
        "draw_expected_rate": 0.24086,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.000393,
        "entropy": 0.696745,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.14,
        "draw_ceiling": 0.42,
        "objective": 0.811684,
        "log_loss": 0.745823,
        "rps": 0.137229,
        "brier": 0.446433,
        "ece": 0.02889,
        "draw_recall": 0.014563,
        "draw_expected_rate": 0.240608,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.000141,
        "entropy": 0.698406,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.8,
        "poisson_weight": 0.2,
        "draw_floor": 0.04,
        "draw_ceiling": 0.42,
        "objective": 0.811733,
        "log_loss": 0.743087,
        "rps": 0.136934,
        "brier": 0.445089,
        "ece": 0.018537,
        "draw_recall": 0.016181,
        "draw_expected_rate": 0.231351,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.009115,
        "entropy": 0.68604,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.82,
        "poisson_weight": 0.18,
        "draw_floor": 0.12,
        "draw_ceiling": 0.42,
        "objective": 0.811768,
        "log_loss": 0.745045,
        "rps": 0.137112,
        "brier": 0.44586,
        "ece": 0.023199,
        "draw_recall": 0.016181,
        "draw_expected_rate": 0.237038,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.003429,
        "entropy": 0.696287,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.8,
        "poisson_weight": 0.2,
        "draw_floor": 0.1,
        "draw_ceiling": 0.42,
        "objective": 0.811783,
        "log_loss": 0.744307,
        "rps": 0.137031,
        "brier": 0.445482,
        "ece": 0.020852,
        "draw_recall": 0.016181,
        "draw_expected_rate": 0.234696,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.005771,
        "entropy": 0.693769,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.9,
        "poisson_weight": 0.1,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.811831,
        "log_loss": 0.740039,
        "rps": 0.136604,
        "brier": 0.443907,
        "ece": 0.017419,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.235177,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.00529,
        "entropy": 0.681408,
        "randomness_penalty": 0.02
      },
      {
        "classifier_weight": 0.82,
        "poisson_weight": 0.18,
        "draw_floor": 0.14,
        "draw_ceiling": 0.46,
        "objective": 0.811834,
        "log_loss": 0.74573,
        "rps": 0.13715,
        "brier": 0.445805,
        "ece": 0.027852,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.241516,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.001049,
        "entropy": 0.70162,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.9,
        "poisson_weight": 0.1,
        "draw_floor": 0.06,
        "draw_ceiling": 0.46,
        "objective": 0.811865,
        "log_loss": 0.73954,
        "rps": 0.136573,
        "brier": 0.443777,
        "ece": 0.017084,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.233973,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.006494,
        "entropy": 0.678609,
        "randomness_penalty": 0.02
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.06,
        "draw_ceiling": 0.38,
        "objective": 0.8119,
        "log_loss": 0.74226,
        "rps": 0.13699,
        "brier": 0.445768,
        "ece": 0.024095,
        "draw_recall": 0.004854,
        "draw_expected_rate": 0.230165,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.010302,
        "entropy": 0.680075,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.08,
        "draw_ceiling": 0.38,
        "objective": 0.811943,
        "log_loss": 0.74274,
        "rps": 0.13702,
        "brier": 0.445894,
        "ece": 0.024624,
        "draw_recall": 0.004854,
        "draw_expected_rate": 0.231343,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.009124,
        "entropy": 0.682817,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.74,
        "poisson_weight": 0.26,
        "draw_floor": 0.06,
        "draw_ceiling": 0.46,
        "objective": 0.812006,
        "log_loss": 0.744046,
        "rps": 0.136995,
        "brier": 0.444857,
        "ece": 0.019719,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.233225,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.007242,
        "entropy": 0.693266,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.74,
        "poisson_weight": 0.26,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.812034,
        "log_loss": 0.744399,
        "rps": 0.137018,
        "brier": 0.444955,
        "ece": 0.0207,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.234215,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.006252,
        "entropy": 0.695609,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.76,
        "poisson_weight": 0.24,
        "draw_floor": 0.12,
        "draw_ceiling": 0.46,
        "objective": 0.812045,
        "log_loss": 0.745741,
        "rps": 0.137134,
        "brier": 0.445496,
        "ece": 0.022185,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.237815,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.002652,
        "entropy": 0.700951,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.78,
        "poisson_weight": 0.22,
        "draw_floor": 0.08,
        "draw_ceiling": 0.42,
        "objective": 0.812098,
        "log_loss": 0.743989,
        "rps": 0.137008,
        "brier": 0.445318,
        "ece": 0.020639,
        "draw_recall": 0.014563,
        "draw_expected_rate": 0.233112,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.007354,
        "entropy": 0.692303,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.78,
        "poisson_weight": 0.22,
        "draw_floor": 0.06,
        "draw_ceiling": 0.42,
        "objective": 0.812106,
        "log_loss": 0.743601,
        "rps": 0.136983,
        "brier": 0.445212,
        "ece": 0.020281,
        "draw_recall": 0.014563,
        "draw_expected_rate": 0.232069,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.008398,
        "entropy": 0.689846,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.14,
        "draw_ceiling": 0.42,
        "objective": 0.812184,
        "log_loss": 0.7462,
        "rps": 0.137254,
        "brier": 0.446477,
        "ece": 0.030467,
        "draw_recall": 0.014563,
        "draw_expected_rate": 0.240356,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.000111,
        "entropy": 0.699997,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.8,
        "poisson_weight": 0.2,
        "draw_floor": 0.14,
        "draw_ceiling": 0.46,
        "objective": 0.812184,
        "log_loss": 0.746184,
        "rps": 0.137191,
        "brier": 0.44589,
        "ece": 0.027855,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.241229,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.000762,
        "entropy": 0.703089,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.06,
        "draw_ceiling": 0.38,
        "objective": 0.812243,
        "log_loss": 0.742705,
        "rps": 0.137013,
        "brier": 0.445826,
        "ece": 0.022652,
        "draw_recall": 0.003236,
        "draw_expected_rate": 0.230156,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.010311,
        "entropy": 0.682064,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.08,
        "draw_ceiling": 0.38,
        "objective": 0.812269,
        "log_loss": 0.743166,
        "rps": 0.137042,
        "brier": 0.445948,
        "ece": 0.023064,
        "draw_recall": 0.003236,
        "draw_expected_rate": 0.231307,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.00916,
        "entropy": 0.684749,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.9,
        "poisson_weight": 0.1,
        "draw_floor": 0.1,
        "draw_ceiling": 0.46,
        "objective": 0.812294,
        "log_loss": 0.741075,
        "rps": 0.136692,
        "brier": 0.44425,
        "ece": 0.018562,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.236862,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.003605,
        "entropy": 0.6849,
        "randomness_penalty": 0.02
      },
      {
        "classifier_weight": 0.8,
        "poisson_weight": 0.2,
        "draw_floor": 0.12,
        "draw_ceiling": 0.42,
        "objective": 0.812324,
        "log_loss": 0.745527,
        "rps": 0.137153,
        "brier": 0.445951,
        "ece": 0.022983,
        "draw_recall": 0.016181,
        "draw_expected_rate": 0.236861,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.003606,
        "entropy": 0.697864,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.9,
        "poisson_weight": 0.1,
        "draw_floor": 0.04,
        "draw_ceiling": 0.46,
        "objective": 0.812336,
        "log_loss": 0.739591,
        "rps": 0.136576,
        "brier": 0.443784,
        "ece": 0.0163,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.233099,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.007367,
        "entropy": 0.676296,
        "randomness_penalty": 0.02
      }
    ],
    "best_by_classifier_weight": [
      {
        "classifier_weight": 0.5,
        "poisson_weight": 0.5,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.822029,
        "log_loss": 0.753307,
        "rps": 0.138178,
        "brier": 0.44781,
        "ece": 0.020577,
        "draw_recall": 0.032362,
        "draw_expected_rate": 0.232772,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.007695,
        "entropy": 0.711497,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.52,
        "poisson_weight": 0.48,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.821115,
        "log_loss": 0.752484,
        "rps": 0.138057,
        "brier": 0.447513,
        "ece": 0.020706,
        "draw_recall": 0.032362,
        "draw_expected_rate": 0.232892,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.007575,
        "entropy": 0.710362,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.54,
        "poisson_weight": 0.46,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.820201,
        "log_loss": 0.751675,
        "rps": 0.13794,
        "brier": 0.447227,
        "ece": 0.020643,
        "draw_recall": 0.030744,
        "draw_expected_rate": 0.233012,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.007455,
        "entropy": 0.709195,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.56,
        "poisson_weight": 0.44,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.819332,
        "log_loss": 0.750879,
        "rps": 0.137828,
        "brier": 0.446952,
        "ece": 0.020954,
        "draw_recall": 0.029126,
        "draw_expected_rate": 0.233133,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.007334,
        "entropy": 0.707997,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.58,
        "poisson_weight": 0.42,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.818339,
        "log_loss": 0.750098,
        "rps": 0.13772,
        "brier": 0.446687,
        "ece": 0.019502,
        "draw_recall": 0.027508,
        "draw_expected_rate": 0.233253,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.007214,
        "entropy": 0.706766,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.6,
        "poisson_weight": 0.4,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.817754,
        "log_loss": 0.749331,
        "rps": 0.137616,
        "brier": 0.446433,
        "ece": 0.022961,
        "draw_recall": 0.027508,
        "draw_expected_rate": 0.233373,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.007094,
        "entropy": 0.705502,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.62,
        "poisson_weight": 0.38,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.816875,
        "log_loss": 0.748579,
        "rps": 0.137517,
        "brier": 0.44619,
        "ece": 0.022526,
        "draw_recall": 0.02589,
        "draw_expected_rate": 0.233493,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.006974,
        "entropy": 0.704203,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.64,
        "poisson_weight": 0.36,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.81613,
        "log_loss": 0.747842,
        "rps": 0.137423,
        "brier": 0.445958,
        "ece": 0.023567,
        "draw_recall": 0.022654,
        "draw_expected_rate": 0.233614,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.006853,
        "entropy": 0.702868,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.66,
        "poisson_weight": 0.34,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.815229,
        "log_loss": 0.74712,
        "rps": 0.137333,
        "brier": 0.445736,
        "ece": 0.022429,
        "draw_recall": 0.019417,
        "draw_expected_rate": 0.233734,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.006733,
        "entropy": 0.701497,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.68,
        "poisson_weight": 0.32,
        "draw_floor": 0.06,
        "draw_ceiling": 0.46,
        "objective": 0.814386,
        "log_loss": 0.746112,
        "rps": 0.137227,
        "brier": 0.445437,
        "ece": 0.020936,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.232944,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.007523,
        "entropy": 0.697917,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.7,
        "poisson_weight": 0.3,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.813666,
        "log_loss": 0.745725,
        "rps": 0.137167,
        "brier": 0.445324,
        "ece": 0.022478,
        "draw_recall": 0.019417,
        "draw_expected_rate": 0.233974,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.006493,
        "entropy": 0.698636,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.72,
        "poisson_weight": 0.28,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.812769,
        "log_loss": 0.745053,
        "rps": 0.13709,
        "brier": 0.445134,
        "ece": 0.020698,
        "draw_recall": 0.019417,
        "draw_expected_rate": 0.234095,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.006372,
        "entropy": 0.697144,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.74,
        "poisson_weight": 0.26,
        "draw_floor": 0.06,
        "draw_ceiling": 0.46,
        "objective": 0.812006,
        "log_loss": 0.744046,
        "rps": 0.136995,
        "brier": 0.444857,
        "ece": 0.019719,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.233225,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.007242,
        "entropy": 0.693266,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.76,
        "poisson_weight": 0.24,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.811199,
        "log_loss": 0.743764,
        "rps": 0.136951,
        "brier": 0.444787,
        "ece": 0.019189,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.234335,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.006132,
        "entropy": 0.694028,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.78,
        "poisson_weight": 0.22,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.810559,
        "log_loss": 0.743151,
        "rps": 0.136888,
        "brier": 0.444629,
        "ece": 0.019822,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.234456,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.006011,
        "entropy": 0.692398,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.8,
        "poisson_weight": 0.2,
        "draw_floor": 0.06,
        "draw_ceiling": 0.46,
        "objective": 0.810028,
        "log_loss": 0.742152,
        "rps": 0.136803,
        "brier": 0.444372,
        "ece": 0.021155,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.233505,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.006962,
        "entropy": 0.688203,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.82,
        "poisson_weight": 0.18,
        "draw_floor": 0.06,
        "draw_ceiling": 0.46,
        "objective": 0.809329,
        "log_loss": 0.741567,
        "rps": 0.136748,
        "brier": 0.444232,
        "ece": 0.020512,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.233599,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.006868,
        "entropy": 0.686411,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.84,
        "poisson_weight": 0.16,
        "draw_floor": 0.06,
        "draw_ceiling": 0.46,
        "objective": 0.808586,
        "log_loss": 0.741009,
        "rps": 0.136698,
        "brier": 0.444102,
        "ece": 0.018965,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.233692,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.006775,
        "entropy": 0.68456,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.86,
        "poisson_weight": 0.14,
        "draw_floor": 0.06,
        "draw_ceiling": 0.46,
        "objective": 0.807963,
        "log_loss": 0.740482,
        "rps": 0.136652,
        "brier": 0.443983,
        "ece": 0.018504,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.233786,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.006681,
        "entropy": 0.682647,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.06,
        "draw_ceiling": 0.46,
        "objective": 0.807422,
        "log_loss": 0.73999,
        "rps": 0.13661,
        "brier": 0.443875,
        "ece": 0.018606,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.233879,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.006588,
        "entropy": 0.680666,
        "randomness_penalty": 0.0
      },
      {
        "classifier_weight": 0.9,
        "poisson_weight": 0.1,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.811831,
        "log_loss": 0.740039,
        "rps": 0.136604,
        "brier": 0.443907,
        "ece": 0.017419,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.235177,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.00529,
        "entropy": 0.681408,
        "randomness_penalty": 0.02
      },
      {
        "classifier_weight": 0.92,
        "poisson_weight": 0.08,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.816555,
        "log_loss": 0.73966,
        "rps": 0.136573,
        "brier": 0.443824,
        "ece": 0.019089,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.235297,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.00517,
        "entropy": 0.679324,
        "randomness_penalty": 0.04
      },
      {
        "classifier_weight": 0.94,
        "poisson_weight": 0.06,
        "draw_floor": 0.08,
        "draw_ceiling": 0.46,
        "objective": 0.821314,
        "log_loss": 0.739348,
        "rps": 0.136546,
        "brier": 0.443751,
        "ece": 0.019383,
        "draw_recall": 0.021036,
        "draw_expected_rate": 0.235418,
        "draw_actual_rate": 0.240467,
        "draw_gap": 0.005049,
        "entropy": 0.677145,
        "randomness_penalty": 0.06
      }
    ],
    "nested_temporal_validation": {
      "version": "nested_temporal_policy_v3_component_ablation_no_leakage_no_draw_xgb",
      "description": "Each outer year trains inner models only before the internal validation window, selects blend components plus classifier/Poisson/draw policy on that later internal window, then refits on all prior data and evaluates the outer year without retuning.",
      "aggregate": {
        "outer_year": 2021.744694,
        "outer_rows": 7963,
        "outer_rho": -0.18,
        "outer_objective": 0.839901,
        "outer_log_loss": 0.767263,
        "outer_rps": 0.141409,
        "outer_brier": 0.452269,
        "outer_ece": 0.036643,
        "outer_draw_recall": 0.022092,
        "outer_draw_expected_rate": 0.233206,
        "outer_draw_actual_rate": 0.233329,
        "outer_draw_gap": 0.011119,
        "outer_entropy": 0.751251,
        "outer_randomness_penalty": 0.0,
        "folds": 8
      },
      "rows": [
        {
          "outer_year": 2018,
          "inner_train_start": "1990-01-12",
          "inner_train_end": "2015-12-28",
          "inner_train_rows": 22413,
          "inner_start": "2015-12-31",
          "inner_end": "2017-12-29",
          "inner_rows": 1812,
          "outer_rows": 896,
          "inner_rho": -0.18,
          "outer_rho": -0.18,
          "selected_classifier_weight": 0.88,
          "selected_poisson_weight": 0.12,
          "selected_draw_floor": 0.04,
          "selected_draw_ceiling": 0.42,
          "inner_objective": 0.884681,
          "outer_objective": 0.909137,
          "outer_log_loss": 0.833329,
          "outer_rps": 0.155875,
          "outer_brier": 0.494501,
          "outer_ece": 0.044302,
          "outer_draw_recall": 0.026087,
          "outer_draw_expected_rate": 0.254847,
          "outer_draw_actual_rate": 0.256696,
          "outer_draw_gap": 0.001849,
          "outer_entropy": 0.810639,
          "outer_randomness_penalty": 0.0
        },
        {
          "outer_year": 2019,
          "inner_train_start": "1990-01-12",
          "inner_train_end": "2016-12-30",
          "inner_train_rows": 23340,
          "inner_start": "2017-01-04",
          "inner_end": "2018-12-31",
          "inner_rows": 1781,
          "outer_rows": 1151,
          "inner_rho": -0.18,
          "outer_rho": -0.18,
          "selected_classifier_weight": 0.88,
          "selected_poisson_weight": 0.12,
          "selected_draw_floor": 0.06,
          "selected_draw_ceiling": 0.46,
          "inner_objective": 0.911792,
          "outer_objective": 0.802147,
          "outer_log_loss": 0.730432,
          "outer_rps": 0.134182,
          "outer_brier": 0.427611,
          "outer_ece": 0.036955,
          "outer_draw_recall": 0.028571,
          "outer_draw_expected_rate": 0.229819,
          "outer_draw_actual_rate": 0.212858,
          "outer_draw_gap": 0.016961,
          "outer_entropy": 0.736083,
          "outer_randomness_penalty": 0.0
        },
        {
          "outer_year": 2020,
          "inner_train_start": "1990-01-12",
          "inner_train_end": "2017-12-18",
          "inner_train_rows": 24212,
          "inner_start": "2017-12-22",
          "inner_end": "2019-12-19",
          "inner_rows": 2060,
          "outer_rows": 366,
          "inner_rho": -0.18,
          "outer_rho": -0.18,
          "selected_classifier_weight": 0.88,
          "selected_poisson_weight": 0.12,
          "selected_draw_floor": 0.04,
          "selected_draw_ceiling": 0.3,
          "inner_objective": 0.846081,
          "outer_objective": 0.991388,
          "outer_log_loss": 0.906004,
          "outer_rps": 0.175853,
          "outer_brier": 0.540677,
          "outer_ece": 0.060249,
          "outer_draw_recall": 0.0,
          "outer_draw_expected_rate": 0.25772,
          "outer_draw_actual_rate": 0.259563,
          "outer_draw_gap": 0.001843,
          "outer_entropy": 0.848106,
          "outer_randomness_penalty": 0.0
        },
        {
          "outer_year": 2021,
          "inner_train_start": "1990-01-12",
          "inner_train_end": "2018-12-06",
          "inner_train_rows": 25094,
          "inner_start": "2018-12-11",
          "inner_end": "2020-12-09",
          "inner_rows": 1544,
          "outer_rows": 1132,
          "inner_rho": -0.18,
          "outer_rho": -0.18,
          "selected_classifier_weight": 0.88,
          "selected_poisson_weight": 0.12,
          "selected_draw_floor": 0.04,
          "selected_draw_ceiling": 0.3,
          "inner_objective": 0.844082,
          "outer_objective": 0.776001,
          "outer_log_loss": 0.711585,
          "outer_rps": 0.12583,
          "outer_brier": 0.415682,
          "outer_ece": 0.037945,
          "outer_draw_recall": 0.0,
          "outer_draw_expected_rate": 0.218485,
          "outer_draw_actual_rate": 0.225265,
          "outer_draw_gap": 0.00678,
          "outer_entropy": 0.718169,
          "outer_randomness_penalty": 0.0
        },
        {
          "outer_year": 2022,
          "inner_train_start": "1990-01-12",
          "inner_train_end": "2019-12-19",
          "inner_train_rows": 26272,
          "inner_start": "2020-01-07",
          "inner_end": "2021-12-31",
          "inner_rows": 1498,
          "outer_rows": 990,
          "inner_rho": -0.18,
          "outer_rho": -0.18,
          "selected_classifier_weight": 0.88,
          "selected_poisson_weight": 0.12,
          "selected_draw_floor": 0.04,
          "selected_draw_ceiling": 0.42,
          "inner_objective": 0.823779,
          "outer_objective": 0.853617,
          "outer_log_loss": 0.777412,
          "outer_rps": 0.145552,
          "outer_brier": 0.459483,
          "outer_ece": 0.038069,
          "outer_draw_recall": 0.039301,
          "outer_draw_expected_rate": 0.246835,
          "outer_draw_actual_rate": 0.231313,
          "outer_draw_gap": 0.015522,
          "outer_entropy": 0.77207,
          "outer_randomness_penalty": 0.0
        },
        {
          "outer_year": 2023,
          "inner_train_start": "1990-01-12",
          "inner_train_end": "2020-12-09",
          "inner_train_rows": 26638,
          "inner_start": "2021-01-12",
          "inner_end": "2022-12-30",
          "inner_rows": 2122,
          "outer_rows": 1059,
          "inner_rho": -0.18,
          "outer_rho": -0.18,
          "selected_classifier_weight": 0.88,
          "selected_poisson_weight": 0.12,
          "selected_draw_floor": 0.04,
          "selected_draw_ceiling": 0.42,
          "inner_objective": 0.808428,
          "outer_objective": 0.849485,
          "outer_log_loss": 0.777528,
          "outer_rps": 0.146629,
          "outer_brier": 0.458038,
          "outer_ece": 0.043174,
          "outer_draw_recall": 0.021008,
          "outer_draw_expected_rate": 0.22858,
          "outer_draw_actual_rate": 0.22474,
          "outer_draw_gap": 0.00384,
          "outer_entropy": 0.739642,
          "outer_randomness_penalty": 0.0
        },
        {
          "outer_year": 2024,
          "inner_train_start": "1990-01-12",
          "inner_train_end": "2021-12-30",
          "inner_train_rows": 27769,
          "inner_start": "2021-12-31",
          "inner_end": "2023-12-31",
          "inner_rows": 2050,
          "outer_rows": 1298,
          "inner_rho": -0.18,
          "outer_rho": -0.18,
          "selected_classifier_weight": 0.86,
          "selected_poisson_weight": 0.14,
          "selected_draw_floor": 0.04,
          "selected_draw_ceiling": 0.3,
          "inner_objective": 0.849476,
          "outer_objective": 0.856215,
          "outer_log_loss": 0.778058,
          "outer_rps": 0.140584,
          "outer_brier": 0.461806,
          "outer_ece": 0.020577,
          "outer_draw_recall": 0.0,
          "outer_draw_expected_rate": 0.225921,
          "outer_draw_actual_rate": 0.252696,
          "outer_draw_gap": 0.026775,
          "outer_entropy": 0.740088,
          "outer_randomness_penalty": 0.0
        },
        {
          "outer_year": 2025,
          "inner_train_start": "1990-01-12",
          "inner_train_end": "2022-12-30",
          "inner_train_rows": 28760,
          "inner_start": "2023-01-02",
          "inner_end": "2024-12-31",
          "inner_rows": 2357,
          "outer_rows": 1071,
          "inner_rho": -0.18,
          "outer_rho": -0.18,
          "selected_classifier_weight": 0.88,
          "selected_poisson_weight": 0.12,
          "selected_draw_floor": 0.06,
          "selected_draw_ceiling": 0.46,
          "inner_objective": 0.844978,
          "outer_objective": 0.796394,
          "outer_log_loss": 0.730398,
          "outer_rps": 0.133781,
          "outer_brier": 0.427968,
          "outer_ece": 0.032153,
          "outer_draw_recall": 0.054622,
          "outer_draw_expected_rate": 0.226726,
          "outer_draw_actual_rate": 0.222222,
          "outer_draw_gap": 0.004503,
          "outer_entropy": 0.7255,
          "outer_randomness_penalty": 0.0
        }
      ],
      "fold_winner_policy": {
        "classifier_weight": 0.88,
        "poisson_weight": 0.12,
        "draw_floor": 0.04,
        "draw_ceiling": 0.42,
        "selected_outer_rows": 2945,
        "selected_folds": 3,
        "selected_avg_inner_objective": 0.836788
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
        "draw_ceiling": 0.46,
        "selected_outer_rows": 7963,
        "selected_folds": 8,
        "selected_avg_inner_objective": 0.838467,
        "selected_avg_inner_entropy": 0.715498,
        "outer_objective": 0.817645,
        "outer_log_loss": 0.750923,
        "outer_rps": 0.139955,
        "outer_brier": 0.448513,
        "outer_ece": 0.008562,
        "outer_draw_recall": 0.010226,
        "outer_draw_expected_rate": 0.237199,
        "outer_draw_actual_rate": 0.233329,
        "outer_draw_gap": 0.00387,
        "outer_entropy": 0.702521,
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
          "draw_ceiling": 0.46,
          "selected_outer_rows": 7963,
          "selected_folds": 8,
          "selected_avg_inner_objective": 0.838467,
          "selected_avg_inner_entropy": 0.715498,
          "outer_objective": 0.817645,
          "outer_log_loss": 0.750923,
          "outer_rps": 0.139955,
          "outer_brier": 0.448513,
          "outer_ece": 0.008562,
          "outer_draw_recall": 0.010226,
          "outer_draw_expected_rate": 0.237199,
          "outer_draw_actual_rate": 0.233329,
          "outer_draw_gap": 0.00387,
          "outer_entropy": 0.702521,
          "outer_randomness_penalty": 0.0
        },
        "top_component_candidates": [
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
            "selected_outer_rows": 7963,
            "selected_folds": 8,
            "avg_inner_objective": 0.837515,
            "avg_inner_entropy": 0.715152
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
            "selected_outer_rows": 7963,
            "selected_folds": 8,
            "avg_inner_objective": 0.838194,
            "avg_inner_entropy": 0.708825
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
            "selected_outer_rows": 7963,
            "selected_folds": 8,
            "avg_inner_objective": 0.838232,
            "avg_inner_entropy": 0.706624
          },
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
            "selected_outer_rows": 7963,
            "selected_folds": 8,
            "avg_inner_objective": 0.839152,
            "avg_inner_entropy": 0.722293
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
            "selected_outer_rows": 7963,
            "selected_folds": 8,
            "avg_inner_objective": 0.839413,
            "avg_inner_entropy": 0.724206
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
            "selected_outer_rows": 7963,
            "selected_folds": 8,
            "avg_inner_objective": 0.839683,
            "avg_inner_entropy": 0.725169
          },
          {
            "component_candidate": "subset__xgb+competitive+poisson",
            "active_components": [
              "xgb",
              "competitive",
              "poisson"
            ],
            "manual_blend_weights": {
              "xgb": 0.588235,
              "competitive": 0.264706,
              "logistic": 0.0,
              "elo": 0.0,
              "poisson": 0.147059,
              "count_poisson": 0.0
            },
            "selected_outer_rows": 7963,
            "selected_folds": 8,
            "avg_inner_objective": 0.840007,
            "avg_inner_entropy": 0.716809
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
            "selected_outer_rows": 7963,
            "selected_folds": 8,
            "avg_inner_objective": 0.840254,
            "avg_inner_entropy": 0.717563
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
            "selected_outer_rows": 7963,
            "selected_folds": 8,
            "avg_inner_objective": 0.840738,
            "avg_inner_entropy": 0.72265
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
            "selected_outer_rows": 7963,
            "selected_folds": 8,
            "avg_inner_objective": 0.841235,
            "avg_inner_entropy": 0.724336
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
            "selected_outer_rows": 7963,
            "selected_folds": 8,
            "avg_inner_objective": 0.841301,
            "avg_inner_entropy": 0.732564
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
            "selected_outer_rows": 7963,
            "selected_folds": 8,
            "avg_inner_objective": 0.841497,
            "avg_inner_entropy": 0.730897
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
            "selected_outer_rows": 7963,
            "selected_folds": 8,
            "avg_inner_objective": 0.841583,
            "avg_inner_entropy": 0.734137
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
            "selected_outer_rows": 7963,
            "selected_folds": 8,
            "avg_inner_objective": 0.842048,
            "avg_inner_entropy": 0.729637
          },
          {
            "component_candidate": "subset__xgb+competitive+poisson+count_poisson",
            "active_components": [
              "xgb",
              "competitive",
              "poisson",
              "count_poisson"
            ],
            "manual_blend_weights": {
              "xgb": 0.512821,
              "competitive": 0.230769,
              "logistic": 0.0,
              "elo": 0.0,
              "poisson": 0.128205,
              "count_poisson": 0.128205
            },
            "selected_outer_rows": 7963,
            "selected_folds": 8,
            "avg_inner_objective": 0.842074,
            "avg_inner_entropy": 0.724913
          },
          {
            "component_candidate": "subset__competitive",
            "active_components": [
              "competitive"
            ],
            "manual_blend_weights": {
              "xgb": 0.0,
              "competitive": 1.0,
              "logistic": 0.0,
              "elo": 0.0,
              "poisson": 0.0,
              "count_poisson": 0.0
            },
            "selected_outer_rows": 7963,
            "selected_folds": 8,
            "avg_inner_objective": 0.842181,
            "avg_inner_entropy": 0.708239
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
            "selected_outer_rows": 7963,
            "selected_folds": 8,
            "avg_inner_objective": 0.843757,
            "avg_inner_entropy": 0.740076
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
            "selected_outer_rows": 7963,
            "selected_folds": 8,
            "avg_inner_objective": 0.843942,
            "avg_inner_entropy": 0.730475
          },
          {
            "component_candidate": "subset__competitive+logistic+poisson",
            "active_components": [
              "competitive",
              "logistic",
              "poisson"
            ],
            "manual_blend_weights": {
              "xgb": 0.0,
              "competitive": 0.5,
              "logistic": 0.222222,
              "elo": 0.0,
              "poisson": 0.277778,
              "count_poisson": 0.0
            },
            "selected_outer_rows": 7963,
            "selected_folds": 8,
            "avg_inner_objective": 0.845492,
            "avg_inner_entropy": 0.742417
          },
          {
            "component_candidate": "subset__competitive+poisson",
            "active_components": [
              "competitive",
              "poisson"
            ],
            "manual_blend_weights": {
              "xgb": 0.0,
              "competitive": 0.642857,
              "logistic": 0.0,
              "elo": 0.0,
              "poisson": 0.357143,
              "count_poisson": 0.0
            },
            "selected_outer_rows": 7963,
            "selected_folds": 8,
            "avg_inner_objective": 0.845583,
            "avg_inner_entropy": 0.725339
          }
        ],
        "fold_best_rows": [
          {
            "outer_year": 2018,
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
            "outer_rows": 896,
            "selected_classifier_weight": 0.86,
            "selected_poisson_weight": 0.14,
            "selected_draw_floor": 0.04,
            "selected_draw_ceiling": 0.46,
            "inner_objective": 0.869668,
            "outer_objective": 0.896467,
            "outer_log_loss": 0.819973,
            "outer_rps": 0.154088,
            "outer_brier": 0.491083,
            "outer_ece": 0.025686,
            "outer_draw_recall": 0.004348,
            "outer_draw_expected_rate": 0.247987,
            "outer_draw_actual_rate": 0.256696,
            "outer_draw_gap": 0.008709,
            "outer_entropy": 0.755047,
            "outer_randomness_penalty": 0.0
          },
          {
            "outer_year": 2019,
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
            "outer_rows": 1151,
            "selected_classifier_weight": 0.88,
            "selected_poisson_weight": 0.12,
            "selected_draw_floor": 0.08,
            "selected_draw_ceiling": 0.46,
            "inner_objective": 0.8981,
            "outer_objective": 0.783981,
            "outer_log_loss": 0.712651,
            "outer_rps": 0.132218,
            "outer_brier": 0.423734,
            "outer_ece": 0.027961,
            "outer_draw_recall": 0.008163,
            "outer_draw_expected_rate": 0.23248,
            "outer_draw_actual_rate": 0.212858,
            "outer_draw_gap": 0.019622,
            "outer_entropy": 0.690579,
            "outer_randomness_penalty": 0.0
          },
          {
            "outer_year": 2020,
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
            "outer_rows": 366,
            "selected_classifier_weight": 0.88,
            "selected_poisson_weight": 0.12,
            "selected_draw_floor": 0.04,
            "selected_draw_ceiling": 0.42,
            "inner_objective": 0.825948,
            "outer_objective": 0.994004,
            "outer_log_loss": 0.910257,
            "outer_rps": 0.178274,
            "outer_brier": 0.547017,
            "outer_ece": 0.034738,
            "outer_draw_recall": 0.0,
            "outer_draw_expected_rate": 0.259241,
            "outer_draw_actual_rate": 0.259563,
            "outer_draw_gap": 0.000322,
            "outer_entropy": 0.814009,
            "outer_randomness_penalty": 0.0
          },
          {
            "outer_year": 2021,
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
            "outer_rows": 1132,
            "selected_classifier_weight": 0.88,
            "selected_poisson_weight": 0.12,
            "selected_draw_floor": 0.04,
            "selected_draw_ceiling": 0.34,
            "inner_objective": 0.825374,
            "outer_objective": 0.765558,
            "outer_log_loss": 0.697686,
            "outer_rps": 0.124904,
            "outer_brier": 0.413269,
            "outer_ece": 0.034398,
            "outer_draw_recall": 0.0,
            "outer_draw_expected_rate": 0.211965,
            "outer_draw_actual_rate": 0.225265,
            "outer_draw_gap": 0.0133,
            "outer_entropy": 0.651989,
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
            "outer_rows": 990,
            "selected_classifier_weight": 0.88,
            "selected_poisson_weight": 0.12,
            "selected_draw_floor": 0.04,
            "selected_draw_ceiling": 0.46,
            "inner_objective": 0.81127,
            "outer_objective": 0.832212,
            "outer_log_loss": 0.755747,
            "outer_rps": 0.143347,
            "outer_brier": 0.454664,
            "outer_ece": 0.032928,
            "outer_draw_recall": 0.0131,
            "outer_draw_expected_rate": 0.250583,
            "outer_draw_actual_rate": 0.231313,
            "outer_draw_gap": 0.01927,
            "outer_entropy": 0.717368,
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
            "outer_rows": 1059,
            "selected_classifier_weight": 0.88,
            "selected_poisson_weight": 0.12,
            "selected_draw_floor": 0.06,
            "selected_draw_ceiling": 0.46,
            "inner_objective": 0.786518,
            "outer_objective": 0.846868,
            "outer_log_loss": 0.775318,
            "outer_rps": 0.147419,
            "outer_brier": 0.460005,
            "outer_ece": 0.031979,
            "outer_draw_recall": 0.0,
            "outer_draw_expected_rate": 0.220534,
            "outer_draw_actual_rate": 0.22474,
            "outer_draw_gap": 0.004206,
            "outer_entropy": 0.682245,
            "outer_randomness_penalty": 0.0
          },
          {
            "outer_year": 2024,
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
            "outer_rows": 1298,
            "selected_classifier_weight": 0.88,
            "selected_poisson_weight": 0.12,
            "selected_draw_floor": 0.04,
            "selected_draw_ceiling": 0.42,
            "inner_objective": 0.832535,
            "outer_objective": 0.838689,
            "outer_log_loss": 0.758293,
            "outer_rps": 0.139389,
            "outer_brier": 0.457178,
            "outer_ece": 0.036044,
            "outer_draw_recall": 0.012195,
            "outer_draw_expected_rate": 0.2235,
            "outer_draw_actual_rate": 0.252696,
            "outer_draw_gap": 0.029196,
            "outer_entropy": 0.66851,
            "outer_randomness_penalty": 0.0
          },
          {
            "outer_year": 2025,
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
            "outer_rows": 1071,
            "selected_classifier_weight": 0.88,
            "selected_poisson_weight": 0.12,
            "selected_draw_floor": 0.06,
            "selected_draw_ceiling": 0.46,
            "inner_objective": 0.828373,
            "outer_objective": 0.782882,
            "outer_log_loss": 0.715402,
            "outer_rps": 0.132355,
            "outer_brier": 0.42517,
            "outer_ece": 0.02073,
            "outer_draw_recall": 0.012605,
            "outer_draw_expected_rate": 0.233787,
            "outer_draw_actual_rate": 0.222222,
            "outer_draw_gap": 0.011565,
            "outer_entropy": 0.686835,
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
      "accuracy": 0.5837,
      "test_rows": 2570
    },
    "baseline_fifa_rank_1x2": {
      "accuracy": 0.5642,
      "test_rows": 2570
    },
    "baseline_elo_winner_no_draw": {
      "accuracy": 0.7802,
      "test_rows": 1952
    },
    "baseline_fifa_rank_winner_no_draw": {
      "accuracy": 0.752,
      "test_rows": 1952
    },
    "logistic_1x2": {
      "accuracy": 0.6218,
      "top2_accuracy": 0.8848,
      "log_loss": 0.7876,
      "draw_recall": 0.3447,
      "test_rows": 2570
    },
    "xgb_1x2": {
      "accuracy": 0.6584,
      "top2_accuracy": 0.8984,
      "log_loss": 0.7416,
      "draw_recall": 0.0583,
      "test_rows": 2570
    },
    "xgb_temperature_calibrated_1x2": {
      "accuracy": 0.6584,
      "top2_accuracy": 0.8984,
      "log_loss": 0.7404,
      "draw_recall": 0.0583,
      "temperature": 1.05,
      "test_rows": 2570
    },
    "winner_xgb_no_draw": {
      "accuracy": 0.8689,
      "log_loss": 0.2957,
      "brier": 0.0928,
      "test_rows": 1952
    },
    "competitive_xgb_1x2": {
      "accuracy": 0.667,
      "top2_accuracy": 0.8969,
      "log_loss": 0.7273,
      "test_rows": 1892
    },
    "poisson_goal_model_1x2": {
      "accuracy": 0.6482,
      "top2_accuracy": 0.8887,
      "log_loss": 0.7785,
      "draw_recall": 0.0194,
      "home_goal_mae": 1.0824,
      "away_goal_mae": 0.7476,
      "test_rows": 2570
    },
    "xgb_count_poisson_1x2": {
      "accuracy": 0.6521,
      "top2_accuracy": 0.8957,
      "log_loss": 0.77,
      "draw_recall": 0.0065,
      "home_goal_mae": 1.0587,
      "away_goal_mae": 0.7459,
      "test_rows": 2570
    },
    "stacking_meta_1x2": {
      "accuracy": 0.6066,
      "top2_accuracy": 0.8778,
      "log_loss": 0.821,
      "draw_recall": 0.343,
      "test_rows": 2570
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
    "temporal_importance_sample_weight": true
  },
  "metric_gains_vs_previous_report": {
    "baseline_elo_1x2": {
      "accuracy": 0.0,
      "test_rows": 0.0
    },
    "baseline_fifa_rank_1x2": {
      "accuracy": 0.0051,
      "test_rows": 0.0
    },
    "baseline_elo_winner_no_draw": {
      "accuracy": 0.0,
      "test_rows": 0.0
    },
    "baseline_fifa_rank_winner_no_draw": {
      "accuracy": 0.0046,
      "test_rows": 0.0
    },
    "logistic_1x2": {
      "accuracy": 0.0031,
      "top2_accuracy": 0.0004,
      "log_loss": -0.0022,
      "draw_recall": 0.0097,
      "test_rows": 0.0
    },
    "xgb_1x2": {
      "accuracy": -0.0004,
      "top2_accuracy": 0.0054,
      "log_loss": -0.0031,
      "draw_recall": -0.0064,
      "test_rows": 0.0
    },
    "xgb_temperature_calibrated_1x2": {
      "accuracy": -0.0004,
      "top2_accuracy": 0.0054,
      "log_loss": -0.0031,
      "draw_recall": -0.0064,
      "temperature": 0.0,
      "test_rows": 0.0
    },
    "winner_xgb_no_draw": {
      "accuracy": 0.0072,
      "log_loss": -0.004,
      "brier": -0.0022,
      "test_rows": 0.0
    },
    "competitive_xgb_1x2": {
      "accuracy": -0.0016,
      "top2_accuracy": 0.0005,
      "log_loss": -0.0056,
      "test_rows": 0.0
    },
    "poisson_goal_model_1x2": {
      "accuracy": 0.0,
      "top2_accuracy": 0.0,
      "log_loss": -0.0008,
      "draw_recall": 0.0016,
      "home_goal_mae": -0.0009,
      "away_goal_mae": -0.0011,
      "test_rows": 0.0
    },
    "xgb_count_poisson_1x2": {
      "accuracy": -0.0004,
      "top2_accuracy": 0.0027,
      "log_loss": -0.0031,
      "draw_recall": 0.0,
      "home_goal_mae": -0.0022,
      "away_goal_mae": -0.0026,
      "test_rows": 0.0
    },
    "stacking_meta_1x2": {
      "accuracy": -0.0059,
      "top2_accuracy": -0.0004,
      "log_loss": -0.0031,
      "draw_recall": -0.013,
      "test_rows": 0.0
    }
  },
  "metric_gains_vs_sota_v1": {
    "baseline_elo_1x2": {
      "accuracy": -0.0056
    },
    "baseline_fifa_rank_1x2": {
      "accuracy": 0.0013
    },
    "baseline_elo_winner_no_draw": {
      "accuracy": -0.0056
    },
    "baseline_fifa_rank_winner_no_draw": {
      "accuracy": -0.0047
    },
    "logistic_1x2": {
      "accuracy": 0.0538,
      "top2_accuracy": 0.0437,
      "log_loss": -0.0966,
      "draw_recall": 0.1149
    },
    "xgb_1x2": {
      "accuracy": 0.0574,
      "top2_accuracy": 0.0623,
      "log_loss": -0.1234,
      "draw_recall": 0.0548
    },
    "winner_xgb_no_draw": {
      "accuracy": 0.0798,
      "log_loss": -0.1452,
      "brier": -0.0511
    },
    "competitive_xgb_1x2": {
      "accuracy": 0.057,
      "top2_accuracy": 0.0539,
      "log_loss": -0.1255
    }
  },
  "world_cup_backtest_aggregate": {
    "baseline_elo_1x2": {
      "accuracy": 0.542,
      "top2_accuracy": 0.646,
      "log_loss": 1.1198,
      "brier": 0.6726,
      "rps": 0.2237,
      "ece": 0.218,
      "draw_recall": 0.1881
    },
    "hybrid_nested_policy_1x2": {
      "log_loss": 0.6341,
      "brier": 0.3764,
      "rps": 0.1026,
      "ece": 0.1065,
      "draw_recall": 0.0193,
      "classifier_weight": 0.8154,
      "poisson_weight": 0.1846,
      "draw_floor": 0.0426,
      "draw_ceiling": 0.4139,
      "objective": 0.7074,
      "draw_expected_rate": 0.2474,
      "draw_actual_rate": 0.236,
      "draw_gap": 0.0361,
      "entropy": 0.7148,
      "randomness_penalty": 0.0,
      "inner_train_rows": 13717.288,
      "inner_rows": 1802.728,
      "inner_rho": -0.18,
      "outer_rho": -0.18
    },
    "logistic_1x2": {
      "accuracy": 0.682,
      "top2_accuracy": 0.982,
      "log_loss": 0.6357,
      "brier": 0.4208,
      "rps": 0.1118,
      "ece": 0.1062,
      "draw_recall": 0.4027
    },
    "poisson_goal_model_1x2": {
      "accuracy": 0.722,
      "top2_accuracy": 0.962,
      "log_loss": 0.6697,
      "brier": 0.3919,
      "rps": 0.1098,
      "ece": 0.0986,
      "draw_recall": 0.0,
      "home_goal_mae": 1.0211,
      "away_goal_mae": 0.6522
    },
    "stacking_meta_1x2": {
      "accuracy": 0.674,
      "top2_accuracy": 0.978,
      "log_loss": 0.6863,
      "brier": 0.4322,
      "rps": 0.1182,
      "ece": 0.1103,
      "draw_recall": 0.3947
    },
    "xgb_1x2": {
      "accuracy": 0.73,
      "top2_accuracy": 0.98,
      "log_loss": 0.5645,
      "brier": 0.3657,
      "rps": 0.0962,
      "ece": 0.0939,
      "draw_recall": 0.0695
    },
    "xgb_count_poisson_1x2": {
      "accuracy": 0.736,
      "top2_accuracy": 0.974,
      "log_loss": 0.67,
      "brier": 0.3887,
      "rps": 0.1077,
      "ece": 0.1323,
      "draw_recall": 0.0,
      "home_goal_mae": 0.9472,
      "away_goal_mae": 0.6533
    },
    "xgb_temperature_calibrated_1x2": {
      "accuracy": 0.73,
      "top2_accuracy": 0.98,
      "log_loss": 0.5684,
      "brier": 0.3699,
      "rps": 0.0972,
      "ece": 0.0988,
      "draw_recall": 0.0695,
      "temperature": 0.8216
    }
  },
  "monte_carlo_runs": 1000,
  "sample_champion_seed_2026": "Netherlands",
  "top_10_champion_odds": [
    {
      "team": "Spain",
      "wins": 185,
      "champion_probability": 0.185
    },
    {
      "team": "Brazil",
      "wins": 128,
      "champion_probability": 0.128
    },
    {
      "team": "Germany",
      "wins": 117,
      "champion_probability": 0.117
    },
    {
      "team": "Mexico",
      "wins": 112,
      "champion_probability": 0.112
    },
    {
      "team": "Czechia",
      "wins": 96,
      "champion_probability": 0.096
    },
    {
      "team": "Netherlands",
      "wins": 73,
      "champion_probability": 0.073
    },
    {
      "team": "Korea Republic",
      "wins": 67,
      "champion_probability": 0.067
    },
    {
      "team": "Uruguay",
      "wins": 36,
      "champion_probability": 0.036
    },
    {
      "team": "France",
      "wins": 31,
      "champion_probability": 0.031
    },
    {
      "team": "Switzerland",
      "wins": 28,
      "champion_probability": 0.028
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
  "runtime_log_loss_lte_0_82": true,
  "runtime_near_ablation_frontier": true,
  "dixon_coles_near_rho_frontier": true,
  "beats_elo_accuracy_by_5pp": true,
  "beats_fifa_accuracy_by_7pp": true,
  "monte_carlo_uncertainty_reported": true,
  "stage_uncertainty_reported": true,
  "advanced_calibration_exhausted": true,
  "team_strength_dixon_coles_exhausted": true,
  "class_calibration_reported": true,
  "block_bootstrap_reported": true,
  "runtime_adjustment_audit_reported": true,
  "runtime_adjustment_max_shift_lte_35pp": true,
  "runtime_adjustment_p95_shift_lte_18pp": true,
  "raw_data_manifest_reported": true,
  "raw_data_manifest_hash_reported": true,
  "raw_data_semantic_sanity_passed": true,
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
    "sha256": "82274d68fd54b3aee18dcd2db137f087a99a516a9850ab73aea105d462d11e78",
    "size_bytes": 4565106,
    "mtime_ns": 1779038639047286897
  },
  "model_report": {
    "path": "/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_model_report.json",
    "sha256": "5192ec1147f2d774e34562a79ebeca0a38d5d6d1092f9e8f13ef204c554b8196",
    "size_bytes": 97095,
    "mtime_ns": 1779038639050470500
  },
  "training_matches": {
    "path": "/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/data/processed/sota_training_matches.csv",
    "sha256": "652da5722831d8ed4ba3bddab1032e1f71d1f328db7484fb0c41929c111038b6",
    "size_bytes": 12323223,
    "mtime_ns": 1779038639027782527
  },
  "sota_pipeline": {
    "path": "/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/src/sota_pipeline.py",
    "sha256": "c4fb0df83adfa66e28dc9f7d69aa3ca61afc71468cb3df2dc6c3edcfbc49bdc4",
    "size_bytes": 164817,
    "mtime_ns": 1779193516621869414
  },
  "stats_qa_script": {
    "path": "/Users/eventanilha/Projects/arena-ai/scripts/model_stats_qa.py",
    "sha256": "af6617163c725190638d77e3b051828dadf1351359b1e4658b5493bb782b2d7d",
    "size_bytes": 86641,
    "mtime_ns": 1779194405417479773
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
  "total_size_bytes": 81082917,
  "manifest_sha256": "508a0b0651216d38d516c37dc22adc8930c9cc1a498cc6718aa912621ea64429",
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

## Política ativa

```json
{
  "classifier_weight": 0.88,
  "poisson_weight": 0.12,
  "draw_floor": 0.04,
  "draw_ceiling": 0.46,
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
  "draw_ceiling": 0.46,
  "selected_outer_rows": 7963,
  "selected_folds": 8,
  "selected_avg_inner_objective": 0.838467,
  "selected_avg_inner_entropy": 0.715498,
  "outer_objective": 0.817645,
  "outer_log_loss": 0.750923,
  "outer_rps": 0.139955,
  "outer_brier": 0.448513,
  "outer_ece": 0.008562,
  "outer_draw_recall": 0.010226,
  "outer_draw_expected_rate": 0.237199,
  "outer_draw_actual_rate": 0.233329,
  "outer_draw_gap": 0.00387,
  "outer_entropy": 0.702521,
  "outer_randomness_penalty": 0.0
}
```

```json
{
  "classifier_weight": 0.88,
  "poisson_weight": 0.12,
  "draw_floor": 0.04,
  "draw_ceiling": 0.46,
  "objective": 0.807857,
  "log_loss": 0.740046,
  "rps": 0.136614,
  "brier": 0.443882,
  "ece": 0.017843,
  "draw_recall": 0.021036,
  "draw_expected_rate": 0.233025,
  "draw_actual_rate": 0.240467,
  "draw_gap": 0.007442,
  "entropy": 0.678402,
  "randomness_penalty": 0.0
}
```

Resumo por classe:

| label | sample_count | weighted_abs_gap | max_abs_gap | mean_predicted_rate | mean_empirical_rate |
| --- | --- | --- | --- | --- | --- |
| empate | 2570 | 0.024132 | 0.111641 | 0.233025 | 0.240467 |
| casa | 2570 | 0.022696 | 0.069399 | 0.596683 | 0.580934 |
| fora | 2570 | 0.017898 | 0.066102 | 0.170291 | 0.178599 |

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
      "calibration_rows": 1298,
      "evaluation_rows": 1272,
      "cal_objective": 0.828091,
      "eval_objective": 0.79148,
      "eval_objective_delta_vs_identity": 0.0,
      "eval_log_loss": 0.725129,
      "eval_log_loss_delta_vs_identity": 0.0,
      "eval_ece": 0.04215,
      "eval_draw_gap": 0.001843,
      "eval_accuracy": 0.660377,
      "promoted": false,
      "decision": "candidate",
      "note": "referencia do runtime atual"
    },
    "best": {
      "experiment": "advanced_calibration",
      "family": "identity",
      "candidate": "runtime_sem_calibracao_extra",
      "calibration_rows": 1298,
      "evaluation_rows": 1272,
      "cal_objective": 0.828091,
      "eval_objective": 0.79148,
      "eval_objective_delta_vs_identity": 0.0,
      "eval_log_loss": 0.725129,
      "eval_log_loss_delta_vs_identity": 0.0,
      "eval_ece": 0.04215,
      "eval_draw_gap": 0.001843,
      "eval_accuracy": 0.660377,
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
      "calibration_rows": 29819,
      "evaluation_rows": 2570,
      "eval_objective": 0.807857,
      "eval_objective_delta_vs_runtime": 0.0,
      "eval_log_loss": 0.740046,
      "eval_log_loss_delta_vs_runtime": 0.0,
      "eval_ece": 0.017843,
      "eval_draw_gap": 0.007442,
      "eval_accuracy": 0.65642,
      "promoted": false,
      "decision": "candidate",
      "note": "referencia Poisson/DC atual do runtime"
    },
    "best": {
      "experiment": "dixon_coles_team_strength",
      "family": "poisson_regressor_plus_team_strength",
      "candidate": "shrink=4_half_life=3_alpha=0.20",
      "calibration_rows": 29819,
      "evaluation_rows": 2570,
      "eval_objective": 0.807557,
      "eval_objective_delta_vs_runtime": -0.0003,
      "eval_log_loss": 0.740529,
      "eval_log_loss_delta_vs_runtime": 0.000483,
      "eval_ece": 0.016488,
      "eval_draw_gap": 0.006112,
      "eval_accuracy": 0.657588,
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
| Spain | 185 | 0.185 | 0.160933 | 0.209067 | 0.024067 |
| Brazil | 128 | 0.128 | 0.107293 | 0.148707 | 0.020707 |
| Germany | 117 | 0.117 | 0.097078 | 0.136922 | 0.019922 |
| Mexico | 112 | 0.112 | 0.092453 | 0.131547 | 0.019547 |
| Czechia | 96 | 0.096 | 0.077741 | 0.114259 | 0.018259 |
| Netherlands | 73 | 0.073 | 0.056877 | 0.089123 | 0.016123 |
| Korea Republic | 67 | 0.067 | 0.051503 | 0.082497 | 0.015497 |
| Uruguay | 36 | 0.036 | 0.024454 | 0.047546 | 0.011546 |

Fases:

| team | stage | probability | lower_95 | upper_95 | margin_95 |
| --- | --- | --- | --- | --- | --- |
| Mexico | Group Stage | 1.0 | 1.0 | 1.0 | 0.0 |
| Mexico | Round of 32 | 0.983 | 0.974988 | 0.991012 | 0.008012 |
| Mexico | Round of 16 | 0.807 | 0.782539 | 0.831461 | 0.024461 |
| Mexico | Quarter-finals | 0.644 | 0.614323 | 0.673677 | 0.029677 |
| Mexico | Semi-finals | 0.21 | 0.184755 | 0.235245 | 0.025245 |
| Mexico | Final | 0.161 | 0.13822 | 0.18378 | 0.02278 |
| Mexico | Champion | 0.112 | 0.092453 | 0.131547 | 0.019547 |
| South Africa | Group Stage | 1.0 | 1.0 | 1.0 | 0.0 |

Bootstrap por bloco temporal/torneio:

| block_type | metric | block_count | mean | lower_95 | upper_95 | width_95 |
| --- | --- | --- | --- | --- | --- | --- |
| ano | log_loss | 3 | 0.740517 | 0.716526 | 0.75852 | 0.041994 |
| ano | rps | 3 | 0.136737 | 0.132931 | 0.139635 | 0.006704 |
| ano | brier | 3 | 0.444115 | 0.42653 | 0.457208 | 0.030678 |
| ano | ece | 3 | 0.027693 | 0.017353 | 0.048364 | 0.031011 |
| ano | draw_gap | 3 | 0.007802 | 0.001344 | 0.015508 | 0.014164 |
| torneio | log_loss | 48 | 0.741012 | 0.697733 | 0.78026 | 0.082527 |
| torneio | rps | 48 | 0.136983 | 0.127984 | 0.14561 | 0.017626 |
| torneio | brier | 48 | 0.445157 | 0.418657 | 0.470103 | 0.051447 |
| torneio | ece | 48 | 0.027212 | 0.016911 | 0.043103 | 0.026192 |
| torneio | draw_gap | 48 | 0.00833 | 0.000297 | 0.020884 | 0.020587 |

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
    2000
  ],
  "stability_gate": {
    "max_top16_abs_delta": 0.015,
    "max_top16_churn": 2,
    "leader_change_allowed": false,
    "max_stage_top16_abs_delta": 0.035,
    "max_stage_top16_churn": 4,
    "max_pair_top8_abs_delta": 0.02,
    "max_pair_top8_churn": 16
  },
  "final_comparison": {
    "baseline": false,
    "previous_runs": 5000,
    "leader_changed": false,
    "comparison": "union_top16_abs_delta",
    "union_team_count": 17,
    "entered_top16": [
      "Japan"
    ],
    "exited_top16": [
      "Belgium"
    ],
    "top16_churn_count": 2,
    "max_top16_abs_delta": 0.0106,
    "mean_top16_abs_delta": 0.003688
  },
  "stage_bracket_final_comparison": {
    "baseline": false,
    "previous_runs": 1000,
    "stage_top16": {
      "Round of 32": {
        "entered": [],
        "exited": [],
        "churn_count": 0,
        "max_abs_delta": 0.007,
        "mean_abs_delta": 0.002281
      },
      "Round of 16": {
        "entered": [],
        "exited": [],
        "churn_count": 0,
        "max_abs_delta": 0.013,
        "mean_abs_delta": 0.005
      },
      "Quarterfinals": {
        "entered": [
          "Germany"
        ],
        "exited": [
          "France"
        ],
        "churn_count": 2,
        "max_abs_delta": 0.026,
        "mean_abs_delta": 0.008912
      },
      "Semifinals": {
        "entered": [],
        "exited": [],
        "churn_count": 0,
        "max_abs_delta": 0.016,
        "mean_abs_delta": 0.007062
      },
      "Final": {
        "entered": [],
        "exited": [],
        "churn_count": 0,
        "max_abs_delta": 0.0185,
        "mean_abs_delta": 0.005875
      }
    },
    "pair_top8": {
      "Round of 32": {
        "entered": [],
        "exited": [],
        "churn_count": 0,
        "max_abs_delta": 0.0115,
        "mean_abs_delta": 0.00575
      },
      "Round of 16": {
        "entered": [],
        "exited": [],
        "churn_count": 0,
        "max_abs_delta": 0.0095,
        "mean_abs_delta": 0.003187
      },
      "Quarterfinals": {
        "entered": [],
        "exited": [],
        "churn_count": 0,
        "max_abs_delta": 0.0125,
        "mean_abs_delta": 0.006563
      },
      "Semifinals": {
        "entered": [],
        "exited": [],
        "churn_count": 0,
        "max_abs_delta": 0.0095,
        "mean_abs_delta": 0.004562
      },
      "Final": {
        "entered": [],
        "exited": [],
        "churn_count": 0,
        "max_abs_delta": 0.007,
        "mean_abs_delta": 0.0035
      }
    },
    "finalist_top16": {
      "entered": [],
      "exited": [],
      "churn_count": 0,
      "max_abs_delta": 0.0185,
      "mean_abs_delta": 0.005875
    },
    "max_stage_top16_abs_delta": 0.026,
    "max_stage_top16_churn": 2,
    "max_pair_top8_abs_delta": 0.0125,
    "max_pair_top8_churn": 0,
    "max_finalist_top16_abs_delta": 0.0185,
    "max_finalist_top16_churn": 0
  },
  "summary": {
    "max_runs": 10000,
    "min_runs": 5000,
    "max_stage_bracket_runs": 2000,
    "min_stage_bracket_runs": 1000,
    "leader_at_max_runs": "Spain",
    "leader_probability_at_max_runs": 0.16,
    "csv_path": "/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_monte_carlo_stability.csv",
    "stage_bracket_csv_path": "/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_monte_carlo_stage_bracket_stability.csv"
  },
  "source_fingerprints": {
    "model_package": {
      "path": "/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/models/model_sota.pkl",
      "sha256": "82274d68fd54b3aee18dcd2db137f087a99a516a9850ab73aea105d462d11e78",
      "size_bytes": 4565106,
      "mtime_ns": 1779038639047286897
    },
    "model_report": {
      "path": "/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_model_report.json",
      "sha256": "5192ec1147f2d774e34562a79ebeca0a38d5d6d1092f9e8f13ef204c554b8196",
      "size_bytes": 97095,
      "mtime_ns": 1779038639050470500
    },
    "training_matches": {
      "path": "/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/data/processed/sota_training_matches.csv",
      "sha256": "652da5722831d8ed4ba3bddab1032e1f71d1f328db7484fb0c41929c111038b6",
      "size_bytes": 12323223,
      "mtime_ns": 1779038639027782527
    },
    "sota_pipeline": {
      "path": "/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/src/sota_pipeline.py",
      "sha256": "c4fb0df83adfa66e28dc9f7d69aa3ca61afc71468cb3df2dc6c3edcfbc49bdc4",
      "size_bytes": 164817,
      "mtime_ns": 1779193516621869414
    },
    "mc_stability_script": {
      "path": "/Users/eventanilha/Projects/arena-ai/scripts/monte_carlo_stability.py",
      "sha256": "0bdafd05b719f21403a6ae5dbb244301596a827511f51f8f86e61e51f0563f5f",
      "size_bytes": 28065,
      "mtime_ns": 1779068608864827481
    }
  }
}
```

## 3. Ablation study

A tabela inclui todos os 63 subconjuntos dos seis sinais (`xgb`, `competitive`, `logistic`, `elo`, `poisson`, `count_poisson`) e compara contra a política ativa do runtime.

| ablation | objective | log_loss | rps | draw_gap | entropy |
| --- | --- | --- | --- | --- | --- |
| subset__xgb+logistic | 0.803398 | 0.738753 | 0.136259 | 0.001285 | 0.687038 |
| subset__xgb+logistic+count_poisson | 0.807506 | 0.742102 | 0.136448 | 0.00183 | 0.700475 |
| runtime_policy | 0.807856 | 0.740045 | 0.136613 | 0.007442 | 0.678396 |
| subset__xgb+competitive+logistic | 0.807856 | 0.740045 | 0.136613 | 0.007442 | 0.678396 |
| subset__competitive+logistic | 0.808864 | 0.742142 | 0.137027 | 0.005058 | 0.697072 |
| subset__xgb+logistic+poisson | 0.809181 | 0.743142 | 0.136726 | 0.002289 | 0.699648 |
| subset__xgb+competitive+logistic+count_poisson | 0.81025 | 0.742402 | 0.136707 | 0.007105 | 0.689984 |
| subset__xgb+competitive+logistic+poisson | 0.81115 | 0.743151 | 0.136902 | 0.007455 | 0.689398 |
| subset__xgb+logistic+poisson+count_poisson | 0.812367 | 0.746023 | 0.136952 | 0.002714 | 0.707764 |
| subset__competitive+logistic+count_poisson | 0.812747 | 0.746827 | 0.137181 | 0.002492 | 0.714662 |

## 4. Draw-specific calibration

```json
{
  "candidate_count": 690,
  "current_policy_rank": 3,
  "current_policy": [
    0.88,
    0.04,
    0.46
  ],
  "top_10": [
    {
      "classifier_weight": 0.88,
      "poisson_weight": 0.12,
      "draw_floor": 0.06,
      "draw_ceiling": 0.46,
      "objective": 0.807422,
      "log_loss": 0.73999,
      "rps": 0.13661,
      "brier": 0.443875,
      "ece": 0.018606,
      "draw_recall": 0.021036,
      "draw_expected_rate": 0.233879,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.006588,
      "entropy": 0.680666,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.88,
      "poisson_weight": 0.12,
      "draw_floor": 0.08,
      "draw_ceiling": 0.46,
      "objective": 0.807465,
      "log_loss": 0.74047,
      "rps": 0.13664,
      "brier": 0.444,
      "ece": 0.019135,
      "draw_recall": 0.021036,
      "draw_expected_rate": 0.235057,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.00541,
      "entropy": 0.683408,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.88,
      "poisson_weight": 0.12,
      "draw_floor": 0.04,
      "draw_ceiling": 0.46,
      "objective": 0.807857,
      "log_loss": 0.740046,
      "rps": 0.136614,
      "brier": 0.443882,
      "ece": 0.017843,
      "draw_recall": 0.021036,
      "draw_expected_rate": 0.233025,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.007442,
      "entropy": 0.678402,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.88,
      "poisson_weight": 0.12,
      "draw_floor": 0.1,
      "draw_ceiling": 0.46,
      "objective": 0.807894,
      "log_loss": 0.741477,
      "rps": 0.136725,
      "brier": 0.444333,
      "ece": 0.020035,
      "draw_recall": 0.021036,
      "draw_expected_rate": 0.236704,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.003763,
      "entropy": 0.686831,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.86,
      "poisson_weight": 0.14,
      "draw_floor": 0.06,
      "draw_ceiling": 0.46,
      "objective": 0.807963,
      "log_loss": 0.740482,
      "rps": 0.136652,
      "brier": 0.443983,
      "ece": 0.018504,
      "draw_recall": 0.021036,
      "draw_expected_rate": 0.233786,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.006681,
      "entropy": 0.682647,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.86,
      "poisson_weight": 0.14,
      "draw_floor": 0.08,
      "draw_ceiling": 0.46,
      "objective": 0.807989,
      "log_loss": 0.740943,
      "rps": 0.136681,
      "brier": 0.444105,
      "ece": 0.018916,
      "draw_recall": 0.021036,
      "draw_expected_rate": 0.234937,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.00553,
      "entropy": 0.685333,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.86,
      "poisson_weight": 0.14,
      "draw_floor": 0.04,
      "draw_ceiling": 0.46,
      "objective": 0.808317,
      "log_loss": 0.740543,
      "rps": 0.136655,
      "brier": 0.443991,
      "ece": 0.017762,
      "draw_recall": 0.021036,
      "draw_expected_rate": 0.232951,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.007516,
      "entropy": 0.680432,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.86,
      "poisson_weight": 0.14,
      "draw_floor": 0.1,
      "draw_ceiling": 0.46,
      "objective": 0.808421,
      "log_loss": 0.741921,
      "rps": 0.136763,
      "brier": 0.444427,
      "ece": 0.02004,
      "draw_recall": 0.021036,
      "draw_expected_rate": 0.236546,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.00392,
      "entropy": 0.688687,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.88,
      "poisson_weight": 0.12,
      "draw_floor": 0.12,
      "draw_ceiling": 0.46,
      "objective": 0.80852,
      "log_loss": 0.742846,
      "rps": 0.136864,
      "brier": 0.444869,
      "ece": 0.022379,
      "draw_recall": 0.021036,
      "draw_expected_rate": 0.239086,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.001381,
      "entropy": 0.691277,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.84,
      "poisson_weight": 0.16,
      "draw_floor": 0.06,
      "draw_ceiling": 0.46,
      "objective": 0.808586,
      "log_loss": 0.741009,
      "rps": 0.136698,
      "brier": 0.444102,
      "ece": 0.018965,
      "draw_recall": 0.021036,
      "draw_expected_rate": 0.233692,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.006775,
      "entropy": 0.68456,
      "randomness_penalty": 0.0
    }
  ],
  "best_by_classifier_weight": [
    {
      "classifier_weight": 0.5,
      "poisson_weight": 0.5,
      "draw_floor": 0.08,
      "draw_ceiling": 0.46,
      "objective": 0.822029,
      "log_loss": 0.753307,
      "rps": 0.138178,
      "brier": 0.44781,
      "ece": 0.020577,
      "draw_recall": 0.032362,
      "draw_expected_rate": 0.232772,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.007695,
      "entropy": 0.711497,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.52,
      "poisson_weight": 0.48,
      "draw_floor": 0.08,
      "draw_ceiling": 0.46,
      "objective": 0.821115,
      "log_loss": 0.752484,
      "rps": 0.138057,
      "brier": 0.447513,
      "ece": 0.020706,
      "draw_recall": 0.032362,
      "draw_expected_rate": 0.232892,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.007575,
      "entropy": 0.710362,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.54,
      "poisson_weight": 0.46,
      "draw_floor": 0.08,
      "draw_ceiling": 0.46,
      "objective": 0.820201,
      "log_loss": 0.751675,
      "rps": 0.13794,
      "brier": 0.447227,
      "ece": 0.020643,
      "draw_recall": 0.030744,
      "draw_expected_rate": 0.233012,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.007455,
      "entropy": 0.709195,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.56,
      "poisson_weight": 0.44,
      "draw_floor": 0.08,
      "draw_ceiling": 0.46,
      "objective": 0.819332,
      "log_loss": 0.750879,
      "rps": 0.137828,
      "brier": 0.446952,
      "ece": 0.020954,
      "draw_recall": 0.029126,
      "draw_expected_rate": 0.233133,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.007334,
      "entropy": 0.707997,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.58,
      "poisson_weight": 0.42,
      "draw_floor": 0.08,
      "draw_ceiling": 0.46,
      "objective": 0.818339,
      "log_loss": 0.750098,
      "rps": 0.13772,
      "brier": 0.446687,
      "ece": 0.019502,
      "draw_recall": 0.027508,
      "draw_expected_rate": 0.233253,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.007214,
      "entropy": 0.706766,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.6,
      "poisson_weight": 0.4,
      "draw_floor": 0.08,
      "draw_ceiling": 0.46,
      "objective": 0.817754,
      "log_loss": 0.749331,
      "rps": 0.137616,
      "brier": 0.446433,
      "ece": 0.022961,
      "draw_recall": 0.027508,
      "draw_expected_rate": 0.233373,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.007094,
      "entropy": 0.705502,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.62,
      "poisson_weight": 0.38,
      "draw_floor": 0.08,
      "draw_ceiling": 0.46,
      "objective": 0.816875,
      "log_loss": 0.748579,
      "rps": 0.137517,
      "brier": 0.44619,
      "ece": 0.022526,
      "draw_recall": 0.02589,
      "draw_expected_rate": 0.233493,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.006974,
      "entropy": 0.704203,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.64,
      "poisson_weight": 0.36,
      "draw_floor": 0.08,
      "draw_ceiling": 0.46,
      "objective": 0.81613,
      "log_loss": 0.747842,
      "rps": 0.137423,
      "brier": 0.445958,
      "ece": 0.023567,
      "draw_recall": 0.022654,
      "draw_expected_rate": 0.233614,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.006853,
      "entropy": 0.702868,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.66,
      "poisson_weight": 0.34,
      "draw_floor": 0.08,
      "draw_ceiling": 0.46,
      "objective": 0.815229,
      "log_loss": 0.74712,
      "rps": 0.137333,
      "brier": 0.445736,
      "ece": 0.022429,
      "draw_recall": 0.019417,
      "draw_expected_rate": 0.233734,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.006733,
      "entropy": 0.701497,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.68,
      "poisson_weight": 0.32,
      "draw_floor": 0.06,
      "draw_ceiling": 0.46,
      "objective": 0.814386,
      "log_loss": 0.746112,
      "rps": 0.137227,
      "brier": 0.445437,
      "ece": 0.020936,
      "draw_recall": 0.021036,
      "draw_expected_rate": 0.232944,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.007523,
      "entropy": 0.697917,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.7,
      "poisson_weight": 0.3,
      "draw_floor": 0.08,
      "draw_ceiling": 0.46,
      "objective": 0.813666,
      "log_loss": 0.745725,
      "rps": 0.137167,
      "brier": 0.445324,
      "ece": 0.022478,
      "draw_recall": 0.019417,
      "draw_expected_rate": 0.233974,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.006493,
      "entropy": 0.698636,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.72,
      "poisson_weight": 0.28,
      "draw_floor": 0.08,
      "draw_ceiling": 0.46,
      "objective": 0.812769,
      "log_loss": 0.745053,
      "rps": 0.13709,
      "brier": 0.445134,
      "ece": 0.020698,
      "draw_recall": 0.019417,
      "draw_expected_rate": 0.234095,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.006372,
      "entropy": 0.697144,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.74,
      "poisson_weight": 0.26,
      "draw_floor": 0.06,
      "draw_ceiling": 0.46,
      "objective": 0.812006,
      "log_loss": 0.744046,
      "rps": 0.136995,
      "brier": 0.444857,
      "ece": 0.019719,
      "draw_recall": 0.021036,
      "draw_expected_rate": 0.233225,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.007242,
      "entropy": 0.693266,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.76,
      "poisson_weight": 0.24,
      "draw_floor": 0.08,
      "draw_ceiling": 0.46,
      "objective": 0.811199,
      "log_loss": 0.743764,
      "rps": 0.136951,
      "brier": 0.444787,
      "ece": 0.019189,
      "draw_recall": 0.021036,
      "draw_expected_rate": 0.234335,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.006132,
      "entropy": 0.694028,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.78,
      "poisson_weight": 0.22,
      "draw_floor": 0.08,
      "draw_ceiling": 0.46,
      "objective": 0.810559,
      "log_loss": 0.743151,
      "rps": 0.136888,
      "brier": 0.444629,
      "ece": 0.019822,
      "draw_recall": 0.021036,
      "draw_expected_rate": 0.234456,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.006011,
      "entropy": 0.692398,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.8,
      "poisson_weight": 0.2,
      "draw_floor": 0.06,
      "draw_ceiling": 0.46,
      "objective": 0.810028,
      "log_loss": 0.742152,
      "rps": 0.136803,
      "brier": 0.444372,
      "ece": 0.021155,
      "draw_recall": 0.021036,
      "draw_expected_rate": 0.233505,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.006962,
      "entropy": 0.688203,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.82,
      "poisson_weight": 0.18,
      "draw_floor": 0.06,
      "draw_ceiling": 0.46,
      "objective": 0.809329,
      "log_loss": 0.741567,
      "rps": 0.136748,
      "brier": 0.444232,
      "ece": 0.020512,
      "draw_recall": 0.021036,
      "draw_expected_rate": 0.233599,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.006868,
      "entropy": 0.686411,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.84,
      "poisson_weight": 0.16,
      "draw_floor": 0.06,
      "draw_ceiling": 0.46,
      "objective": 0.808586,
      "log_loss": 0.741009,
      "rps": 0.136698,
      "brier": 0.444102,
      "ece": 0.018965,
      "draw_recall": 0.021036,
      "draw_expected_rate": 0.233692,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.006775,
      "entropy": 0.68456,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.86,
      "poisson_weight": 0.14,
      "draw_floor": 0.06,
      "draw_ceiling": 0.46,
      "objective": 0.807963,
      "log_loss": 0.740482,
      "rps": 0.136652,
      "brier": 0.443983,
      "ece": 0.018504,
      "draw_recall": 0.021036,
      "draw_expected_rate": 0.233786,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.006681,
      "entropy": 0.682647,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.88,
      "poisson_weight": 0.12,
      "draw_floor": 0.06,
      "draw_ceiling": 0.46,
      "objective": 0.807422,
      "log_loss": 0.73999,
      "rps": 0.13661,
      "brier": 0.443875,
      "ece": 0.018606,
      "draw_recall": 0.021036,
      "draw_expected_rate": 0.233879,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.006588,
      "entropy": 0.680666,
      "randomness_penalty": 0.0
    },
    {
      "classifier_weight": 0.9,
      "poisson_weight": 0.1,
      "draw_floor": 0.08,
      "draw_ceiling": 0.46,
      "objective": 0.811831,
      "log_loss": 0.740039,
      "rps": 0.136604,
      "brier": 0.443907,
      "ece": 0.017419,
      "draw_recall": 0.021036,
      "draw_expected_rate": 0.235177,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.00529,
      "entropy": 0.681408,
      "randomness_penalty": 0.02
    },
    {
      "classifier_weight": 0.92,
      "poisson_weight": 0.08,
      "draw_floor": 0.08,
      "draw_ceiling": 0.46,
      "objective": 0.816555,
      "log_loss": 0.73966,
      "rps": 0.136573,
      "brier": 0.443824,
      "ece": 0.019089,
      "draw_recall": 0.021036,
      "draw_expected_rate": 0.235297,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.00517,
      "entropy": 0.679324,
      "randomness_penalty": 0.04
    },
    {
      "classifier_weight": 0.94,
      "poisson_weight": 0.06,
      "draw_floor": 0.08,
      "draw_ceiling": 0.46,
      "objective": 0.821314,
      "log_loss": 0.739348,
      "rps": 0.136546,
      "brier": 0.443751,
      "ece": 0.019383,
      "draw_recall": 0.021036,
      "draw_expected_rate": 0.235418,
      "draw_actual_rate": 0.240467,
      "draw_gap": 0.005049,
      "entropy": 0.677145,
      "randomness_penalty": 0.06
    }
  ]
}
```

## 5. Dixon-Coles

| rho | hybrid_objective | hybrid_log_loss | hybrid_draw_gap | poisson_only_log_loss |
| --- | --- | --- | --- | --- |
| -0.18 | 0.807857 | 0.740046 | 0.007442 | 0.778513 |
| -0.16 | 0.808071 | 0.740096 | 0.007877 | 0.779064 |
| -0.14 | 0.808271 | 0.740147 | 0.008312 | 0.779704 |
| -0.12 | 0.808506 | 0.7402 | 0.008747 | 0.780436 |
| -0.1 | 0.808727 | 0.740254 | 0.009182 | 0.781261 |
| -0.08 | 0.808947 | 0.740311 | 0.009618 | 0.782181 |
| -0.06 | 0.809151 | 0.740368 | 0.010053 | 0.783196 |
| -0.04 | 0.809422 | 0.740428 | 0.010488 | 0.784309 |

## Benchmark externo disponível

```json
{
  "status": "available_public_style_baselines_only",
  "market_odds_benchmark": {
    "available": false,
    "reason": "O pacote atual nao contem odds historicas limpas de casas de aposta; nao inventamos benchmark externo sem dados auditaveis."
  },
  "available_baselines": {
    "elo_accuracy": 0.5837,
    "fifa_rank_accuracy": 0.5642,
    "xgb_calibrated_log_loss": 0.7404,
    "competitive_xgb_log_loss": 0.7273
  },
  "runtime_policy": {
    "accuracy": 0.65642,
    "log_loss": 0.740046,
    "accuracy_gain_vs_elo_pp": 7.272,
    "accuracy_gain_vs_fifa_pp": 9.222,
    "log_loss_gap_vs_xgb_calibrated": -0.000354,
    "log_loss_gap_vs_competitive_xgb": 0.012746
  },
  "interpretation": "O runtime precisa superar baselines publicos simples de forca/ranking e ficar perto da fronteira XGBoost, mas preservando Poisson/Dixon-Coles para placar, empate e variancia de futebol."
}
```

## Auditoria dos ajustes 2026

Os ajustes de elenco, Transfermarkt e contexto entram no runtime porque a Copa 2026 precisa refletir força atual de elenco. Como não existem snapshots históricos equivalentes no pacote, eles são auditados por limite de deslocamento probabilístico, e não usados para retunar a validação temporal.

```json
{
  "path": "/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_runtime_adjustment_audit.csv",
  "method": "all ordered 2026 qualified-team pairs; compare base classifier blend before squad/Transfermarkt/context proxies against pre-draw and final runtime probabilities",
  "teams": 48,
  "pairs": 2256,
  "max_abs_shift_pre_draw": 0.182885,
  "p95_abs_shift_pre_draw": 0.112826,
  "mean_abs_shift_pre_draw": 0.037299,
  "argmax_flip_rate_pre_draw": 0.057181,
  "max_abs_shift_final": 0.267877,
  "p95_abs_shift_final": 0.122401,
  "decision": "audit_only_runtime_kept",
  "reason": "Ajustes de elenco/Transfermarkt/contexto usam proxies 2026 sem backtest historico limpo; entram como camada operacional auditada, nao como novo tuning escondido. Limites de sanidade reprovam se o maximo passar de 35pp ou p95 passar de 18pp.",
  "top_10": [
    {
      "home_team": "Cura\u00e7ao",
      "away_team": "France",
      "base_home": 0.169778,
      "base_draw": 0.226162,
      "base_away": 0.60406,
      "adjusted_home": 0.012976,
      "adjusted_draw": 0.200079,
      "adjusted_away": 0.786945,
      "final_home": 0.012976,
      "final_draw": 0.200079,
      "final_away": 0.786945,
      "max_abs_shift_pre_draw": 0.182885,
      "sum_abs_shift_pre_draw": 0.365769,
      "max_abs_shift_final": 0.182885,
      "base_argmax": "fora",
      "adjusted_argmax": "fora",
      "final_argmax": "fora",
      "argmax_changed_pre_draw": false,
      "argmax_changed_final": false,
      "squad_top26_diff": -20.312821,
      "tm_market_value_log_diff": -2.137546,
      "tm_caps_diff": -0.498147,
      "tm_recent_injury_days_diff": -0.575738,
      "context_shift": 0.0
    },
    {
      "home_team": "South Africa",
      "away_team": "France",
      "base_home": 0.43365,
      "base_draw": 0.425441,
      "base_away": 0.140909,
      "adjusted_home": 0.275326,
      "adjusted_draw": 0.405562,
      "adjusted_away": 0.319112,
      "final_home": 0.275326,
      "final_draw": 0.405562,
      "final_away": 0.319112,
      "max_abs_shift_pre_draw": 0.178203,
      "sum_abs_shift_pre_draw": 0.356406,
      "max_abs_shift_final": 0.178203,
      "base_argmax": "casa",
      "adjusted_argmax": "empate",
      "final_argmax": "empate",
      "argmax_changed_pre_draw": true,
      "argmax_changed_final": true,
      "squad_top26_diff": -19.596154,
      "tm_market_value_log_diff": -4.04689,
      "tm_caps_diff": -1.406595,
      "tm_recent_injury_days_diff": -0.83586,
      "context_shift": 0.0
    },
    {
      "home_team": "South Africa",
      "away_team": "Spain",
      "base_home": 0.44456,
      "base_draw": 0.442608,
      "base_away": 0.112832,
      "adjusted_home": 0.286667,
      "adjusted_draw": 0.423322,
      "adjusted_away": 0.29001,
      "final_home": 0.286667,
      "final_draw": 0.423322,
      "final_away": 0.29001,
      "max_abs_shift_pre_draw": 0.177178,
      "sum_abs_shift_pre_draw": 0.354356,
      "max_abs_shift_final": 0.177178,
      "base_argmax": "casa",
      "adjusted_argmax": "empate",
      "final_argmax": "empate",
      "argmax_changed_pre_draw": true,
      "argmax_changed_final": true,
      "squad_top26_diff": -19.480769,
      "tm_market_value_log_diff": -4.22087,
      "tm_caps_diff": -1.287854,
      "tm_recent_injury_days_diff": -0.80335,
      "context_shift": 0.0
    },
    {
      "home_team": "Iraq",
      "away_team": "Spain",
      "base_home": 0.43494,
      "base_draw": 0.49245,
      "base_away": 0.072611,
      "adjusted_home": 0.277104,
      "adjusted_draw": 0.47505,
      "adjusted_away": 0.247846,
      "final_home": 0.285049,
      "final_draw": 0.46,
      "final_away": 0.254951,
      "max_abs_shift_pre_draw": 0.175235,
      "sum_abs_shift_pre_draw": 0.35047,
      "max_abs_shift_final": 0.182341,
      "base_argmax": "empate",
      "adjusted_argmax": "empate",
      "final_argmax": "empate",
      "argmax_changed_pre_draw": false,
      "argmax_changed_final": false,
      "squad_top26_diff": -19.302198,
      "tm_market_value_log_diff": -4.191882,
      "tm_caps_diff": -0.884997,
      "tm_recent_injury_days_diff": -0.604977,
      "context_shift": 0.0
    },
    {
      "home_team": "Iraq",
      "away_team": "France",
      "base_home": 0.480538,
      "base_draw": 0.487049,
      "base_away": 0.032413,
      "adjusted_home": 0.324212,
      "adjusted_draw": 0.469465,
      "adjusted_away": 0.206324,
      "final_home": 0.329995,
      "final_draw": 0.46,
      "final_away": 0.210005,
      "max_abs_shift_pre_draw": 0.173911,
      "sum_abs_shift_pre_draw": 0.347821,
      "max_abs_shift_final": 0.177591,
      "base_argmax": "empate",
      "adjusted_argmax": "empate",
      "final_argmax": "empate",
      "argmax_changed_pre_draw": false,
      "argmax_changed_final": false,
      "squad_top26_diff": -19.417582,
      "tm_market_value_log_diff": -4.017903,
      "tm_caps_diff": -1.003737,
      "tm_recent_injury_days_diff": -0.637486,
      "context_shift": 0.0
    },
    {
      "home_team": "Cura\u00e7ao",
      "away_team": "England",
      "base_home": 0.161388,
      "base_draw": 0.261112,
      "base_away": 0.5775,
      "adjusted_home": 0.01324,
      "adjusted_draw": 0.237536,
      "adjusted_away": 0.749224,
      "final_home": 0.01324,
      "final_draw": 0.237536,
      "final_away": 0.749224,
      "max_abs_shift_pre_draw": 0.171724,
      "sum_abs_shift_pre_draw": 0.343448,
      "max_abs_shift_final": 0.171724,
      "base_argmax": "fora",
      "adjusted_argmax": "fora",
      "final_argmax": "fora",
      "argmax_changed_pre_draw": false,
      "argmax_changed_final": false,
      "squad_top26_diff": -19.082051,
      "tm_market_value_log_diff": -2.29835,
      "tm_caps_diff": -0.623029,
      "tm_recent_injury_days_diff": -0.78873,
      "context_shift": 0.0
    },
    {
      "home_team": "South Africa",
      "away_team": "England",
      "base_home": 0.4444,
      "base_draw": 0.48868,
      "base_away": 0.066921,
      "adjusted_home": 0.294901,
      "adjusted_draw": 0.47191,
      "adjusted_away": 0.23319,
      "final_home": 0.301551,
      "final_draw": 0.46,
      "final_away": 0.238449,
      "max_abs_shift_pre_draw": 0.166269,
      "sum_abs_shift_pre_draw": 0.332537,
      "max_abs_shift_final": 0.171528,
      "base_argmax": "empate",
      "adjusted_argmax": "empate",
      "final_argmax": "empate",
      "argmax_changed_pre_draw": false,
      "argmax_changed_final": false,
      "squad_top26_diff": -18.365385,
      "tm_market_value_log_diff": -4.207694,
      "tm_caps_diff": -1.531476,
      "tm_recent_injury_days_diff": -1.048851,
      "context_shift": 0.0
    },
    {
      "home_team": "Iraq",
      "away_team": "England",
      "base_home": 0.501265,
      "base_draw": 0.469225,
      "base_away": 0.02951,
      "adjusted_home": 0.353997,
      "adjusted_draw": 0.451985,
      "adjusted_away": 0.194018,
      "final_home": 0.353997,
      "final_draw": 0.451985,
      "final_away": 0.194018,
      "max_abs_shift_pre_draw": 0.164508,
      "sum_abs_shift_pre_draw": 0.329016,
      "max_abs_shift_final": 0.164508,
      "base_argmax": "casa",
      "adjusted_argmax": "empate",
      "final_argmax": "empate",
      "argmax_changed_pre_draw": true,
      "argmax_changed_final": true,
      "squad_top26_diff": -18.186813,
      "tm_market_value_log_diff": -4.178707,
      "tm_caps_diff": -1.128619,
      "tm_recent_injury_days_diff": -0.850478,
      "context_shift": 0.0
    },
    {
      "home_team": "Cura\u00e7ao",
      "away_team": "Brazil",
      "base_home": 0.206057,
      "base_draw": 0.325198,
      "base_away": 0.468745,
      "adjusted_home": 0.063882,
      "adjusted_draw": 0.303825,
      "adjusted_away": 0.632294,
      "final_home": 0.063882,
      "final_draw": 0.303825,
      "final_away": 0.632294,
      "max_abs_shift_pre_draw": 0.163549,
      "sum_abs_shift_pre_draw": 0.327097,
      "max_abs_shift_final": 0.163549,
      "base_argmax": "fora",
      "adjusted_argmax": "fora",
      "final_argmax": "fora",
      "argmax_changed_pre_draw": false,
      "argmax_changed_final": false,
      "squad_top26_diff": -19.158974,
      "tm_market_value_log_diff": -1.962302,
      "tm_caps_diff": 0.241019,
      "tm_recent_injury_days_diff": -0.853268,
      "context_shift": 0.0
    },
    {
      "home_team": "Cura\u00e7ao",
      "away_team": "Portugal",
      "base_home": 0.177596,
      "base_draw": 0.293023,
      "base_away": 0.529381,
      "adjusted_home": 0.036093,
      "adjusted_draw": 0.271826,
      "adjusted_away": 0.692081,
      "final_home": 0.036093,
      "final_draw": 0.271826,
      "final_away": 0.692081,
      "max_abs_shift_pre_draw": 0.1627,
      "sum_abs_shift_pre_draw": 0.325399,
      "max_abs_shift_final": 0.1627,
      "base_argmax": "fora",
      "adjusted_argmax": "fora",
      "final_argmax": "fora",
      "argmax_changed_pre_draw": false,
      "argmax_changed_final": false,
      "squad_top26_diff": -18.235897,
      "tm_market_value_log_diff": -1.756089,
      "tm_caps_diff": -0.797011,
      "tm_recent_injury_days_diff": -0.356375,
      "context_shift": 0.0
    }
  ],
  "policy_reference": {
    "classifier_weight": 0.88,
    "poisson_weight": 0.12,
    "draw_floor": 0.04,
    "draw_ceiling": 0.46
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
| ajustes 2026 de elenco/Transfermarkt/contexto como modelo calibrado historicamente | foram auditados por limite de deslocamento probabilistico, mas nao promovidos a tuning academico porque faltam snapshots historicos equivalentes |

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
- `/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_monte_carlo_stability.json`
- `/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_raw_data_manifest.json`
- `/Users/eventanilha/Projects/arena-ai/modeling/worldcup_2026_ml/reports/sota_raw_data_manifest.csv`
