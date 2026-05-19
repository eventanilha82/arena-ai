# Modelo Estatístico E Runtime Da Copa

Este é o documento canônico do modelo do Arena AI. Ele consolida os antigos documentos avulsos de modelo, sorteio estatístico, runtime e fontes de dados. A auditoria estatística profunda fica em [STATISTICAL_AUDIT.md](STATISTICAL_AUDIT.md).

O objetivo é deixar claro o que o projeto faz, quais dados usa, como os modelos trabalham juntos, como a aleatoriedade é controlada e por que o carimbo atual é SOTA/KISS para o jogo com os dados disponíveis.

## Veredito Atual

O pacote ativo é `worldcup_2026_sota_v3`.

Carimbo atual:

```text
SOTA/KISS acadêmico aplicado aos dados disponíveis
```

Esse carimbo significa:

- a seleção da política usa validação temporal aninhada, sem retunar no mesmo holdout;
- a ablação completa dos sinais do blend foi executada;
- o modelo dedicado de empate (`draw_xgb`) foi removido porque tinha peso zero;
- o empate continua ativo por faixa calibrada e pela massa Poisson/Dixon-Coles;
- o Poisson/Dixon-Coles segue ativo no sorteio de placar;
- o Monte Carlo reporta incerteza e estabilidade;
- os ajustes 2026 de elenco, Transfermarkt e contexto foram auditados por limites de sanidade;
- os experimentos internos sem dataset externo foram esgotados até o ponto em que não havia ganho material para promover.

Ele não significa que o modelo venceu mercado de apostas ou odds de bookmakers. O pacote atual não contém uma base histórica limpa de odds de mercado, então esse benchmark externo não é declarado.

## Perguntas Que O Sistema Responde

O projeto separa quatro perguntas:

| Pergunta | Motor principal | Saída |
| --- | --- | --- |
| Quem tende a vencer em 90 minutos? | Classificador 1X2/XGBoost calibrado | casa, empate, fora |
| Qual placar é plausível? | Poisson/Dixon-Coles | matriz de placares, top 5, over/under, ambos marcam |
| Quem avança quando precisa haver vencedor? | `winner_xgb_no_draw` | vencedor obrigatório |
| Quem ganha a Copa em muitas simulações? | Monte Carlo | ranking, fase, caminho e campanha em destaque |

Essa separação é a decisão central do projeto: futebol tem placar como contagem de gols, mas resultado 1X2 como classificação. Usar um único modelo para tudo deixava a simulação determinística ou estatisticamente confusa.

## Arquitetura Do Modelo

### Classificador 1X2 Em 90 Minutos

O classificador final usa três sinais ativos:

```text
XGBoost principal     60,6061%
XGBoost competitivo   27,2727%
Logistic Regression   12,1212%
```

Sinais auditados, mas com peso zero no blend interno final:

```text
ELO                   0,0000%
Poisson 1X2           0,0000%
XGB count-poisson     0,0000%
```

Isso não remove ELO nem Poisson do projeto:

- ELO e ranking FIFA continuam como baselines, features históricas e comparação pública simples.
- Poisson/Dixon-Coles continua ativo no sorteio final de placar e no pacote 1X2 com peso `12%`.

### XGBoost Principal

O `xgb_1x2` é o motor forte de classificação multiclasses. Ele aprende padrões não lineares de histórico internacional, ranking, ELO, forma, contexto de jogo, força ofensiva/defensiva e sinais derivados.

Métrica de holdout temporal 2024+:

```text
accuracy = 0.6584
log_loss = 0.7416
top2_accuracy = 0.8984
```

Com calibração por temperatura:

```text
accuracy = 0.6584
log_loss = 0.7404
temperature = 1.05
```

### XGBoost Competitivo

O `competitive_xgb_1x2` usa a mesma família de features, mas é focado em jogos mais competitivos, com `tournament_weight >= 1.0`. Ele ajuda a capturar comportamento de partidas de competição real, em vez de deixar amistosos dominarem a leitura.

Métrica de holdout:

```text
accuracy = 0.6670
log_loss = 0.7273
test_rows = 1892
```

### Logistic Regression

O `logistic_1x2` é a âncora linear e explicável. Ele não substitui o XGBoost, mas estabiliza a composição do classificador. Ele ajuda a reduzir exageros locais quando o modelo não linear encontra cortes muito específicos.

Métrica de holdout:

```text
accuracy = 0.6218
top2_accuracy = 0.8848
log_loss = 0.7876
draw_recall = 0.3447
```

### Vencedor Obrigatório

O mata-mata não pode terminar empatado. Para isso existe o `winner_xgb_no_draw`, treinado em jogos com vencedor e usado na camada de avanço quando há prorrogação, pênaltis ou necessidade de desempate.

Métrica de holdout:

```text
accuracy = 0.8689
log_loss = 0.2957
brier = 0.0928
test_rows = 1952
```

### Poisson/Dixon-Coles

O placar é uma contagem de gols. A matriz Poisson/Dixon-Coles calcula:

```text
P(0x0), P(1x0), P(1x1), P(2x1), ...
```

Dela saem:

- placar mais provável;
- top 5 placares;
- mais de 2,5 gols;
- ambos marcam;
- massa de placares por pacote 1X2: casa, empate, fora.

O `rho` Dixon-Coles ativo é `-0.18`, escolhido por sensibilidade temporal. Na auditoria, ele ficou na fronteira operacional:

| rho | objective | log_loss | draw_gap |
| --- | ---: | ---: | ---: |
| `-0.18` | `0.807857` | `0.740046` | `0.007442` |
| `-0.16` | `0.808071` | `0.740096` | `0.007877` |
| `-0.14` | `0.808271` | `0.740147` | `0.008312` |

Uma versão mais pesada por ataque/defesa de seleção, com shrinkage e decaimento temporal, foi testada sem dataset externo. Ela teve ganho diagnóstico insuficiente e piorou levemente log loss:

```text
candidate = shrink=4_half_life=3_alpha=0.20
objective_delta_vs_runtime = -0.000300
log_loss_delta_vs_runtime = +0.000483
decision = not_promoted
```

Por isso ela não entrou no runtime.

## Empate

Empate é a classe mais frágil: tem massa estatística real, mas raramente vence no `argmax`.

A versão anterior tinha um modelo binário dedicado (`draw_xgb`), mas a calibração sempre escolhia peso zero para ele. Manter um modelo desligado é ruim para SOTA/KISS: cria complexidade sem função.

Por isso:

```text
draw_xgb = removido
```

O empate continua ativo por três caminhos:

1. probabilidade do classificador 1X2;
2. faixa calibrada `draw_floor` e `draw_ceiling`;
3. massa dos placares empatados da matriz Poisson/Dixon-Coles.

Política ativa:

```text
draw_floor   = 0.04
draw_ceiling = 0.46
```

Resumo por classe no diagnóstico:

| Classe | weighted_abs_gap | max_abs_gap | predicted_rate | empirical_rate |
| --- | ---: | ---: | ---: | ---: |
| empate | `0.024132` | `0.111641` | `0.233025` | `0.240467` |
| casa | `0.022696` | `0.069399` | `0.596683` | `0.580934` |
| fora | `0.017898` | `0.066102` | `0.170291` | `0.178599` |

## Sorteio Estatístico Influenciado

O jogo não usa o placar mais provável de forma determinística. Ele sorteia com peso estatístico.

Para cada jogo:

```text
p_classificador = [P(casa vence), P(empate), P(fora vence)]
p_poisson       = [
  soma dos placares com casa vencendo,
  soma dos placares empatados,
  soma dos placares com fora vencendo
]

p_mix = classifier_weight * p_classificador
      + poisson_weight    * p_poisson
```

Política persistida:

```text
classifier_weight = 0.88
poisson_weight    = 0.12
draw_floor        = 0.04
draw_ceiling      = 0.46
selected_by       = strict_nested_temporal_component_ablation_no_leakage_no_draw_xgb
```

Depois:

1. sorteia o pacote 1X2 usando `p_mix`;
2. filtra a matriz Poisson/Dixon-Coles para placares compatíveis com o pacote;
3. sorteia o placar dentro desse pacote.

Exemplo:

- Se o pacote sorteado for vitória do Brasil, o placar pode ser `1x0`, `2x0`, `2x1`, `3x1`.
- Se o pacote sorteado for empate, o placar pode ser `0x0`, `1x1`, `2x2`.

Esse mecanismo resolve dois extremos ruins:

- XGBoost sozinho deixava o jogo determinístico demais;
- Poisson bruto podia escolher `1x1` como placar individual mais provável mesmo quando o pacote geral apontava favoritismo forte.

## Por Que 0.88 / 0.12

A busca avaliou:

- os `63` subconjuntos não vazios dos seis sinais 1X2;
- `690` políticas por subconjunto;
- pesos de classificador de `0.50` a `0.94`;
- pisos de empate entre `0.04` e `0.14`;
- tetos de empate entre `0.30` e `0.46`.

Função objetivo:

```text
objective =
  log_loss
  + 0.30 * RPS
  + 0.08 * ECE
  + 0.05 * Brier
  + 0.42 * erro_da_taxa_esperada_de_empate
  + penalidade_de_baixa_entropia
  + penalidade_se_Poisson_cair_abaixo_de_12%
```

Política ativa:

```text
objective = 0.807857
log_loss = 0.740046
RPS = 0.136614
Brier = 0.443882
ECE = 0.017843
draw_expected_rate = 0.233025
draw_actual_rate = 0.240467
draw_gap = 0.007442
entropy = 0.678402
```

O peso `0.88 / 0.12` é o melhor equilíbrio dentro da restrição de variância mínima. Pesos como `0.90`, `0.92` e `0.94` melhoram um pouco o log loss bruto, mas deixam o Poisson abaixo de `12%`, o que torna a simulação menos futebol e mais classificador determinístico.

Comparação rápida:

| Política | log_loss | draw_gap | observação |
| --- | ---: | ---: | --- |
| `0.62 / 0.38` | `0.752794` | `0.009544` | mais variância, pior objetivo |
| `0.80 / 0.20` | `0.747682` | `0.009178` | política anterior, pior objetivo |
| `0.88 / 0.12` | `0.740046` | `0.007442` | política ativa |
| `0.94 / 0.06` | `0.739348` | `0.005049` | log loss menor, mas penalizada por baixa variância |

## Validação Temporal Aninhada

A política não é escolhida no mesmo holdout usado para reportar.

Fluxo:

1. escolhe um ano externo;
2. separa uma janela interna de validação dos 24 meses anteriores;
3. treina modelos internos apenas antes dessa janela;
4. escolhe componentes do blend e política nessa janela posterior;
5. congela pesos, `classifier_weight`, `draw_floor` e `draw_ceiling`;
6. retreina modelos externos com todos os dados anteriores ao ano avaliado;
7. avalia o ano externo sem retunar;
8. repete para vários anos.

Resumo da política selecionada:

```text
selected_outer_rows = 7963
selected_folds = 8
selected_avg_inner_objective = 0.838467
outer_objective = 0.817645
outer_log_loss = 0.750923
outer_RPS = 0.139955
outer_Brier = 0.448513
outer_ECE = 0.008562
outer_draw_expected_rate = 0.237199
outer_draw_actual_rate = 0.233329
outer_draw_gap = 0.003870
```

O holdout 2024+ continua como auditoria operacional. A seleção enviada ao runtime vem da validação temporal aninhada.

## Auditoria Estatística

O gate principal é:

```bash
make stats-qa
```

Ele não retreina o pickle. Ele valida o pacote atual e gera:

- `modeling/worldcup_2026_ml/reports/sota_statistical_report.json`
- `docs/STATISTICAL_AUDIT.md`
- `sota_calibration_bins.csv`
- `sota_class_calibration_summary.csv`
- `sota_block_bootstrap_intervals.csv`
- `sota_uncertainty_intervals.csv`
- `sota_stage_uncertainty_intervals.csv`
- `sota_ablation_study.csv`
- `sota_dixon_coles_rho_sensitivity.csv`
- `sota_internal_frontier_experiments.csv`
- `sota_runtime_adjustment_audit.csv`
- `sota_raw_data_manifest.json/.csv`

Os JSON/CSV ficam em `modeling/worldcup_2026_ml/reports/`. O relatório Markdown consolidado fica em `docs/STATISTICAL_AUDIT.md`.

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

Freshness dos artefatos que sustentam esse carimbo:

| Artefato | SHA-256 | Tamanho |
| --- | --- | ---: |
| `modeling/worldcup_2026_ml/models/model_sota.pkl` | `82274d68fd54b3aee18dcd2db137f087a99a516a9850ab73aea105d462d11e78` | `4565106` |
| `modeling/worldcup_2026_ml/reports/sota_model_report.json` | `5192ec1147f2d774e34562a79ebeca0a38d5d6d1092f9e8f13ef204c554b8196` | `97095` |
| `modeling/worldcup_2026_ml/data/processed/sota_training_matches.csv` | `652da5722831d8ed4ba3bddab1032e1f71d1f328db7484fb0c41929c111038b6` | `12323223` |
| `modeling/worldcup_2026_ml/src/sota_pipeline.py` | `124cf8ff7dad64d6b3fdc175be48160c395b759b210acf803cfa5bb535b29f9d` | `167197` |
| `scripts/model_stats_qa.py` | `e3a2e4e3a9c31d45b8ed79796bece5b8329c58731eb7d471d2b16b4b491e8a0b` | `85643` |

Manifesto bruto:

```text
path = modeling/worldcup_2026_ml/reports/sota_raw_data_manifest.json
csv_path = modeling/worldcup_2026_ml/reports/sota_raw_data_manifest.csv
file_count = 14
csv_file_count = 14
total_size_bytes = 81082917
manifest_sha256 = 508a0b0651216d38d516c37dc22adc8930c9cc1a498cc6718aa912621ea64429
```

## Ablação

A auditoria testa todos os `63` subconjuntos dos seis sinais:

```text
xgb
competitive
logistic
elo
poisson
count_poisson
```

Top da fronteira:

| ablation | objective | log_loss | draw_gap | entropy |
| --- | ---: | ---: | ---: | ---: |
| `subset__xgb+logistic` | `0.803398` | `0.738753` | `0.001285` | `0.687038` |
| `subset__xgb+logistic+count_poisson` | `0.807506` | `0.742102` | `0.001830` | `0.700475` |
| `runtime_policy` | `0.807856` | `0.740045` | `0.007442` | `0.678396` |
| `subset__xgb+competitive+logistic` | `0.807856` | `0.740045` | `0.007442` | `0.678396` |

O diagnóstico 2024+ favorece alguns candidatos pontualmente, mas o runtime usa a escolha nested temporal para evitar escolher e reportar no mesmo recorte.

## Calibração Avançada

Foram testados calibradores extras sem dataset externo:

- identity/runtime sem calibrador extra;
- temperature;
- isotonic;
- vector scaling;
- Dirichlet/logistic calibration.

Regra de promoção:

```text
promover apenas se melhorar objective, log_loss e draw_gap no split temporal sem vazamento
```

Resultado:

```text
promoted = false
best = runtime_sem_calibracao_extra
eval_objective = 0.791480
eval_log_loss = 0.725129
eval_draw_gap = 0.001843
```

Nenhum calibrador venceu o runtime no conjunto objetivo + log loss + empate.

## Baselines E Benchmark Externo

Benchmark externo de mercado:

```text
market_odds_benchmark = indisponível
motivo = não há odds históricas limpas no pacote atual
```

Baselines públicos simples disponíveis:

| Baseline | Accuracy |
| --- | ---: |
| ELO 1X2 | `0.5837` |
| FIFA ranking 1X2 | `0.5642` |
| runtime policy | `0.65642` |

Ganho do runtime:

```text
accuracy_gain_vs_elo = +7.272 pp
accuracy_gain_vs_fifa = +9.222 pp
```

## Ajustes 2026

Os ajustes 2026 de elenco, Transfermarkt e contexto entram porque a Copa precisa refletir força atual. Como o pacote não possui snapshots históricos equivalentes para todos esses sinais, eles não são usados como tuning histórico escondido. Eles são auditados por deslocamento probabilístico.

Auditoria em todos os confrontos ordenados das 48 seleções:

```text
teams = 48
pairs = 2256
max_abs_shift_pre_draw = 0.182885
p95_abs_shift_pre_draw = 0.112826
mean_abs_shift_pre_draw = 0.037299
argmax_flip_rate_pre_draw = 0.057181
limites = max <= 35pp, p95 <= 18pp
decision = audit_only_runtime_kept
```

Sinais expostos como `MatchDrivers`:

- `squad_top26_diff`: força agregada do elenco;
- `attack_strength_diff`: ataque;
- `midfield_strength_diff`: meio-campo;
- `defense_strength_diff`: defesa;
- `gk_strength_diff`: goleiro;
- `tm_market_value_log_diff`: valor de mercado;
- `tm_caps_diff`: experiência internacional;
- `tm_recent_injury_days_diff`: lesões recentes;
- `context_shift`: descanso, viagem e vantagem de sede.

`MatchDrivers` é uma dataclass congelada em `src/arena_ai/worldcup_model.py`. Console e game renderizam esses campos por contrato tipado, sem indexar strings soltas do pacote bruto.

## Dados

Somente dados usados pelo pipeline final ficam em `modeling/worldcup_2026_ml/data/raw/`.

### Estrutura 2026

Arquivos:

- `teams.csv`
- `matches.csv`
- `host_cities.csv`
- `tournament_stages.csv`

Fonte:

```text
Kaggle areezvisram12/fifa-world-cup-2026-match-data-unofficial
```

Função:

- grupos 2026;
- fixtures;
- cidades-sede;
- labels de fase;
- slots do bracket;
- contexto de kickoff.

Finalização:

- os placeholders de play-offs foram atualizados depois do fim da classificação, usando relatórios finais FIFA/UEFA de seleções classificadas;
- os nomes finais foram reconciliados para nomenclatura FIFA em inglês, para evitar drift entre dataset, modelo, game e console.

Placeholders finais resolvidos após classificação:

| Slot | Seleção |
| --- | --- |
| UEFA Path A | Bosnia and Herzegovina |
| UEFA Path B | Sweden |
| UEFA Path C | Türkiye |
| UEFA Path D | Czechia |
| FIFA Play-Off Path 1 | Congo DR |
| FIFA Play-Off Path 2 | Iraq |

Nomes reconciliados:

```text
USA
IR Iran
Korea Republic
Congo DR
Türkiye
Czechia
Côte d'Ivoire
Curaçao
```

Validação final:

```text
48 seleções locais
48 seleções FIFA
sem seleções faltantes
sem seleções extras
sem placeholders
```

### Força E Ranking

Arquivos:

- `fc26_players.csv`
  - fonte: Kaggle `rovnez/fc-26-fifa-26-player-data`
  - função: proxy de força de elenco por rating, valor, idade e posição.
- `fifa_rankings_1992_2024.csv`
  - fonte: Kaggle `cashncarry/fifaworldranking`
  - função: ranking FIFA temporal, pontos, variação e confederação.

### Histórico E Candidatos Mantidos

Arquivos:

- `candidates/pataterie_all_matches.csv`
- `candidates/pataterie_countries_names.csv`
- `candidates/saifalnimri_eloratings.csv`
- `candidates/transfermarkt_player_profiles.csv`
- `candidates/transfermarkt_player_market_value.csv`
- `candidates/transfermarkt_player_injuries.csv`
- `candidates/transfermarkt_player_national_performances.csv`
- `candidates/transfermarkt_team_details.csv`

Função:

- histórico internacional;
- ELO independente;
- forma;
- backtests walk-forward;
- valor de mercado;
- lesões;
- caps;
- cobertura de elenco.

Fontes dos candidatos mantidos:

| Arquivos | Fonte |
| --- | --- |
| `candidates/pataterie_all_matches.csv`, `candidates/pataterie_countries_names.csv` | Kaggle `patateriedata/all-international-football-results` |
| `candidates/saifalnimri_eloratings.csv` | Kaggle `saifalnimri/international-football-elo-ratings` |
| `candidates/transfermarkt_*` | Kaggle `xfkzujqjvx97n/football-datasets` |

### Dados Descartados

Descartados por auditoria:

- Kaggle `lchikry/international-football-match-features-and-statistics`: benchmark forte, mas o XGBoost externo não bateu o SOTA principal em log loss no holdout.
- FiveThirtyEight/SPI snapshots: úteis como referência metodológica, mas não alteraram o modelo final.
- datasets sintéticos pequenos de features/probabilidades 2026: provenance ou realismo insuficientes.
- odds/xG de clubes: forte domain shift contra seleções.
- fallbacks antigos e outputs da pipeline não SOTA.

## Manifesto Dos Dados Brutos

O relatório estatístico inclui manifesto com hash, tamanho, linhas, colunas e sanidade semântica.

Resumo atual:

```text
file_count = 14
csv_file_count = 14
total_size_bytes = 81082917
required_files_present = true
checked_file_count = 14
passed_file_count = 14
semantic_passed = true
manifest_sha256 = 508a0b0651216d38d516c37dc22adc8930c9cc1a498cc6718aa912621ea64429
```

Checks semânticos:

- IDs de seleções nos jogos existem;
- IDs de cidades existem;
- IDs de fases existem;
- partidas 1 a 104 são únicas;
- fase de grupos tem IDs de seleções;
- slots de mata-mata são placeholders ou IDs conhecidos;
- 48 seleções;
- grupos A-L com quatro seleções;
- 104 partidas;
- 72 jogos de fase de grupos.

## Monte Carlo

No modo Copa, cada partida usa a mesma política híbrida do confronto.

Fluxo:

1. simula fase de grupos;
2. classifica 1º e 2º;
3. escolhe melhores terceiros;
4. simula mata-mata;
5. resolve empate com prorrogação/vencedor obrigatório quando necessário;
6. repete muitas vezes;
7. conta campeões, fase, finalistas e caminhos.

Volumes:

| Volume | Uso |
| --- | --- |
| `1000` | UI do jogo e snapshot atual |
| `10000` | default do CLI de rebuild completo |
| `5k/10k` | estabilidade offline do ranking |
| `1k/2k` | estabilidade offline por fase, finalistas e confrontos de chave |

No Pygame:

```text
1000 Copas na UI
Monte Carlo fresh por padrão
cache runtime só para predições
banco de campanhas só no modo turbo explícito
progress bar sem travar
```

O runtime não deve recalcular o miolo ML de todas as partidas a cada abertura da tela. Esse miolo envolve XGBoost, regressão logística, sinais competitivos e Poisson; ele é caro porque a Copa tem fase de grupos, melhores terceiros e mata-mata. O pacote do jogo carrega `modeling/worldcup_2026_ml/models/runtime_prediction_cache.pkl`, gerado por:

```bash
make runtime-cache
```

Esse arquivo guarda:

- `prediction_base_cache`: predições base por confronto antes dos ajustes contextuais;
- `prediction_cache`: predições com contexto já vistas no aquecimento;
- `scenario_bank`: 1000 campanhas Monte Carlo completas no nível de campeão/final.

No jogo, a tela da Copa roda `1000` Copas frescas por padrão, em uma thread dedicada, com `8` workers por default. A barra de progresso vem dos callbacks do Monte Carlo real; o resultado só é revelado depois da amostra completa e de uma janela mínima de loading (`ARENA_AI_TOURNAMENT_MIN_LOADING_SECONDS`, default `3.2s`). O `scenario_bank` continua existindo, mas virou modo turbo explícito (`ARENA_AI_TOURNAMENT_MC_BOOTSTRAP=1`) para builds, executáveis e auditorias rápidas. Se alguém definir `ARENA_AI_TOURNAMENT_MC_FRESH=1`, essa variável legada também força fresh; sem variáveis, o jogo não usa o banco de campanhas.

### Incerteza

A UI e o console podem mostrar intervalo aproximado:

```text
erro ≈ 1.96 * sqrt(p * (1 - p) / n)
```

Exemplo:

```text
ESP 16,0% ± 2,3% IC 95%
```

Snapshot de Monte Carlo no model card:

```text
runs = 1000
sample champion seed 2026 = Netherlands
```

Top odds do snapshot:

| Seleção | Probabilidade | Vitórias |
| --- | ---: | ---: |
| Spain | `18.50%` | `185` |
| Brazil | `12.80%` | `128` |
| Germany | `11.70%` | `117` |
| Mexico | `11.20%` | `112` |
| Czechia | `9.60%` | `96` |
| Netherlands | `7.30%` | `73` |
| Korea Republic | `6.70%` | `67` |
| Uruguay | `3.60%` | `36` |

### Estabilidade Offline

Gate:

```bash
make mc-stability
```

Estado atual:

```text
available = true
fresh = true
passed = true
stage_bracket_passed = true
runs = 5000, 10000
stage_bracket_runs = 1000, 2000
leader_at_max_runs = Spain
leader_probability_at_max_runs = 0.16
max_top16_abs_delta = 0.0106
top16_churn_count = 2
max_stage_top16_abs_delta = 0.026
max_pair_top8_abs_delta = 0.0125
```

## Campanha Em Destaque

O Ranking Monte Carlo é frequência bruta de campeões. A Campanha em destaque é uma Copa concreta escolhida para narrativa visual.

A regra:

1. escolhe um destaque dentro do top 5, ponderado pelas odds;
2. procura uma Copa real da amostra em que esse destaque foi campeão;
3. favorece finalista que frequentemente chega à final;
4. penaliza finais muito elásticas, como `5 x 1`;
5. permite zebra, mas como `surpresa_controlada` ou `zebra_controlada`.

Assim, se Alemanha venceu `97/1000` Copas, o jogo não precisa mostrar uma final caricata contra Tunísia só porque aquela seed existiu. Ele procura uma campanha representativa dentro das Copas em que Alemanha venceu.

## Game E Console

### Confronto

O modo confronto usa `WorldCupModel.analyze_match(...)`.

Ele retorna `MatchAnalysis` com:

- probabilidades XGBoost 1X2 base;
- classificador final calibrado;
- massa Poisson/DC por pacote;
- mix final usado no sorteio;
- pacote sorteado;
- placar sorteado, mas revelado visualmente só no apito final;
- xG;
- top placares;
- `over_25`;
- `btts`;
- avanço/vencedor obrigatório;
- `MatchDrivers`.

HUD padronizado:

- campo: cinematics;
- painel lateral: auditoria do modelo;
- barra inferior: placar ao vivo, probabilidades e suspense do sorteio.

Durante os 90 minutos simulados, o jogo não mostra o placar escolhido pelo
sorteio híbrido. A tela mostra apenas o placar ao vivo, as probabilidades
1X2, os sinais de contexto usados pelo modelo e as possibilidades Poisson/DC.
O placar sorteado aparece como `PLACAR REVELADO` somente no fim da simulação.

Termos:

- `Tendência 1X2/XGBoost calibrada`
- `Poisson/DC`
- `Sorteio híbrido`
- `Resultado oculto até o apito final`
- `Placar revelado`
- `Chance desse placar`

### Copa

Tela da Copa:

- `Ranking Monte Carlo`;
- `Favorito da amostra`;
- `Campanha em destaque`;
- fase de grupos e mata-mata revelados só após a amostra completa.

O ranking não deve aparecer antes do fim da amostra para não misturar prévia com resultado final.

### Console

Comando:

```bash
make console
```

Principais telas:

1. `Métricas`
2. `Odds de campeão`
3. `Odds por fase`
4. `Prever confronto`
5. `Simular uma Copa`
6. `Artefatos`
7. `Força das seleções`
8. `Jogos de grupos`
9. `Backtest por Copa`
10. `Explicar confronto`
11. `Caminho de seleção`
12. `Estabilidade seeds`
13. `Monte Carlo ao vivo`

## Pipeline

Benchmark de workers Monte Carlo:

```bash
make benchmark-mc-workers
```

Ele testa a carga real em `2/4/8/16/32` workers. O padrão do jogo fica em `8` workers porque foi o melhor ponto local entre velocidade e overhead, mas a tela aceita override por ambiente quando for necessário testar outra máquina.

Regerar pacote SOTA:

```bash
PYTHONPATH=modeling/worldcup_2026_ml/src .venv/bin/python -m sota_pipeline
```

Com volume customizado:

```bash
PYTHONPATH=modeling/worldcup_2026_ml/src .venv/bin/python -m sota_pipeline --runs 100000
```

Gera:

- `models/model_sota.pkl`
- `reports/sota_model_report.json`
- `docs/STATISTICAL_AUDIT.md`
- `reports/sota_champion_odds.csv`
- `reports/sota_stage_odds.csv`
- `reports/sota_match_probabilities.csv`
- `reports/sota_tournament_simulation.csv`
- `reports/sota_world_cup_backtest.json`
- `reports/sota_world_cup_backtest_folds.csv`
- `data/processed/sota_training_matches.csv`
- `data/processed/sota_team_strength_2026.csv`
- `data/processed/sota_fixtures_2026.csv`

## Runtime Pygame

O Pygame deve ser tratado como runtime de jogo, não como script de desenho.

Regras:

- `update(dt)` avança tempo, áudio, eventos de gol e Monte Carlo;
- `draw_*` apenas desenha estado atual;
- `WorldCupModel` decide probabilidades e simulações;
- o jogo consome snapshots prontos;
- UI reutilizável fica em `src/arena_ai/ui.py`;
- caches e transforms caros ficam em `src/arena_ai/rendering.py`;
- assets gerados ficam em `assets/generated/`;
- scripts de preparo ficam em `scripts/`;
- runtime não deve recortar ou regenerar sprite durante o jogo.
- o contrato operacional do som fica em `src/arena_ai/audio_manifest.py`;
- assets de áudio aprovados ficam documentados em `assets/sounds/audio_manifest.json`;
- `make audio-qa` valida duração, transiente, clipping, buses, sequência de eventos, ticks da Copa e provenance.

Performance:

- `SurfaceCache` para `smoothscale`, `flip`, `rotozoom` e cover;
- `TextCache` para HUD;
- evitar `pygame.Surface` temporária em loop de draw;
- Monte Carlo roda em thread daemon e comunica por `Queue`.

## Gates

Uso diário:

```bash
make smoke
make validate
```

Quando mexer em modelo, pipeline, dados ou scripts estatísticos:

```bash
make mc-stability
make stats-qa
make validate
```

Atalho equivalente para fechamento estatístico completo:

```bash
make mc-stability && make stats-qa && make validate
```

Para QA visual/performance:

```bash
make aaa-qa
make visual-qa
```

Para som:

```bash
make audio-qa
```

O `validate` fica leve o bastante para iteração diária, mas reprova artefatos estatísticos stale, manifesto bruto stale, Monte Carlo stale, assets faltantes, pureza de render e contrato básico de áudio.

O `aaa-qa` é o gate pesado de cinematics, sprites, 60 FPS, partida completa, uniformes e tela da Copa.

## Limitações

- FC26 e Transfermarkt são proxies de elenco, não convocação oficial.
- Ajustes 2026 são camada operacional auditada, não tuning histórico completo.
- Não há benchmark de odds de mercado no pacote.
- O modelo é didático/experimental, não modelo de apostas.
- Dixon-Coles hierárquico completo segue como possibilidade futura, mas a aproximação sem dataset externo não entregou ganho material.
- Calibradores extras foram testados e não promovidos.

## Fontes De Verdade No Código

- `modeling/worldcup_2026_ml/src/sota_pipeline.py`
  - treino, features, política, Monte Carlo, bracket e relatórios.
- `src/arena_ai/worldcup_model.py`
  - fachada única para game e console.
- `modeling/worldcup_2026_ml/src/console.py`
  - auditoria textual e Monte Carlo ao vivo.
- `src/arena_ai/main.py`
  - runtime Pygame, HUD, tela da Copa e cinematic.
- `scripts/model_stats_qa.py`
  - auditoria estatística.
- `scripts/monte_carlo_stability.py`
  - estabilidade offline do Monte Carlo.
