# Arena AI

Jogo/simulador Pygame de IA aplicada a Copa do Mundo 2026.

O jogador escolhe duas seleções e decide entre simular um confronto ou simular a Copa inteira.

## O que aparece

- Tela de abertura com identidade visual de estádio + IA.
- Seleção de confronto com as 48 seleções do fixture final da Copa 2026.
- `SIMULAR CONFRONTO`: usa sorteio híbrido entre XGBoost e Poisson/Dixon-Coles para resultado em 90 minutos; o placar escolhido fica oculto até o apito final para manter suspense.
- `SIMULAR COPA`: usa Monte Carlo com o mesmo sorteio estatístico influenciado para rodar grupos, melhores terceiros e mata-mata.
- Cena cinematográfica de confronto: atacante em corrida, bola viva, parallax de estádio, gol 3D, goleiro saltando, rede estufando no gol, defesa antes da linha e chute para fora visível além da trave.
- Movimento do atacante vinculado à leitura XGBoost + matriz de placar Poisson.
- Bola, gols, placar e pressão guiados por probabilidades e xG do modelo.
- HUD de confronto em camadas: placar ao vivo, probabilidades 1X2, sinais do modelo e possibilidades Poisson/DC; o placar sorteado só aparece como `PLACAR REVELADO` no 90'.
- Som em camadas: cama de estádio, air, tensão, chant, chute, rede, bass hit, roar e reveal da Copa sincronizados por timeline de áudio.

## Modelo

Documentação canônica consolidada: [docs/MODEL.md](docs/MODEL.md).

O jogo consome o pacote SOTA em `modeling/worldcup_2026_ml/`:

- `models/model_sota.pkl`: pacote carregado pelo Pygame.
- `src/sota_pipeline.py`: treino, predição, bracket e Monte Carlo.
- `reports/sota_model_report.json`: metricas e metadados.
- `reports/sota_statistical_report.json`: auditoria estatística técnica. A leitura Markdown consolidada fica em [docs/STATISTICAL_AUDIT.md](docs/STATISTICAL_AUDIT.md).
- `reports/sota_champion_odds.csv`: snapshot de odds quando o pipeline completo e regenerado.
- `reports/sota_match_probabilities.csv`: probabilidades e xG dos jogos de grupo.

Motores usados no jogo:

- `XGBoost`: motor principal de classificação para vitória, empate e derrota em 90 minutos.
- `Poisson/Dixon-Coles`: motor de placar, top 5 placares, over/under e ambos marcam. No confronto, sua matriz também vota no pacote do resultado, então empates e zebras continuam possíveis quando têm peso estatístico.
- `Monte Carlo`: motor de Copa completa. Cada jogo usa o mix classificador + Poisson/Dixon-Coles: primeiro sorteia tendência estatística de vitória/empate/derrota, depois sorteia o placar dentro dessa tendência.
- `ELO`, ranking FIFA e regressão logística (`logistic_1x2`): sinais/baselines auditados dentro do pacote, não modos isolados no jogo. A ablação nested atual zerou `ELO` no blend final, mas ele segue documentado como baseline e feature histórica.

Política atual do sorteio híbrido: `0.88` classificador 1X2 + `0.12` Poisson/Dixon-Coles, com empate limitado pela faixa calibrada `4%` a `30%`. A ablação completa dos 63 subconjuntos escolheu o blend interno `XGBoost 60,6% + XGBoost competitivo 27,3% + regressão logística 12,1%`; `ELO`, `Poisson` e `count_poisson` ficaram com peso `0` dentro do classificador. O Poisson/Dixon-Coles continua ativo no sorteio final de placar e pacote com peso `12%`. O `draw_xgb` foi removido: como recebia peso zero, manter um modelo desligado quebrava o padrão SOTA/KISS. A política final vem de validação nested temporal sem vazamento: em cada ano externo, os modelos internos treinam só antes da janela de validação interna, os componentes do blend e os parâmetros de sorteio são escolhidos nessa janela posterior, e o ano externo é avaliado depois de retreino apenas com dados anteriores. Documento didático completo: [docs/MODEL.md](docs/MODEL.md).

Na tela da Copa, a lista de campeões prováveis usa `1000` Copas frescas no runtime. O jogo roda a amostra em thread dedicada, usa cache apenas para predições de confronto já aquecidas e mantém a barra de progresso ligada ao cálculo real, sem revelar o resultado de cara. O banco de campanhas em `modeling/worldcup_2026_ml/models/runtime_prediction_cache.pkl` existe só como modo turbo explícito (`ARENA_AI_TOURNAMENT_MC_BOOTSTRAP=1 make run`) para builds e auditorias rápidas. O ranking só aparece quando a amostra completa termina, para não misturar prévia com resultado final. O caminho mostrado escolhe um destaque dentro do top 5 de campeões, ponderado pelas odds, e então seleciona uma Copa concreta em que esse destaque foi campeão usando plausibilidade narrativa: finalista com frequência real de final, placar menos caricato e zebra controlada.

Há três volumes diferentes, de propósito:

- `1000` Copas: modo jogável/UI e snapshot salvo no relatório atual.
- `10000` Copas: default do CLI de rebuild completo do pipeline, quando rodado sem `--runs`.
- `5k/10k` Copas: auditoria offline operacional de estabilidade do ranking em `make mc-stability`.
- `1k/2k/5k` Copas completas: auditoria offline de estabilidade por fase, finalistas e confrontos de chave.

O padrão de `8` threads é usado pela UI no cálculo fresh. `ARENA_AI_TOURNAMENT_MC_BOOTSTRAP=1 make run` ativa o banco de cenários como modo turbo opcional, e `make benchmark-mc-workers` mede o caminho completo/fallback. Quando `model_sota.pkl` ou `sota_pipeline.py` mudarem, rode `make runtime-cache` para renovar esse banco opcional.

Documentação detalhada do conceito, dados, validação e runtime: [docs/MODEL.md](docs/MODEL.md).

## Rodar

O projeto está padronizado em Python `3.12.x`. O benchmark local com a carga real de Monte Carlo deixou o Python 3.14 mais lento nesta stack, então mantemos `.python-version` em `3.12.12` e `requires-python = ">=3.12,<3.13"`.

```bash
make sync
make run
```

Console de auditoria do modelo:

```bash
make console
```

No console, use `Monte Carlo ao vivo` para ver a mesma lógica da tela da Copa. O modo padrão é `fresh`, igual ao jogo; `bootstrap` fica disponível apenas como modo turbo explícito com o banco de cenários.

Console do bolão com fase fixa e escolha de campeão:

```bash
make bolao
```

O bolão é um utilitário Rich do projeto: não gera aplicativo, binário ou ZIP de
release próprio. Quando for a hora de distribuir o produto, `make build-release`
gera apenas os artefatos Mac e Windows do ArenaAI.

O bolão lê os placares registrados em `modeling/worldcup_2026_ml/data/observed/worldcup_2026_group_stage_results.csv` junto com a metadata local `worldcup_2026_group_stage_snapshot.json`. A metadata declara o `as_of` com timezone, contagem e hash do CSV, proveniência manual local e `official_source: false`; o programa confere identidade do confronto, grupo, ordem cronológica e que cada jogo observado já teria terminado (kickoff + duas horas) no `as_of`. A metadata também aceita `fair_play_scores` agregado por seleção, desde que cubra as 48 equipes. Sem esse dado, qualquer empate de grupo que dependa de fair play é recusado em vez de cair para o ranking FIFA. Isso não é uma fonte FIFA nem uma validação independente dos resultados. Jogos eliminatórios concluídos ficam em `worldcup_2026_knockout_results.csv`, com o placar de 90 minutos, prorrogação e pênaltis quando houver; os 32 jogos até a final estão registrados ali. Eles são travados na chave e uma seleção eliminada não pode voltar em uma trilha condicionada.

Para cada seleção, os gols registrados na Copa são comparados ao xG histórico previsto antes do jogo. Nos jogos vindos do CSV, o console exibe esse xG pré-jogo/base, sem recalculá-lo depois de observar todos os 72 resultados. O ajuste candidato de forma só é avaliado depois da fotografia completa dos 72 jogos: é um posterior Gamma-Poisson por ataque e defesa, aplicado apenas às projeções futuras nos lambdas da matriz Poisson/Dixon-Coles. O prior é escolhido somente nos primeiros dois terços cronológicos do CSV e precisa superar o baseline histórico no terço final. Se não superar, o bolão preserva o híbrido histórico; ele não força um peso mínimo para a forma atual.

Os 16 avos funcionam como uma checagem externa posterior: a forma dos grupos reduziu levemente log-loss 1X2, Brier, log-loss de placar e as perdas de avanço em relação ao histórico puro. Por isso o prior de `8,0` gols equivalentes permanece congelado e os placares regulamentares dos 16 jogos são acrescentados ao posterior. Gols de prorrogação e cobranças de disputa não atualizam a taxa de 90 minutos; eles servem somente para resolver a chave. O peso mediano da foto atual passou de `34,6%` depois dos grupos para `40,0%` depois dos 16 avos. A amostra não retreina XGBoost, não troca os pesos globais do híbrido e não recalibra empate ou pênaltis. Esse é um gate temporal local, não um selo de calibração SOTA para o bolão.

As oito oitavas são avaliadas contra a foto congelada ao fim dos 16 avos. Nessa rodada, o histórico puro ficou ligeiramente melhor que a forma atual: `0,7766` contra `0,7834` de log-loss 1X2. Todas as diferenças de perda ficaram abaixo da margem operacional de `0,01`, reutilizada do limiar de materialidade temporal já existente; ela é um guardrail, não um teste formal de não inferioridade. Por isso não há evidência para retreinar ou trocar a política: o prior continua congelado, somente os placares de 90 minutos são acrescentados e o peso mediano passa para `41,9%`. A atualização altera a probabilidade de avanço nas quartas em no máximo `0,62 p.p.`.

As quatro quartas usam a foto congelada após as oitavas como novo teste prospectivo. Ela identificou os quatro classificados e melhorou marginalmente o log-loss 1X2 (`0,9099` para `0,9089`) e o Brier (`0,5576` para `0,5557`) frente ao histórico puro. Placar exato e perdas de avanço ficaram ligeiramente piores, mas todas as diferenças permaneceram abaixo do mesmo guardrail de `0,01`. Os dois empates em 90 minutos, contra `1,00` esperado, produzem resíduo padronizado de `1,16` e não sustentam recalibração de empate ou prorrogação. Só os placares regulamentares entram no posterior: o peso mediano passa de `41,9%` para `42,3%`, e o maior deslocamento de avanço nas semifinais é `0,32 p.p.`.

Os quatro jogos finais são avaliados em três fotos prospectivas: antes das semifinais, antes do terceiro lugar e antes da final. A foto acertou os dois finalistas e indicou a Espanha como campeã: `35,4%` no Monte Carlo pré-semifinal e `57,9%` antes da final. No conjunto, acertou três dos quatro vencedores; o erro foi França `4 x 6` Inglaterra, placar cuja probabilidade exata era `0,0069%`. Mesmo assim, as cinco diferenças agregadas de perda contra o histórico ficaram dentro do guardrail de `0,01`. O dataset processado não contém a fase histórica do jogo, portanto não permite estimar um regime próprio de terceiro lugar: nenhum multiplicador foi inventado. O peso mediano final passa de `42,3%` para `43,4%`, e o gol espanhol da prorrogação permanece fora do posterior de 90 minutos.

Enquanto existem jogos abertos, o console roda 1000 Copas Monte Carlo para listar até 10 campeões possíveis. Empates de mata-mata seguem a matriz Poisson/Dixon-Coles na prorrogação e usam pênaltis neutros, sem extrapolar uma probabilidade de 90 minutos para a disputa. O intervalo de Wilson de 95% mostrado por seleção representa somente erro de amostragem da simulação; não mede a incerteza total do modelo, da forma ou do snapshot manual. Depois que a final é travada, o console não executa Monte Carlo nem apresenta `100%` como previsão: exibe a Espanha como campeã observada e a chave encerrada.

Confrontos neutros também são calculados nos dois sentidos da ordem da chave e espelhados antes de produzir o 1X2, xG, prorrogação e avanço. Assim, aparecer como mandante nominal não cria vantagem estatística; descanso, viagem e sede continuam podendo entrar quando o fixture fornece esse contexto real.

Ao incluir uma rodada nova, atualize CSV e metadata juntos: `match_number`, `group`, mandante/visitante oficiais do fixture e placar no CSV; `as_of`, `result_count` e `results_sha256` na metadata. O bolão recusa números duplicados, grupo incorreto, confronto que não corresponda ao calendário oficial, hash stale, uma foto que pule jogos cronologicamente anteriores ou resultado cujo término mínimo (kickoff + duas horas) seja posterior ao `as_of`.

Para ver só a fase de grupos:

```bash
make bolao-grupos
```

Para abrir direto uma história sem prompt, use rank ou nome do top: `uv run arena-bolao --campeao 2` ou `uv run arena-bolao --campeao Brasil`. Para reduzir a saída inicial, filtre grupos: `uv run arena-bolao --grupo C`.

Auditoria da transição para as oitavas:

```bash
make bolao-knockout-audit
```

O alvo reavalia os 16 avos sem look-ahead, compara histórico puro e forma validada nos grupos e grava `bolao_round32_calibration.json/.csv`. Também resolve a chave e gera `bolao_round16_predictions.csv` com xG, 1X2 de 90 minutos, probabilidade de avanço, placar modal e o delta causado pela nova rodada. O gate recusa snapshot incompleto, piora das métricas auditadas, atualização que misture prorrogação/pênaltis na taxa regulamentar, deslocamento acima de 5 pontos percentuais em um confronto ou uma chave de oitavas incompleta.

Auditoria da transição para as quartas:

```bash
make bolao-round16-audit
```

O alvo preserva a foto pré-oitavas, audita os oito resultados sem look-ahead e grava `bolao_round16_calibration.json/.csv`. A chave seguinte fica em `bolao_quarterfinal_predictions.csv`. Além das métricas e do guardrail de 5 pontos percentuais por confronto, a auditoria registra a correção factual do jogo 96: Suíça `0–0` Colômbia após 120 minutos, com vitória suíça por `4–3` nos pênaltis.

Auditoria da transição para as semifinais:

```bash
make bolao-quarterfinal-audit
```

O alvo preserva a foto pré-quartas, audita os quatro resultados sem look-ahead e grava `bolao_quarterfinal_calibration.json/.csv`. A nova foto fica em `bolao_semifinal_predictions.csv`, com `França x Espanha` e `Inglaterra x Argentina`. A auditoria exige os 28 resultados travados, exclui os três gols de prorrogação do posterior de 90 minutos e impede que quatro jogos sejam usados para retreinar XGBoost ou selecionar novos pesos globais.

Auditoria final do torneio:

```bash
make bolao-final-audit
```

O alvo congela snapshots antes das semifinais, do terceiro lugar e da final. Ele grava `bolao_final_calibration.json`, `bolao_semifinal_calibration.csv` e `bolao_medal_matches_calibration.csv`, recupera as probabilidades Monte Carlo prospectivas da Espanha, mede o desvio do `6 x 4` e valida que a final foi `0 x 0` em 90 minutos antes do `1 x 0` na prorrogação. O relatório mantém 2026 como bloco fora da amostra para qualquer retreino futuro.

Auditoria de estabilidade do próprio bolão:

```bash
make bolao-mc-stability
```

O gate roda prefixos MC aninhados de `1k` e `2k` Copas e três repetições de `2k` com seeds independentes sobre os grupos fixos, a forma temporal e o mata-mata do bolão. Ele grava `modeling/worldcup_2026_ml/reports/bolao_monte_carlo_stability.json`, registra fingerprints do código, modelo, cache, auditoria e snapshot fixo, e falha se o delta máximo de probabilidade, a sobreposição do top ou o z-score de duas amostras configurados não passarem. Com a final observada, o resultado é deterministicamente Espanha em todas as execuções; isso valida o lock da chave, não a calibração de uma probabilidade de título. Nesse estado o relatório marca a incerteza preditiva como não aplicável e não publica intervalo Wilson para o 100% observado.

Auditoria de viés do top 10:

```bash
make bolao-top10-audit
```

Esse alvo compara até dez candidatos ainda vivos com e sem a forma atual, verifica cada candidato contra todos os demais classificados nas duas ordens nominais e reprova diferença de 1X2, xG, avanço ou pênaltis além de erro numérico. O relatório fica em `modeling/worldcup_2026_ml/reports/bolao_top10_bias_audit.json` e os pares auditados em `bolao_top10_bias_audit.csv`. Após a final, sobra apenas a campeã observada; a auditoria continua útil para simetria e integridade, não como ranking preditivo, e deixa vazios os intervalos Monte Carlo da campeã travada.

Auditoria estatística do pacote:

```bash
make stats-qa
```

Esse alvo gera `sota_statistical_report.json`, atualiza [docs/STATISTICAL_AUDIT.md](docs/STATISTICAL_AUDIT.md), bins de calibração, calibração detalhada por classe, bootstrap por bloco temporal/torneio, intervalos de incerteza de campeão/fase, ablação completa dos 63 subconjuntos 1X2, sensibilidade do `rho` Dixon-Coles, auditoria dos ajustes 2026 de elenco/Transfermarkt/contexto, auditoria de ordem neutra no runtime e `sota_internal_frontier_experiments.csv` com os experimentos limite sem dataset externo. Ele compara o runtime com ELO por log-loss e RPS no mesmo recorte temporal, em vez de usar uma meta estática de acurácia. Ele não retreina o pickle: valida o pacote atual e documenta se a política continua SOTA/KISS.

Para estabilidade de relatório fora do jogo:

```bash
make mc-stability
```

Esse alvo roda Monte Carlo offline em duas camadas. A primeira usa o caminho otimizado de campeão em `5k` e `10k` Copas. A segunda roda Copas completas em `1k`, `2k` e `5k` para medir estabilidade de fases, finalistas e confrontos de chave. Ele grava `sota_monte_carlo_stability.json/.csv` e `sota_monte_carlo_stage_bracket_stability.csv`, registra fingerprint do pacote e falha se os limites configurados forem violados, incluindo churn e z-score das probabilidades de confrontos top 8 em amostras aninhadas. A auditoria é de convergência: o volume maior estende a mesma amostra base, em vez de comparar seeds independentes. Ele é propositalmente separado do `validate`, porque é uma auditoria pesada e não deve atrapalhar o ciclo rápido; `20k/50k` continua possível via argumentos manuais quando quisermos auditoria archival.

O carimbo **SOTA/KISS acadêmico** é uma avaliação interna do pacote histórico disponível. Ele exige nested temporal sem vazamento, ablação completa, comparação ELO no mesmo recorte por log-loss/RPS, empate calibrado, Poisson/Dixon-Coles preservado para placar, simetria casa/fora no treino e no runtime de jogos neutros, intervalo Monte Carlo, incerteza por fase, bootstrap por bloco, auditoria dos proxies 2026, manifesto/hash completo dos dados brutos com sanidade semântica e esgotamento dos experimentos internos sem dados externos. Ele não é uma alegação de calibração de mercado, nem se estende automaticamente ao snapshot manual ou à forma atual do bolão. O `stats-qa` também grava hash do pickle, relatório do modelo, CSV de treino, `sota_pipeline.py`, scripts de QA e todos os arquivos em `data/raw`; o `validate` reprova relatório estatístico, manifesto bruto, sanidade raw ou Monte Carlo stale.

## Empacotamento Mac / Windows

O projeto tem um `Makefile` para padronizar validação, staging de assets e
empacotamento:

```bash
make validate
make build-assets-qa
make build-mac
```

O PyInstaller não faz cross-compile confiável. Por isso:

- `make build-mac` gera o `.app` no macOS.
- `make build-windows` gera o `.exe` dentro do Windows.
- No macOS, `make build-windows` usa a VM Windows configurada em `win/`.

Fluxo Windows remoto:

```bash
cp win/.env.example win/.env
# edite ARENA_WIN_HOST, ARENA_WIN_USER, ARENA_WIN_REMOTE_ROOT e ARENA_WIN_QA
make runtime-cache
make build-windows
```

O alvo remoto exige um worktree limpo, fixa commit/tree antes do QA, confirma
que a identidade não mudou e empacota exatamente o SHA capturado com
`git archive`. Depois envia o bundle e sua proveniência para a VM por SSH, roda
`uv sync --dev`, executa o QA definido por `ARENA_WIN_QA` e roda o PyInstaller
no Windows. O ZIP e o sidecar obrigatório do build são baixados para:

```text
win/artifacts/ArenaAI-windows-latest.zip
win/artifacts/build-result.json
```

Depois o Makefile valida que o ZIP abre, contém `ArenaAI.exe` e inclui
`runtime_prediction_cache.pkl`, usado para aquecer predições de confronto e para
o modo turbo opcional da tela da Copa. O sidecar vincula o hash do ZIP e do
executável ao mesmo fingerprint de source usado pelo `.app`; um binário antigo,
um sidecar ausente ou qualquer divergência de commit/tree reprova o release.

Se estiver trabalhando diretamente em uma máquina Windows, rode:

```powershell
make sync
make build-windows
```

Esse caminho também exige checkout Git limpo e constrói em uma extração
temporária do SHA aprovado; `build-current` sem `.git` é reservado ao wrapper
que fornece `RELEASE_SOURCE_PROVENANCE`.

O build usa `make build-assets-qa` para montar `build/release_assets/` só com
assets de runtime e pacote SOTA mínimo; `assets/sounds/candidates/`, docs e
fontes brutas não entram no bundle. O staging e o ZIP Windows precisam conter
todo o inventário declarado, inclusive os 54 sheets legados, os 18 sheets POC2
ativos e os 16 quadros do goleiro por direção.
Guia completo da VM, bootstrap RDP/SSH,
variáveis e cuidados de segurança: [docs/BUILD.md](docs/BUILD.md).

## Controles

- `Enter` / `Espaço`: avança no menu.
- `←` / `→`: troca a seleção da esquerda.
- `A` / `D`: troca a seleção da direita.
- `Espaço` ou `Enter`: simula o confronto selecionado.
- `T`: roda nova amostra Monte Carlo e mostra ranking + Copa em destaque.
- `R` ou `Espaço` na simulação: roda novamente.
- `Backspace` ou botão voltar: retorna para seleção.
- `Esc`: sai.

## Assets Usados

Documentação canônica consolidada de assets, áudio, fontes e licenças: [docs/ASSETS.md](docs/ASSETS.md).

- `assets/generated/title_stadium_ai.png`: background da abertura.
- `assets/generated/stadium_parallax_real.png`: estádio/campo realista usado no parallax do confronto.
- `assets/generated/parallax_sources/imagen_turf_*.png`: sprites-fonte do tapete de campo gerados pelo `image_gen`.
- `assets/generated/parallax/turf_*_strip.png`: recortes derivados dos sprites-fonte, preparados por `scripts/generate_parallax_turf.py` para o parallax contínuo do gramado.
- Uniformes disponíveis: azul, azul claro/celeste, vermelho, vinho/bordô, branco, verde, amarelo/dourado, laranja e preto.
- `assets/generated/cinematic/poc2_runner_{right,left}_*.png` e `poc2_runner_motion.json`: corrida ativa promovida. São 18 sheets para 9 uniformes e 2 direções, com 8 poses GPT Image diretas por ciclo, timing autoral variável e duração de `1,6 s`. Esquerda e direita são artes independentes; `ORACLE` faz parte dos pixels originais da camisa, sem espelho, recoloração, morph, crossfade ou texto sobreposto.
- `assets/generated/cinematic/runner*_smooth_*.png`: inventário final de 54 sheets que preserva as 16 poses de chute, 8 de parada e a corrida legada de 16 poses por uniforme/direção. O lance promovido usa POC2 na condução e faz handoff para o chute autoral.
- `assets/generated/cinematic/runner_motion.json`: contrato runtime v8 de chute/parada, pivô, baseline, landmarks, compensação autoral de altura e política alpha-safe de contato. Os sheets e os dois contratos de movimento são artefatos promovidos; o repositório não mantém geradores ou fontes intermediárias das POCs.
- `assets/generated/cinematic/keeper_anim_{right,left}_{0..15}.png` e `keeper_motion.json`: 16 quadros runtime por direção, formados por 15 poses autorais únicas e um quadro aprovado de reset. O runtime escolhe um único quadro por update, ancora o pouso no gramado, mantém a pose derrotada após gol sofrido e usa a recuperação completa em defesa ou chute para fora.
- `assets/generated/cinematic/poc7_runtime_contract.json`: contrato cinematográfico v5 promovido, com estado completo a 60 Hz para 30 sequências direcionais de gol, defesa e chute para fora.
- `assets/generated/cinematic/poc7_net/`: 58 camadas finais protegidas por hash. Cada campanha de gol possui 70 estados traseiros a 30 Hz e contato frontal amostrado a 60 Hz, empacotados em atlas e selecionados sem crossfade.
- `assets/generated/balls3d/ball_{0..31}.png`: 32 quadros finais da bola; o runtime seleciona um quadro nativo por update, sem crossfade.
- `src/arena_ai/cinematic_uniforms.py`: define as 9 cores de uniforme, variações de bermuda e mapeamento para seleções.
- `assets/generated/balls3d/*.png`: frames da bola.
- `assets/generated/flags/*.png`: 48 bandeiras em sprite geradas por `image_gen`; a validação falha se alguma seleção esperada não tiver PNG.
- `assets/asset_manifest.json`: manifesto canônico dos assets ativos, fontes ainda mantidas e cardinalidades exatas do payload de release.
- `assets/fonts/Oxanium.ttf`: fonte OFL empacotada para uso direto no Pygame.
- `assets/sounds/runtime_assets/`: sons efetivamente usados pelo jogo. O contrato operacional fica em `src/arena_ai/audio_manifest.py`; a governança, provenance, licenças e hashes ficam em `assets/sounds/audio_manifest.json`.
- `assets/sounds/audio_manifest.json`: manifesto único de áudio aprovado. `assets/sounds/candidates/` é biblioteca de curadoria e nunca deve ser usado direto pelo código.
- Camadas de áudio aprovadas: abertura, base de estádio, air, torcida leve, tensão, chant, ataque, chute, whoosh, rede, explosão de gol, bass hit, roar, reverb, apito inicial/final e stingers da Copa. A política de cada cue fica centralizada no `AudioEngine`: o estádio não some no impacto, o `crowd_attack_rise` vira arco de gol/ataque e a fila da Copa é limpa ao trocar de cena.

## Validação Visual

`make validate` compila o projeto e roda o gate essencial: smoke do modelo, relatório estatístico SOTA/KISS já gerado, manifesto/hash dos dados brutos com sanidade semântica, Monte Carlo fresco, inventário de sprites/som, manifesto de assets, contrato de áudio, layout da tela de confronto, pureza de render e áudio essencial. Ele fica leve para a iteração diária, mas reprova artefatos estatísticos stale. Quando mexer em modelo, pipeline, dados ou scripts de auditoria, rode `make mc-stability && make stats-qa && make validate`.

`make aaa-qa` roda o gate pesado do renderer ativo: inventário final, `ORACLE` nativo e legível, matriz cinematográfica promovida, direção e limites do goleiro, contato alpha da bola, rede e z-order, sincronismo de áudio/placar, determinismo, partida completa, caches e orçamento de 60 FPS. CPU do processo é sempre bloqueante; wall-clock só bloqueia com `ARENA_AI_STRICT_WALL_CLOCK_QA=1` em host controlado, evitando classificar contenção do scheduler como regressão. O wordmark é medido nos pixels finais em 864 combinações: 720 quadros legados de corrida/chute/parada mais as 144 poses POC2 ativas.

`make cinematic-game-qa` valida os contratos promovidos no `App`. A matriz contém 30 sequências: 18 campanhas de gol (`2 direções × 3 alturas × 3 medoids`) e 12 campanhas sem gol (`save/wide`, `2 direções × 3 alturas`). Os medoids de gol foram escolhidos offline entre 1.000 planos e carregam pesos inteiros que somam 10.000 por direção/altura; no jogo a escolha continua determinística para o mesmo lance. O gate rasteriza 150 checkpoints, percorre 2.502 amostras sequenciais pós-impacto e valida cadência, hashes, regiões de atlas e contato da rede; também prova ator/bola/goleiro/gol no framebuffer em oito casos, confirma a progressão POC2 de 8 poses e o chute de 16 poses nas 18 combinações uniforme/direção, executa 162 casos de movimento e 270 handoffs POC2 (`30 sequências × 9 uniformes`), valida os 58 assets do preload assíncrono, executa seis timelines reais por `update -> draw -> áudio -> placar` e mede 2.430 contatos pós-chute (`30 sequências × 9 uniformes × 9 instantes`), sempre com sobreposição alpha máxima de 3%.

`make cinematic-game-qa-capture` recria `artifacts/cinematic_game_qa/current/frames/` com os 150 checkpoints do renderer real e um manifesto schema 14. `make cinematic-game-qa-check` repete o gate ao vivo, valida hashes das fontes runtime e dos 198 itens cinematográficos finais (197 arquivos de mídia/metadata mais o contrato POC7) e renderiza novamente os 150 frames em diretório temporário para comparação RGBA. O campo geral `runtime_assets` do manifesto possui 200 arquivos porque agrega três camadas de parallax; o contrato POC7 é registrado separadamente e o contrato POC2 integra o inventário runtime. O gate também confere cadências de rede (`30 Hz` traseira e contato limitado a `180 ms`), seleção de variante, direção e recuperação do goleiro, geometria, trajetória, materialização no framebuffer, ausência de I/O durante o lance, `flush` de áudio e transição exata do placar. O pan estéreo aplicado é provado por `audio-qa`; `aaa-qa` executa os dois gates. Não há aprovação baseada apenas em imagens antigas. Esses gates comprovam integridade e determinismo dos contratos promovidos; a aceitação estética final continua sendo uma revisão visual humana.

`make visual-qa` gera `artifacts/visual_qa/current/` do zero: contact sheets com as 8 poses POC2 ativas de corrida, 16 de chute e 8 de parada por uniforme/direção, os 16 quadros do goleiro em cada direção, vídeos mandante/visitante a 60 fps, close temporal em câmera lenta `2,5x` e CSV/JSON quadro a quadro. A captura segue até o repouso da bola na rede; `metadata.json` registra contagem, duração, hashes e inventário recursivo. `make visual-qa-check` reprova evidência stale, órfã, alterada ou não determinística.

Fontes dos sons:

- Mixkit Sound Effects: https://mixkit.co/free-sound-effects/
- Mixkit License: https://mixkit.co/license/
- Manifesto operacional: `src/arena_ai/audio_manifest.py`
- Manifesto de governança/provenance: `assets/sounds/audio_manifest.json`
