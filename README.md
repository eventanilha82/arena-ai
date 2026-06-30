# Arena AI

Jogo/simulador Pygame de IA aplicada a Copa do Mundo 2026.

O jogador escolhe duas seleções e decide entre simular um confronto ou simular a Copa inteira.

## O que aparece

- Tela de abertura com identidade visual de estádio + IA.
- Seleção de confronto com as 48 seleções do fixture final da Copa 2026.
- `SIMULAR CONFRONTO`: usa sorteio híbrido entre XGBoost e Poisson/Dixon-Coles para resultado em 90 minutos; o placar escolhido fica oculto até o apito final para manter suspense.
- `SIMULAR COPA`: usa Monte Carlo com o mesmo sorteio estatístico influenciado para rodar grupos, melhores terceiros e mata-mata.
- Cena cinematográfica de confronto: atacante em corrida, bola viva, parallax de estádio, gol 3D, goleiro saltando e rede estufando no lance de gol.
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

Política atual do sorteio híbrido: `0.88` classificador 1X2 + `0.12` Poisson/Dixon-Coles, com empate limitado pela faixa calibrada `4%` a `46%`. A ablação completa dos 63 subconjuntos escolheu o blend interno `XGBoost 60,6% + XGBoost competitivo 27,3% + regressão logística 12,1%`; `ELO`, `Poisson` e `count_poisson` ficaram com peso `0` dentro do classificador. O Poisson/Dixon-Coles continua ativo no sorteio final de placar e pacote com peso `12%`. O `draw_xgb` foi removido: como recebia peso zero, manter um modelo desligado quebrava o padrão SOTA/KISS. A política final vem de validação nested temporal sem vazamento: em cada ano externo, os modelos internos treinam só antes da janela de validação interna, os componentes do blend e os parâmetros de sorteio são escolhidos nessa janela posterior, e o ano externo é avaliado depois de retreino apenas com dados anteriores. Documento didático completo: [docs/MODEL.md](docs/MODEL.md).

Na tela da Copa, a lista de campeões prováveis usa `1000` Copas frescas no runtime. O jogo roda a amostra em thread dedicada, usa cache apenas para predições de confronto já aquecidas e mantém a barra de progresso ligada ao cálculo real, sem revelar o resultado de cara. O banco de campanhas em `modeling/worldcup_2026_ml/models/runtime_prediction_cache.pkl` existe só como modo turbo explícito (`ARENA_AI_TOURNAMENT_MC_BOOTSTRAP=1 make run`) para builds e auditorias rápidas. O ranking só aparece quando a amostra completa termina, para não misturar prévia com resultado final. O caminho mostrado escolhe um destaque dentro do top 5 de campeões, ponderado pelas odds, e então seleciona uma Copa concreta em que esse destaque foi campeão usando plausibilidade narrativa: finalista com frequência real de final, placar menos caricato e zebra controlada.

Há três volumes diferentes, de propósito:

- `1000` Copas: modo jogável/UI e snapshot salvo no relatório atual.
- `10000` Copas: default do CLI de rebuild completo do pipeline, quando rodado sem `--runs`.
- `5k/10k` Copas: auditoria offline operacional de estabilidade do ranking em `make mc-stability`.
- `1k/2k` Copas completas: auditoria offline de estabilidade por fase, finalistas e confrontos de chave.

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

O bolão lê os placares registrados em `modeling/worldcup_2026_ml/data/observed/worldcup_2026_group_stage_results.csv` junto com a metadata local `worldcup_2026_group_stage_snapshot.json`. A metadata declara o `as_of` com timezone, contagem e hash do CSV, proveniência manual local e `official_source: false`; o programa confere identidade do confronto, grupo, ordem cronológica e que cada jogo observado já teria terminado (kickoff + duas horas) no `as_of`. A metadata também aceita `fair_play_scores` agregado por seleção, desde que cubra as 48 equipes. Sem esse dado, qualquer empate de grupo que dependa de fair play é recusado em vez de cair para o ranking FIFA. Isso não é uma fonte FIFA nem uma validação independente dos resultados. Jogos eliminatórios concluídos ficam em `worldcup_2026_knockout_results.csv`, com o placar de 90 minutos, prorrogação e pênaltis quando houver; eles são travados na chave e uma seleção eliminada não pode voltar em uma trilha condicionada.

Para cada seleção, os gols registrados na Copa são comparados ao xG histórico previsto antes do jogo. Nos jogos vindos do CSV, o console exibe esse xG pré-jogo/base, sem recalculá-lo depois de observar todos os 72 resultados. O ajuste candidato de forma só é avaliado depois da fotografia completa dos 72 jogos: é um posterior Gamma-Poisson por ataque e defesa, aplicado apenas às projeções futuras nos lambdas da matriz Poisson/Dixon-Coles. O prior é escolhido somente nos primeiros dois terços cronológicos do CSV e precisa superar o baseline histórico no terço final. Se não superar, o bolão preserva o híbrido histórico; ele não força um peso mínimo para a forma atual. Esse é um gate temporal local, não um selo de calibração SOTA para o bolão. O status e o delta de log-score aparecem no console.

Depois, o console roda 1000 Copas Monte Carlo somente nos jogos ainda abertos da chave que nasce dessa fase fixa para listar o top 10 de campeões. Empates de mata-mata seguem a matriz Poisson/Dixon-Coles na prorrogação e usam pênaltis neutros, sem extrapolar uma probabilidade de 90 minutos para a disputa. O intervalo de Wilson de 95% mostrado por seleção representa somente erro de amostragem da simulação; não mede a incerteza total do modelo, da forma ou do snapshot manual. A partir desse ranking, você escolhe um campeão e o console monta uma trilha modal condicionada para esse time ser campeão. Essa trilha é explicativa, não uma amostra de `P(chave | campeão)`.

Confrontos neutros também são calculados nos dois sentidos da ordem da chave e espelhados antes de produzir o 1X2, xG, prorrogação e avanço. Assim, aparecer como mandante nominal não cria vantagem estatística; descanso, viagem e sede continuam podendo entrar quando o fixture fornece esse contexto real.

Ao incluir uma rodada nova, atualize CSV e metadata juntos: `match_number`, `group`, mandante/visitante oficiais do fixture e placar no CSV; `as_of`, `result_count` e `results_sha256` na metadata. O bolão recusa números duplicados, grupo incorreto, confronto que não corresponda ao calendário oficial, hash stale, uma foto que pule jogos cronologicamente anteriores ou resultado cujo término mínimo (kickoff + duas horas) seja posterior ao `as_of`.

Para ver só a fase de grupos:

```bash
make bolao-grupos
```

Para abrir direto uma história sem prompt, use rank ou nome do top: `uv run arena-bolao --campeao 2` ou `uv run arena-bolao --campeao Brasil`. Para reduzir a saída inicial, filtre grupos: `uv run arena-bolao --grupo C`.

Auditoria de estabilidade do próprio bolão:

```bash
make bolao-mc-stability
```

O gate roda prefixos MC aninhados de `1k` e `2k` Copas e três repetições de `2k` com seeds independentes sobre os grupos fixos, a forma temporal e o mata-mata do bolão. Ele grava `modeling/worldcup_2026_ml/reports/bolao_monte_carlo_stability.json`, registra fingerprints do código, modelo, cache, auditoria e snapshot fixo, e falha se o delta máximo de probabilidade, a sobreposição do top ou o z-score de duas amostras configurados não passarem em qualquer um dos dois testes. Os intervalos de Wilson presentes nesse relatório também são apenas de erro de amostragem MC, não de incerteza total do modelo.

Auditoria de viés do top 10:

```bash
make bolao-top10-audit
```

Esse alvo compara os dez candidatos do ranking com e sem a forma atual, verifica cada candidato contra todos os demais classificados nas duas ordens nominais e reprova diferença de 1X2, xG, avanço ou pênaltis além de erro numérico. O relatório fica em `modeling/worldcup_2026_ml/reports/bolao_top10_bias_audit.json` e os pares auditados em `bolao_top10_bias_audit.csv`. A comparação de forma é uma análise de sensibilidade; ela não trata o intervalo Monte Carlo como incerteza total do modelo.

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

O alvo remoto empacota o workspace atual, envia para a VM por SSH, roda
`uv sync --dev`, executa o QA definido por `ARENA_WIN_QA`, roda o PyInstaller no
Windows e baixa o artefato para:

```text
win/artifacts/ArenaAI-windows-latest.zip
```

Depois o Makefile valida que o ZIP abre, contém `ArenaAI.exe` e inclui
`runtime_prediction_cache.pkl`, usado para aquecer predições de confronto e para
o modo turbo opcional da tela da Copa.

Se estiver trabalhando diretamente em uma máquina Windows, rode:

```powershell
make sync
make build-windows
```

O build usa `make build-assets-qa` para montar `build/release_assets/` só com
assets de runtime e pacote SOTA mínimo; `assets/sounds/candidates/`, docs e
fontes brutas não entram no bundle. Guia completo da VM, bootstrap RDP/SSH,
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
- `assets/generated/cinematic_sources/imagen_oracle_*.png`: sheets-fonte de jogadores gerados por `image_gen` com a única marca permitida, `ORACLE`, no centro da camisa; sem escudo no shorts, sem brasão, sem número e sem outras marcas.
- Uniformes disponíveis: azul, azul claro/celeste, vermelho, vinho/bordô, branco, verde, amarelo/dourado, laranja e preto.
- `assets/generated/cinematic/runner_*.png`: ciclo de corrida do atacante em quatro frames, recortado dos sheets `imagen_oracle_*` por uniforme.
- `assets/generated/cinematic/keeper_anim_*.png`: animação do goleiro saltando no lance de gol.
- `assets/generated/cinematic/goal_net_*.png`: camada traseira do sprite 3D da trave/rede.
- `assets/generated/cinematic/goal_front_*.png`: camada frontal com postes/trave limpos, sem rede duplicada, desenhada por cima do lance depois de jogador, bola e goleiro; z-order oficial: `goal_back -> jogador/bola/goleiro -> goal_front/impact`.
- `assets/generated/cinematic/goal_impact_*.png`: deformação da rede no ponto de impacto da bola.
- `assets/generated/ball_sources/plain_ball_sheet_8frames.png`: sheet da bola gerado por `image_gen`.
- `assets/generated/cinematic/*_idle.png`, `*_run1.png`, `*_dribble.png`, `*_kick.png`, `*_keeper.png`: sprites de posse, chute e empate.
- `src/arena_ai/cinematic_uniforms.py`: define as 9 cores de uniforme, variações de bermuda e mapeamento para seleções.
- `assets/generated/balls3d/*.png`: frames da bola.
- `assets/generated/flags/*.png`: 48 bandeiras em sprite geradas por `image_gen`; a validação falha se alguma seleção esperada não tiver PNG.
- `assets/asset_manifest.json`: manifesto dos assets ativos, globs gerados e allowlist de legados não usados.
- `assets/fonts/Oxanium.ttf`: fonte OFL empacotada para uso direto no Pygame.
- `assets/sounds/runtime_assets/`: sons efetivamente usados pelo jogo. O contrato operacional fica em `src/arena_ai/audio_manifest.py`; a governança, provenance, licenças e hashes ficam em `assets/sounds/audio_manifest.json`.
- `assets/sounds/audio_manifest.json`: manifesto único de áudio aprovado. `assets/sounds/candidates/` é biblioteca de curadoria e nunca deve ser usado direto pelo código.
- Camadas de áudio aprovadas: abertura, base de estádio, air, torcida leve, tensão, chant, ataque, chute, whoosh, rede, explosão de gol, bass hit, roar, reverb, apito inicial/final e stingers da Copa. A política de cada cue fica centralizada no `AudioEngine`: o estádio não some no impacto, o `crowd_attack_rise` vira arco de gol/ataque e a fila da Copa é limpa ao trocar de cena.

## Validação Visual

`make validate` compila o projeto e roda o gate essencial: smoke do modelo, relatório estatístico SOTA/KISS já gerado, manifesto/hash dos dados brutos com sanidade semântica, Monte Carlo fresco, inventário de sprites/som, manifesto de assets, contrato de áudio, layout da tela de confronto, pureza de render e áudio essencial. Ele fica leve para a iteração diária, mas reprova artefatos estatísticos stale. Quando mexer em modelo, pipeline, dados ou scripts de auditoria, rode `make mc-stability && make stats-qa && make validate`.

`make aaa-qa` roda o gate pesado: fonte antiga `oracle_*`, sprite extra no runtime, jogador sem `ORACLE` no peito, jogador sem área de pernas suficiente, escala diferente do padrão visual de pose `192px`, chute sem usar a âncora real do pé, bola duplicada, chute perto demais do goleiro, goleiro fora do gol, bola fora da rede, rede pouco visível, overlay de `GOOOL!` cobrindo a zona da trave/rede, lances sem gol `save/wide`, fade de goleiro/trave, parallax sem scroll acumulado, parallax com faixas idênticas/seam/banda, arquivo de asset órfão fora da allowlist, áudio fora da ordem `kick -> whoosh -> net -> bass -> cheer -> reverb`, sync quantizado de chute/rede, cama de estádio preservada no impacto, tick da Copa consumido sem tocar, reveal antes dos ticks drenarem, render puro, determinismo visual, canais de crowd base/air/light/tension/chant, reação aleatória, duck de narração, camadas de gol, recuperação ordenada de áudio após stutter, uma partida completa de 45 segundos simulados até o placar final e orçamento visual de 60 FPS.

`make visual-qa` gera os frames em `artifacts/visual_qa/current/`, incluindo `contact_sheet.png`, para revisar a posse inicial, corrida/chute, bola na rede, gol invertido e empate final.

Fontes dos sons:

- Mixkit Sound Effects: https://mixkit.co/free-sound-effects/
- Mixkit License: https://mixkit.co/license/
- Manifesto operacional: `src/arena_ai/audio_manifest.py`
- Manifesto de governança/provenance: `assets/sounds/audio_manifest.json`
