# Qualidade, QA E Gates Do Jogo

Este é o documento canônico dos gates de qualidade do Arena AI. Ele consolida o antigo `docs/runtime_quality.md` e a seção de validação do `README.md`.

## Loop Recomendado

```bash
make smoke      # rápido: compile/import/modelo/1 predição
make validate   # gate essencial diário
make audio-qa   # contrato e mixagem de áudio
make cinematic-game-qa          # contrato promovido no runtime real
make cinematic-game-qa-capture  # matriz visual goal/save/wide
make cinematic-game-qa-check    # repete contrato e evidência no runtime atual
make visual-qa  # frames e contact sheet para inspeção
make aaa-qa     # gate pesado de jogo completo
```

Regra central: `draw_*` só desenha estado pronto. Eventos, áudio, Monte Carlo, transições e mutações vivem no `update()`.

---

## Histórico Consolidado

O conteúdo operacional antigo foi consolidado aqui e nos docs canônicos:

- `MODEL.md`: arquitetura do runtime Pygame, separacao `update()`/`draw_*`, gates e performance.
- `ASSETS.md`: governanca de assets, audio, sprites, z-order, licencas e validacao visual/sonora.

Resumo operacional atual:

```bash
make smoke
make validate
make audio-qa
make visual-qa
make aaa-qa
```

Regra central: `draw_*` desenha estado pronto; eventos, audio, Monte Carlo e transicoes vivem no `update()`.

Este arquivo substitui a antiga ponte `docs/runtime_quality.md`.

---

## Validação Visual

`make validate` compila o projeto e roda o gate essencial: smoke do modelo, relatório estatístico SOTA/KISS já gerado, manifesto/hash dos dados brutos com sanidade semântica, Monte Carlo fresco, inventário de sprites/som, manifesto de assets, contrato de áudio, layout da tela de confronto, pureza de render e áudio essencial. Ele fica leve para a iteração diária, mas reprova artefatos estatísticos stale. Quando mexer em modelo, pipeline, dados ou scripts de auditoria, rode `make mc-stability && make stats-qa && make validate`.

`make aaa-qa` combina inventário e integridade dos sprites finais, legibilidade do `ORACLE` nativo, contrato de áudio, UI, determinismo, orçamento headless e a matriz cinematográfica descrita abaixo. No orçamento de render, tempo de CPU do processo é sempre bloqueante; wall-clock é registrado, mas vira bloqueante somente com `ARENA_AI_STRICT_WALL_CLOCK_QA=1`, destinado a um host controlado sem contenção externa. Assim, pausa do scheduler não é rotulada como regressão do jogo. O nome do target é histórico: um passe automatizado não é, sozinho, uma certificação estética "AAA". A aceitação visual do jogo continua exigindo revisão humana no executável.

### Contrato Cinematográfico Ativo

`assets/generated/cinematic/poc7_runtime_contract.json` é o artefato promovido e a autoridade executável do lance. O cabeçalho exige `artifact=arena_cinematic_runtime_contract` e `status=promoted`; o jogo o carrega por `PocSequenceBank` e não reconstrói câmera, trajetória, goleiro ou rede com políticas experimentais. As POCs, geradores e fontes intermediárias que produziram a aprovação não fazem parte do runtime nem do repositório final.

O estado ativo de `cinematic_scene_state` publica `ball_pos`, `ball_prev_pos`, `ball_velocity_px_s`, `ball_ground_pos`, `ball_depth`, `ball_rotation_degrees`, `ball_scale`, `ball_phase` e `raw_shot_progress`. Posição e ângulo usam interpolação contínua; o ângulo cruza `359 -> 0` pelo arco curto. Frames de jogador, goleiro, fase e ordem de profundidade permanecem discretos.

Cada uniforme possui 16 poses GPT Image diretas de corrida, 16 de chute e 8 de parada por direção. Esquerda e direita são artes independentes. O runtime não espelha, não recolore, não faz warp anatômico e não sobrepõe texto: `ORACLE` faz parte dos pixels originais da camisa. O contrato possui oito estágios lógicos de corrida e oito de chute; o runtime os distribui pelos 16 frames autorais e aplica a escala de referência específica de cada uniforme/direção, mantendo uma única dimensão durante todo o lance.

A bola usa os 32 quadros finais promovidos. O corredor pré-contato preserva a posição do contrato e aplica apenas o mínimo deslocamento dirigido necessário para o contorno alpha do uniforme e da rotação corrente não atravessar o jogador; o mesmo contrato vale para os rastros do chute contra a pose atualmente exibida. `make cinematic-game-qa` mede a superfície realmente escalada em 2.430 casos (`30 sequências x 9 uniformes x 9 instantes`) e reprova sobreposição bola/jogador acima de 3%.

O goleiro usa 16 poses GPT Image independentes por direção. A direção do asset é oposta ao sentido do ataque, conforme o contrato promovido, e não é inferida pela posição horizontal da trave. A posição e o apoio no gramado vêm das amostras congeladas; gol sofrido, defesa e chute para fora preservam seus próprios finais.

O gol mantém a geometria autoral `993x497`. A rede ativa possui 70 estados traseiros a 30 Hz por campanha de gol e contato frontal amostrado a 60 Hz até `180 ms`; os estados ficam empacotados em atlas protegidos por hash. O preload começa em segundo plano ao entrar na partida, instala no máximo um asset por frame e mantém a timeline em zero até concluir; durante o lance, o renderer não lê, calcula hash nem decodifica PNG. O runtime aplica o arredondamento temporal declarado no contrato, desenha a região traseira deformada e fecha a oclusão com contato frontal localizado e postes. A ordem é:

```text
rede/trave traseira -> jogador -> bola/goleiro por profundidade -> contato local -> postes/trave frontais
```

### Matriz De Paridade

`make cinematic-game-qa` valida:

- 30 sequências: 18 gols (`2 direções x 3 alturas x 3 medoids`) e 12 lances `save/wide` (`2 direções x 3 alturas`);
- 150 checkpoints rasterizados pelo renderer real, incluindo o estado terminal de cada sequência;
- 2.502 amostras sequenciais pós-impacto, validando cadência, ordem temporal, hashes, limites das regiões de atlas e presença útil da animação de contato;
- oito provas isoladas de materialização no framebuffer para ator, bola, goleiro e composição do gol nos dois sentidos;
- 18 matrizes de progressão que exigem os 16 frames autorais de corrida e chute em cada combinação uniforme/direção;
- 2.430 casos de folga alpha da bola, cobrindo todas as sequências, uniformes e nove instantes entre a liberação e os 133 ms seguintes;
- 58 assets aprovados por hash, carregados de forma assíncrona e disponíveis no cache antes da timeline avançar;
- seis partidas completas de integração por `App.update() -> App.draw() -> flush de áudio`;
- direção e final do goleiro, geometria `993x497`, presença raster da bola/trave, progressão dos frames, ordem das alturas, trajetória, impulso/decaimento da rede, placar no frame de impacto e inventário/fila/execução dos cues;
- cadência traseira da rede a 30 Hz, contato frontal a 60 Hz e ausência de I/O ou decodificação durante o lance;
- identidade promovida, schema, cardinalidade e hashes do contrato e de todas as camadas localizadas da rede.

Os três medoids de cada campanha de gol são selecionados offline a partir de 1.000 planos por perfil. Seus pesos inteiros somam 10.000 em cada combinação direção/altura; o runtime faz uma escolha determinística por seed, minuto, lado e resultado. `save` e `wide` usam um medoid por perfil.

`make cinematic-game-qa-capture` gera 150 PNGs em `artifacts/cinematic_game_qa/current/frames/`, sempre pelo renderer real. O manifesto schema 13 registra fontes runtime, contrato, 179 itens cinematográficos finais (178 arquivos de mídia/metadata mais o contrato), relatório e hashes binário/RGBA dos frames. Seu campo geral `runtime_assets` possui 181 arquivos porque agrega três camadas de parallax e registra o contrato separadamente. `make cinematic-game-qa-check` repete ao vivo a matriz, valida o manifesto e renderiza os 150 frames novamente em diretório temporário para comparação de pixels. O relatório corrente fica em `build/qa/cinematic_game_integration.json`.

O gate automatizado comprova integridade do contrato promovido, coerência estrutural e integração. Ele não transforma gosto visual em verdade objetiva, não certifica compositor/GPU ou áudio físico de todo Mac/Windows e não substitui a validação humana antes do build. A implementação continua sendo `screen-space` 2D com sprites discretos, sem rig, IK ou motion capture.

### Evidência Visual

`make visual-qa` gera os frames em `artifacts/visual_qa/current/`. A `contact_sheet.png` preserva a razão `1280:760`, com a legenda fora da imagem. A captura também produz:

- `cinematic_sequence/`: 12 PNGs entre aproximação, chute, voo, impacto e repouso;
- `cinematic_shot_sequence.png`: strip em quatro colunas sem deformação;
- `cinematic_ball_roll/`: 15 PNGs em progresso estritamente crescente da condução no chão até o início do voo;
- `cinematic_ball_roll_sequence.png`: strip dedicado à rolagem, com 13 quadros até o contato e dois depois dele;
- `cinematic_dribble_flow/`: 17 PNGs que atravessam corrida livre, entrada no lance, condução, domínio, planta, contato, rede e retorno em `p=1,40`;
- `cinematic_dribble_flow_sequence.png`: contact sheet fim a fim da locomoção, incluindo índice do runner, fase/pé do toque, landmark diagnóstico e distância alpha realmente renderizada no metadata;
- `cinematic_dribble_home_60fps.mp4` e `cinematic_dribble_away_60fps.mp4`: evidência temporal a 60 fps em cada sentido, do início da corrida até depois do repouso na rede e da recuperação; contagem, duração e progresso final ficam registrados em `metadata.json`;
- `cinematic_dribble_*_foot_slowmo.mp4`: close estabilizado em câmera lenta para apoio, passada e toque;
- `cinematic_runner_all_frames_{right,left}.png`, `cinematic_kick_all_frames_{right,left}.png` e `cinematic_stop_all_frames_{right,left}.png`: 16 poses de corrida, 16 de chute e 8 de parada para cada um dos 9 uniformes e direções;
- `cinematic_keeper_all_directions.png`: as 16 poses diretas do goleiro em cada direção;
- `cinematic_dribble_motion.csv` e `.json`: pelve, suporte, fase, sola, pé, bola e folga por quadro;
- `cinematic_variants/`: voo, contato e absorção dos seis perfis, alternando mandante/visitante, mais sequências de defesa e trave;
- `cinematic_variants_contact_sheet.png`: 24 quadros em três colunas para revisão humana dos casos críticos;
- `metadata.json`: schema 3, commit, estado limpo/sujo, hash agregado de código/assets, tempos, progresso solicitado, contrato da bola, template real do `GOOOL!`, mutações adversariais e inventário recursivo de todo arquivo produzido;
- três frames com `raw_shot_progress > 1.0`, para provar assentamento pós-impacto.

Os 12 frames são ordenados por `raw_shot_progress`; o quadro estático `04_bola_em_voo.png` é capturado depois de `SHOT_RELEASE_END`. O gate exige deformação e dissipação da rede pelo estado/render canônico, mas não exige o burst decorativo de `cached_goal_impact_burst`.

`ffmpeg` participa apenas da captura dos vídeos de QA; não participa dos sprites, do runtime nem dos gates de inventário. A câmera lenta é amostrada a `150 Hz`, desacelerada em `2,5x` e verificada por `ffprobe` como saída real de `60 fps`, sem repetir artificialmente os frames de uma captura a 60 Hz.

`make visual-qa-check` não regrava nada: ele compara commit, estado limpo/sujo, fingerprint exato das fontes, incluindo todos os `docs/*.md` e `scripts/*.py`, e o hash/tamanho de cada artefato, reprovando ausência, alteração ou órfão. Em seguida, recria a captura em diretório temporário e exige igualdade semântica do metadata e igualdade byte a byte do inventário. Como o metadata atual declara `dirty_worktree`, essa foto é honesta para revisão pré-commit; um commit posterior muda a proveniência e obriga nova captura antes do release.

Limites honestos desta implementação: a física continua em `screen-space` 2D; bola, jogadores e goleiro usam sequências discretas 2D sem esqueleto IK, malha volumétrica ou motion capture. As 16/16/8 poses dos jogadores, 32 vistas da bola e 16 poses do goleiro por direção são diretas, mas ainda podem exibir diferenças autorais pequenas entre quadros. Os gates reprovam floating, foot sliding forte, troca de identidade, fase reversa, contato abstrato, salto fora do envelope autoral, silhueta dupla, oclusão incoerente e regressão de performance no viewport canônico `1280x760`. Eles não certificam todos os viewports, GPUs ou dispositivos físicos Mac/Windows.

Regressões do inventário e dos contratos cinematográficos essenciais reprovam `standard`; a validação completa do contrato promovido pertence a `aaa-qa` e à captura temporal. Depois de mudanças cinematográficas, rode `make validate`, `make cinematic-game-qa-capture`, `make aaa-qa`, `make audio-qa` e `make visual-qa`.

Fontes dos sons:

- Mixkit Sound Effects: https://mixkit.co/free-sound-effects/
- Mixkit License: https://mixkit.co/license/
- Manifesto operacional: `src/arena_ai/audio_manifest.py`
- Manifesto de governança/provenance: `assets/sounds/audio_manifest.json`
