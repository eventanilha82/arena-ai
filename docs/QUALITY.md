# Qualidade, QA E Gates Do Jogo

Este é o documento canônico dos gates de qualidade do Arena AI. Ele consolida o antigo `docs/runtime_quality.md` e a seção de validação do `README.md`.

## Loop Recomendado

```bash
make smoke      # rápido: compile/import/modelo/1 predição
make validate   # gate essencial diário
make audio-qa   # contrato e mixagem de áudio
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

`make aaa-qa` roda o gate pesado: fonte antiga `oracle_*`, sprite extra no runtime, jogador sem `ORACLE` no peito, jogador sem área de pernas suficiente, escala diferente do padrão visual de pose `192px`, chute sem usar a âncora real do pé, bola duplicada, chute perto demais do goleiro, goleiro fora do gol, bola fora da rede, rede pouco visível, overlay de `GOOOL!` cobrindo a zona da trave/rede, lances sem gol `save/wide`, fade de goleiro/trave, parallax sem scroll acumulado, parallax com faixas idênticas/seam/banda, arquivo de asset órfão fora da allowlist, áudio fora da ordem `kick -> whoosh -> net -> bass -> cheer -> reverb`, sync quantizado de chute/rede, cama de estádio preservada no impacto, tick da Copa consumido sem tocar, reveal antes dos ticks drenarem, render puro, determinismo visual, canais de crowd base/air/light/tension/chant, reação aleatória, duck de narração, camadas de gol, recuperação ordenada de áudio após stutter, uma partida completa de 45 segundos simulados até o placar final e orçamento CPU/headless do loop de 60 FPS.

### Contrato Cinematográfico Da Bola

O runtime de `cinematic_scene_state` deve publicar, em todos os lances ativos, `ball_pos`, `ball_prev_pos`, `ball_velocity_px_s`, `ball_ground_pos`, `ball_depth`, `ball_rotation_degrees`, `ball_scale` e `raw_shot_progress`. Os vetores são tuplas 2D finitas, `ball_depth` fica em `0..1` e o frame anterior representa a posição de um frame de 60 Hz antes.

Os tamanhos canônicos são `CINEMATIC_BALL_SIZE=38` e `CINEMATIC_SHOT_BALL_SIZE=32`. A proporção é calculada sobre o alpha visível de cada sprite, não sobre o canvas: a bola de condução precisa ocupar entre `14%` e `18%` da altura visível do jogador.

O contato centro da bola -> dedo do pé usa o raio alpha visível máximo mais `3px` de tolerância de drible. Assim o canvas transparente não cria um falso gap nem mascara separação perceptível.

No impacto, a legibilidade da bola de `32px` é medida por densidade de highlights sobre a área alpha visível do material canônico, com retenção mínima de `72%` da própria referência e piso de `18%`. Não há limiar absoluto herdado das bolas antigas de `48/50px`.

Na sequência, a visibilidade é a contribuição real do draw com/sem bola dividida pela área alpha do material na escala corrente. A perda permitida é proporcional à área alpha da bola realmente coberta pelo goleiro, usando o quadrado da fração ainda visível para acomodar highlights não uniformes; um único pixel de interseção mantém praticamente toda a exigência. Sem oclusão, a densidade não pode cair abaixo de `16%` nem de `30%` do melhor quadro comparável.

O z-order acompanha `ball_depth`: rede traseira primeiro; bola profunda, reação e plano frontal antes do goleiro na boca do gol; bola rasa depois do goleiro. Assim luvas/corpo ocultam fisicamente a bola no contato sem anéis artificiais.

O mergulho do goleiro é orientado pelo lado real do alvo, não pelo sentido geral do ataque. A pose estendida permanece ativa até o contato; a recuperação só começa depois da colisão. Na defesa, a bola pode ficar parcialmente ocluída pelas luvas no contato, mas o recuo posterior precisa voltar a ser legível sem depender do texto do HUD.

Lances sem gol não dependem de flash, ring ou brilho de payoff. Em três frames pré/contato/pós, a defesa é provada por âncora da luva, frame/action, desaceleração/recuo e oclusão real pelo goleiro; o `wide`, pela menor distância alpha-normalizada da trave e continuidade da saída. Material, sombra e profundidade continuam validados em ambos.

O gate denso de `make aaa-qa` amostra a 240 Hz gol local, gol visitante, defesa e chute para fora. No `wide`, os contratos numéricos seguem validados após a saída, enquanto os gates perceptuais acompanham a bola até ela deixar o viewport por dois diâmetros. Ele reprova:

- posição incompatível com a velocidade integrada;
- descontinuidade grosseira de velocidade em `SHOT_KICK_AT`, `SHOT_NET_AT` ou no contato físico `SHOT_NET_VISUAL_CONTACT_AT`;
- reversão horizontal ou de profundidade antes da colisão;
- mudança de escala acima de `1px` por frame de 60 Hz;
- `ball_prev_pos` diferente da posição do frame anterior;
- sombra fora do plano inferior do gramado ou acima da bola;
- spin acima de `20 graus` por amostra de 240 Hz ou `80 graus` por frame de 60 Hz, e variação acima de `14 graus` entre passos angulares consecutivos;
- ganho de velocidade dentro da rede acima de `3%` da velocidade de entrada ou recuperação acima da tolerância numérica acumulada de `0,5%` em relação ao menor valor já observado, varrendo 202 combinações de seed e direção e os seis perfis de chute;
- bola ainda em movimento após o limite dinâmico de dissipação, cujo teto é `SHOT_NET_SETTLE_PROGRESS=0.38`; a evidência final em `p=1.382` já deve estar em repouso.

A trajetória pós-impacto é percorrida por comprimento de arco com desaceleração até zero. A duração nasce da distância real dentro da rede e da velocidade de entrada; não existe mais um tempo fixo que obrigue chutes profundos a ganhar energia. O mesmo sweep exige diferença mensurável entre os arcos de `rasteiro forte`, `meia altura`, `alto firme` e `angulo seco`.

O orçamento de frame roda com drivers SDL `dummy` e mede o custo CPU de `event pump + update + draw/present + audio flush`. Os limites são `p95 <= 14ms`, `p99 <= 16,67ms`, máximo `<= 33,3ms` e ausência de cauda sustentada acima de dois frames. Ele não certifica compositor/GPU, dispositivo físico de áudio nem FPS sustentado em todo hardware Mac/Windows; os pacotes ainda passam pelos checks de build e conteúdo. Os contratos existentes de ordem de áudio, recuperação após stutter, pureza de render e determinismo continuam obrigatórios.

### Evidência Visual

`make visual-qa` gera os frames em `artifacts/visual_qa/current/`. A `contact_sheet.png` preserva a razão `1280:760`, com a legenda fora da imagem. A captura também produz:

- `cinematic_sequence/`: 12 PNGs entre aproximação, chute, voo, impacto e repouso;
- `cinematic_shot_sequence.png`: strip em quatro colunas sem deformação;
- `cinematic_variants/`: voo, contato e absorção dos seis perfis, alternando mandante/visitante, mais sequências de defesa e trave;
- `cinematic_variants_contact_sheet.png`: 24 quadros em três colunas para revisão humana dos casos críticos;
- `metadata.json`: hashes, tempos, progresso solicitado e o contrato completo da bola em cada frame;
- três frames com `raw_shot_progress > 1.0`, para provar assentamento pós-impacto.

Os 12 frames são ordenados por `raw_shot_progress`; o quadro estático `04_bola_em_voo.png` é capturado depois de `SHOT_RELEASE_END`. O gate exige deformação e dissipação da rede pelo estado/render canônico, mas não exige o burst decorativo de `cached_goal_impact_burst`.

`ffmpeg` não participa do gate e nunca é requisito. Um vídeo pode ser montado externamente a partir dos PNGs, sem alterar a evidência canônica.

Limites honestos desta implementação: a física continua em `screen-space` 2D, a bola usa oito vistas-base interpoladas e o goleiro possui quatro poses por ação. Os gates eliminam aceleração artificial, saltos, orientação errada, oclusão incoerente e regressões de performance, mas não transformam os sprites em captura volumétrica, simulação de malha 3D ou motion capture. A evidência canônica é produzida em `1280x760`; outros tamanhos continuam cobertos pelos contratos de layout, não por uma segunda captura cinematográfica completa.

Qualquer regressão do contrato reprova `standard`, `aaa` e a captura temporal pelo tamanho incorreto ou pela lista explícita de campos ausentes. Depois de mudanças cinematográficas, rode `make validate`, `make aaa-qa`, `make audio-qa` e `make visual-qa`.

Fontes dos sons:

- Mixkit Sound Effects: https://mixkit.co/free-sound-effects/
- Mixkit License: https://mixkit.co/license/
- Manifesto operacional: `src/arena_ai/audio_manifest.py`
- Manifesto de governança/provenance: `assets/sounds/audio_manifest.json`
