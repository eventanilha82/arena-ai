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

`make aaa-qa` roda o gate pesado: fonte antiga `oracle_*`, sprite extra no runtime, jogador sem `ORACLE` no peito, jogador sem área de pernas suficiente, escala diferente do padrão visual de pose `192px`, chute sem usar a âncora real do pé, bola duplicada, chute perto demais do goleiro, goleiro fora do gol, bola fora da rede, rede pouco visível, overlay de `GOOOL!` cobrindo a zona da trave/rede, lances sem gol `save/wide`, fade de goleiro/trave, parallax sem scroll acumulado, parallax com faixas idênticas/seam/banda, arquivo de asset órfão fora da allowlist, áudio fora da ordem `kick -> whoosh -> net -> bass -> cheer -> reverb`, sync quantizado de chute/rede, cama de estádio preservada no impacto, tick da Copa consumido sem tocar, reveal antes dos ticks drenarem, render puro, determinismo visual, canais de crowd base/air/light/tension/chant, reação aleatória, duck de narração, camadas de gol, recuperação ordenada de áudio após stutter, uma partida completa de 45 segundos simulados até o placar final e orçamento visual de 60 FPS.

`make visual-qa` gera os frames em `artifacts/visual_qa/current/`, incluindo `contact_sheet.png`, para revisar a posse inicial, corrida/chute, bola na rede, gol invertido e empate final.

Fontes dos sons:

- Mixkit Sound Effects: https://mixkit.co/free-sound-effects/
- Mixkit License: https://mixkit.co/license/
- Manifesto operacional: `src/arena_ai/audio_manifest.py`
- Manifesto de governança/provenance: `assets/sounds/audio_manifest.json`
