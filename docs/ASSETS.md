# Assets, Áudio, Fontes E Licenças

Este é o documento canônico de assets do Arena AI. Ele consolida as antigas páginas avulsas de áudio, curadoria, imagens FIFA externas e fontes. O manifesto único de áudio aprovado fica em `assets/sounds/audio_manifest.json`.

O objetivo é deixar claro o que entra no runtime, o que é candidato, o que precisa de licença confirmada e qual é o processo para promover assets sem bagunçar o jogo.

## Regra Central

O runtime só deve carregar assets aprovados e manifestados.

Para áudio:

```text
runtime_assets/ = aprovado e carregado pelo jogo
candidates/     = biblioteca de curadoria
rejected_assets/ = descartado ou substituído
```

Nenhum código deve carregar `assets/sounds/candidates/` diretamente.

Um som só entra no jogo depois de:

1. passar por curadoria A/B;
2. ser promovido para `assets/sounds/runtime_assets/`;
3. receber nome estável;
4. ser registrado em `assets/sounds/audio_manifest.json`;
5. ser listado no contrato operacional `src/arena_ai/audio_manifest.py`;
6. passar em `make audio-qa`.

Se o status do som for `Acervo do projeto`, o próprio `assets/sounds/audio_manifest.json` precisa trazer
o bloco `project_archive_receipt` com autor, origem, data, licença/status, SHA-256 do arquivo fonte
promovido e SHA-256 do arquivo runtime. O QA reprova qualquer asset de acervo sem receipt completo
ou com hash stale.

## Manifestos

Fontes de verdade:

- `src/arena_ai/audio_manifest.py`: contrato operacional do áudio.
- `assets/sounds/audio_manifest.json`: governança, provenance, licenças, hashes e receipts dos sons aprovados.
- `assets/sounds/downloaded_audio_manifest.csv`: inventário técnico dos candidatos baixados.
- `assets/asset_manifest.json`: manifesto geral de assets ativos.

`assets/sounds/downloaded_audio_manifest.csv` é o inventário bruto dos downloads manuais. Ele guarda o detalhe técnico que não deve ser duplicado em texto narrativo: caminho, duração, camada, origem e status de cada candidato.

No `assets/asset_manifest.json`, `generated_runtime_globs` deve apontar apenas para arquivos finais usados pelo jogo. Sheets brutos de `image_gen`, fontes de recorte, flag sheets e parallax sources ficam em `generated_source_globs`; candidatos de áudio ficam em `curation_asset_globs`. O empacotador reprova qualquer glob de runtime que aponte para fontes brutas ou diretórios de curadoria.

O contrato de áudio define:

- filename;
- bus;
- papel funcional;
- duração mínima e máxima;
- limite de transiente para sons de impacto;
- buses obrigatórios;
- sequência oficial do gol;
- política única de cue em `AudioCuePolicy`;
- marcadores sonoros do Monte Carlo.

Sequência oficial do gol:

```text
kick -> whoosh -> net -> bass -> cheer -> reverb
```

Marcadores do Monte Carlo:

```text
25%, 50%, 75%, 100%
```

## Buses De Áudio

O `AudioEngine` usa buses:

| Bus | Função |
| --- | --- |
| `ambience` | cama de estádio e ar contínuo |
| `crowd` | torcida leve, tensão, chants e reações |
| `ball` | chute, whoosh e rede |
| `goal` | bass hit, explosão, roar e reverb |
| `ui` | apitos, ticks e stingers de interface |
| `music` | abertura/menu |

`draw()` deve ser puro. Eventos de áudio são disparados no `update()` e em transições de estado, com cooldown, ducking, pan, randomização, fade e sound-bags que evitam repetição imediata. A regra de cada cue fica centralizada em `AudioCuePolicy`: o `App` apenas arma/enfileira eventos, e o `AudioEngine` decide suppressão, boost, foco de impacto e ducking.

O ducking não pode matar o estádio. Em chute/gol, a cama `stadium_base + air` permanece audível; o corte leve acontece nas camadas de música/tensão/chant para abrir espaço para `kick`, `net`, `bass` e explosão. O `crowd_attack_rise` deixou de ser reação randômica e virou arco de gol/ataque, enquanto reações aleatórias usam takes curtos.

## Timeline Sonora

| Momento | Áudio |
| --- | --- |
| Abertura/menu | `opening_theme` + base de estádio |
| Seleção | estádio vivo baixo + UI sutil |
| Início do confronto | apito inicial |
| Ataque perigoso | tensão e crowd rise progressivo |
| Chute | `kick_grass_*` real, cortado no transiente |
| Bola | whoosh leve com pan discreto |
| Rede | impacto real de rede no tempo zero |
| Gol | bass hit + explosão + roar + cauda de arena |
| Fim | apito final |
| Monte Carlo | ticks 25/50/75/100 + stinger de reveal, sem som de gol |

No Monte Carlo, ticks e reveal são fila exclusiva da tela de Copa. Ao sair da Copa para menu/seleção/confronto, a fila é limpa para não tocar tick ou stinger em outra cena.

## Runtime Audio Assets

Somente os arquivos abaixo são carregados em tempo de execução.

Quando a origem aparece como `promoted_sources/...`, o take aprovado fica preservado em `assets/sounds/candidates/promoted_sources/` como recibo da curadoria antes da cópia normalizada para `runtime_assets/`.
Quando a licença/status é `Acervo do projeto`, o receipt obrigatório fica no campo
`project_archive_receipt` de `assets/sounds/audio_manifest.json`; ele é mais forte que a linha da
tabela porque valida hash fonte/runtime.

| Runtime asset | Bus | Papel | Origem | Licença/status | Uso |
| --- | --- | --- | --- | --- | --- |
| `stadium_base_loop.mp3` | ambience | stadium_base | `promoted_sources/stadium_crowd_loop.mp3` | Acervo do projeto | cama principal de estádio |
| `stadium_air_loop.wav` | ambience | stadium_air | Freesound Sandermotions pack 27815, `494350` | Creative Commons 0 | ar longo de arquibancada |
| `crowd_light_loop.mp3` | crowd | light_crowd | `promoted_sources/light_crowd_mixkit.mp3` | Mixkit Free License | torcida leve |
| `crowd_tension_loop.mp3` | crowd | crowd_tension | `promoted_sources/crowd_tension_mixkit.mp3` | Mixkit Free License | tensão de ataque |
| `crowd_attack_rise.wav` | crowd | crowd_attack | Freesound Sandermotions pack 27815, `494362` | Creative Commons 0 | reação humana de ataque |
| `crowd_attack_short.wav` | crowd | crowd_attack_short | Freesound Sandermotions pack 27815, `494361` | Creative Commons 0 | reação curta |
| `crowd_chant_loop.mp3` | crowd | chant | `promoted_sources/stadium_chant_mixkit.mp3` | Mixkit Free License | chant com fade |
| `goal_roar_main.mp3` | goal | goal_roar | `promoted_sources/crowd_goal_roar.mp3` | Acervo do projeto | roar principal no gol |
| `goal_roar_pixabay_01.wav` | goal | goal_roar | Pixabay `mrmark81-stadium-roar-concert-471943.mp3` | Pixabay Content License confirmado em 2026-05-18 | roar forte randomizado no gol |
| `goal_roar_pixabay_02.wav` | goal | goal_roar | Pixabay `vishiv-crowd-cheering-in-stadium-435357.mp3` | Pixabay Content License confirmado em 2026-05-18 | cheer forte randomizado no gol |
| `goal_crowd_cc0.wav` | goal | goal_crowd_tail | Freesound Sandermotions pack 27815, `494352` | Creative Commons 0 | cauda alternativa de crowd |
| `goal_explosion_01.wav` | goal | goal_explosion | `promoted_sources/goal_explosion_mixkit_a.mp3` | Mixkit Free License | explosão de gol A |
| `goal_explosion_02.wav` | goal | goal_explosion | Mixkit `crowd_at_stadium_2111.mp3` | Mixkit Free License | explosão de gol B |
| `goal_explosion_03.wav` | goal | goal_explosion_alt | Mixkit `stadium_joy_shouting_crowd_3022.mp3` | Mixkit Free License | explosão de gol C |
| `bass_hit_01.wav` | goal | bass_hit | `promoted_sources/bass_hit_mixkit.mp3` | Mixkit Free License | impacto emocional com ataque cortado em WAV |
| `stadium_reverb_tail.mp3` | goal | reverb_tail | `promoted_sources/stadium_reverb_tail_mixkit.mp3` | Mixkit Free License | cauda/reverb |
| `kick_grass_01.wav` | ball | kick | Freesound `555042__bittermelonheart__soccer-ball-kick.wav`, https://freesound.org/s/555042/ | Creative Commons 0 confirmado em 2026-05-17 | chute principal |
| `kick_grass_02.wav` | ball | kick_alt | Mixkit `soccer_ball_kick_2108.mp3` | Mixkit Free License | variação de chute |
| `kick_grass_03.wav` | ball | kick_alt | Mixkit `hitting_soccer_ball_2112.mp3` | Mixkit Free License | variação seca de chute |
| `kick_grass_04.wav` | ball | kick_alt | Mixkit `sports_ball_hit_2082.mp3` | Mixkit Free License | impacto curto de bola |
| `ball_whoosh_01.wav` | ball | whoosh | Mixkit `fast_sweep_transition_174.mp3` | Mixkit Free License | whoosh da bola |
| `ball_whoosh_02.wav` | ball | whoosh_alt | Mixkit `cinematic_trailer_riser_790.mp3` | Mixkit Free License | whoosh alternativo |
| `net_ripple_01.wav` | ball | net | Pixabay `forza1903-a-football-hits-the-net-goal-313216.mp3`, https://pixabay.com/de/sound-effects/film-spezialeffekte-a-football-hits-the-net-goal-313216/ | Pixabay Content License confirmado em 2026-05-18 | bola na rede |
| `net_ripple_02.wav` | ball | net_alt | Mixkit `basketball_ball_hitting_net_2084.mp3` | Mixkit Free License | rede alternativa curta |
| `crowd_reaction_01.wav` | crowd | crowd_reaction | Mixkit `crowd_yelling_at_stadium_2097.mp3` | Mixkit Free License | reação de defesa/pressão |
| `crowd_reaction_02.wav` | crowd | crowd_reaction | Freesound Sandermotions pack 27815, `494357` | Creative Commons 0 conforme `_readme_and_license.txt` | reação adicional de ambiente |
| `crowd_near_miss_01.wav` | crowd | near_miss_reaction | Freesound Sandermotions pack 27815, `494359` | Creative Commons 0 conforme `_readme_and_license.txt` | reação em chance perdida |
| `ui_chime_01.wav` | ui | ui_chime | `promoted_sources/analysis_chime.wav` | Acervo do projeto | UI/chime |
| `cup_progress_tick_01.wav` | ui | cup_tick | trecho curto de `analysis_chime.wav` | Acervo do projeto | tick Monte Carlo |
| `cup_reveal_stinger.wav` | ui | cup_reveal | Mixkit `movie_trailer_epic_impact_2908.mp3` | Mixkit Free License | reveal da Copa com ataque cortado em WAV |
| `opening_theme.mp3` | music | opening_theme | Pixabay `bombinsound-football-football-soccer-game-music-15-second-490555.mp3`, https://pixabay.com/sound-effects/musical-football-football-soccer-game-music-15-second-490555/ | Pixabay Content License confirmado em 2026-05-18 | abertura/menu |
| `whistle_start_01.wav` | ui | whistle_start | Freesound `538422__rosa-orenes256__referee-whistle-sound.wav`, https://freesound.org/s/538422/ | Creative Commons 0 confirmado em 2026-05-17 | apito inicial |
| `whistle_final_01.wav` | ui | whistle_final | Freesound `218318__splicesound__referee-whistle-blow-gymnasium.wav`, https://freesound.org/s/218318/ | Creative Commons 0 confirmado em 2026-05-17 | apito final |

Regras técnicas:

- Chute, rede, whoosh, apitos e tick são cortados para começar no impacto.
- Não usar `start_ms` frágil em arquivo longo para evento de impacto.
- Loops longos ficam em buses com crossfade.
- Reações curtas entram com cooldown e randomização.
- Monte Carlo usa ticks/stinger, nunca sons de gol.
- Ticks 25/50/75/100 do Monte Carlo entram em fila; o reveal espera a fila drenar para não marcar áudio como tocado sem som real.
- A timeline do gol não despeja todos os cues quando há stutter: `kick`, `whoosh`, `net`, `bass`, `cheer` e `reverb` recuperam em ordem nos frames seguintes.

## Biblioteca Manual Do Usuário

Origem local recebida:

```text
/Users/eventanilha/Downloads/audio-game
```

Os arquivos foram copiados para `assets/sounds/candidates/` sem apagar os originais.

Resumo:

| Camada | Arquivos | Uso provável |
| --- | ---: | --- |
| `ball_dribble` | 1 | drible/bola rolando na grama |
| `chant` | 2 | torcida contínua ou pós-gol |
| `crowd_reaction` | 2 | ataque perigoso, aplauso, reação curta |
| `goal_roar` | 3 | explosão/cauda de gol; dois roars Pixabay viraram runtime randomizado |
| `kick` | 1 | chute seco |
| `license` | 1 | documentação de licença |
| `net_goal` | 1 | bola na rede |
| `opening_music` | 1 | abertura/splash |
| `stadium_ambience` | 16 | ambiente base, air, loops de estádio |
| `whistle` | 3 | apito |

Observações:

- O pack Freesound `27815__sandermotions__soccer-match-stadium-sounds.zip` foi extraído em `assets/sounds/candidates/freesound/sandermotions_soccer_match_stadium_sounds_pack_27815/`.
- O `_readme_and_license.txt` desse pack indica Creative Commons 0.
- Arquivos Freesound avulsos preservam o ID no nome; confirmar a licença individual antes de promover.
- Não foi detectado padrão claro de ZapSplat nessa leva.

## Freesound Sandermotions Pack 27815

Pack:

```text
Soccer match stadium sounds
Autor: Sandermotions
URL: https://freesound.org/people/Sandermotions/packs/27815/
Licença: Creative Commons 0
Licença URL: http://creativecommons.org/publicdomain/zero/1.0/
```

Arquivos documentados no pack:

| ID | Arquivo | URL | Licença |
| --- | --- | --- | --- |
| `494362` | soccer-stadium-oehh | https://freesound.org/s/494362/ | Creative Commons 0 |
| `494361` | soccer-stadium-oehh-2 | https://freesound.org/s/494361/ | Creative Commons 0 |
| `494360` | soccer-stadium-10 | https://freesound.org/s/494360/ | Creative Commons 0 |
| `494359` | soccer-stadium-booohh | https://freesound.org/s/494359/ | Creative Commons 0 |
| `494358` | soccer-stadium-08 | https://freesound.org/s/494358/ | Creative Commons 0 |
| `494357` | soccer-stadium-09 | https://freesound.org/s/494357/ | Creative Commons 0 |
| `494356` | soccer-stadium-04 | https://freesound.org/s/494356/ | Creative Commons 0 |
| `494355` | soccer-stadium-05 | https://freesound.org/s/494355/ | Creative Commons 0 |
| `494354` | soccer-stadium-06 | https://freesound.org/s/494354/ | Creative Commons 0 |
| `494353` | soccer-stadium-07 | https://freesound.org/s/494353/ | Creative Commons 0 |
| `494352` | goal | https://freesound.org/s/494352/ | Creative Commons 0 |
| `494351` | soccer-stadium-01 | https://freesound.org/s/494351/ | Creative Commons 0 |
| `494350` | soccer-stadium-02 | https://freesound.org/s/494350/ | Creative Commons 0 |
| `494349` | soccer-stadium-03 | https://freesound.org/s/494349/ | Creative Commons 0 |

## Mixkit

Fonte:

- https://mixkit.co/free-sound-effects/
- https://mixkit.co/license/

Status:

- Uso permitido em projetos pessoais e comerciais, sem atribuição obrigatória conforme licença Mixkit Free License.
- Conferir a página oficial antes de publicar build externo.

Candidatos baixados:

| Arquivo | Origem direta | Camada candidata | Observação |
| --- | --- | --- | --- |
| `stadium_joy_shouting_crowd_3022.mp3` | https://assets.mixkit.co/active_storage/sfx/3022/3022-preview.mp3 | gol/abertura | crowd forte |
| `crowd_yelling_at_stadium_2097.mp3` | https://assets.mixkit.co/active_storage/sfx/2097/2097-preview.mp3 | ambiente base | loop longo de jogo |
| `stadium_chaotic_applause_drums_chants_363.mp3` | https://assets.mixkit.co/active_storage/sfx/363/363-preview.mp3 | chant/ataque | usar baixo no mix |
| `crowd_at_stadium_2111.mp3` | https://assets.mixkit.co/active_storage/sfx/2111/2111-preview.mp3 | reação curta | ataque ou gol |
| `crowd_stadium_chant_2110.mp3` | https://assets.mixkit.co/active_storage/sfx/2110/2110-preview.mp3 | chant | pós-gol |
| `ambient_sports_crowd_2099.mp3` | https://assets.mixkit.co/active_storage/sfx/2099/2099-preview.mp3 | textura curta | reação discreta |
| `soccer_ball_kick_2108.mp3` | https://assets.mixkit.co/active_storage/sfx/2108/2108-preview.mp3 | chute | variação real |
| `hitting_soccer_ball_2112.mp3` | https://assets.mixkit.co/active_storage/sfx/2112/2112-preview.mp3 | chute | mais corpo/cauda |
| `sports_ball_hit_2082.mp3` | https://assets.mixkit.co/active_storage/sfx/2082/2082-preview.mp3 | chute/impacto | variação |
| `ball_bouncing_ground_2077.mp3` | https://assets.mixkit.co/active_storage/sfx/2077/2077-preview.mp3 | bola | micro-layer de drible |
| `basketball_ball_hitting_net_2084.mp3` | https://assets.mixkit.co/active_storage/sfx/2084/2084-preview.mp3 | rede | usar baixo se aproveitado |
| `short_bass_hit_2299.mp3` | https://assets.mixkit.co/active_storage/sfx/2299/2299-preview.mp3 | bass hit | impacto seco |
| `knocking_sub_bass_2300.mp3` | https://assets.mixkit.co/active_storage/sfx/2300/2300-preview.mp3 | sub impact | mais longo |
| `pulsating_bass_transition_2295.mp3` | https://assets.mixkit.co/active_storage/sfx/2295/2295-preview.mp3 | tensão/abertura | build-up |
| `movie_trailer_epic_impact_2908.mp3` | https://assets.mixkit.co/active_storage/sfx/2908/2908-preview.mp3 | abertura/gol | cinemático |
| `cinematic_trailer_riser_790.mp3` | https://assets.mixkit.co/active_storage/sfx/790/790-preview.mp3 | riser | antes do chute |
| `reverse_cinematic_impact_trailer_784.mp3` | https://assets.mixkit.co/active_storage/sfx/784/784-preview.mp3 | abertura | dramático |
| `fast_sweep_transition_174.mp3` | https://assets.mixkit.co/active_storage/sfx/174/174-preview.mp3 | UI/transição | sweep curto |

## Pixabay

Fonte:

- https://pixabay.com/sound-effects/
- https://pixabay.com/service/license-summary/

Status:

- Normalmente permite uso comercial sem atribuição obrigatória.
- Conferir a página original antes de build público.
- Alguns assets já foram baixados manualmente pelo usuário e catalogados como candidatos.

Arquivos candidatos catalogados:

| Arquivo | Camada | Duração | Status |
| --- | --- | ---: | --- |
| `freesound_community-soccer-fans-vocals-field-recording-with-zoom-h2n-25401.mp3` | chant | 18.41s | confirmar página |
| `djartmusic-powerful-stomps-claps-cheering-sport-rhythmic-applause-317290.mp3` | crowd_reaction | 7.78s | confirmar página |
| `mrmark81-stadium-roar-concert-471943.mp3` | goal_roar | 26.04s | promovido para `goal_roar_pixabay_01.wav`; Pixabay Content License confirmado em 2026-05-18 |
| `vishiv-crowd-cheering-in-stadium-435357.mp3` | goal_roar | 30.02s | promovido para `goal_roar_pixabay_02.wav`; Pixabay Content License confirmado em 2026-05-18 |
| `forza1903-a-football-hits-the-net-goal-313216.mp3` | net_goal | 3.08s | runtime; Pixabay Content License confirmado em 2026-05-18 |
| `bombinsound-football-football-soccer-game-music-15-second-490555.mp3` | opening_music | 15.05s | runtime; Pixabay Content License confirmado em 2026-05-18 |
| `freesound_community-soccer-stadium-10-6709.mp3` | stadium_ambience | 156.36s | confirmar página |
| `framptones-referee-whistle-coach-whistle-sports-whistle-291815.mp3` | whistle | 16.51s | confirmar página |

## Freesound

Fonte:

- https://freesound.org/

Status:

- Download de arquivo original costuma exigir login.
- Via API, original exige OAuth2.
- Preferir CC0 ou Attribution; evitar NonCommercial se o build puder ser comercial.

Arquivos candidatos avulsos catalogados:

| Arquivo | Camada | Duração | Status |
| --- | --- | ---: | --- |
| `788268__amsaenz03__soccer-ball-kick-dribble-in-grass.wav` | ball_dribble | 43.64s | confirmar licença |
| `417783__garuda1982__soccer-fans-vocals-field-recording-with-zoom-h2n.wav` | chant | 18.42s | confirmar licença |
| `268932__icmusic__crowd-reaction-at-ice-hockey-match.wav` | crowd_reaction | 225.71s | confirmar licença |
| `555042__bittermelonheart__soccer-ball-kick.wav` | kick | 0.74s | runtime, Creative Commons 0 confirmado em 2026-05-17 |
| `274516__stomachache__stadium-crowd.wav` | stadium_ambience | 8.19s | confirmar licença |
| `494360__sandermotions__soccer-stadium-10.wav` | stadium_ambience | 156.30s | CC0 no pack |
| `218318__splicesound__referee-whistle-blow-gymnasium.wav` | whistle | 3.33s | runtime, Creative Commons 0 confirmado em 2026-05-17 |
| `538422__rosa-orenes256__referee-whistle-sound.wav` | whistle | 0.54s | runtime, Creative Commons 0 confirmado em 2026-05-17 |

## ZapSplat E Sonniss

ZapSplat:

- https://www.zapsplat.com/
- geralmente exige login;
- plano gratuito costuma exigir atribuição;
- Gold remove atribuição e libera formatos melhores.

Sonniss/GameAudioGDC:

- https://sonniss.com/gameaudiogdc
- bundles grandes, em GB;
- usar como biblioteca externa local, não como asset bruto do repo.

Fluxo recomendado:

1. baixar fora do repo;
2. extrair em biblioteca local;
3. copiar apenas takes selecionados para `assets/sounds/candidates/sonniss/`;
4. registrar pacote, arquivo original e licença;
5. promover só após A/B.

## Curadoria Recomendada

### Abertura

Prioridade:

1. `reverse_cinematic_impact_trailer_784.mp3` em volume baixo com fade-in.
2. `stadium_joy_shouting_crowd_3022.mp3` como estádio cheio.
3. `movie_trailer_epic_impact_2908.mp3` apenas no botão de entrada ou splash.

Evitar riser contínuo alto.

### Jogo / Ambiente

Mix sugerido:

```text
base longo
+ air leve
+ chant baixo com fade lento
+ reações curtas randômicas
```

Prioridade:

1. `crowd_yelling_at_stadium_2097.mp3`;
2. `stadium_chaotic_applause_drums_chants_363.mp3`;
3. `crowd_at_stadium_2111.mp3`;
4. `ambient_sports_crowd_2099.mp3`.

### Ataque Perigoso

Mix sugerido:

```text
crowd_tension
+ riser curto
+ bass muito baixo
+ duck leve no ambiente base
```

Prioridade:

1. `pulsating_bass_transition_2295.mp3`;
2. `cinematic_trailer_riser_790.mp3`;
3. `crowd_at_stadium_2111.mp3`.

### Chute / Bola

Prioridade:

1. `soccer_ball_kick_2108.mp3`;
2. `hitting_soccer_ball_2112.mp3`;
3. `sports_ball_hit_2082.mp3`.

Evitar tocar todos no mesmo frame sem contrato de volume. O runtime atual usa matriz validada: chute/whoosh/rede + bass + roar randomizado + explosão randomizada + tail Sandermotions + swell de arquibancada + reverb.

### Gol / Rede

Mix sugerido:

```text
kick
+ whoosh
+ net
+ bass hit
+ crowd explosion
+ roar humano
+ chant posterior
+ reverb tail
```

Prioridade:

1. `goal_explosion_mixkit_a.mp3` como transiente de impacto;
2. `mrmark81-stadium-roar-concert-471943.mp3` e `vishiv-crowd-cheering-in-stadium-435357.mp3` como roars alternados;
3. Sandermotions pack 27815 para cauda humana e reações de ambiente;
4. `crowd_stadium_chant_2110.mp3` e cama de estádio para sustentação posterior.

## Imagens FIFA Externas

Baixadas em `2026-05-11` a partir de URLs fornecidas pelo usuário para composição local das telas da Copa 2026.

Essas imagens são externas, não sprites gerados. Confirmar direitos de uso antes de redistribuição pública.

| Asset | Origem |
| --- | --- |
| `fifa_maple.jpg` | https://digitalhub.fifa.com/transform/36208906-a2c3-4abb-bf22-7e8a4fa0aefc/Maple?&io=transform:fill,aspectratio:16x9,width:1366&quality=75 |
| `fifa_zayu.jpg` | https://digitalhub.fifa.com/transform/a2bd0cc0-5f5a-4aa2-bd67-6a251a955826/Zayu?&io=transform:fill,aspectratio:16x9,width:1366&quality=75 |
| `fifa_clutch.jpg` | https://digitalhub.fifa.com/transform/7f6547c8-d6f4-4b56-98e9-a425be706d13/Clutch?&io=transform:fill,aspectratio:16x9,width:1366&quality=75 |
| `fifa_club_world_cup_final_2025.jpg` | https://digitalhub.fifa.com/transform/1ce727ba-aa20-4c3c-8867-36b3de424e67/Chelsea-FC-v-Paris-Saint-Germain-Final-FIFA-Club-World-Cup-2025?&io=transform:fill,aspectratio:16x9,width:1920&quality=75 |
| `fifa_mexico_opening_ceremony.jpg` | https://digitalhub.fifa.com/transform/f6dbd3cd-ce55-4a88-b398-76a6e6bd4b2a/Mexico-Opening-Ceremony-graphic?&io=transform:fill,height:910,width:1536&quality=75 |
| `fifa_mexico_opening_ceremony_clean.png` | edição `image_gen` derivada de `fifa_mexico_opening_ceremony.jpg`; removeu apenas o texto embutido "MEXICO OPENING CEREMONY" para uso como fundo no jogo |
| `fifa_detail_image_03.jpg` | https://digitalhub.fifa.com/transform/ac004beb-13cb-4cb9-923b-dc5f030eb34f/Detail-Image-03_3200x1800?&io=transform:fill,aspectratio:16x9,width:1366&quality=75 |
| `fwc26_ecomm_photo_update_b.jpg` | convertido de https://cdn.prod.website-files.com/689fd0a66c26ce8fe1446c25/69d95f4c71d00684ba1d8e76_FWC26_Ecomm_Photo_Update_B_1000x720.webp |

Regra:

- usar como apoio visual local;
- não tratar como sprite gerado pelo projeto;
- antes de release público, validar direitos de redistribuição.

## Fontes

Fonte empacotada:

```text
assets/fonts/Oxanium.ttf
```

Licença:

```text
SIL Open Font License 1.1
Copyright 2019 The Oxanium Project Authors
https://github.com/sevmeyer/oxanium
https://scripts.sil.org/OFL
```

Resumo operacional:

- pode usar, estudar, copiar, embutir, modificar, redistribuir e vender software que inclua a fonte;
- a fonte não pode ser vendida isoladamente;
- redistribuições devem manter copyright e licença;
- versões modificadas não podem usar nome reservado sem permissão;
- documentos gerados usando a fonte não precisam ficar sob OFL.

O texto integral da licença permanece em `assets/fonts/Oxanium-OFL.txt`.

## Sprites E Assets Gerados

Embora este documento consolide principalmente fontes/licenças, estes grupos também fazem parte da governança visual do jogo:

- `assets/generated/title_stadium_ai.png`: abertura.
- `assets/generated/stadium_parallax_real.png`: estádio/campo do confronto.
- `assets/generated/parallax_sources/imagen_turf_*.png`: fontes do gramado geradas por `image_gen`.
- `assets/generated/parallax/turf_*_strip.png`: faixas preparadas para parallax.
- `assets/generated/cinematic_sources/imagen_oracle_*.png`: sheets de jogadores gerados por `image_gen`, com `ORACLE` no centro da camisa.
- `assets/generated/cinematic/runner_*.png`: corrida por uniforme.
- `assets/generated/cinematic/keeper_anim_*.png`: goleiro.
- `assets/generated/cinematic/goal_net_*.png`: camada traseira da trave/rede.
- `assets/generated/cinematic/goal_front_*.png`: camada frontal com postes/trave limpos, sem rede duplicada.
- `assets/generated/cinematic/goal_impact_*.png`: deformação da rede.
- `assets/generated/ball_sources/plain_ball_sheet_8frames.png`: fonte da bola.
- `assets/generated/balls3d/*.png`: frames da bola.
- `assets/generated/flags/*.png`: 48 bandeiras em sprite.

Z-order oficial do gol:

```text
goal_back -> jogador/bola -> goal_front/impact -> goleiro
```

Regra de sprites:

- sprites principais do jogo devem ser gerados pelo `image_gen`;
- recorte e preparação ficam em scripts;
- runtime não deve fazer recorte/chroma key pesado;
- nenhum jogador deve ter escudo no shorts, brasão, número ou marcas além de `ORACLE` no centro da camisa.

## Validação

Gates:

```bash
make audio-qa
make visual-qa
make aaa-qa
make validate
```

`make audio-qa` valida:

- arquivos do manifest;
- recibos de origem/licenca dos runtime assets;
- duração;
- transiente;
- clipping;
- mix combinado `kick -> whoosh -> net -> bass -> cheer/reverb`;
- buses;
- ducking/cooldown;
- política única de cue e preservação da cama de estádio no impacto;
- ordem dos eventos de gol, quase gol, defesa e apito final;
- sincronismo quantizado de frame para chute/rede;
- ticks e reveal do Monte Carlo;
- reset da fila de áudio da Copa ao trocar de cena;
- pureza de `draw()`.

`make visual-qa` valida sincronização visual e gera frames em:

```text
artifacts/visual_qa/current/
```

`make aaa-qa` adiciona gates pesados:

- sprites antigos no runtime;
- `ORACLE` legível no peito;
- pernas e bbox;
- escala de pose;
- chute com âncora do pé;
- bola duplicada;
- goleiro no gol;
- bola na rede;
- rede visível;
- parallax sem seam;
- z-order do gol;
- partida completa;
- orçamento visual de 60 FPS.

## Processo Para Promover Novo Asset

1. Baixar ou gerar o asset.
2. Guardar em `candidates/` ou pasta de fontes apropriada.
3. Registrar origem, autor, licença, URL e data.
4. Fazer curadoria A/B.
5. Preparar/cortar/normalizar por script.
6. Mover para runtime com nome estável.
7. Atualizar manifesto operacional.
8. Atualizar provenance.
9. Atualizar `assets/sounds/downloaded_audio_manifest.csv` quando o asset veio de download manual.
10. Rodar QA.

Para áudio:

```bash
uv run python scripts/promote_audio_assets.py
make audio-qa
```

Para visual/cinematic:

```bash
make visual-qa
make aaa-qa
```

## Riscos Pendentes

- Assets runtime de Freesound avulsos confirmados como Creative Commons 0 em 2026-05-17: `kick_grass_01.wav`, `whistle_start_01.wav`, `whistle_final_01.wav`.
- Assets runtime de Pixabay confirmados em 2026-05-18: `net_ripple_01.wav`, `opening_theme.mp3`, `goal_roar_pixabay_01.wav`, `goal_roar_pixabay_02.wav`.
- Imagens FIFA externas precisam de validação de direitos antes de redistribuição pública.
- ZapSplat e Sonniss não estão integrados ao runtime atual.
