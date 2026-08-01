# Changelog

## v0.2.0 - 2026-07-31

- Cinematics POC2 promovida para o jogo, com corrida e condução mais naturais nos dois sentidos e em nove uniformes.
- Chute, goleiro, trajetória da bola, gol 3D, rede e áudio sincronizados pelo contrato POC7 aprovado.
- Mais variedade de gols, defesas e chutes para fora, com continuidade validada em 270 handoffs e 150 checkpoints visuais.
- Bolão atualizado com os 104 resultados observados, auditorias temporais e Monte Carlo estável.
- Runtime, caches, assets e proveniência de release reforçados para os builds macOS e Windows.
- Cache neutro agora exige pares espelhados completos, eliminando deriva numérica entre macOS e Windows.
- Fixture sintético de bundle macOS respeita a ausência de bits POSIX no Windows; artefatos reais continuam falhando fechado sem permissão de execução.
- Gate de 60 FPS trata relógios de CPU quantizados com janelas de throughput e wall-time por frame obrigatório.
- Release revalida ZIPs, snapshot final e vínculo da tag anotada ao SHA comum dos builds antes da publicação.
- Relatórios Monte Carlo e top-10 passam a fingerprintar também cache runtime e adaptador `worldcup_model.py`.

## v0.1.0 - 2026-05-19

- Primeira release pública do Arena AI.
