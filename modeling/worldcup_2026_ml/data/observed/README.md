# Snapshot do Bolão

O arquivo `worldcup_2026_group_stage_results.csv` registra o snapshot fornecido
diretamente pelo usuário em 2026-06-28. A proveniência de cada linha é
`user_reported_group_stage`.

O código valida a identidade do confronto, grupo, placar inteiro não negativo
e a ordem cronológica do snapshot contra o fixture. Ele não substitui uma
fonte oficial nem afirma verificação independente dos resultados.

O arquivo `worldcup_2026_knockout_results.csv` registra jogos eliminatórios
encerrados. Cada linha precisa reproduzir o confronto resolvido pela chave,
separar placar de 90 minutos, prorrogação e, quando existir, a disputa de
pênaltis. A foto final de 2026-07-19 contém os 32 jogos eliminatórios: 23
resolvidos em 90 minutos, cinco na prorrogação e quatro nos pênaltis. As linhas
dos 16 avos usam `user_reported_and_web_verified_2026-07-04`; as sete oitavas
resolvidas em 90 minutos usam `user_reported_and_web_verified_2026-07-07`; e as
quatro quartas usam `user_reported_and_web_verified_2026-07-12`. Semifinais,
terceiro lugar e final usam `user_reported_and_web_verified_2026-07-19`.

O jogo 96 foi informado inicialmente com o classificado invertido. A verificação
externa confirmou Suíça 0 x 0 Colômbia após 120 minutos e vitória suíça por 4 x 3
nos pênaltis; a linha usa `web_verified_correction_2026-07-07`.

Nos jogos 99 e 100, o placar ao fim dos 90 minutos foi `1 x 1`. A Inglaterra
marcou `0 x 1` na prorrogação contra a Noruega, enquanto a Argentina marcou
`2 x 0` na prorrogação contra a Suíça. A ordem interna segue o fixture oficial:
`Norway x England` no jogo 99.

Na etapa final, a ordem do fixture é `France x Spain`, `England x Argentina`,
`France x England` e `Spain x Argentina`. A final terminou `0 x 0` aos 90
minutos; o `1 x 0` espanhol foi marcado na prorrogação e aparece separado nas
colunas `extra_time_*`.

Esses jogos ficam travados; o modelo não os reprojeta nem permite que uma trilha
condicionada ressuscite uma seleção eliminada. Com o jogo 104 registrado, a
Espanha é campeã observada e o console não apresenta esse lock como previsão.
Quando a forma atual está habilitada pelo gate temporal dos grupos, somente o
placar ao fim dos 90 minutos atualiza o posterior Gamma-Poisson. Gols da
prorrogação e cobranças da disputa permanecem fora da taxa regulamentar.
