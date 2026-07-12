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
pênaltis. A foto de 2026-07-12 contém 28 jogos eliminatórios encerrados: 20
resolvidos em 90 minutos, quatro na prorrogação e quatro nos pênaltis. As linhas
dos 16 avos usam `user_reported_and_web_verified_2026-07-04`; as sete oitavas
resolvidas em 90 minutos usam `user_reported_and_web_verified_2026-07-07`; e as
quatro quartas usam `user_reported_and_web_verified_2026-07-12`.

O jogo 96 foi informado inicialmente com o classificado invertido. A verificação
externa confirmou Suíça 0 x 0 Colômbia após 120 minutos e vitória suíça por 4 x 3
nos pênaltis; a linha usa `web_verified_correction_2026-07-07`.

Nos jogos 99 e 100, o placar ao fim dos 90 minutos foi `1 x 1`. A Inglaterra
marcou `0 x 1` na prorrogação contra a Noruega, enquanto a Argentina marcou
`2 x 0` na prorrogação contra a Suíça. A ordem interna segue o fixture oficial:
`Norway x England` no jogo 99.

Esses jogos ficam travados nas simulações seguintes; o modelo não os reprojeta
nem permite que uma trilha condicionada ressuscite uma seleção já eliminada.
Quando a forma atual está habilitada pelo gate temporal dos grupos, somente o
placar ao fim dos 90 minutos atualiza o posterior Gamma-Poisson. Gols da
prorrogação e cobranças da disputa permanecem fora da taxa regulamentar.
