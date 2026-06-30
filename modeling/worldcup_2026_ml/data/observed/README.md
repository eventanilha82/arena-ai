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
pênaltis. Esses jogos ficam travados nas simulações seguintes; o modelo não os
reprojeta nem permite que uma trilha condicionada ressuscite uma seleção já
eliminada.
