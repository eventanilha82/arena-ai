# Galeria Visual Versionada

Esta pasta contém a galeria pública usada pelo `README.md`. As imagens foram promovidas da evidência gerada pelo renderer real em `artifacts/visual_qa/current/` para que permaneçam disponíveis no GitHub, onde o diretório de artefatos locais é ignorado de propósito.

## Proveniência Atual

| Campo | Valor |
| --- | --- |
| Release de referência | `v0.2.0` |
| Commit do jogo capturado | `37d693fd494c4874952b28a7e9e4da496d542edc` |
| Geração da evidência | `2026-08-01T11:06:18+00:00` |
| Viewport | `1280x760` |
| Estado do worktree capturado | limpo |
| Capturas na folha completa | `23` |
| Fonte de evidência | `artifacts/visual_qa/current/metadata.json` schema 3 |

## Imagens Promovidas

| Arquivo | Estado documentado |
| --- | --- |
| `contact_sheet.png` | visão conjunta das 23 capturas |
| `00_menu.png` | abertura |
| `00b_selecao.png` | escolha do confronto |
| `01_posse_inicial.png` | partida ao vivo e parallax |
| `04_bola_em_voo.png` | chute, trajetória da bola e goleiro em extensão |
| `05b_gol_overlay.png` | gol confirmado após impacto na rede |
| `09_placar_final.png` | encerramento e placar final |
| `10_copa_calculando.png` | Monte Carlo em execução |
| `11_copa_grupos.png` | classificação da fase de grupos |
| `12_copa_mata_mata.png` | chave eliminatória e campeã |

## Limite De Uso

Estas imagens são documentação, não assets de runtime. Elas não entram em `assets/asset_manifest.json`, no staging do PyInstaller nem nos ZIPs de macOS e Windows. A fonte de verdade para QA continua sendo a captura completa ignorada pelo Git, acompanhada por `metadata.json`, vídeos e inventário de hashes.

Uma imagem versionada demonstra o estado visual do commit indicado acima. Ela não substitui `make visual-qa-check`, não comprova comportamento temporal e não deve ser reutilizada como evidência depois de uma mudança de UI ou cinemáticas.

## Como Atualizar

1. Implemente e valide a mudança visual.
2. Rode `make visual-qa` e faça a revisão humana da folha completa, dos vídeos a 60 fps e dos closes em câmera lenta.
3. Rode `make visual-qa-check` para confirmar hashes, inventário e reprodução determinística.
4. Substitua nesta pasta somente as capturas públicas selecionadas para o README.
5. Atualize a tabela de proveniência e revise os textos alternativos do README.
6. Depois do commit, rode novamente `make visual-qa && make visual-qa-check` em worktree limpo antes de preparar uma release.

Os detalhes dos gates e as limitações honestas da evidência visual ficam em [../../QUALITY.md](../../QUALITY.md).
