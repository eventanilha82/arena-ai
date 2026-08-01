# Build, Empacotamento E Release

Este documento consolida a seção de empacotamento do `README.md` e o antigo `win/README.md`. O `README.md` da raiz fica como capa do projeto; os detalhes de build vivem aqui.

---

## Empacotamento Mac / Windows

O projeto tem um `Makefile` para padronizar validação, staging de assets e
empacotamento:

```bash
make validate
make build-assets-qa
make release-qa-local
make build-mac
```

`make build-assets-qa` é o caminho iterativo: valida e monta
`build/release_assets/` a partir do worktree atual, mesmo antes de novos assets
serem adicionados ao Git. O gate de release é mais estrito:

```bash
make build-assets-release-qa
```

Ele exige que todos os assets finais ativos do inventário estejam rastreados no
Git. Essa exigência cobre imagens, fontes, áudio, flags, bolas, poses, sheets e
metadados cinematográficos; não se limita ao áudio.

## Gates Local/CI E De Host

O release usa dois gates complementares:

- `make release-qa-local` roda no workspace local ou no CI, onde existem Git e
  `artifacts/visual_qa/current`. Ele regenera o cache, executa validação,
  estabilidade Monte Carlo, AAA QA, valida a evidência visual corrente, repete
  a captura cinematográfica em diretório temporário e exige todos os assets
  finais ativos rastreados no Git. O primeiro passo, `release-source-qa`, exige
  um snapshot commitado e sem alterações e fixa commit, tree e fingerprint.
- `make release-qa-host` roda sobre um source bundle ou clean checkout. Ele não
  consulta `.git` nem depende de evidência visual preexistente. Em vez disso,
  valida o cache recebido, executa `validate`, `aaa-runtime-qa` e
  reconstrói/valida o staging de runtime no próprio host.

`make aaa-runtime-qa` executa os mesmos contratos de código, assets, física,
áudio e performance do jogo, mas não exige o diretório `artifacts/`. O alvo
local `make aaa-qa` acrescenta o selo da evidência cinematográfica corrente;
por isso ele continua sendo o gate usado antes de enviar o source bundle.

`make release-qa` é um alias de `make release-qa-local`. Os dois gates apenas
validam e montam staging; nenhum deles chama PyInstaller, cria ZIP ou publica
release.

`make build-current` é o passo de host: roda `release-qa-host`, empacota para o
sistema atual e grava um sidecar com hash do payload, do executável e do source.
Ele serve para source bundles sem Git quando recebe
`RELEASE_SOURCE_PROVENANCE`, mas não substitui a aprovação local/CI. `make
build` e `make build-mac` capturam a identidade antes do QA, confirmam que ela
não mudou, extraem `git archive <SHA capturado>` em diretório temporário e só
então chamam o passo de host nessa cópia imutável. O artefato só volta para
`dist/` se o checkout original ainda corresponder ao mesmo snapshot.

O PyInstaller não faz cross-compile confiável. Por isso:

- `make build-mac` gera o `.app` no macOS.
- `make build-windows` gera o `.exe` dentro do Windows.
- No macOS, `make build-windows` usa a VM Windows configurada em `win/`.

Fluxo Windows remoto:

```bash
cp win/.env.example win/.env
# edite ARENA_WIN_HOST, ARENA_WIN_USER, ARENA_WIN_REMOTE_ROOT e ARENA_WIN_QA
make build-windows
```

O wrapper remoto fixa commit/tree/fingerprint antes do
`release-qa-local`, confirma a mesma identidade depois do QA e cria
`git archive <SHA capturado>`. Em seguida envia o snapshot e sua proveniência
para a VM, roda `uv sync --dev`, executa o QA adicional definido por
`ARENA_WIN_QA`, passa pelo `release-qa-host`, roda o PyInstaller no Windows e
baixa o artefato e seu sidecar para:

```text
win/artifacts/ArenaAI-windows-latest.zip
win/artifacts/build-result.json
```

Depois o Makefile valida que o ZIP abre, contém `ArenaAI.exe` e inclui
`runtime_prediction_cache.pkl`, usado para aquecer predições de confronto e para
o modo turbo opcional da tela da Copa. Todos os caminhos lógicos de runtime são
normalizados sem distinção de caixa, inclusive o prefixo `_internal/`, e cada
payload obrigatório é comparado por SHA-256 com o source enviado ao host.
Além disso, o sidecar registra o hash do ZIP e do `ArenaAI.exe`; o checker
compara commit, tree e fingerprint com o snapshot local limpo. Assim, um build
antigo não pode receber a proveniência do código atual.

Se estiver trabalhando diretamente em uma máquina Windows, rode:

```powershell
make sync
make build-windows
```

Esse comando direto exige checkout Git limpo e evidência visual corrente; ele
usa o mesmo build por snapshot imutável do Mac. O caminho sem `.git` é
exclusivo do host de build e deve chamar `make build-current` com um
`RELEASE_SOURCE_PROVENANCE` criado pelo wrapper local/CI.

O build usa `make build-assets-qa` para montar `build/release_assets/` só com
assets finais de runtime e o pacote SOTA mínimo. O checker compara cada arquivo
do staging byte a byte com o source. O ZIP Windows rejeita caminhos duplicados,
extras, fontes brutas e divergência de bytes depois de normalizar `_internal/`,
separadores e caixa, como exige um filesystem case-insensitive. O `.app` passa
por `make mac-app-check`: ele exige `Info.plist`, executável e o inventário
canônico byte a byte em `Contents/Resources` ou
`Contents/Frameworks/_internal`. Mac e Windows também exigem sidecars gerados no
host de build com hash do payload e do executável. O empacotador repete os
checks, exige que ambos apontem para o mesmo snapshot Git limpo e só depois cria
qualquer saída. Isso certifica o conteúdo
PyInstaller, não um smoke visual do binário. `assets/sounds/candidates/`, docs e
fontes brutas podem existir no source bundle para fins de QA, mas não entram no
payload. Guia completo da VM, bootstrap RDP/SSH, variáveis e cuidados de
segurança: este próprio documento.

---

## Windows Via VM

Esta pasta automatiza uma VM Windows para gerar o pacote PyInstaller do Arena AI.
Ela existe porque o PyInstaller não faz cross-compile confiável: o `.app` deve
ser gerado no macOS e o `.exe` deve ser gerado dentro do Windows.

A senha da instância Windows na OCI não deve ser commitada. Guarde no gerenciador
de senhas ou digite de forma interativa quando o SSH/RDP pedir.

## Bootstrap Da VM

Use a imagem `Windows Server 2022 Standard` com desktop experience. Evite a
imagem `Core`, porque RDP ajuda muito na validação visual do Pygame.

A senha inicial da OCI deve ser usada só no primeiro login por RDP. Depois que o
Windows pedir a troca de senha, rode o bootstrap para instalar ferramentas,
habilitar SSH e configurar chave pública.

O arquivo `win/authorized_keys.pub` é ignorado pelo Git de propósito. Gere ou
copie a sua chave pública local para esse caminho antes de rodar o bootstrap:

```bash
cp ~/.ssh/id_ed25519.pub win/authorized_keys.pub
```

## Caminho Recomendado Por RDP

Este é o caminho mais simples quando o repo local está compartilhado no RDP como
`\\tsclient\repo`.

1. Conecte na VM via RDP e compartilhe esta pasta do repo como `repo`.
2. Faça login com o usuário `opc` e a senha inicial da OCI.
3. Troque a senha quando o Windows pedir.
4. Rode:

```cmd
\\tsclient\repo\win\bootstrap-rdp.cmd
```

5. Aceite o prompt de UAC do Windows.
6. Acompanhe o log localmente:

```bash
tail -f win/logs/bootstrap-windows.log
```

O wrapper relança o processo como Administrador, chama
`win/bootstrap-windows.ps1`, instala a chave pública de
`win/authorized_keys.pub` e grava o log em `win/logs/bootstrap-windows.log`.

## Caminho Manual PowerShell

Use este caminho se não estiver compartilhando a pasta local no RDP.

1. Conecte na VM via RDP com as credenciais iniciais da OCI.
2. Abra PowerShell como Administrador.
3. Copie `win/bootstrap-windows.ps1` para a VM.
4. Rode:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\bootstrap-windows.ps1
```

Opcionalmente, instale a chave pública já no bootstrap:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\bootstrap-windows.ps1 -PublicKey "ssh-ed25519 AAAA..."
```

O bootstrap instala Git, GNU Make, uv, Python 3.12 gerenciado pelo uv, habilita
OpenSSH Server, abre a porta `22` no firewall do Windows e cria `C:\arena-ai`.

## Configuração Local

Copie o exemplo de variáveis:

```bash
cp win/.env.example win/.env
```

Edite `win/.env`:

```bash
ARENA_WIN_HOST=147.224.175.141
ARENA_WIN_USER=opc
ARENA_WIN_REMOTE_ROOT=C:/arena-ai
ARENA_WIN_QA=smoke
ARENA_WIN_INITIAL_PASSWORD=
ARENA_WIN_PASSWORD=
```

`ARENA_WIN_INITIAL_PASSWORD` e `ARENA_WIN_PASSWORD` são espaços locais para esta
máquina. Eles existem para não perder a senha durante o setup, mas `win/.env` é
ignorado pelo Git e nunca deve ser commitado.

## Empacotamento A Partir Do Mac

Depois do bootstrap da VM, rode pelo Makefile:

```bash
make build-windows
```

No macOS, esse alvo chama `make build-windows-remote`, que executa
`win/remote-build.sh` com os valores de `win/.env`. No Windows, o mesmo alvo
roda o build local do PyInstaller.

O build remoto faz:

1. Exige worktree limpo e grava commit, tree e fingerprint em
   `release-source-provenance.json`.
2. Roda `make release-qa-local`. É neste ponto, antes do upload, que
   `artifacts/visual_qa/current` é validado contra as fontes atuais e que o
   inventário exige todos os assets finais ativos rastreados no Git.
3. Confirma que commit/tree/fingerprint continuam idênticos e cria o bundle
   exclusivamente com `git archive <SHA capturado>`; arquivos ignorados,
   experimentos, builds locais, segredos e mudanças não commitadas não entram.
4. Envia o bundle e a proveniência para a VM Windows via SSH e roda
   `uv sync --dev`.
5. Mantém no source bundle somente os arquivos rastreados do snapshot. O corte
   para runtime ocorre em `make build-assets-qa`; por isso `candidates/`, docs e
   fontes brutas não entram no executável final.
6. Roda o diagnóstico adicional escolhido por `ARENA_WIN_QA`.
7. Roda `make build-current` dentro do Windows. Esse alvo chama
   `release-qa-host`, que repete os gates executáveis sem consultar Git ou a
   evidência visual local, e depois chama o PyInstaller.
8. Compacta `dist/ArenaAI` e grava `build-result.json` com os hashes do ZIP,
   `ArenaAI.exe` e snapshot de source.
9. Baixa obrigatoriamente ZIP e sidecar para `win/artifacts/`.
10. O Makefile valida inventário, bytes e a correspondência completa do
    sidecar antes de aceitar o artefato.

`ARENA_WIN_QA` pode ser:

- `smoke`: diagnóstico rápido antes do gate de host.
- `validate`: diagnóstico intermediário antes do gate de host.
- `aaa`: repete antecipadamente o gate visual/performance completo.
- `none`: pula somente esse diagnóstico adicional.

Nenhuma dessas opções desativa `release-qa-host`; portanto `none`, `smoke` ou
`validate` não reduzem o gate obrigatório executado por `build-current`.

Se `model_sota.pkl` ou `sota_pipeline.py` mudarem, rode antes:

```bash
make runtime-cache
```

Isso renova `runtime_prediction_cache.pkl`, usado pela tela da Copa para manter
o Monte Carlo de 1000 Copas rápido no executável Windows.

Em um clean checkout que ainda não possui `artifacts/visual_qa/current`, gere e
revise a evidência antes do preflight local/CI:

```bash
make visual-qa
make release-qa-local
```

Um source bundle recebido pelo host não precisa desses artefatos e deve usar
`make release-qa-host` ou `make build-current`.

## Validação Do Artefato

Depois do build remoto, valide o ZIP local:

```bash
make windows-artifact-check
```

Esse alvo reprova o pacote se faltar `ArenaAI.exe`, qualquer arquivo do inventário
completo de runtime (inclusive sheets e os 16 quadros do goleiro por direção), se entrar asset
bruto, ou se `runtime_prediction_cache.pkl` estiver ausente/stale/incompleto. O
mesmo contrato é aplicado ao staging local e ao ZIP Windows já empacotado.

`generated_runtime_globs` e `release_runtime_globs` descrevem o mesmo payload
canônico; o gate reprova qualquer divergência entre eles. A cardinalidade fixa
dos globs finais inclui 32 quadros de bola,
16 quadros de goleiro por direção e seis sheets por uniforme (corrida, chute e
parada em cada direção), além do contrato promovido, redes e flags. Um payload
válido exige somente o inventário final declarado; os checks do staging/ZIP
rejeitam tanto ausências quanto extras sob `assets/` e `modeling/`.
O verificador também rejeita colisões após normalização, por exemplo o mesmo
asset presente simultaneamente em `assets/...` e `_internal/assets/...`, além de
caminhos absolutos ou com travessia `..`.

## Artefatos De Release Opcionais

Os builds normais ficam ignorados pelo Git:

```text
dist/
build/
win/artifacts/
release/
```

Para gerar Mac + Windows e montar uma pasta de release local:

```bash
make build-release
```

Esse alvo roda:

1. `make build-mac`
2. `make build-windows`
3. `make release-artifacts`

A saída fica em:

```text
release/ArenaAI-mac-latest.zip
release/ArenaAI-windows-latest.zip
release/ArenaAI-mac-build-result.json
release/ArenaAI-windows-build-result.json
release/SHA256SUMS
release/release-manifest.json
```

Se os builds já existem e você quer só montar a pasta `release/`:

```bash
make release-artifacts
```

O bolão é um utilitário Rich executado dentro do projeto por `make bolao`; ele
não gera binário nem ZIP próprio de release.

Para publicar como GitHub Release, depois de configurar o repositório remoto e
ter o GitHub CLI autenticado:

```bash
git tag -a v0.2.0 -m "ArenaAI v0.2.0"
git push origin v0.2.0
make release-github TAG=v0.2.0
```

O empacotamento reabre o ZIP macOS, valida CRC, inventário, bytes, launcher,
permissão de execução e o fingerprint canônico da árvore contra a proveniência
do build; o ZIP Windows copiado também é revalidado. Ao final, o snapshot Git
precisa continuar limpo e idêntico. A publicação exige tag anotada apontando
exatamente para o SHA comum aos dois builds e usa `gh --verify-tag`, impedindo
que o GitHub crie silenciosamente uma tag em outro commit.

Se, excepcionalmente, quiser versionar os ZIPs no Git, use o alvo explícito:

```bash
make release-stage
```

Ele roda `git add -f` apenas nos seis arquivos da pasta `release/`. Esse passo
é intencionalmente separado porque binários de release normalmente pertencem a
GitHub Releases, OCI Object Storage ou outro repositório de artefatos, não ao
commit normal do código.

Para validação visual final, abra RDP e execute:

```powershell
C:\arena-ai\source\dist\ArenaAI\ArenaAI.exe
```

## Cuidados

- Restrinja as regras de ingress da OCI para RDP `3389` e SSH `22` ao seu IP.
- Não commite ZIPs de release. Use GitHub Releases, OCI Object Storage ou outro
  repositório de artefatos.
- `win/artifacts/`, `win/logs/`, `win/.env`, chaves SSH e `known_hosts` são
  ignorados de propósito.
