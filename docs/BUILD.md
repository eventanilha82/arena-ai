# Build, Empacotamento E Release

Este documento consolida a seção de empacotamento do `README.md` e o antigo `win/README.md`. O `README.md` da raiz fica como capa do projeto; os detalhes de build vivem aqui.

---

## Empacotamento Mac / Windows

O projeto tem um `Makefile` para padronizar validação, staging de assets e
empacotamento:

```bash
make validate
make build-assets-qa
make build-mac
```

O PyInstaller não faz cross-compile confiável. Por isso:

- `make build-mac` gera o `.app` no macOS.
- `make build-windows` gera o `.exe` dentro do Windows.
- No macOS, `make build-windows` usa a VM Windows configurada em `win/`.

Fluxo Windows remoto:

```bash
cp win/.env.example win/.env
# edite ARENA_WIN_HOST, ARENA_WIN_USER, ARENA_WIN_REMOTE_ROOT e ARENA_WIN_QA
make runtime-cache
make build-windows
```

O alvo remoto empacota o workspace atual, envia para a VM por SSH, roda
`uv sync --dev`, executa o QA definido por `ARENA_WIN_QA`, roda o PyInstaller no
Windows e baixa o artefato para:

```text
win/artifacts/ArenaAI-windows-latest.zip
```

Depois o Makefile valida que o ZIP abre, contém `ArenaAI.exe` e inclui
`runtime_prediction_cache.pkl`, usado para aquecer predições de confronto e para
o modo turbo opcional da tela da Copa.

Se estiver trabalhando diretamente em uma máquina Windows, rode:

```powershell
make sync
make build-windows
```

O build usa `make build-assets-qa` para montar `build/release_assets/` só com
assets de runtime e pacote SOTA mínimo; `assets/sounds/candidates/`, docs e
fontes brutas não entram no bundle. Guia completo da VM, bootstrap RDP/SSH,
variáveis e cuidados de segurança: este próprio documento.

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

1. Empacota a área de trabalho local atual, incluindo mudanças não commitadas.
2. Exclui `.git`, `.venv`, `build`, `dist`, artefatos Windows locais e segredos
   locais da VM, como `win/.env`, chaves SSH, `known_hosts` e
   `authorized_keys.pub`. O upload mantém fontes geradas e assets de curadoria
   porque o QA valida a rastreabilidade do manifesto antes do build. O corte de
   runtime acontece depois, em `make build-assets-qa`, para garantir que
   `candidates/`, docs e fontes brutas não entrem no executável final.
3. Envia o bundle para a VM Windows via SSH.
4. Roda `uv sync --dev`.
5. Roda o gate de QA escolhido.
6. Roda `make build-windows` dentro do Windows.
7. Compacta `dist/ArenaAI`.
8. Baixa o ZIP para `win/artifacts/ArenaAI-windows-latest.zip`.
9. O Makefile valida que o ZIP abre, contém `ArenaAI.exe`, não inclui assets
   brutos e traz um `runtime_prediction_cache.pkl` fresco por hash/conteúdo.

`ARENA_WIN_QA` pode ser:

- `smoke`: gate rápido antes do build.
- `validate`: gate mais amplo.
- `aaa`: gate visual/performance completo. Use apenas quando esse gate fizer
  sentido para o bundle enxuto enviado à VM; o upload padrão exclui raw,
  candidates e sources.
- `none`: apenas empacota.

Se `model_sota.pkl` ou `sota_pipeline.py` mudarem, rode antes:

```bash
make runtime-cache
```

Isso renova `runtime_prediction_cache.pkl`, usado pela tela da Copa para manter
o Monte Carlo de 1000 Copas rápido no executável Windows.

## Validação Do Artefato

Depois do build remoto, valide o ZIP local:

```bash
make windows-artifact-check
```

Esse alvo reprova o pacote se faltar `ArenaAI.exe`, se entrar asset bruto, ou
se `runtime_prediction_cache.pkl` estiver ausente/stale/incompleto.

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
release/SHA256SUMS
release/release-manifest.json
```

Se os builds já existem e você quer só montar a pasta `release/`:

```bash
make release-artifacts
```

Esse alvo também roda `make windows-artifact-check`. Ou seja: ele só monta a
pasta de release se o ZIP Windows tiver `ArenaAI.exe` e todos os assets/cache
obrigatórios do runtime atual.

Para publicar como GitHub Release, depois de configurar o repositório remoto e
ter o GitHub CLI autenticado:

```bash
make release-github TAG=v0.1.0
```

Se, excepcionalmente, quiser versionar os ZIPs no Git, use o alvo explícito:

```bash
make release-stage
```

Ele roda `git add -f` apenas nos quatro arquivos da pasta `release/`. Esse passo
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
