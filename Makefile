.PHONY: help sync run cinematic-game-qa cinematic-game-qa-capture cinematic-game-qa-check console bolao bolao-grupos bolao-qa bolao-knockout-audit bolao-round16-audit bolao-quarterfinal-audit bolao-final-audit bolao-mc-stability bolao-top10-audit smoke validate stats-qa mc-stability runtime-cache runtime-cache-check aaa-runtime-qa aaa-qa visual-qa visual-qa-check audio-smoke audio-qa benchmark-mc-workers build-assets-qa build-assets-release-qa release-provenance-qa release-source-qa release-qa release-qa-local release-qa-host build build-current build-mac build-windows build-windows-local build-windows-remote mac-app-check windows-artifact-check build-release release-artifacts release-stage release-github release-clean clean-build

APP_NAME := ArenaAI
ifeq ($(OS),Windows_NT)
	PYTHON ?= .venv/Scripts/python.exe
else
	PYTHON ?= .venv/bin/python
endif
PYINSTALLER := uv run pyinstaller
PYI_COMMON := --noconfirm --clean --onedir --windowed --name $(APP_NAME) --paths src
PYI_COMMON += --hidden-import pandas --hidden-import numpy
PYI_COMMON += --hidden-import sklearn.pipeline
PYI_COMMON += --hidden-import sklearn.linear_model._logistic
PYI_COMMON += --hidden-import sklearn.linear_model._glm.glm
PYI_COMMON += --hidden-import sklearn.preprocessing._data
PYI_COMMON += --hidden-import sklearn._loss.link --hidden-import sklearn._loss.loss
PYI_COMMON += --hidden-import xgboost.sklearn --hidden-import xgboost.core
PYI_COMMON += --collect-binaries xgboost --collect-data xgboost
RELEASE_ASSETS := build/release_assets
CINEMATIC_GAME_QA_DIR := artifacts/cinematic_game_qa/current
CINEMATIC_GAME_QA_REPORT := build/qa/cinematic_game_integration.json
RELEASE_DIR := release
MAC_RELEASE_ARTIFACT := $(RELEASE_DIR)/$(APP_NAME)-mac-latest.zip
WINDOWS_RELEASE_ARTIFACT := $(RELEASE_DIR)/$(APP_NAME)-windows-latest.zip
MAC_RELEASE_PROVENANCE := $(RELEASE_DIR)/$(APP_NAME)-mac-build-result.json
WINDOWS_RELEASE_PROVENANCE := $(RELEASE_DIR)/$(APP_NAME)-windows-build-result.json
MAC_APP := dist/$(APP_NAME).app
MAC_BUILD_PROVENANCE := dist/$(APP_NAME).app.build-result.json
TAG ?=
RELEASE_SOURCE_PROVENANCE ?=
RELEASE_SOURCE_PROVENANCE_ARG = $(if $(strip $(RELEASE_SOURCE_PROVENANCE)),--source-provenance "$(RELEASE_SOURCE_PROVENANCE)",)
BOLAO_QA_RUNS ?= 100
BOLAO_QA_SEED ?= 2026
BOLAO_MC_RUNS ?= 1000,2000
BOLAO_MC_SEED ?= 20260628
BOLAO_MC_INDEPENDENT_SEEDS ?= 20260628,20260629,20260630

ifeq ($(OS),Windows_NT)
	PYI_DATA_SEP := ;
	CURRENT_PLATFORM := windows
	CURRENT_BUILD_ARTIFACT := dist/$(APP_NAME)
	CURRENT_BUILD_PROVENANCE := dist/$(APP_NAME).build-result.json
else
	PYI_DATA_SEP := :
	CURRENT_PLATFORM := macos
	CURRENT_BUILD_ARTIFACT := $(MAC_APP)
	CURRENT_BUILD_PROVENANCE := $(MAC_BUILD_PROVENANCE)
endif

PYI_DATA := --add-data "$(RELEASE_ASSETS)/assets$(PYI_DATA_SEP)assets"
PYI_DATA += --add-data "$(RELEASE_ASSETS)/modeling/worldcup_2026_ml$(PYI_DATA_SEP)modeling/worldcup_2026_ml"
WINDOWS_REMOTE_BUILD := win/remote-build.sh
WINDOWS_ARTIFACT := win/artifacts/$(APP_NAME)-windows-latest.zip
WINDOWS_BUILD_PROVENANCE := win/artifacts/build-result.json

help:
	@printf "Arena AI\n\n"
	@printf "make sync          instala dependencias com uv\n"
	@printf "make run           abre o jogo Pygame\n"
	@printf "make cinematic-game-qa valida o contrato cinematografico no runtime real\n"
	@printf "make cinematic-game-qa-capture salva a matriz visual do runtime real\n"
	@printf "make console       abre o console do modelo SOTA\n"
	@printf "make bolao         fase fixa + top 10 Monte Carlo + escolha interativa\n"
	@printf "make bolao-grupos  lista placares e classificacao fixa dos grupos\n"
	@printf "make bolao-qa      valida CSV, grupos e Monte Carlo curto sem interacao\n"
	@printf "make bolao-knockout-audit audita os 16 avos e gera a foto probabilistica das oitavas\n"
	@printf "make bolao-round16-audit audita as oitavas e gera a foto probabilistica das quartas\n"
	@printf "make bolao-quarterfinal-audit audita as quartas e gera a foto probabilistica das semifinais\n"
	@printf "make bolao-final-audit audita semifinais, terceiro lugar e final sem look-ahead\n"
	@printf "make bolao-mc-stability audita grupos fixos + forma + mata-mata em 1k/2k e seeds independentes\n"
	@printf "make bolao-top10-audit audita os favoritos por simetria, forma e pênaltis neutros\n"
	@printf "make smoke         compila, importa modelo e roda 1 predicao\n"
	@printf "make validate      roda gate essencial de assets/UI/audio-smoke/modelo\n"
	@printf "make stats-qa      gera auditoria estatistica: calibracao, IC, ablation, empate e Dixon-Coles\n"
	@printf "make mc-stability  roda Monte Carlo offline 5k/10k + fases/chaves 1k/2k/5k\n"
	@printf "make runtime-cache gera cache de predições e banco de 1000 Copas para a UI\n"
	@printf "make runtime-cache-check valida o cache runtime sem regenerar\n"
	@printf "make aaa-runtime-qa roda QA AAA executável sem exigir artifacts preexistentes\n"
	@printf "make aaa-qa        roda QA visual/performance completo\n"
	@printf "make visual-qa     roda QA completo e salva frames-chave em artifacts/visual_qa/current\n"
	@printf "make visual-qa-check reprova evidência visual órfã, alterada ou ligada a fonte antiga\n"
	@printf "make audio-smoke   valida wiring essencial de audio sem mixes/timeline completos\n"
	@printf "make audio-qa      valida governanca, buses e timeline de audio\n"
	@printf "make benchmark-mc-workers testa 2/4/8/16/32 workers no Monte Carlo\n"
	@printf "make build-assets-qa monta staging iterativo sem exigir Git\n"
	@printf "make build-assets-release-qa exige todos os assets ativos rastreados no Git\n"
	@printf "make release-provenance-qa prova rejeicao de sidecar/binario adulterado\n"
	@printf "make release-source-qa exige snapshot Git limpo e identifica commit/tree\n"
	@printf "make release-qa-local valida Git + evidência visual no workspace local/CI\n"
	@printf "make release-qa-host valida o bundle no host sem depender de Git/artifacts prévios\n"
	@printf "make release-qa    alias do gate local/CI completo\n"
	@printf "make build         valida e empacota um snapshot Git imutavel\n"
	@printf "make build-current empacota no host atual; não substitui o gate local/CI\n"
	@printf "make build-mac     empacota .app no macOS\n"
	@printf "make build-windows empacota .exe; no macOS usa a VM Windows da OCI\n"
	@printf "make build-windows-remote usa win/.env para buildar na VM Windows da OCI\n"
	@printf "make mac-app-check valida inventário e bytes do .app contra o source\n"
	@printf "make windows-artifact-check valida o zip Windows baixado em win/artifacts\n"
	@printf "make build-release gera Mac + Windows e monta release/*.zip\n"
	@printf "make release-artifacts monta release/*.zip a partir dos builds existentes\n"
	@printf "make release-stage faz git add -f dos artefatos em release/ se quiser versionar\n"
	@printf "make release-github TAG=vX.Y.Z publica os artefatos em GitHub Release\n"
	@printf "make clean-build   remove artefatos de build\n"

sync:
	uv sync --dev

run:
	uv run arena-ai

cinematic-game-qa:
	PYGAME_HIDE_SUPPORT_PROMPT=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy $(PYTHON) scripts/validate_poc_runtime_parity.py --output $(CINEMATIC_GAME_QA_REPORT)

cinematic-game-qa-capture:
	PYGAME_HIDE_SUPPORT_PROMPT=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy $(PYTHON) scripts/validate_poc_runtime_parity.py --capture-dir $(CINEMATIC_GAME_QA_DIR)/frames --output $(CINEMATIC_GAME_QA_DIR)/integration_report.json

cinematic-game-qa-check:
	PYGAME_HIDE_SUPPORT_PROMPT=1 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy $(PYTHON) scripts/validate_poc_runtime_parity.py --check-evidence $(CINEMATIC_GAME_QA_DIR) --output $(CINEMATIC_GAME_QA_REPORT)

console:
	uv run python modeling/worldcup_2026_ml/src/console.py

bolao:
	uv run arena-bolao

bolao-grupos:
	uv run arena-bolao --somente-grupos

bolao-qa:
	@printf "[make] bolao semantic QA\n"
	$(PYTHON) scripts/bolao_qa.py --runs $(BOLAO_QA_RUNS) --seed $(BOLAO_QA_SEED)

bolao-knockout-audit:
	@printf "[make] bolao Round-of-32 calibration audit\n"
	$(PYTHON) -m compileall -q src/arena_ai/bolao.py scripts/bolao_knockout_calibration.py
	$(PYTHON) scripts/bolao_knockout_calibration.py

bolao-round16-audit:
	@printf "[make] bolao Round-of-16 calibration audit\n"
	$(PYTHON) -m compileall -q src/arena_ai/bolao.py scripts/bolao_knockout_calibration.py scripts/bolao_round16_calibration.py
	$(PYTHON) scripts/bolao_round16_calibration.py

bolao-quarterfinal-audit:
	@printf "[make] bolao quarterfinal calibration audit\n"
	$(PYTHON) -m compileall -q src/arena_ai/bolao.py scripts/bolao_knockout_calibration.py scripts/bolao_round16_calibration.py scripts/bolao_quarterfinal_calibration.py
	$(PYTHON) scripts/bolao_quarterfinal_calibration.py

bolao-final-audit:
	@printf "[make] bolao final calibration audit\n"
	$(PYTHON) -m compileall -q src/arena_ai/bolao.py scripts/bolao_knockout_calibration.py scripts/bolao_round16_calibration.py scripts/bolao_quarterfinal_calibration.py scripts/bolao_final_calibration.py
	$(PYTHON) scripts/bolao_final_calibration.py

bolao-mc-stability:
	@printf "[make] compile bolao Monte Carlo stability\n"
	$(PYTHON) -m compileall -q src/arena_ai/bolao.py scripts/bolao_mc_stability.py
	@printf "[make] bolao Monte Carlo stability $(BOLAO_MC_RUNS)\n"
	$(PYTHON) scripts/bolao_mc_stability.py --runs $(BOLAO_MC_RUNS) --seed $(BOLAO_MC_SEED) --independent-seeds $(BOLAO_MC_INDEPENDENT_SEEDS)

bolao-top10-audit:
	@printf "[make] top-10 bias audit\n"
	$(PYTHON) -m compileall -q src/arena_ai/bolao.py modeling/worldcup_2026_ml/src/sota_pipeline.py scripts/bolao_top10_bias_audit.py
	$(PYTHON) scripts/bolao_top10_bias_audit.py --runs 2000 --seed 20260629 --top 10

smoke:
	@printf "[make] compile smoke\n"
	$(PYTHON) -m compileall -q src/arena_ai modeling/worldcup_2026_ml/src/sota_pipeline.py modeling/worldcup_2026_ml/src/console.py scripts/smoke_model.py
	@printf "[make] model smoke\n"
	$(PYTHON) scripts/smoke_model.py
	@printf "[make] asset smoke\n"
	$(PYTHON) scripts/validate_visuals.py --suite smoke

validate:
	@printf "[make] compile validate\n"
	$(PYTHON) -m compileall -q src/arena_ai modeling/worldcup_2026_ml/src/sota_pipeline.py modeling/worldcup_2026_ml/src/console.py scripts
	$(MAKE) release-provenance-qa
	@printf "[make] model smoke\n"
	$(PYTHON) scripts/smoke_model.py
	$(MAKE) bolao-qa
	@printf "[make] standard validation\n"
	$(PYTHON) scripts/validate_visuals.py --suite standard
	@printf "[make] audio smoke\n"
	$(PYTHON) scripts/audio_qa.py --suite smoke

stats-qa:
	@printf "[make] compile stats-qa\n"
	$(PYTHON) -m compileall -q modeling/worldcup_2026_ml/src/sota_pipeline.py scripts/model_stats_qa.py
	@printf "[make] statistical model QA\n"
	$(PYTHON) scripts/model_stats_qa.py

mc-stability:
	@printf "[make] compile mc-stability\n"
	$(PYTHON) -m compileall -q modeling/worldcup_2026_ml/src/sota_pipeline.py scripts/monte_carlo_stability.py
	@printf "[make] Monte Carlo offline stability 5k/10k + fases/chaves 1k/2k/5k\n"
	$(PYTHON) scripts/monte_carlo_stability.py --runs 5000,10000 --stage-runs 1000,2000,5000 --workers 8 --chunk-size 25 --stage-chunk-size 25

runtime-cache:
	@printf "[make] compile runtime-cache\n"
	$(PYTHON) -m compileall -q src/arena_ai/worldcup_model.py modeling/worldcup_2026_ml/src/sota_pipeline.py scripts/build_runtime_prediction_cache.py
	@printf "[make] runtime Monte Carlo cache\n"
	$(PYTHON) scripts/build_runtime_prediction_cache.py --runs 1000 --workers 8

runtime-cache-check:
	@printf "[make] compile runtime-cache-check\n"
	$(PYTHON) -m compileall -q src/arena_ai/worldcup_model.py modeling/worldcup_2026_ml/src/sota_pipeline.py scripts/build_runtime_prediction_cache.py
	@printf "[make] check runtime Monte Carlo cache\n"
	$(PYTHON) scripts/build_runtime_prediction_cache.py --runs 1000 --workers 8 --check

aaa-runtime-qa:
	@printf "[make] compile aaa-runtime-qa\n"
	$(PYTHON) -m compileall -q src/arena_ai modeling/worldcup_2026_ml/src/sota_pipeline.py modeling/worldcup_2026_ml/src/console.py scripts
	@printf "[make] deep visual/performance validation\n"
	$(PYTHON) scripts/validate_visuals.py --suite aaa
	@printf "[make] audio validation\n"
	$(PYTHON) scripts/audio_qa.py
	@printf "[make] approved cinematic contract in real game runtime\n"
	$(MAKE) cinematic-game-qa

aaa-qa: aaa-runtime-qa
	@printf "[make] current cinematic evidence integrity\n"
	$(MAKE) cinematic-game-qa-check

audio-smoke:
	$(PYTHON) scripts/audio_qa.py --suite smoke

audio-qa:
	$(PYTHON) scripts/audio_qa.py

visual-qa: aaa-qa
	$(PYTHON) scripts/capture_visual_qa.py
	$(PYTHON) scripts/capture_visual_qa.py --check-current

visual-qa-check:
	$(PYTHON) scripts/capture_visual_qa.py --check-current

benchmark-mc-workers:
	$(PYTHON) scripts/benchmark_monte_carlo_workers.py

build-assets-qa:
	$(PYTHON) scripts/build_assets_qa.py --stage $(RELEASE_ASSETS)

build-assets-release-qa:
	$(PYTHON) scripts/build_assets_qa.py --stage $(RELEASE_ASSETS) --require-git-tracked

release-provenance-qa:
	$(PYTHON) scripts/release_provenance.py self-test
	$(PYTHON) scripts/build_from_snapshot.py --self-test

release-source-qa:
	$(PYTHON) scripts/release_provenance.py check-source

release-qa:
	$(MAKE) release-qa-local

release-qa-local:
	$(MAKE) release-source-qa
	$(MAKE) runtime-cache
	$(MAKE) validate
	$(MAKE) bolao-mc-stability
	$(MAKE) aaa-qa
	$(MAKE) visual-qa-check
	$(MAKE) build-assets-release-qa

release-qa-host:
	$(MAKE) runtime-cache-check
	$(MAKE) validate
	$(MAKE) aaa-runtime-qa
	$(MAKE) build-assets-qa

build:
	$(PYTHON) scripts/build_from_snapshot.py --platform "$(CURRENT_PLATFORM)"

build-current: clean-build
	$(MAKE) release-qa-host
	$(PYINSTALLER) $(PYI_COMMON) $(PYI_DATA) src/arena_ai/main.py
	$(PYTHON) scripts/release_provenance.py write-build --platform "$(CURRENT_PLATFORM)" --artifact "$(CURRENT_BUILD_ARTIFACT)" --output "$(CURRENT_BUILD_PROVENANCE)" $(RELEASE_SOURCE_PROVENANCE_ARG)
	@printf "\nBuild pronto em dist/$(APP_NAME) para $(CURRENT_PLATFORM).\n"

build-mac:
	@case "$$(uname -s)" in Darwin) true ;; *) printf "build-mac precisa rodar no macOS.\n"; exit 1 ;; esac
	$(MAKE) build
	$(MAKE) mac-app-check

build-windows:
ifeq ($(OS),Windows_NT)
	$(MAKE) build-windows-local
else
	$(MAKE) build-windows-remote
endif

build-windows-local:
ifeq ($(OS),Windows_NT)
	$(PYTHON) scripts/build_from_snapshot.py --platform windows
else
	@printf "build-windows-local precisa rodar dentro do Windows.\n"
	@exit 1
endif

build-windows-remote:
	@test -f "$(WINDOWS_REMOTE_BUILD)" || (printf "Script nao encontrado: $(WINDOWS_REMOTE_BUILD)\n"; exit 1)
	bash "$(WINDOWS_REMOTE_BUILD)"
	$(MAKE) windows-artifact-check

windows-artifact-check:
	@test -f "$(WINDOWS_ARTIFACT)" || (printf "Artefato nao encontrado: $(WINDOWS_ARTIFACT)\n"; exit 1)
	@test -f "$(WINDOWS_BUILD_PROVENANCE)" || (printf "Proveniencia Windows nao encontrada: $(WINDOWS_BUILD_PROVENANCE)\n"; exit 1)
	$(PYTHON) scripts/build_assets_qa.py --check-zip "$(WINDOWS_ARTIFACT)"
	$(PYTHON) scripts/release_provenance.py check-build --platform windows --artifact "$(WINDOWS_ARTIFACT)" --provenance "$(WINDOWS_BUILD_PROVENANCE)"
	@printf "\nArtefato Windows OK: $(WINDOWS_ARTIFACT)\n"

mac-app-check:
	@test -d "$(MAC_APP)" || (printf "App Mac nao encontrado: $(MAC_APP). Rode make build-mac.\n"; exit 1)
	@test -f "$(MAC_BUILD_PROVENANCE)" || (printf "Proveniencia Mac nao encontrada: $(MAC_BUILD_PROVENANCE)\n"; exit 1)
	$(PYTHON) scripts/build_assets_qa.py --check-mac-app "$(MAC_APP)"
	$(PYTHON) scripts/release_provenance.py check-build --platform macos --artifact "$(MAC_APP)" --provenance "$(MAC_BUILD_PROVENANCE)"
	@printf "\nApp macOS OK: $(MAC_APP)\n"

build-release: build-mac build-windows release-artifacts

release-artifacts: mac-app-check windows-artifact-check
	@test -d "$(MAC_APP)" || (printf "App Mac nao encontrado: $(MAC_APP). Rode make build-mac.\n"; exit 1)
	@test -f "$(WINDOWS_ARTIFACT)" || (printf "ZIP Windows nao encontrado: $(WINDOWS_ARTIFACT). Rode make build-windows.\n"; exit 1)
	$(PYTHON) scripts/package_release_artifacts.py --app-name "$(APP_NAME)" --mac-app "$(MAC_APP)" --mac-provenance "$(MAC_BUILD_PROVENANCE)" --windows-zip "$(WINDOWS_ARTIFACT)" --windows-provenance "$(WINDOWS_BUILD_PROVENANCE)" --out "$(RELEASE_DIR)"
	@printf "\nArtefatos de release prontos em $(RELEASE_DIR)/\n"

release-stage: release-artifacts
	git add -f "$(MAC_RELEASE_ARTIFACT)" "$(WINDOWS_RELEASE_ARTIFACT)" "$(MAC_RELEASE_PROVENANCE)" "$(WINDOWS_RELEASE_PROVENANCE)" "$(RELEASE_DIR)/SHA256SUMS" "$(RELEASE_DIR)/release-manifest.json"
	@printf "\nArtefatos adicionados ao Git com -f. Use só se quiser versionar binarios no repo.\n"

release-github: release-artifacts
	@test -n "$(TAG)" || (printf "Informe a tag: make release-github TAG=v0.1.0\n"; exit 1)
	@test -x "$$(command -v gh)" || (printf "GitHub CLI nao encontrado. Instale gh ou publique manualmente os arquivos de $(RELEASE_DIR)/.\n"; exit 1)
	$(PYTHON) scripts/release_provenance.py check-tag --tag "$(TAG)" --manifest "$(RELEASE_DIR)/release-manifest.json" --remote origin
	gh release create "$(TAG)" "$(MAC_RELEASE_ARTIFACT)" "$(WINDOWS_RELEASE_ARTIFACT)" "$(MAC_RELEASE_PROVENANCE)" "$(WINDOWS_RELEASE_PROVENANCE)" "$(RELEASE_DIR)/SHA256SUMS" "$(RELEASE_DIR)/release-manifest.json" --verify-tag --title "$(APP_NAME) $(TAG)" --notes "Builds Mac e Windows do ArenaAI."

release-clean:
	rm -rf "$(RELEASE_DIR)"

clean-build:
	rm -rf build dist *.spec
