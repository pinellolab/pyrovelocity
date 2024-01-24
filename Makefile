.DEFAULT_GOAL := help

ENV_PREFIX ?= ./
ENV_FILE := $(wildcard $(ENV_PREFIX)/.env)

ifeq ($(strip $(ENV_FILE)),)
$(info $(ENV_PREFIX)/.env file not found, skipping inclusion)
else
include $(ENV_PREFIX)/.env
export
endif

GIT_SHA_SHORT = $(shell git rev-parse --short HEAD)
GIT_REF = $(shell git rev-parse --abbrev-ref HEAD)

#-------
##@ help
#-------

# based on "https://gist.github.com/prwhite/8168133?permalink_comment_id=4260260#gistcomment-4260260"
help: ## Display this help. (Default)
	@grep -hE '^(##@|[A-Za-z0-9_ \-]*?:.*##).*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; /^##@/ {print "\n" substr($$0, 5)} /^[A-Za-z0-9_ \-]*?:.*##/ {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

help-sort: ## Display alphabetized version of help (no section headings).
	@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | sort | \
	awk 'BEGIN {FS = ":.*?## "}; /^[A-Za-z0-9_ \-]*?:.*##/ {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

HELP_TARGETS_PATTERN ?= test
help-targets: ## Print commands for all targets matching a given pattern. eval "$(make help-targets HELP_TARGETS_PATTERN=test | sed 's/\x1b\[[0-9;]*m//g')"
	@make help-sort | awk '{print $$1}' | grep '$(HELP_TARGETS_PATTERN)' | xargs -I {} printf "printf '___\n\n{}:\n\n'\nmake -n {}\nprintf '\n'\n"

#-------------------------
##@ primary python package
#-------------------------

test: ## Run tests. See pyproject.toml for configuration.
	poetry run pytest

test-bazel: ## Run tests with Bazel.
test-bazel:
	bazel test //src/...

test-bazel-debug: ## Run tests with Bazel in debug mode. See .aspect/bazelrc/debug.bazelrc.
	bazel test --config=debug //src/...

test-bazel-nodocs: ## Run tests with Bazel excluding doctests.
	bazel test //src/... -- -//src/pyrovelocity:xdoctest

test-cov-xml: ## Run tests with coverage
	poetry run pytest --cov-report=xml

precommit-run: ## Run pre-commit hooks
	pre-commit run -a

precommit-install:
	pre-commit install -t pre-commit
	pre-commit install -t post-commit
	pre-commit install -t post-checkout

lint: ## Run linter
	poetry run ruff format .
	poetry run ruff --fix .

lint-check: ## Run linter in check mode
	poetry run ruff format --check .
	poetry run ruff .

typecheck: ## Run typechecker
	poetry run pyright
	
docs-build: ## Build documentation
	poetry run sphinx-build docs site

docs-serve: ## Serve documentation
docs-serve:
	poetry run sphinx-autobuild docs site --open-browser

lock-poetry: ## Lock poetry dependencies.
	poetry lock --no-update

PIP_REQUIREMENTS_NAME ?= requirements

lock-pip: ## Export requirements.txt for pip.
lock-pip:
	poetry export \
	--format=requirements.txt \
	--with=test \
	--with=workflows \
	--output=$(PIP_REQUIREMENTS_NAME).txt \
	--without-hashes
	poetry export \
	--format=requirements.txt \
	--with=test \
	--with=workflows \
	--output=$(PIP_REQUIREMENTS_NAME)-hashed.txt

lock-pip-cpu: ## Export requirements-cpu.txt for pip.
lock-pip-cpu: lock-poetry
	make lock-pip PIP_REQUIREMENTS_NAME=requirements-cpu

lock-conda: ## Export environment yaml and lock files for conda. (see pyproject.toml).
	poe conda-lock

lock-bazel: ## Export requirements-bazel.txt for bazel.
	touch requirements-bazel.txt
	bazel run //:requirements.update
	make cache-requirements-bazel

# make lock-bazel
lock: ## Lock poetry, pip, and conda lock files.
lock: lock-poetry 
	make lock-pip 
	make lock-conda
	@echo "updated poetry, pip, bazel, and conda lock files"

meta-bazel: ## Print bazel meta information.
meta-bazel:
	bazel version
	bazel info
	bazel query /...

clean-bazel: ## Clean local and remote bazel build caches.
	bazel clean --async
	gsutil -m rm gs://pyrovelocity/build/**

set-requirements-bazel: ## Set bazel python requirements from OS specific requirements.
	@if [ "$$(uname -s)" = "Darwin" ]; then \
		cp requirements-darwin.txt requirements-bazel.txt; \
		echo "Copied requirements-darwin.txt to requirements-bazel.txt"; \
	elif [ "$$(uname -s)" = "Linux" ]; then \
		cp requirements-linux.txt requirements-bazel.txt; \
		echo "Copied requirements-linux.txt to requirements-bazel.txt"; \
	else \
		echo "OS unsupported for automatic copying of Bazel python requirements."; \
	fi

cache-requirements-bazel: ## Cache bazel python requirements as OS specific requirements.
	@if [ "$$(uname -s)" = "Darwin" ]; then \
		cp requirements-bazel.txt requirements-darwin.txt; \
		echo "Cached requirements-bazel.txt as requirements-darwin.txt"; \
		PAGER=cat git diff requirements-darwin.txt || true; \
	elif [ "$$(uname -s)" = "Linux" ]; then \
		cp requirements-bazel.txt requirements-linux.txt; \
		echo "Cached requirements-bazel.txt as requirements-linux.txt"; \
		PAGER=cat git diff requirements-linux.txt || true; \
	else \
		echo "OS unsupported for automatic copying of Bazel python requirements."; \
	fi

#---------------------
##@ workflow execution
#---------------------

run_help: ## Print hydra help for execute script.
	poetry run pyrovelocity --help

# Capture additional arguments to pass to hydra-zen cli
# converting them to make do-nothing targets
# supports passing hydra overrides as ARGS, e.g.:
#   make run HYDRA_OVERRIDES="entity_config.inputs.logistic_regression.max_iter=2000 execution_context=local_shell"
HYDRA_OVERRIDES = $(filter-out $@,$(MAKECMDGOALS))
%:
	@:

.PHONY: run
run: ## Run registered workflow in remote dev mode. (default)
	poetry run pyrovelocity $(HYDRA_OVERRIDES)

run-dev: ## Run registered workflow in remote dev mode. (ci default)
	poetry run pyrovelocity execution_context=remote_dev $(HYDRA_OVERRIDES)

run-prod: ## Run registered workflow in remote prod mode. (ci default)
	poetry run pyrovelocity execution_context=remote_prod $(HYDRA_OVERRIDES)

run-local-cluster: ## Run registered workflow in local cluster dev mode.
	poetry run pyrovelocity execution_context=local_cluster_dev $(HYDRA_OVERRIDES)

run-local: ## Run registered workflow in local shell mode. (only with all python tasks)
	poetry run pyrovelocity execution_context=local_shell $(HYDRA_OVERRIDES)

run-async: ## Run registered workflow (async).
	poetry run pyrovelocity execution_context.wait=False

run-check-image: ## Check workflow image exists.
	crane ls $(WORKFLOW_IMAGE) | grep "$(GIT_REF)\|$(GIT_SHA)\|$(GIT_SHA_SHORT)"

#---------
##@ github
#---------

browse: ## Open github repo in browser at HEAD commit.
	gh browse $(GIT_SHA_SHORT)

GH_ACTIONS_DEBUG ?= false

cid: ## Run CID (GH_ACTIONS_DEBUG default is false).
	gh workflow run "CID" --ref $(GIT_REF) -f debug_enabled=$(GH_ACTIONS_DEBUG)

build-images: ## Run Build Images (GH_ACTIONS_DEBUG default is false).
	gh workflow run "Build Images" --ref $(GIT_REF) -f debug_enabled=$(GH_ACTIONS_DEBUG)

ci-view-workflow: ## Open CI workflow summary.
	gh workflow view "CI"

build-images-view-workflow: ## Open Build Images workflow summary.
	gh workflow view "Build Images"

# CPU | MEM | DISK | MACHINE_TYPE
# ----|-----|------|----------------
#   2 |   8 |   32 | basicLinux32gb
#   4 |  16 |   32 | standardLinux32gb
#   8 |  32 |   64 | premiumLinux
#  16 |  64 |  128 | largePremiumLinux
MACHINE_TYPE ?= premiumLinux
codespace-create: ## Create codespace. make -n codespace_create MACHINE_TYPE=largePremiumLinux
	gh codespace create -R $(GH_REPO) -b $(GIT_REF) -m $(MACHINE_TYPE)

code: ## Open codespace in browser.
	gh codespace code -R $(GH_REPO) --web

codespace-list: ## List codespace.
	PAGER=cat gh codespace list

codespace-stop: ## Stop codespace.
	gh codespace stop

codespace-delete: ## Delete codespace.
	gh codespace delete

docker-login: ## Login to ghcr docker registry. Check regcreds in $HOME/.docker/config.json.
	docker login ghcr.io -u $(GH_ORG) -p $(GITHUB_TOKEN)

# gh secret set GOOGLE_APPLICATION_CREDENTIALS_DATA --repo="$(GH_REPO)" --body='$(shell cat $(GCP_GACD_PATH))'
ghsecrets: ## Update github secrets for GH_REPO from ".env" file.
	@echo "secrets before updates:"
	@echo
	PAGER=cat gh secret list --repo=$(GH_REPO)
	@echo
	gh secret set FLYTE_CLUSTER_ENDPOINT --repo="$(GH_REPO)" --body="$(FLYTE_CLUSTER_ENDPOINT)"
	gh secret set FLYTE_OAUTH_CLIENT_SECRET --repo="$(GH_REPO)" --body="$(FLYTE_OAUTH_CLIENT_SECRET)"
	gh secret set FLYTECTL_CONFIG --repo="$(GH_REPO)" --body="$(FLYTECTL_CONFIG)"
	# gh secret set CODECOV_TOKEN --repo="$(GH_REPO)" --body="$(CODECOV_TOKEN)"
	gh secret set GCP_PROJECT_ID --repo="$(GH_REPO)" --body="$(GCP_PROJECT_ID)"
	gh secret set GCP_STORAGE_SCOPES --repo="$(GH_REPO)" --body="$(GCP_STORAGE_SCOPES)"
	gh secret set GCP_STORAGE_CONTAINER --repo="$(GH_REPO)" --body="$(GCP_STORAGE_CONTAINER)"
	gh secret set GCP_ARTIFACT_REGISTRY_PATH --repo="$(GH_REPO)" --body="$(GCP_ARTIFACT_REGISTRY_PATH)"
	@echo
	@echo secrets after updates:
	@echo
	PAGER=cat gh secret list --repo=$(GH_REPO)

ghvars: ## Update github secrets for GH_REPO from ".env" file.
	@echo "variables before updates:"
	@echo
	PAGER=cat gh variable list --repo=$(GH_REPO)
	@echo
	gh variable set WORKFLOW_IMAGE --repo="$(GH_REPO)" --body="$(WORKFLOW_IMAGE)"
	@echo
	@echo variables after updates:
	@echo
	PAGER=cat gh variable list --repo=$(GH_REPO)

EXISTING_IMAGE_TAG ?= main
NEW_IMAGE_TAG ?= $(GIT_REF)

# Default bumps main to the checked out branch for dev purposes
tag-images: ## Add tag to existing images, (default main --> branch, override with make -n tag_images NEW_IMAGE_TAG=latest).
	crane tag $(WORKFLOW_IMAGE):$(EXISTING_IMAGE_TAG) $(NEW_IMAGE_TAG)
	crane tag ghcr.io/$(GH_ORG)/$(GH_REPO):$(EXISTING_IMAGE_TAG) $(NEW_IMAGE_TAG)

list-gcr-workflow-image-tags: ## List images in gcr.
	gcloud container images list --repository=$(GCP_ARTIFACT_REGISTRY_PATH)                                                                                                                             │
	gcloud container images list-tags $(WORKFLOW_IMAGE)

#------
##@ nix
#------

meta: ## Generate nix flake metadata.
	nix flake metadata --impure --accept-flake-config
	nix flake show --impure --accept-flake-config

up: ## Update nix flake lock file.
	nix flake update --impure --accept-flake-config
	nix flake check --impure --accept-flake-config

dup: ## Debug update nix flake lock file.
	nix flake update --impure --accept-flake-config
	nix flake check --show-trace --print-build-logs --impure --accept-flake-config

nix-lint: ## Lint nix files.
	nix fmt

NIX_DERIVATION_PATH ?= $(shell which python)

closure-size: ## Print nix closure size for a given path. make -n NIX_DERIVATION_PATH=$(shell which python)
	nix path-info -Sh $(NIX_DERIVATION_PATH)

re: ## Reload direnv.
	direnv reload

al: ## Enable direnv.
	direnv allow

devshell-info: ## Print devshell info.
	nix build .#devShells.$(shell nix eval --impure --expr 'builtins.currentSystem').default --impure
	nix path-info --recursive ./result
	du -chL ./result
	rm ./result

cache: ## Push devshell to cachix
	nix build --json \
	.#devShells.$(shell nix eval --impure --expr 'builtins.currentSystem').default \
	--impure \
	--accept-flake-config | \
	jq -r '.[].outputs | to_entries[].value' | \
	cachix push $(CACHIX_CACHE_NAME)

container-script: ## Build devcontainer build script.
	nix build .#containerStream --accept-flake-config --impure --show-trace
	cat ./result

container: ## Build container.
container: container-script
	./result | docker load

devcontainer-script: ## Build devcontainer build script.
	nix build .#devcontainerStream --accept-flake-config --impure --show-trace
	cat ./result

# Or the rough equivalent with nix2container
# nix run .#devcontainerNix2Container.copyToDockerDaemon --accept-flake-config --impure
devcontainer: ## Build devcontainer.
devcontainer: devcontainer-script
	./result | docker load

# The default value for DEVCONTAINER_IMAGE FQN can be completely overriden to
# support specification of tags or digests (see .example.env and create .env)
DEVCONTAINER_IMAGE ?= ghcr.io/pinellolab/pyrovelocitydev
# DEVCONTAINER_IMAGE=ghcr.io/pinellolab/pyrovelocitydev:main
# DEVCONTAINER_IMAGE=ghcr.io/pinellolab/pyrovelocitydev@sha256:

drundc: ## Run devcontainer. make drundc DEVCONTAINER_IMAGE=
	docker run --rm -it $(DEVCONTAINER_IMAGE)

adhocpkgs: ## Install adhoc nix packages. make adhocpkgs ADHOC_NIX_PKGS="gnugrep fzf"
	nix profile list
	$(foreach pkg, $(ADHOC_NIX_PKGS), nix profile install nixpkgs#$(pkg);)
	nix profile list

findeditable: ## Find *-editable.pth files in the nix store.
	rg --files --glob '*editable.pth' --hidden --no-ignore --follow /nix/store/

image-digests: ## Print image digests.
	@echo
	docker images -a --digests $(DEVCONTAINER_IMAGE)
	@echo

.PHONY: digest
digest: ## Print image digest from tag. make digest DEVCONTAINER_IMAGE=
	@echo
	docker inspect --format='{{index .RepoDigests 0}}' $(DEVCONTAINER_IMAGE)
	@echo

#----------------
##@ vscode server
#----------------

VSCODE_EXTENSIONS := \
	"vscodevim.vim" \
	"Catppuccin.catppuccin-vsc" \
	"jnoortheen.nix-ide" \
	"tamasfe.even-better-toml" \
	"donjayamanne.python-extension-pack" \
	"charliermarsh.ruff" \
	"redhat.vscode-yaml" \
	"ms-kubernetes-tools.vscode-kubernetes-tools" \
	"eamodio.gitlens" \
	"GitHub.vscode-pull-request-github" \
	"ms-azuretools.vscode-docker" \
	"ms-toolsai.jupyter" \
	"njzy.stats-bar" \
	"vscode-icons-team.vscode-icons"

vscode-install-extensions: ## Install vscode extensions.
	@echo "Listing currently installed extensions..."
	@openvscode-server --list-extensions --show-versions
	@echo ""
	@$(foreach extension,$(VSCODE_EXTENSIONS),openvscode-server --install-extension $(extension) --force; echo "";)
	@echo "Listing extensions after installation..."
	@openvscode-server --list-extensions --show-versions

vscode-server: ## Run vscode server.
	openvscode-server --host 0.0.0.0 --without-connection-token .


#-----------------
##@ jupyter server
#-----------------

.PHONY: jupyter
jupyter: ## Run jupyter lab in devcontainer. make jupyter DEVCONTAINER_IMAGE=ghcr.io/pinellolab/pyrovelocitydev@sha256:
	@echo "Attempting to start jupyter lab in"
	@echo
	@echo "DEVCONTAINER_IMAGE: $(DEVCONTAINER_IMAGE)"
	@echo
	docker compose -f containers/compose.yaml up -d jupyter
	@echo
	$(MAKE) jupyter-logs

jupyter-logs: ## Print docker-compose logs.
	@echo
	@echo "Ctrl/cmd + click the http://127.0.0.1:8888/lab?token=... link to open jupyter lab in your default browser"
	@echo
	@trap 'printf "\n  use \`make jupyter-logs\` to reattach to logs or \`make jupyter-down\` to terminate\n\n"; exit 2' SIGINT; \
	while true; do \
		docker compose -f containers/compose.yaml logs -f jupyter; \
	done

jupyter-down: compose_list
jupyter-down: ## Stop docker-compose containers.
	docker compose -f containers/compose.yaml down jupyter
	$(MAKE) compose_list

compose-list: ## List docker-compose containers.
	@echo
	docker compose ls
	@echo
	docker compose -f containers/compose.yaml ps --services
	@echo
	docker compose -f containers/compose.yaml ps
	@echo

jupyter-manual: ## Prefer `make -n jupyter` to this target. make jupyter-manual DEVCONTAINER_IMAGE=
	docker run --rm -it -p 8888:8888 \
	$(DEVCONTAINER_IMAGE) \
	jupyter lab --allow-root --ip=0.0.0.0 /root/pyrovelocity

jupyter-local: ## Run jupyter lab locally. See make -n setup_dev.
	SHELL=zsh \
	jupyter lab \
	--ServerApp.terminado_settings="shell_command=['zsh']" \
	--allow-root \
	--ip=0.0.0.0 ./


#--------------------------------------
##@ setup local development environment
#--------------------------------------

uninstall-nix: ## Uninstall nix.
	(cat /nix/receipt.json && \
	/nix/nix-installer uninstall) || echo "nix not found, skipping uninstall"

install-nix: ## Install nix. Check script before execution: https://install.determinate.systems/nix .
install-nix: uninstall-nix
	@which nix > /dev/null || \
	curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install

install-direnv: ## Install direnv to `/usr/local/bin`. Check script before execution: https://direnv.net/ .
	@which direnv > /dev/null || \
	(curl -sfL https://direnv.net/install.sh | bash && \
	sudo install -c -m 0755 direnv /usr/local/bin && \
	rm -f ./direnv)
	@echo "see https://direnv.net/docs/hook.html"

setup-dev: ## Setup nix development environment.
setup-dev: install-direnv install-nix
	@. /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh && \
	nix profile install nixpkgs#cachix && \
	echo "trusted-users = root $$USER" | sudo tee -a /etc/nix/nix.conf && sudo pkill nix-daemon && \
	cachix use devenv

.PHONY: devshell
devshell: ## Enter nix devshell. See use_flake in `direnv stdlib`.
	./scripts/flake

cdirenv: ## !!Enable direnv in zshrc.!!
	@if ! grep -q 'direnv hook zsh' "${HOME}/.zshrc"; then \
		printf '\n%s\n' 'eval "$$(direnv hook zsh)"' >> "${HOME}/.zshrc"; \
	fi

cstarship: ## !!Enable starship in zshrc.!!
	@if ! grep -q 'starship init zsh' "${HOME}/.zshrc"; then \
		printf '\n%s\n' 'eval "$$(starship init zsh)"' >> "${HOME}/.zshrc"; \
	fi

catuin: ## !!Enable atuin in zshrc.!!
	@if ! grep -q 'atuin init zsh' "${HOME}/.zshrc"; then \
		printf '\n%s\n' 'eval "$$(atuin init zsh)"' >> "${HOME}/.zshrc"; \
	fi

czsh: ## !!Enable zsh with command line info and searchable history.!!
czsh: catuin cstarship cdirenv


#-------------------------
##@ in-cluster development
#-------------------------

CLUSTER_DEV_IMAGE_TAG ?= $(GIT_REF)
CLUSTER_DEV_DEPLOYMENT_NAME ?= $(GH_REPO_NAME)

cluster-config-export: ## Export kube config for cluster in current context.
	kubectl config view --minify --flatten > $(CLUSTER_DEV_CONFIG)

cluster-config: ## Set kube context for cluster in CLUSTER_DEV_CONFIG.
	$(eval CLUSTER_DEV_CONTEXT_NAME=$(shell kubectl config view --kubeconfig='$(CLUSTER_DEV_CONFIG)' -o jsonpath='{.contexts[0].name}'))
	kubectl config use-context --kubeconfig=$(CLUSTER_DEV_CONFIG) $(CLUSTER_DEV_CONTEXT_NAME)

CLUSTER_DEV_MODULE_PATH ?=./dev/cluster/pyrovelocity/pyrovelocitydev

cluster-dev-lint: ## Lint dev module.
	cue fmt dev.cue dev.example.cue
	cue fmt $(CLUSTER_DEV_MODULE_PATH)/...
	timoni mod vet $(CLUSTER_DEV_MODULE_PATH)

CLUSTER_DEV_INSTANCE_NAME ?= dev
CUE_DEV_VALUES ?= dev.cue

cluster-dev-render: ## Render dev package yaml.
	timoni build $(CLUSTER_DEV_INSTANCE_NAME) $(CLUSTER_DEV_MODULE_PATH) \
	-n $(CLUSTER_DEV_NAMESPACE) \
	-f $(CUE_DEV_VALUES) > $(CLUSTER_DEV_MODULE_PATH)/manifest.yaml
	@if command -v bat > /dev/null; then \
		bat -P -l yaml $(CLUSTER_DEV_MODULE_PATH)/manifest.yaml; \
	else \
		cat $(CLUSTER_DEV_MODULE_PATH)/manifest.yaml; \
	fi
	@echo "bat -pp -l yaml $(CLUSTER_DEV_MODULE_PATH)/manifest.yaml"

cluster-dev-apply-check: ## Check dev package deployment status.
cluster-dev-apply-check: cluster-dev-render
	timoni apply $(CLUSTER_DEV_INSTANCE_NAME) $(CLUSTER_DEV_MODULE_PATH) \
	-n $(CLUSTER_DEV_NAMESPACE) \
	-f $(CUE_DEV_VALUES) \
	--diff

cluster-dev-apply: ## Deploy dev package resources.
cluster-dev-apply: cluster-dev-render
	timoni apply $(CLUSTER_DEV_INSTANCE_NAME) $(CLUSTER_DEV_MODULE_PATH) \
	-n $(CLUSTER_DEV_NAMESPACE) \
	-f $(CUE_DEV_VALUES) \
	--timeout 12m0s

cluster-dev-delete-check: ## Delete dev package resources.
cluster-dev-delete-check: cluster-dev-render
	timoni delete $(CLUSTER_DEV_INSTANCE_NAME) \
	-n $(CLUSTER_DEV_NAMESPACE) \
	--dry-run

cluster-dev-delete: ## Delete dev package resources.
cluster-dev-delete: cluster-dev-render
	timoni delete $(CLUSTER_DEV_INSTANCE_NAME) \
	-n $(CLUSTER_DEV_NAMESPACE)

cluster-dev-package-test: ## Test oci packaging of dev module.
	docker container stop registry || true
	docker run --rm -d -p 5001:5000 --name registry registry:2
	timoni mod push $(CLUSTER_DEV_MODULE_PATH) \
	oci://localhost:5001/deploydev --version=0.0.0-dev1 --latest=false
	skopeo copy --src-tls-verify=false docker://localhost:5001/deploydev:0.0.0-dev1 oci:deploydev_oci:0.0.0-dev1
	tree --du -ah deploydev_oci
	@for tarball in deploydev_oci/blobs/sha256/*; do \
		if file $$tarball | grep -q 'gzip compressed data'; then \
			echo "Contents of $$tarball:"; \
			tar -tzf $$tarball; \
			echo "----------------------"; \
		else \
			echo "$$tarball is not a gzipped tarball."; \
		fi \
	done
	
cluster-deploy: ## Deploy latest container_image in current kube context (invert: terminate)
	skaffold deploy

cluster-stop: ## Stop latest container_image in current kube context (invert: start)
	kubectl scale deployment/$(CLUSTER_DEV_DEPLOYMENT_NAME) --replicas=0 -n $(CLUSTER_DEV_DEPLOYMENT_NAME)

cluster-start: ## Start latest container_image in current kube context (invert: stop)
	kubectl scale deployment/$(CLUSTER_DEV_DEPLOYMENT_NAME) --replicas=1 -n $(CLUSTER_DEV_DEPLOYMENT_NAME)

cluster-terminate: ## Delete deployment for container_image in current kube context (invert: deploy)
	kubectl delete -f cluster/resources/deployment.yaml

cluster-delete: ## Delete all resources created by skaffold
	skaffold delete

cluster-info: ## Print skaffold info
	skaffold version
	skaffold --help
	skaffold options
	skaffold config list
	skaffold diagnose

cluster-render: ## Render skaffold yaml with latest container_image
	skaffold render

cluster-build: ## Build image with skaffold (disabled by default: see skaffold.yaml)
	skaffold build


#----------------------------
##@ extra system dependencies
#----------------------------

# https://github.com/GoogleContainerTools/skaffold/releases
SKAFFOLD_RELEASE ?= latest # v2.9.0

ifeq ($(SKAFFOLD_RELEASE),latest)
	SKAFFOLD_BINARY_URL := "https://github.com/GoogleContainerTools/skaffold/releases/latest/download/skaffold-$(shell uname -s)-$(shell uname -m)"
else
	SKAFFOLD_BINARY_URL := "https://github.com/GoogleContainerTools/skaffold/releases/download/$(SKAFFOLD_RELEASE)/skaffold-$(shell uname -s)-$(shell uname -m)"
endif

install-skaffold: ## Install skaffold. (check/set: SKAFFOLD_RELEASE).
	curl -L -o skaffold $(SKAFFOLD_BINARY_URL) && \
	sudo install -c -m 0755 skaffold /usr/local/bin && \
	rm -f skaffold
	which skaffold
	skaffold version

install-just: ## Install just. Check script before execution: https://just.systems/ .
	@which cargo > /dev/null || (curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh)
	@cargo install just

install-poetry: ## Install poetry. Check script before execution: https://python-poetry.org/docs/#installation .
	@which poetry > /dev/null || (curl -sSL https://install.python-poetry.org | python3 -)

install-crane: ## Install crane. Check docs before execution: https://github.com/google/go-containerregistry/blob/main/cmd/crane/doc/crane.md .
	@which crane > /dev/null || ( \
		set -e; \
		CRANE_VERSION="0.16.1"; \
		OS=$$(uname -s | tr '[:upper:]' '[:lower:]'); \
		ARCH=$$(uname -m); \
		case $$ARCH in \
			x86_64|amd64) ARCH="x86_64" ;; \
			aarch64|arm64) ARCH="arm64" ;; \
			*) echo "Unsupported architecture: $$ARCH" && exit 1 ;; \
		esac; \
		TMP_DIR=$$(mktemp -d); \
		trap 'rm -rf "$$TMP_DIR"' EXIT; \
		echo "Downloading crane $$CRANE_VERSION for $$OS $$ARCH to $$TMP_DIR"; \
		FILENAME="go-containerregistry_$$OS"_$$ARCH".tar.gz"; \
		URL="https://github.com/google/go-containerregistry/releases/download/v$$CRANE_VERSION/$$FILENAME"; \
		curl -sSL "$$URL" | tar xz -C $$TMP_DIR; \
		sudo mv $$TMP_DIR/crane /usr/local/bin/crane; \
		echo "Crane installed successfully to /usr/local/bin/crane" \
	)

#------------
##@ utilities
#------------

env-print: ## Print a subset of environment variables defined in ".env" file.
	env | grep "TF_VAR\|GIT\|GH_\|GCP_\|MLFLOW|FLYTE\|WORKFLOW" | sort

update-config: ## Update flytectl config file from template.
	yq e '.admin.endpoint = strenv(FLYTE_CLUSTER_ENDPOINT) | .storage.stow.config.project_id = strenv(GCP_PROJECT_ID) | .storage.stow.config.scopes = strenv(GCP_STORAGE_SCOPES) | .storage.container = strenv(GCP_STORAGE_CONTAINER)' \
	$(FLYTECTL_CONFIG_TEMPLATE) > $(FLYTECTL_CONFIG)

set-git-env: ## Set git environment variables.
	@grep "GIT_.*" .env
	./scripts/git/set-git-env.sh
	@grep "GIT_.*" .env
	@echo ""

tree: ## Print directory tree.
	tree -a --dirsfirst -L 4 -I ".git|.direnv|*pycache*|*ruff_cache*|*pytest_cache*|outputs|multirun|conf|scripts|site|*venv*|.coverage"

approve-prs: ## Approve github pull requests from bots: PR_ENTRIES="2-5 10 12-18"
	for entry in $(PR_ENTRIES); do \
		if [[ "$$entry" == *-* ]]; then \
			start=$${entry%-*}; \
			end=$${entry#*-}; \
			for pr in $$(seq $$start $$end); do \
				@gh pr review $$pr --approve; \
			done; \
		else \
			@gh pr review $$entry --approve; \
		fi; \
	done

PREVIOUS_VERSION := 0.2.0b6
NEXT_VERSION := 0.2.0b7

VERSION_FILES := \
	conda/colab/construct.yaml \
	containers/gpu.Dockerfile \
	containers/pkg.Dockerfile \
	docs/source/notebooks/pyrovelocity_colab_template.ipynb \
	MODULE.bazel \
	src/pyrovelocity/workflows/main_workflow.py

update-version: ## Update version in VERSION_FILES.
	@for file in $(VERSION_FILES); do \
		if [ -f $$file ]; then \
			gsed -i 's/$(PREVIOUS_VERSION)/$(NEXT_VERSION)/g' $$file; \
			echo "Updated $$file"; \
		else \
			echo "$$file does not exist"; \
		fi \
	done

GHA_WORKFLOWS := \
	.github/actions/setup_environment/action.yml \
	.github/workflows/app.yaml \
	.github/workflows/build-images.yaml \
	.github/workflows/cid.yaml \
	.github/workflows/cml-images.yml \
	.github/workflows/cml.yml \
	.github/workflows/colab.yml \
	.github/workflows/labeler.yml

ratchet = docker run -it --rm -v "${PWD}:${PWD}" -w "${PWD}" ghcr.io/sethvargo/ratchet:0.5.1 $1

ratchet-pin: ## Pin all workflow versions to hash values. (requires docker).
	$(foreach workflow,$(GHA_WORKFLOWS),$(call ratchet,pin $(workflow));)

ratchet-unpin: ## Unpin hashed workflow versions to semantic values. (requires docker).
	$(foreach workflow,$(GHA_WORKFLOWS),$(call ratchet,unpin $(workflow));)

ratchet-update: ## Unpin hashed workflow versions to semantic values. (requires docker).
	$(foreach workflow,$(GHA_WORKFLOWS),$(call ratchet,update $(workflow));)

module-deps-graph: ## Generate module dependency graph with pydeps.
	rm -f pyrovelocity.svg pyrovelocity.pdf
	pydeps src/pyrovelocity \
	--max-bacon=3 \
	--cluster \
	--max-cluster-size=10 \
	--min-cluster-size=2 \
	--keep-target-cluster \
	--rankdir TB \
	--no-show \
	--show-dot \
	--exclude pyrovelocity.tests.* pyrovelocity._velocity_model_* pyrovelocity.workflows.lrwine \
	--rmprefix pyrovelocity.
	svg2pdf pyrovelocity.svg pyrovelocity.pdf

make-storage-bucket: ## Create storage bucket.
ifdef GCP_STORAGE_BUCKET
	gsutil mb -p $(GCP_PROJECT_ID) -c standard -l $(GCP_REGION) -b on gs://$(GCP_STORAGE_BUCKET)
else
	@echo 'Run "make help" and define necessary variables'
endif

list-storage-bucket-objects: ## List storage bucket objects.
ifdef GCP_STORAGE_BUCKET
	gsutil ls -lhR gs://$(GCP_STORAGE_BUCKET)
else
	@echo 'Run "make help" and define necessary variables'
endif
	

#------------------------------
##@ web application development
#------------------------------

st: ## Run streamlit app in local environment.
	streamlit run app/app.py \
	--server.port=8080 \
	--server.address=0.0.0.0

stcloud: ## Run streamlit app in local environment.
	streamlit run app/app.py \
	--server.port=8080 \
	--server.enableCORS=false \
	--server.enableXsrfProtection=false \
	--server.address=0.0.0.0

# --progress=plain
# --platform linux/amd64
app-build: ## Cloud build of pyrovelocity application container image.
	docker build \
	--progress=plain \
	-t pyrovelocityapp \
	-f containers/Dockerfile.app .

local-app-build: ## Local build of pyrovelocity application container image.
	docker build \
	--progress=plain \
	-t pyrovelocityapp \
	-f containers/Dockerfile.app.local .

cloud-build: ## Build with google cloud build: make cloud_build PROJECT_ID="gcp-projectID"
ifdef PROJECT_ID
	dvc stage list \
    --name-only reproducibility/figures/dvc.yaml | \
    grep -E "summarize" | \
    xargs -t -I {} dvc pull {}
	PROJECT_ID=$(PROJECT_ID) gcloud builds submit
else
	@echo 'Run "make help" and define PROJECT_ID'
endif

app-run: ## Run the pyrovelocity web user interface.
app-run: \
app_build
	docker run --rm -p 8080:8080 pyrovelocityapp

app-dev: ## Run the pyrovelocity web user interface in development mode.
app-dev: \
# app_build
	docker run --rm -it \
		-v ${PWD}:/pyrovelocity \
		-p 8080:8080 \
		--label=pyrovelocityappdev \
		pyrovelocityapp \
		"--server.enableCORS=false" \
		"--server.enableXsrfProtection=false"

app-run-shell: ## Attach to shell in running container: make app_run_shell CONTAINER_ID="98aca71ab536"
ifdef CONTAINER_ID
	docker exec -it $(CONTAINER_ID) /bin/bash
else
	@echo 'Run "make help" and define CONTAINER_ID'
endif

app-shell: ## Run a shell inside the pyrovelocity application container image.
app-shell: \
# app_build
	docker run --rm -it \
	--entrypoint /bin/bash pyrovelocityapp

deploy: ## Deploy application manually with cloud run
ifdef GCP_RUN_SERVICE_NAME
	gcloud run deploy $(GCP_RUN_SERVICE_NAME) \
	--image=$(PKG_ARCHIVE_URL)/$(PKG_APP):$(PKG_IMAGE_TAG) \
	--platform=managed \
	--project=$(GCP_PROJECT_ID) \
	--region=$(GCP_REGION) \
	--allow-unauthenticated
else
	@echo 'Run "make help" and define necessary variables'
endif
