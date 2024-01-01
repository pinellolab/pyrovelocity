.DEFAULT_GOAL := help

ENV_PREFIX ?= ./
ENV_FILE := $(wildcard $(ENV_PREFIX)/.env)

ifeq ($(strip $(ENV_FILE)),)
$(info $(ENV_PREFIX)/.env file not found, skipping inclusion)
else
include $(ENV_PREFIX)/.env
export
endif

GIT_SHORT_SHA = $(shell git rev-parse --short HEAD)
GIT_BRANCH = $(shell git rev-parse --abbrev-ref HEAD)

##@ Utility
help: ## Display this help. (Default)
# based on "https://gist.github.com/prwhite/8168133?permalink_comment_id=4260260#gistcomment-4260260"
	@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

##@ Utility
help_sort: ## Display alphabetized version of help.
	@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

#--------
# package
#--------

test: ## Run tests. See pyproject.toml for configuration.
	poetry run pytest

test-cov-xml: ## Run tests with coverage
	poetry run pytest --cov-report=xml

lint: ## Run linter
	poetry run ruff format .
	poetry run ruff --fix .

lint-check: ## Run linter in check mode
	poetry run ruff format --check .
	poetry run ruff .

typecheck: ## Run typechecker
	poetry run pyright
	
docs-build: ## Build documentation
	poetry run mkdocs build

docs-serve: ## Serve documentation
docs-serve: docs-build
	poetry run mkdocs serve

lock: ## Lock dependencies.
	poetry lock --no-update

export_pip_requirements: ## Export requirements.txt for pip.
export_pip_requirements: lock
	poetry export \
	--format=requirements.txt \
	--with=test \
	--output=requirements.txt \
	--without-hashes


#--------------
# setup dev env
#--------------

uninstall_nix: ## Uninstall nix.
	(cat /nix/receipt.json && \
	/nix/nix-installer uninstall) || echo "nix not found, skipping uninstall"

install_nix: ## Install nix. Check script before execution: https://install.determinate.systems/nix .
install_nix: uninstall_nix
	@which nix > /dev/null || \
	curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install

install_direnv: ## Install direnv to `/usr/local/bin`. Check script before execution: https://direnv.net/ .
	@which direnv > /dev/null || \
	(curl -sfL https://direnv.net/install.sh | bash && \
	sudo install -c -m 0755 direnv /usr/local/bin && \
	rm -f ./direnv)
	@echo "see https://direnv.net/docs/hook.html"

setup_dev: ## Setup nix development environment.
setup_dev: install_direnv install_nix
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

#-------------
# system / dev
#-------------

install_just: ## Install just. Check script before execution: https://just.systems/ .
	@which cargo > /dev/null || (curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh)
	@cargo install just

install_poetry: ## Install poetry. Check script before execution: https://python-poetry.org/docs/#installation .
	@which poetry > /dev/null || (curl -sSL https://install.python-poetry.org | python3 -)

install_crane: ## Install crane. Check docs before execution: https://github.com/google/go-containerregistry/blob/main/cmd/crane/doc/crane.md .
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

precommit: ## Run pre-commit hooks using nox.
	nox -x -rs pre-commit

env_print: ## Print a subset of environment variables defined in ".env" file.
	env | grep "TF_VAR\|GITHUB\|GH_\|GCP_\|MLFLOW|FLYTE\|WORKFLOW" | sort

# gh secret set GOOGLE_APPLICATION_CREDENTIALS_DATA --repo="$(GH_REPO)" --body='$(shell cat $(GCP_GACD_PATH))'
ghsecrets: ## Update github secrets for GH_REPO from ".env" file.
	@echo "secrets before updates:"
	@echo
	PAGER=cat gh secret list --repo=$(GH_REPO)
	@echo
	gh secret set FLYTE_CLUSTER_ENDPOINT --repo="$(GH_REPO)" --body="$(FLYTE_CLUSTER_ENDPOINT)"
	gh secret set FLYTE_OAUTH_CLIENT_SECRET --repo="$(GH_REPO)" --body="$(FLYTE_OAUTH_CLIENT_SECRET)"
	gh secret set FLYTECTL_CONFIG --repo="$(GH_REPO)" --body="$(FLYTECTL_CONFIG)"
	gh secret set CODECOV_TOKEN --repo="$(GH_REPO)" --body="$(CODECOV_TOKEN)"
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

update_config: ## Update flytectl config file from template.
	yq e '.admin.endpoint = strenv(FLYTE_CLUSTER_ENDPOINT) | .storage.stow.config.project_id = strenv(GCP_PROJECT_ID) | .storage.stow.config.scopes = strenv(GCP_STORAGE_SCOPES) | .storage.container = strenv(GCP_STORAGE_CONTAINER)' \
	$(FLYTECTL_CONFIG_TEMPLATE) > $(FLYTECTL_CONFIG)

approve_prs: ## Approve github pull requests from bots: PR_ENTRIES="2-5 10 12-18"
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


#----------------
# web application
#----------------

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
app_build: ## Cloud build of pyrovelocity application container image.
	docker build \
	--progress=plain \
	-t pyrovelocityapp \
	-f dockerfiles/Dockerfile.app .

local_app_build: ## Local build of pyrovelocity application container image.
	docker build \
	--progress=plain \
	-t pyrovelocityapp \
	-f dockerfiles/Dockerfile.app.local .

cloud_build: ## Build with google cloud build: make cloud_build PROJECT_ID="gcp-projectID"
ifdef PROJECT_ID
	dvc stage list \
    --name-only reproducibility/figures/dvc.yaml | \
    grep -E "summarize" | \
    xargs -t -I {} dvc pull {}
	PROJECT_ID=$(PROJECT_ID) gcloud builds submit
else
	@echo 'Run "make help" and define PROJECT_ID'
endif

app_run: ## Run the pyrovelocity web user interface.
app_run: \
app_build
	docker run --rm -p 8080:8080 pyrovelocityapp

app_dev: ## Run the pyrovelocity web user interface in development mode.
app_dev: \
# app_build
	docker run --rm -it \
		-v ${PWD}:/pyrovelocity \
		-p 8080:8080 \
		--label=pyrovelocityappdev \
		pyrovelocityapp \
		"--server.enableCORS=false" \
		"--server.enableXsrfProtection=false"

app_run_shell: ## Attach to shell in running container: make app_run_shell CONTAINER_ID="98aca71ab536"
ifdef CONTAINER_ID
	docker exec -it $(CONTAINER_ID) /bin/bash
else
	@echo 'Run "make help" and define CONTAINER_ID'
endif

app_shell: ## Run a shell inside the pyrovelocity application container image.
app_shell: \
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
