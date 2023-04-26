.DEFAULT_GOAL := help

##@ Utility
help: ## Display this help. (Default)
# based on "https://gist.github.com/prwhite/8168133?permalink_comment_id=4260260#gistcomment-4260260"
	@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

##@ Utility
help_sort: ## Display alphabetized version of help.
	@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

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
app_build: ## Build pyrovelocity application container image.
	docker build \
	--progress=plain \
	-t pyrovelocityapp \
	-f dockerfiles/Dockerfile.app .

local_app_build: ## Build pyrovelocity application container image.
	docker build \
	--progress=plain \
	-t pyrovelocityapp \
	-f dockerfiles/Dockerfile.app.local .

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
