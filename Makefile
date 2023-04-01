.DEFAULT_GOAL := help

##@ Utility
help: ## Display this help. (Default)
# based on "https://gist.github.com/prwhite/8168133?permalink_comment_id=4260260#gistcomment-4260260"
	@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

##@ Utility
help_sort: ## Display alphabetized version of help.
	@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

app_build: ## Build pyrovelocity application container image.
	docker build -t pyrovelocityapp -f dockerfiles/Dockerfile.app .

app_run: ## Run the pyrovelocity web user interface.
app_run: \
app_build
	docker run -p 8080:8080 pyrovelocityapp

app_dev: ## Run the pyrovelocity web user interface in development mode.
app_dev: \
# app_build
	docker run -it \
		-v ${PWD}/app:/pyrovelocity \
		-p 8080:8080 \
		--label=pyrovelocityappdev \
		pyrovelocityapp

app_shell: ## Run a shell inside the pyrovelocity application container image.
app_shell: \
app_build
	docker run -it --entrypoint /bin/bash pyrovelocityapp
