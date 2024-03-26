.DEFAULT_GOAL := help

ENV_PREFIX ?= .
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
help-targets: ## Print commands for all targets matching a given pattern. eval "$(make help-targets HELP_TARGETS_PATTERN=render | sed 's/\x1b\[[0-9;]*m//g')"
	@make help-sort | awk '{print $$1}' | grep '$(HELP_TARGETS_PATTERN)' | xargs -I {} printf "printf '___\n\n{}:\n\n'\nmake -n {}\nprintf '\n'\n"

#-----------------
##@ render article
#-----------------

DOCUMENT_NAME ?= user_example

render-latex: ## Render the article via LaTeX
	quarto render $(DOCUMENT_NAME).qmd --to nature-pdf

render: ## Render all article formats including pdf, html, and docx
	quarto render $(DOCUMENT_NAME).qmd --to all

update-freeze: ## Update the hash for the _freeze directory
	dvc status
	dvc add ../../_freeze -v
	dvc status

sync-nb: ## Sync the notebook to google drive
	@echo "Syncing $(DOCUMENT_NAME).ipynb to pyrovelocitydrive:pyrovelocity/notebooks"
	cp ../../_site/templates/$(DOCUMENT_NAME)/$(DOCUMENT_NAME).ipynb pyrovelocity_$(DOCUMENT_NAME).ipynb
	rclone copy pyrovelocity_$(DOCUMENT_NAME).ipynb pyrovelocitydrive:pyrovelocity/notebooks
	@$(eval DRIVE_LINK=$(shell rclone link pyrovelocitydrive:pyrovelocity/notebooks/pyrovelocity_$(DOCUMENT_NAME).ipynb))
	@echo ""
	@echo "Google Drive Link: "
	@echo $(DRIVE_LINK)
	@$(eval COLAB_LINK=$(shell echo $(DRIVE_LINK) | awk -F'=' '{print "https://colab.research.google.com/drive/" $$2}'))
	@echo ""
	@echo "Google Colab Link: "
	@echo $(COLAB_LINK)
	@echo
	rm -f pyrovelocity_$(DOCUMENT_NAME).ipynb

sync: ## Sync notebook _freeze data and ipynb to CAS and google drive
	$(MAKE) render
	$(MAKE) update-freeze
	dvc push
	$(MAKE) sync-nb

clean: ## Clean compilation artifacts
	rm sn-*.{bst,cls} || true

clean-all: ## Clean all files including output files
clean-all: clean
	rm $(DOCUMENT_NAME).{tex,docx,ipynb,html,pdf,docx.md,html.md,ipynb.md,pdf.md} || true
	rm -r $(DOCUMENT_NAME)_files/ || true

refresh-workflow-execution: ## Refresh workflow execution environment by removing the caches and output files
refresh-workflow-execution: clean-all
	pyflyte local-cache clear
	rm -rf ./.jupyter_cache/ || true
	rm -rf ./data/ ./models/ ./reports/ || true
