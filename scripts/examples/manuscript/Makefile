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


#-----------------
##@ install extensions
#-----------------

install-extensions: ## Install quarto extensions
	@if [ ! -d "_extensions/cameronraysmith/nature" ]; then \
		quarto add --no-prompt cameronraysmith/nature@citeproc; \
	fi
	@if [ ! -f "bibstyle.csl" ]; then \
		cp _extensions/cameronraysmith/nature/csl/springer-basic-brackets-no-et-al.csl bibstyle.csl; \
	fi
	@echo "Extensions installed"

list-texlive-packages: ## List installed TeX Live packages
	$(eval TEXLIVE_TEXMF_PATH=$(shell find /nix/store -type d -name '*-texlive-combined-*-texmfdist' -print -quit))
	@echo "TeX Live texmf distribution path: $(TEXLIVE_TEXMF_PATH)"
	@ls -alh $(TEXLIVE_TEXMF_PATH)/tex/latex

#-----------------
##@ render article
#-----------------

DOCUMENT_NAME ?= index

check-environment: ## Check quarto environment
	echo "QUARTO_PYTHON: $$QUARTO_PYTHON"
	quarto check install
	quarto check jupyter

install-tinytex: ## Install tinytex (requires non-hermetic environment)
	quarto install --no-prompt tinytex

render-latex: ## Render pdf format via LaTeX
render-latex: install-extensions
	quarto render $(DOCUMENT_NAME).qmd --to nature-pdf

render: ## Render all article formats including docx, html, ipynb, pdf, and tex
render: install-extensions
	quarto render $(DOCUMENT_NAME).qmd --to all

render-refresh: ## (EXAMPLE) Render with parameter specification and cache refresh
render-refresh: install-extensions
	quarto render index.qmd --to all -P TEST_MODE:False --cache-refresh

clean: ## Clean compilation artifacts
	rm sn-*.{bst,cls} || true

clean-all: ## Clean all files including output files
clean-all: clean
	rm $(DOCUMENT_NAME).{docx,html,ipynb,pdf,tex} || true
	rm -rf $(DOCUMENT_NAME)_files/
