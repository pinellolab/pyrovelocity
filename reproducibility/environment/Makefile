#############
# makefile for terraform google_notebooks_instance
#
# see README.md for usage
#
# prerequisites:
#
# - install gcloud sdk
#   - gcloud init # set project
# 	- gcloud auth login
# 	- gcloud auth application-default login
#
# - install terraform
# 	- terraform init
#
# - install gh cli
#   - gh auth login
#   - gh auth status
#
# - edit dotenv-gen.sh
# 	- set variables using [pass][pass] or manually
#
# - review/edit terraform.tfvars
#
# [pass]: https://www.passwordstore.org/
#
# usage:
#
#     make up - create -OR- update the instance
# 	  make stop - stop the instance
# 	  make start - start the instance
#     make down - delete the instance
#
# all other targets are auxiliary
##############
.DEFAULT_GOAL := help

##@ Utility
help: ## Display this help. (Default) 
# based on "https://gist.github.com/prwhite/8168133?permalink_comment_id=4260260#gistcomment-4260260"
	@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

##@ Utility
help_sort: ## Display alphabetized version of help.
	@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# set variables with shell scripts
# or manually edit .env and
# remove/comment the following line(s)
# $(shell ./dotenv-gen.sh > /dev/null 2>&1)
$(shell ./startup-script-gen.sh > /dev/null 2>&1)
include .env
export

up: ## Initialize terraform and startup cloud machine. Deps: apply, update_os_login, info, config_ssh.
up: \
apply \
update_os_login \
info \
config_ssh
	@echo "It may take 15 minutes for the conda development environment to complete installation."

stop: ## Stop running cloud machine.
	gcloud compute instances stop $(TF_VAR_notebooks_name)

start: ## Start existing cloud machine. Deps: start_instance, info, config_ssh.
start: \
start_instance \
info \
config_ssh
	@echo "It may take 5-10 minutes for the machine to complete startup."

down: ## Destroy existing cloud machine. Boot disk will be deleted and data disk will be preserved.
	terraform destroy -auto-approve

############################################
############################################

setup: test ## Setup terraform with init, fmt, validate, and plan output to "tfplan".
	terraform init
	terraform fmt
	terraform validate
	terraform plan -out=tfplan

apply: setup ## Apply terraform configuration in "tfplan".
	terraform apply -auto-approve "tfplan"

info: ## List existing GCP instances.
	gcloud compute instances list
	@printf "\nsee the following url\n\n== https://console.cloud.google.com/vertex-ai/workbench/list/instances?project=$(TF_VAR_project) ==\n\nfor running instances and connection to jupyter server\n\n"

config_ssh: ## Configure ssh with gcloud sdk. See .ssh/config or similar for ssh config.
	rm -f ~/.ssh/google_compute_known_hosts
	gcloud compute config-ssh

start_instance: ## Start up GCP cloud machine named TF_VAR_notebooks_name.
	gcloud compute instances start $(TF_VAR_notebooks_name)

show_disk_images: ## List names of disk images available in the deeplearning-platform-release.
	gcloud compute images list --project=deeplearning-platform-release

show_container_images:
	gcloud container images list --repository="gcr.io/deeplearning-platform-release"

GCP_ZONE=$(shell gcloud compute instances list --filter="name=$(TF_VAR_notebooks_name)" --format "csv[no-heading](zone)")

update_gcp_zone: ## Update GCP zone to GCP_ZONE based on automatic search (see Makefile).
	@echo "setting default GCP zone: $(GCP_ZONE)"
	gcloud config set compute/zone $(GCP_ZONE)

# to be removed when tf google_notebooks_instance API supports enable-oslogin
update_os_login: update_gcp_zone ## Update GCP enable-oslogin to FALSE.
	gcloud compute instances add-metadata $(TF_VAR_notebooks_name) --metadata=enable-oslogin="FALSE"
	gcloud compute instances describe $(TF_VAR_notebooks_name) --format="value(metadata)"

ssh_gcp: ## SSH to cloud machine with GCP SDK.
	gcloud compute ssh $(TF_VAR_notebooks_name) --command="bash -i -c 'cd /home/jupyter && sudo su jupyter'" -- -t

update_gcp_machine_type: ## Update GCP machine type based on TF_VAR_notebooks_name and TF_VAR_notebook_machine_type.
	gcloud compute instances set-machine-type $(TF_VAR_notebooks_name) --machine-type $(TF_VAR_notebook_machine_type)

list_gcs_bucket_contents: ## List the contents of a GCS bucket: make list_gcs_bucket_contents BUCKET_URL="gs://pyrovelocity/dvcstore"
ifdef BUCKET_URL
	gsutil ls -lhR $(BUCKET_URL)
else
	@echo 'define BUCKET_URL'
endif

list_dvc_cache_contents: ## List the contents of the local DVC cache in "../../.dvc/cache".
	tree --prune --du -h ../../.dvc/cache/

delete_gist: ## Delete github gist for machine startup script.
	@echo "github gist ID to be removed: $(GITHUB_STARTUP_SCRIPT_GIST_ID)"
	gh gist delete $(GITHUB_STARTUP_SCRIPT_GIST_ID)

delete_data_disk: ## Delete data disk.
	gcloud compute disks delete $(TF_VAR_notebooks_name)-data

# relevant when using container, as opposed to vm, images
restart_container:
	gcloud compute ssh $(TF_VAR_notebooks_name) --command 'docker restart payload-container'

env_print: ## Print environment variables with name containing "TF_VAR" or "GITHUB" including those defined in ".env" file.
	env | grep "TF_VAR\|GITHUB"


############################################
############################################
