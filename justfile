# Default command when 'just' is run without arguments
# Run 'just <command>' to execute a command.
default: list

# Display help
help:
  @printf "\nSee Makefile targets for just and direnv installation."
  @printf "\nRun 'just -n <command>' to print what would be executed...\n\n"
  @just --list --unsorted
  @echo "\n...by running 'just <command>'.\n"
  @echo "This message is printed by 'just help'."
  @echo "Just 'just' will just list the available recipes.\n"

# List just recipes
list:
  @just --list --unsorted

# List evaluated just variables
vars:
  @just --evaluate

builder := env_var_or_default('BUILDER', 'podman')
container_user := "runner"
container_home := "/home" / container_user
container_work := container_home / "work"
git_username := env_var_or_default('GITHUB_USERNAME', 'pinellolab')
git_org_name := env_var_or_default('GITHUB_ORG_NAME', 'pinellolab')
git_repo_name := env_var_or_default('GITHUB_REPO_NAME', 'pyrovelocity')
git_branch_name := env_var_or_default('GITHUB_BRANCH_NAME', 'main')
container_registry := "ghcr.io/" + git_org_name + "/"
pod_accelerator_type := env_var_or_default('POD_ACCELERATOR_TYPE', 'nvidia-tesla-t4')
accelerator_node_selector := "gpu-type=" + pod_accelerator_type

container_type := "dev" # or "app"
container_image := if container_type == "dev" {
    "pyrovelocitygpu"
  } else if container_type == "app" {
    "pyrovelocityapp"
  } else {
    error("container_type must be either 'dev' or 'app'")
  }
container_tag := "latest"

pod_source_type := env_var_or_default('POD_SOURCE_TYPE', 'git')
pod_git_provider := env_var_or_default('POD_GIT_PROVIDER', 'github')
pod_disk_size := env_var_or_default('POD_DISK_SIZE', '400Gi')
pod_min_cpu := env_var_or_default('POD_MIN_CPU', '16')
pod_min_mem := env_var_or_default('POD_MIN_MEM', '64Gi')
pod_max_cpu := env_var_or_default('POD_MAX_CPU', '32')
pod_max_mem := env_var_or_default('POD_MAX_MEM', '96Gi')
pod_max_accel := env_var_or_default('POD_MAX_ACCEL', '1')
pod_resources := "requests.cpu=" + pod_min_cpu + ",requests.memory=" + pod_min_mem + ",limits.cpu=" + pod_max_cpu + ",limits.memory=" + pod_max_mem + ",limits.nvidia.com/gpu=" + pod_max_accel

architecture := if arch() == "x86_64" {
    "amd64"
  } else if arch() == "aarch64" {
    "arm64"
  } else {
    error("unsupported architecture must be amd64 or arm64")
  }

opsys := if os() == "macos" {
    "darwin"
  } else if os() == "linux" {
    "linux"
  } else {
    error("unsupported operating system must be darwin or linux")
  }

devpod_release := "latest" # or "v0.3.7" or "v0.4.0-alpha.4"

devpod_binary_url := if devpod_release == "latest" {
  "https://github.com/loft-sh/devpod/releases/latest/download/devpod-" + opsys + "-" + architecture
} else {
  "https://github.com/loft-sh/devpod/releases/download/" + devpod_release + "/devpod-" + opsys + "-" + architecture
}

# Install devpod (check/set: devpod_release)
[unix]
install-devpod:
  curl -L -o devpod {{devpod_binary_url}} && \
  sudo install -c -m 0755 devpod /usr/local/bin && \
  rm -f devpod
  which devpod
  devpod version

# Print devpod info
devpod:
  devpod version && echo
  devpod context list
  devpod provider list
  devpod list

# Install and use devpod kubernetes provider
provider:
  devpod provider add kubernetes --silent || true \
  && devpod provider use kubernetes

# Run latest container_image in current kube context
pod:
  devpod up \
  --debug \
  --devcontainer-image {{container_registry}}{{container_image}}:{{container_tag}} \
  --provider kubernetes \
  --ide vscode \
  --open-ide \
  --source {{pod_source_type}}:https://{{pod_git_provider}}.com/{{git_username}}/{{git_repo_name}} \
  --provider-option DISK_SIZE={{pod_disk_size}} \
  --provider-option NODE_SELECTOR={{accelerator_node_selector}} \
  --provider-option RESOURCES={{pod_resources}} \
  {{container_image}}

# Stop devpod (check container_image hasn't changed with -n)
stop:
  devpod stop {{container_image}}

# Delete devpod (check container_image hasn't changed with -n)
delete:
  devpod delete {{container_image}}

## secrets

# Define the project variable
gcp_project_id := env_var_or_default('GCP_PROJECT_ID', 'development')

# Show existing secrets
show:
  @teller show

# Create a secret with the given name
create-secret name:
  @gcloud secrets create {{name}} --replication-policy="automatic" --project {{gcp_project_id}}

# Populate a single secret with the contents of a dotenv-formatted file
populate-single-secret name path:
  @gcloud secrets versions add {{name}} --data-file={{path}} --project {{gcp_project_id}}

# Populate each line of a dotenv-formatted file as a separate secret
populate-separate-secrets path:
  @while IFS= read -r line; do \
     KEY=$(echo $line | cut -d '=' -f 1); \
     VALUE=$(echo $line | cut -d '=' -f 2); \
     gcloud secrets create $KEY --replication-policy="automatic" --project {{gcp_project_id}} 2>/dev/null; \
     printf "$VALUE" | gcloud secrets versions add $KEY --data-file=- --project {{gcp_project_id}}; \
   done < {{path}}

# Complete process: Create a secret and populate it with the entire contents of a dotenv file
create-and-populate-single-secret name path:
  @just create-secret {{name}}
  @just populate-single-secret {{name}} {{path}}

# Complete process: Create and populate separate secrets for each line in the dotenv file
create-and-populate-separate-secrets path:
  @just populate-separate-secrets {{path}}

# Retrieve the contents of a given secret
get-secret name:
  @gcloud secrets versions access latest --secret={{name}} --project={{gcp_project_id}}

# Create empty dotenv from template
seed-dotenv:
  @cp .template.env .env

# Export unique secrets to dotenv format
export:
  @teller export env | sort | uniq | grep -v '^$' > .secrets.env

# Check secrets are available in teller shell.
check-secrets:
  @printf "Check teller environment for secrets\n\n"
  @teller run -s -- env | grep -E 'GITHUB|CACHIX' | teller redact

get-kubeconfig:
  @teller run -s -- printenv KUBECONFIG > dev-kubeconfig.yaml

# Setup docs for building and deploying
docs-setup:
  dvc pull --force --allow-missing --verbose
  quartodoc build --verbose --config nbs/_quarto.yml
  (cd nbs && quartodoc interlinks)
  quarto render nbs

# Run docs locally
docs-dev: docs-setup
  yarn dlx wrangler pages dev nbs/_site

# Deploy docs
docs-deploy: docs-setup
  yarn dlx wrangler pages deploy nbs/_site
