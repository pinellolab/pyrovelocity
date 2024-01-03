# development environment

This folder contains [infrastructure as code][iac] (IaC) for a minimal development environment that supports swapping backend machines and associated GPU(s) to meet the demands of a given development task. It currently uses [terraform][terraform] with the [google cloud platform][gcpsdk] provider and the [google notebooks instance][gni] resource, but it could be adapted for other cloud platforms, providers, or resources (see the [terraform documentation][tfmdocs] for further reference). It is assumed all commands are run from within this folder.

## workflow

The expected workflow is to

- set up a development machine with `make up`,
- connect to the machine via the associated jupyter lab server accessible from the [google cloud platform user interface][gcpui] for interactive use,
- [ssh](#ssh) to the machine from a terminal or IDE such as [VS Code][vscodessh] for library development,
- toggle the machine off and on with `make stop` and `make start`, and
- destroy associated compute resources with `make down`.

## prerequisites

### software

- install [google cloud sdk][gcpsdk]
  - `gcloud init` to set project and [application default credentials][adc]
    - `gcloud auth login`
    - `gcloud auth application-default login`
  - request [GCP GPU quota increase][gcpgpuquota] via [google cloud console compute engine API][gcpconsolequota] to at least `1` for `NVIDIA_T4_GPUS` or other accelerator types you plan to use
- install [terraform][terraform]
  - `terraform init`
- install and authenticate with [github cli][ghcli] to use github gists for the [post startup script](#startup-script)
  - check `gh auth status` when complete

### configuration

#### environment variables

- set environment variables

  - [dotenv-gen.sh](./dotenv-gen.sh) is provided to help construct a `.env` file that is read by the [Makefile](./Makefile) to set environment variables. If you do not want to use [dotenv-gen.sh](./dotenv-gen.sh), you can create a `.env` file as informally described, for example, in [dotenv][python-dotenv] containing all variables written to `.env` at the end of [dotenv-gen.sh](./dotenv-gen.sh) and remove reference to [dotenv-gen.sh](./dotenv-gen.sh) in the [Makefile](./Makefile)
  - example `.env` file (see below for variables related to the startup script)

    ```shell
    TF_VAR_project=<GCP Project ID> # your google cloud platform project ID
    TF_VAR_email=<GCP account email address> # your google cloud platform account email address
    TF_VAR_credentials_file=~/.config/gcloud/application_default_credentials.json # local path to your application default credentials
    TF_VAR_notebooks_name=reponame-dev-notebook # name to assign to your development virtual machine
    GITHUB_USERNAME=username # github username associated to uploading startup scripts as github gists
    GITHUB_ORG_NAME=githuborg # name of the github org or user containing the github repository with code for development
    GITHUB_REPO_NAME=reponame # name of a github repository with a conda environment yaml file
    GITHUB_BRANCH_NAME=main # name of github repository branch to checkout
    GITHUB_REPO_CONDA_ENV_PATH_NAME=conda/environment-gpu.yml # path to conda environment yaml file in the github repository
    GH_PAT=ghp_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX # github personal access token with repo scope
    GH_REPO=${GITHUB_USERNAME}/${GITHUB_REPO_NAME} # derived
    GCP_GACD=$(shell cat service-account-credentials.json) # GCP service account credentials
    GCP_SERVICE_ACCOUNT=111111111111-compute@developer.gserviceaccount.com # GCP service account email
    GCP_PROJECT_ID=${TF_VAR_project} # this is an alias for the gcp project ID
    GCP_REGION=us-central1 # the google cloud platform region for application deployment
    GCP_RUN_SERVICE_NAME=app-test # the service name for application deployment
    PKG_ARCHIVE_URL=us-central1-docker.pkg.dev/${GCP_PROJECT_ID}/${GITHUB_REPO_NAME} # the url to the GAR package
    PKG_APP=${GITHUB_REPO_NAME}app # the package name containing the application
    PKG_IMAGE_TAG=latest # the package tag to deploy
    MLFLOW_TRACKING_URI=https://server.mlflow # the url to the mlflow tracking server
    MLFLOW_TRACKING_USERNAME=username # the username for the mlflow tracking server
    MLFLOW_TRACKING_PASSWORD=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX # the key for the mlflow tracking server
    TF_VAR_post_startup_script_url=https://gist.githubusercontent.com/githubusername/b6c8cd158b00f99d21511a905cc7626a/raw/post-startup-script-dev-notebook.sh # publicly accessible URL to a startup script
    GITHUB_STARTUP_SCRIPT_GIST_ID=b6c8cd158b00f99d21511a905cc7626a # the github gist ID if you would like to use a github gist
    ```

  - set variables using [pass][pass] or manually

    - execute `pass insert github_username`
    - complete the same process for `gcp_credentials_file`, `gcp_email`, `gcp_project`, `gcp_notebooks_name`, `github_org`, `github_repo`, `github_branch`, and `github_repo_conda_env_path`
    - `gcp_credentials_file` contains the path to appication default credentials. The most common value is `~/.config/gcloud/application_default_credentials.json`
    - check these are all defined with `$ pass`

      ```shell
      $ pass
      Password Store
      ├── gcp_credentials_file
      ├── gcp_email
      ├── gcp_project
      ├── gcp_notebooks_name
      ├── github_org
      ├── github_repo
      ├── github_branch
      ├── github_repo_conda_env_path
      └── github_username
      ```

- if there is a variable you would like to set that is not currently exposed in the environment, review/edit [terraform.tfvars](./terraform.tfvars)
  - you can optionally set parameters not currently read from environment variables in this file
  - for example, you may want to set the machine type, accelerator/GPU type, disk size, etc

#### startup script

- edit/generate startup script

  - review/edit [startup-script-gen.sh](./startup-script-gen.sh)

    - this script is executed by default at the top level of the [Makefile](./Makefile) to set variables and upload `post-startup-script.sh` to a publicly accessible location for consumption by the virtual machine. A copy of the latter will be downloaded to and executed from the path `/opt/c2d/post_start.sh` on the remote machine.
    - if you would like to avoid using this script, add values for the following variables to `.env` and comment reference to [startup-script-gen.sh](./startup-script-gen.sh) in the [Makefile](./Makefile)

      ```shell
      TF_VAR_post_startup_script_url=https://gist.githubusercontent.com/githubusername/b6c8cd158b00f99d21511a905cc7626a/raw/post-startup-script-dev-notebook.sh # publicly accessible URL to a startup script
      GITHUB_STARTUP_SCRIPT_GIST_ID=b6c8cd158b00f99d21511a905cc7626a # the github gist ID if you would like to use a github gist
      ```

  - edit [template-post-startup-script.sh](./template-post-startup-script.sh)
    - execution of [startup-script-gen.sh](./startup-script-gen.sh) will upload your current local copy of `post-startup-script-$(TF_VAR_notebooks_name).sh` automatically generated from [template-post-startup-script.sh](./template-post-startup-script.sh) to a github gist by default

- Uploading multiple revisions of the startup script to an associated github gist in succession may cause it to get out of sync with the github server cache. You may find it helpful to run

  ```shell
  gh gist list
  make -n delete_gist
  make delete_gist
  ```

  to refresh the github gist ID associated to your startup script. If you are confident in how this works in your environment, you can likely just run `make delete_gist`.

#### test

- when the requirements above are satisfied, `make test` will do the following
  - upload `post-startup-script-$(TF_VAR_notebooks_name).sh` to github gist
  - print `TF_VAR*` and `GITHUB*` environment variables

## usage

### Makefile

The primary interface is via the [Makefile](./Makefile), which is being used here as a modular collection of short shell scripts rather than as a build system. You can fill environment variables and print each command prior to running with `make -n <target>` such as `make -n up`. Please see [GNU make][make] for further reference. The primary targets are

```shell
make up - create -OR- update the instance
make stop - stop the instance
make start - start the instance
make down - delete the instance
```

All other targets are auxiliary. The [Makefile](./Makefile) is primarily to document commands that are commonly used to work with the terraform resource(s). You can simply copy the command from the Makefile and run it manually in the terminal if you do not want to use [make][make].

### data disk management

The data disk associated to a given `notebooks_name` is retained and reattached even after running `make down` and `make up`. This is useful to avoid losing work, especially when [spot/preemptibility][gcpspot] is enabled (not currently supported by the [terraform google_notebooks_instance resource][gni]). However, this is associated to a cost for retaining the persistent disk. If you want to disable this behavior and delete the data disk automatically when destroying a machine, set `no_remove_data_disk = false` in [terraform.tfvars](./terraform.tfvars). To manually delete the data disk associated to the current value of `notebooks_name`, run `make -n delete_data_disk` to verify the correct disk would be deleted and then rerun without `-n` to delete the data disk.

## machine images

Check available machine images from the [deeplearning-platform-release](https://gcr.io/deeplearning-platform-release) by running `make show_disk_images`. You can modify the machine image by setting `vm_image_project` and `vm_image_family` in [terraform.tfvars](./terraform.tfvars). You can alternatively use a docker image by reviewing and editing the content of [notebooks-instance.tf](./notebooks-instance.tf) to use `container_image` instead of `vm_image`. You can also run `make show_container_images` to list available images. Note however that using a container image as opposed to a disk image would require a different post-startup configuration process. This can be incorporated into a [derivative container image][dci].

## remote usage

### ssh

The [Makefile](./Makefile) will run

```shell
gcloud compute config-ssh
```

to update your ssh configuration file and print the configured hostname at the end of `make up`. From a terminal, you can ssh into the hostname printed at the end of `make up` or try `make ssh_gcp`. In order to connect from IDEs or otherwise, it may be helpful to update your `~/.ssh/config` file with something similar to (updated to reference the key files you use with google cloud platform)

```shell
Host gcp
    HostName <IP_ADDRESS>
    IdentityFile ~/.ssh/google_compute_engine
    UserKnownHostsFile=~/.ssh/google_compute_known_hosts
    IdentitiesOnly=yes
    CheckHostIP=no
    StrictHostKeyChecking=no
    RequestTTY Yes
    RemoteCommand cd /home/jupyter && sudo su jupyter
```

The `IP_ADDRESS` of the remote host is printed at the end of `make up`. You can run `gcloud compute instances list` to display the `IP_ADDRESS` of the virtual machine if you need to reference it. If you are using [VS Code][vscodessh] you may need to manually set `"remote.SSH.enableRemoteCommand": true` in order to respect execution of the `RemoteCommand` within the ssh session.

If you use the container rather than disk image to setup the virtual machine, you may find an alternative `RemoteCommand` useful

```shell
Host gcp
    ...
    RemoteCommand sudo docker exec -it payload-container /bin/bash
```

### github

You may find it useful to execute a script similar to the following

```shell
mkdir -p $HOME/.config/gh && \
printf "\n[user]
    name = Your Name
    email = your@email
[credential]
    helper = store\n" >> $HOME/.gitconfig && \
printf "github.com:
    oauth_token: ghp_github_oauth_token
    user: githubusername
    git_protocol: https" > $HOME/.config/gh/hosts.yml && \
printf "https://githubusername:ghp_github_oauth_token@github.com\n" > $HOME/.git-credentials
```

to support github integration from the remote server.

[iac]: https://en.wikipedia.org/wiki/Infrastructure_as_code
[terraform]: https://developer.hashicorp.com/terraform/tutorials/gcp-get-started/install-cli
[gcpsdk]: https://cloud.google.com/sdk/docs/install
[gni]: https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/notebooks_instance
[tfmdocs]: https://developer.hashicorp.com/terraform/docs
[gcpui]: https://console.cloud.google.com/vertex-ai/workbench/list/instances
[vscodessh]: https://code.visualstudio.com/docs/remote/ssh
[adc]: https://cloud.google.com/docs/authentication/provide-credentials-adc
[gcpgpuquota]: https://cloud.google.com/compute/quotas#gpu_quota
[gcpconsolequota]: https://console.cloud.google.com/apis/api/compute.googleapis.com/quotas
[python-dotenv]: https://github.com/theskumar/python-dotenv#file-format
[gcpspot]: https://cloud.google.com/compute/docs/instances/spot
[ghcli]: https://cli.github.com
[pass]: https://www.passwordstore.org/
[make]: https://www.gnu.org/software/make/
[dci]: https://cloud.google.com/deep-learning-containers/docs/derivative-container
