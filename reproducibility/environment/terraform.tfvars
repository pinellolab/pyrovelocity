##########
# public variables
##########

# see dotenv-gen.sh for private variables

# notebooks_name        = <see TF_VAR_notebooks_name in dotenv-gen.sh>

##########
# machine / accelerator pairings
##########

# see https://cloud.google.com/compute/vm-instance-pricing#general-purpose_machine_type_family
notebook_machine_type = "n1-standard-4" # 15GB
# notebook_machine_type = "n1-highmem-4" # 26GB
# notebook_machine_type = "n1-highmem-8" # 52GB
# notebook_machine_type = "n1-highmem-16" # 104GB
# notebook_machine_type = "n1-highmem-32" # 208GB
accelerator_type = "NVIDIA_TESLA_T4"
# see https://cloud.google.com/compute/vm-instance-pricing#accelerator-optimized
# notebook_machine_type = "a2-highgpu-1g" # 12cpu 85GB
# accelerator_type   = "NVIDIA_TESLA_A100"
# notebook_machine_type = "a2-ultragpu-1g" # 12cpu 170GB
# accelerator_type   = "NVIDIA_A100_80GB"

#########
# accelerator
#########

# https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/notebooks_instance#type
install_gpu_driver = true
accelerator_number = 1

#########
# image
#########

# image-based
vm_image_project = "deeplearning-platform-release"
vm_image_family  = "pytorch-1-12-cu113-notebooks"

# container-based
# container_image       = "gcr.io/deeplearning-platform-release/pytorch-gpu"
# container_tag         = "latest"
# container_image       = "docker.io/cameronraysmith/notebooks"
# container_tag         = "develop"
# container_image       = "nvcr.io/nvidia/pytorch"
# container_tag         = "22.09-py3"

# startup script: 
#
# this can be set to any public url
# for testing you could use a public paste bin
#   post_startup_script_url = "https://hastebin.com/raw/ledahubuqe" 
# by default it is set by exporting the current local copy of
# post-startup-script.sh to github gist in startup-script-gen.sh
#
# post_startup_script_url = <see TF_VARS_post_startup_script_url>


##########
# disks
##########

# https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/notebooks_instance#boot_disk_type
boot_disk_size_gb   = 100
boot_disk_type      = "PD_STANDARD"
data_disk_size_gb   = 100
data_disk_type      = "PD_STANDARD"
no_remove_data_disk = true




#########
# location
#########

# toggle based on availability
# https://cloud.google.com/vertex-ai/docs/general/locations#user-managed-notebooks-locations
# region                  = "us-central1"
# zone                    = "us-central1-b"
region = "us-east4"
zone   = "us-east4-c"
# region                  = "northamerica-northeast1"
# zone                    = "northamerica-northeast1-b"
# region                  = "us-east1"
# zone                    = "us-east1-c"
# region                  = "us-west1"
# zone                    = "us-west1-b"
# region                  = "us-west4"
# zone                    = "us-west4-b"
# region                  = "us-west2"
# zone                    = "us-west2-b"
# region                  = "southamerica-east1"
# zone                    = "southamerica-east1-c"