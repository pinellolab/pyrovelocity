// VARIABLES ==============================================
// see: terraform.tfvars for public
//      dotenv-gen.sh for private
variable "project" {}
variable "email" {}
variable "credentials_file" {}
variable "notebooks_name" {}
variable "notebook_machine_type" {}
variable "vm_image_project" {}
variable "vm_image_family" {}
# variable "container_image" {}
# variable "container_tag" {}
variable "install_gpu_driver" {}
variable "accelerator_type" {}
variable "accelerator_number" {}
variable "boot_disk_size_gb" {}
variable "boot_disk_type" {}
variable "data_disk_size_gb" {}
variable "data_disk_type" {}
variable "no_remove_data_disk" {}
variable "post_startup_script_url" {}
variable "region" {
  default = "us-central1"
}
variable "zone" {
  default = "us-central1-a"
}
