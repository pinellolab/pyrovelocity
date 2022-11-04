// RESOURCES ==============================================
resource "google_notebooks_instance" "instance" {
  name                = var.notebooks_name
  project             = var.project
  location            = var.zone
  machine_type        = var.notebook_machine_type
  install_gpu_driver  = var.install_gpu_driver
  instance_owners     = [var.email]
  boot_disk_type      = var.boot_disk_type
  boot_disk_size_gb   = var.boot_disk_size_gb
  data_disk_type      = var.data_disk_type
  data_disk_size_gb   = var.data_disk_size_gb
  no_remove_data_disk = var.no_remove_data_disk
  post_startup_script = var.post_startup_script_url
  metadata = {
    proxy-mode     = "service_account"
    enable-oslogin = "FALSE" # fails to be set; see make target "update_os_login"
    terraform      = "true"
  }
  vm_image {
    project      = var.vm_image_project
    image_family = var.vm_image_family
  }
  # container_image {
  #   repository = var.container_image
  #   tag        = var.container_tag
  # }
  accelerator_config {
    type       = var.accelerator_type
    core_count = var.accelerator_number
  }
}
