terraform {
  required_providers {
    crusoe = {
      source = "registry.terraform.io/crusoecloud/crusoe"
    }
  }
}

locals {
  my_ssh_key = file("~/.ssh/id_ed25519.pub")
}

// new VM
resource "crusoe_compute_instance" "vllm_vm" {
  name     = "vllm-valid"
  type     = "l40s-48gb.8x"
  location = "us-east1-a"

  # specify the base image
  image = "ubuntu22.04-nvidia-slurm:12.4"

  disks = [
    {
      id              = crusoe_storage_disk.vllm_data_disk.id
      mode            = "read-write"
      attachment_type = "data"
    }
  ]

  ssh_key = local.my_ssh_key
}

resource "crusoe_storage_disk" "vllm_data_disk" {
  name     = "vllm-valid-disk"
  size     = "256GiB"
  location = "us-east1-a"
}

output "instance_public_ip" {
  value = crusoe_compute_instance.vllm_vm.network_interfaces[0].public_ipv4.address
}
