values: {
	replicas: 1
	image: {
		repository: "ghcr.io/cameronraysmith/nixpod"
		digest:     ""
		tag:        "latest"
	}
	resources: {
		requests: {
			cpu:    "2000m"
			memory: "8Gi"
		}
		limits: {
			cpu:    "12000m"
			memory: "32Gi"
			// "nvidia.com/gpu":    "0"
			"ephemeral-storage": "35Gi"
		}
	}
	persistence: {
		name:         "nixbuild0"
		size:         "100Gi"
		storageClass: "ceph-filesystem"
		accessMode:   "ReadWriteMany"
	}
	job: {
		backoffLimit:            8
		activeDeadlineSeconds:   14400
		ttlSecondsAfterFinished: 172800
	}
	nodeSelector: {
		// "cloud.google.com/gke-accelerator": "nvidia-tesla-t4"
		// "cloud.google.com/gke-accelerator": "nvidia-tesla-a100"
		// "cloud.google.com/gke-accelerator": "nvidia-l4"

		// "cloud.google.com/gke-provisioning": "standard"
		"cloud.google.com/gke-provisioning": "spot"
	}
	securityContext: {
		allowPrivilegeEscalation: true
		privileged:               true
		capabilities:
		{
			drop: [""]
			add: ["ALL"]
		}
	}
	containerCommand: ["/bin/sh"]
	containerCommandArgs: ["-c", "sleep 3h"]
}
