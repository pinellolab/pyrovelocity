values: {
	replicas: 1
	image: {
		repository: "us-central1-docker.pkg.dev/pyro-284215/pyrovelocity/pyrovelocitydev"
		digest:     ""
		tag:        "536-input"
	}
	resources: {
		requests: {
			cpu:    "8000m"
			memory: "16Gi"
		}
		limits: {
			cpu:    "12000m"
			memory: "32Gi"
			// "nvidia.com/gpu":    "1"
			"ephemeral-storage": "35Gi"
		}
	}
	persistence: {
		name:         "dev0"
		size:         "100Gi"
		storageClass: "nfs"
		accessMode:   "ReadWriteMany"
		// storageClass: "standard-rwo"
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

		"cloud.google.com/gke-provisioning": "standard"
		// "cloud.google.com/gke-provisioning": "spot"
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
