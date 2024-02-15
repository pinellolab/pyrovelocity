values: {
	replicas: 1
	image: {
		repository: "ghcr.io/pinellolab/pyrovelocitydev"
		digest:     ""
		tag:        "beta"
	}
	resources: {
		requests: {
			cpu:    "16000m"
			memory: "32Gi"
		}
		limits: {
			cpu:                 "30000m"
			memory:              "116Gi"
			"nvidia.com/gpu":    "1"
			"ephemeral-storage": "245Gi"
		}
	}
	persistence: {
		name: "dev"
		size: "100Gi"
	}
	job: {
		backoffLimit:            8
		activeDeadlineSeconds:   14400
		ttlSecondsAfterFinished: 172800
	}
	nodeSelector: {
		"cloud.google.com/gke-accelerator": "nvidia-tesla-t4"
		// "cloud.google.com/gke-accelerator": "nvidia-tesla-a100"
		// "cloud.google.com/gke-accelerator": "nvidia-l4"

		"cloud.google.com/gke-provisioning": "standard"
		// "cloud.google.com/gke-provisioning": "spot"
	}
	// securityContext: {
	// 	allowPrivilegeEscalation: true
	// 	privileged:               true
	// 	capabilities:
	// 	{
	// 		drop: [""]
	// 		add: ["ALL"]
	// 	}
	// }
	containerCommand: ["/bin/sh"]
	containerCommandArgs: ["-c", "sleep 16h"]
}
