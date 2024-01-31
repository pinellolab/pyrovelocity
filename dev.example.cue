values: {
	replicas: 1
	image: {
		repository: "ghcr.io/pinellolab/pyrovelocitydev"
		digest:     ""
		tag:        "main"
	}
	resources: {
		requests: {
			cpu:    "16000m"
			memory: "32Gi"
		}
		limits: {
			cpu:              "30000m"
			memory:           "116Gi"
			"nvidia.com/gpu": "1"
		}
	}
	nodeSelector: {
		"cloud.google.com/gke-accelerator": "nvidia-tesla-t4"
		spot:                               "true"
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
