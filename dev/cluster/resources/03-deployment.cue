package templates

deployment: pyrovelocity: {
	apiVersion: "apps/v1"
	kind:       "Deployment"
	metadata: {
		name:      "pyrovelocity"
		namespace: "pyrovelocity"
	}
	spec: {
		replicas: 1
		strategy: {
			type: "RollingUpdate"
			rollingUpdate: {
				maxSurge:       1
				maxUnavailable: 1
			}
		}
		selector: matchLabels: app: "pyrovelocity"
		template: {
			metadata: labels: app: "pyrovelocity"
			spec: {
				containers: [{
					name:            "pyrovelocitydev"
					image:           "ghcr.io/pinellolab/pyrovelocitydev"
					imagePullPolicy: "Always"
					command: [
						"/bin/sh",
						"-c",
						"sleep 16h",
					]
					resources: {
						requests: {
							cpu:    "16"
							memory: "64Gi"
						}
						limits: {
							cpu:              "30"
							memory:           "96Gi"
							"nvidia.com/gpu": "1"
						}
					}
					volumeMounts: [{
						name:      "pyrovelocity"
						mountPath: "/workspace"
					}]
				}]
				nodeSelector: {
					"cloud.google.com/gke-accelerator": "nvidia-tesla-t4"
					spot:                               "false"
				}
				volumes: [{
					name: "pyrovelocity"
					persistentVolumeClaim: claimName: "pyrovelocity-claim"
				}]
			}
		}
	}
}
