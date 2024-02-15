package templates

import (
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
)

#Job: batchv1.#Job & {
	#config:    #Config
	apiVersion: "batch/v1"
	kind:       "Job"
	metadata:   #config.metadata
	spec: batchv1.#JobSpec & {
		ttlSecondsAfterFinished: #config.job.ttlSecondsAfterFinished
		backoffLimit:            #config.job.backoffLimit
		activeDeadlineSeconds:   #config.job.activeDeadlineSeconds
		template: {
			metadata: {
				labels: #config.selector.labels
				if #config.podAnnotations != _|_ {
					annotations: #config.podAnnotations
				}
			}
			spec: corev1.#PodSpec & {
				restartPolicy:      "OnFailure"
				serviceAccountName: #config.metadata.name
				containers: [
					{
						name:            #config.metadata.name
						image:           #config.image.reference
						imagePullPolicy: #config.image.pullPolicy
						command:         #config.containerCommand
						args:            #config.containerCommandArgs
						volumeMounts: [
							{
								mountPath: "/workspace"
								name:      #config.persistence.name
							},
						]
						resources:       #config.resources
						securityContext: #config.securityContext
					},
				]
				volumes: [
					{
						name: #config.persistence.name
						persistentVolumeClaim: {
							claimName: #config.persistence.name
						}
					},
				]
				if #config.nodeSelector != _|_ {
					nodeSelector: #config.nodeSelector
				}
				if #config.podSecurityContext != _|_ {
					securityContext: #config.podSecurityContext
				}
				if #config.topologySpreadConstraints != _|_ {
					topologySpreadConstraints: #config.topologySpreadConstraints
				}
				if #config.affinity != _|_ {
					affinity: #config.affinity
				}
				if #config.tolerations != _|_ {
					tolerations: #config.tolerations
				}
				if #config.imagePullSecrets != _|_ {
					imagePullSecrets: #config.imagePullSecrets
				}
			}
		}
	}
}
