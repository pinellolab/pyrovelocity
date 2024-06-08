package templates

import (
	autoscalingv2 "k8s.io/api/autoscaling/v2"
)

#HPA: autoscalingv2.#HorizontalPodAutoscaler & {
	#config:    #Config
	apiVersion: "autoscaling/v2"
	kind:       "HorizontalPodAutoscaler"
	metadata:   #config.metadata
	spec: autoscalingv2.#HorizontalPodAutoscalerSpec & {
		scaleTargetRef: {
			apiVersion: #config.scaleTarget.apiVersion
			kind:       #config.scaleTarget.kind
			name:       #config.scaleTarget.name
		}
		minReplicas: #config.scaleTarget.minReplicas
		maxReplicas: #config.scaleTarget.maxReplicas
	}
}
