package templates

import (
	corev1 "k8s.io/api/core/v1"
	timoniv1 "timoni.sh/core/v1alpha1"
)

#Config: {
	kubeVersion!:                                 string
	clusterVersion: timoniv1.#SemVer & {#Version: kubeVersion, #Minimum: "1.20.0"}
	moduleVersion!:                           string
	metadata: timoniv1.#Metadata & {#Version: moduleVersion}
	metadata: labels:       timoniv1.#Labels
	metadata: annotations?: timoniv1.#Annotations
	selector: timoniv1.#Selector & {#Name: metadata.name}

	image!: timoniv1.#Image
	image: pullPolicy: *"Always" | timoniv1.#Image.pullPolicy

	containerCommand: [...string]
	containerCommandArgs: [...string]

	#GPUQuantity: string & =~"^[1-8]$"

	#ExtendedResourceRequirement: {
		cpu?:              timoniv1.#CPUQuantity
		memory?:           timoniv1.#MemoryQuantity
		"nvidia.com/gpu"?: #GPUQuantity
	}

	#ExtendedResourceRequirements: {
		limits?:   #ExtendedResourceRequirement
		requests?: timoniv1.#ResourceRequirement
	}

	resources: #ExtendedResourceRequirements & {
		limits: {
			cpu:               *"32000m" | timoniv1.#CPUQuantity
			memory:            *"64Gi" | timoniv1.#MemoryQuantity
			"nvidia.com/gpu"?: *"1" | #GPUQuantity
		}
		requests: {
			cpu:    *"16000m" | timoniv1.#CPUQuantity
			memory: *"8Gi" | timoniv1.#MemoryQuantity
		}
	}

	nodeSelector: {
		"cloud.google.com/gke-accelerator"?: *"nvidia-tesla-t4" | "nvidia-tesla-a100" | "nvidia-l4"
		spot:                                *"false" | "true"
	}

	replicas: *1 | >=0 & <5

	securityContext: corev1.#SecurityContext & {
		allowPrivilegeEscalation: *false | true
		privileged:               *false | true
		capabilities:
		{
			drop: *["ALL"] | [string]
			add: *["CHOWN", "NET_BIND_SERVICE", "SETGID", "SETUID"] | [string]
		}
	}

	persistence: {
		enabled:      *true | bool
		storageClass: *"standard" | string
		size:         *"500Gi" | string
	}

	podAnnotations?: {[string]: string}
	podSecurityContext?: corev1.#PodSecurityContext
	imagePullSecrets?: [...timoniv1.ObjectReference]
	tolerations?: [...corev1.#Toleration]
	affinity?: corev1.#Affinity
	topologySpreadConstraints?: [...corev1.#TopologySpreadConstraint]

	test: {
		enabled: *false | bool
		image!:  timoniv1.#Image
	}

	message!: string
}

#Instance: {
	config: #Config

	objects: {
		ns: #Namespace & {#config: config}
		sa: #ServiceAccount & {#config: config}
		pvc: #PVC & {#config: config}
		deploy: #Deployment & {
			#config: config
		}
	}

	tests: {
	}
}
