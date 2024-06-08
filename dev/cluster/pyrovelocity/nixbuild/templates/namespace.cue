package templates

#Namespace: {
	#config:    #Config
	apiVersion: "v1"
	kind:       "Namespace"
	metadata: {
		name:   #config.metadata.namespace
		labels: #config.metadata.labels
		if #config.metadata.annotations != _|_ {
			annotations: #config.metadata.annotations
		}
	}
}
