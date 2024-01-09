package templates

persistentvolumeclaim: "pyrovelocity-claim": {
	apiVersion: "v1"
	kind:       "PersistentVolumeClaim"
	metadata: {
		name:      "pyrovelocity-claim"
		namespace: "pyrovelocity"
	}
	spec: {
		accessModes: ["ReadWriteOnce"]
		resources: requests: storage: "400Gi"
		storageClassName: "standard-rwo"
		volumeMode:       "Filesystem"
	}
}
