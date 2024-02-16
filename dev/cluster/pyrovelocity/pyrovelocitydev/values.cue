@if(!debug)

package main

values: {
	message: "pyrovelocitydev"
	image: {
		repository: "ghcr.io/pinellolab/pyrovelocitydev"
		digest:     "" // "sha256:bbbbb"
		tag:        "" // "latest"
	}
	test: image: {
		repository: "cgr.dev/chainguard/curl"
		digest:     ""
		tag:        "latest"
	}
}
