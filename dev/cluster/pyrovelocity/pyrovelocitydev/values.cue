@if(!debug)

package main

values: {
	message: "pyrovelocitydev"
	image: {
		repository: "us-central1-docker.pkg.dev/pyro-284215/pyrovelocity/pyrovelocitydev"
		digest:     "" // "sha256:bbbbb"
		tag:        "" // "latest"
	}
	test: image: {
		repository: "cgr.dev/chainguard/curl"
		digest:     ""
		tag:        "latest"
	}
}
