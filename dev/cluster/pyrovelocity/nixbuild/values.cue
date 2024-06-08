@if(!debug)

package main

values: {
	message: "nixbuild"
	image: {
		repository: "nixos/nix"
		digest:     "" // "sha256:bbbbb"
		tag:        "" // "latest"
	}
	test: image: {
		repository: "cgr.dev/chainguard/curl"
		digest:     ""
		tag:        "latest"
	}
}
