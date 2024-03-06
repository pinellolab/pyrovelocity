{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
    nixpkgs-2305.url = "github:NixOS/nixpkgs/nixos-23.05";
    nixpkgs-unstable.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    flake-parts.url = "github:hercules-ci/flake-parts";
    flake-utils.url = github:numtide/flake-utils;
    poetry2nix = {
      # url = github:nix-community/poetry2nix;
      url = github:cameronraysmith/poetry2nix/patch;
      inputs = {
        nixpkgs.follows = "nixpkgs";
        flake-utils.follows = "flake-utils";
        systems.follows = "systems";
      };
    };
    flocken = {
      url = "github:mirkolenz/flocken/v2";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-parts.follows = "flake-parts";
      inputs.systems.follows = "systems";
    };
  };

  nixConfig = {
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
      "poetry2nix.cachix.org-1:2EWcWDlH12X9H76hfi5KlVtHgOtLa1Xeb7KjTjaV/R8="
      "pyrovelocity.cachix.org-1:+aX2YY45ZywieTsD2CnXLedN8RfKuRl6vL7+rLTCgnc="
    ];
    extra-substituters = [
      "https://nix-community.cachix.org"
      "https://poetry2nix.cachix.org"
      "https://pyrovelocity.cachix.org"
    ];
  };

  outputs = inputs @ {
    self,
    flake-parts,
    ...
  }:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = import inputs.systems;

      perSystem = {
        self',
        system,
        ...
      }: let
        pkgs = import inputs.nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
          overlays = [inputs.poetry2nix.overlays.default];
        };
        pkgs_2305 = import inputs.nixpkgs-2305 {
          inherit system;
          config = {
            allowUnfree = true;
          };
          overlays = [inputs.poetry2nix.overlays.default];
        };
        pkgs_unstable = import inputs.nixpkgs-unstable {
          inherit system;
          config = {
            allowUnfree = true;
          };
          overlays = [inputs.poetry2nix.overlays.default];
        };

        poetry2nixOverrides = import ./nix/poetry {
          inherit
            pkgs
            pkgs_unstable
            self
            ;
        };

        appBuildInputs = with pkgs; [
        ];
        mkPoetryAttrs = {
          projectDir = ./.;
          overrides = poetry2nixOverrides;
          python = pkgs.python310;
          # aarch64 cross-compilation on x86_64 may be intolerably slow if
          # preferWheels is disabled. If all of the individual contributors to
          # this are identified, it may be possible to use the library-specific
          # overrides above and disable the global usage of wheels
          preferWheels = true;
          checkGroups = ["test"];
          extras = [];
        };

        poetryEnv = pkgs.poetry2nix.mkPoetryEnv (
          mkPoetryAttrs
          // {
            extraPackages = ps:
              with pkgs; [
                python310Packages.pip
              ];
          }
        );

        gitHubOrg = "pinellolab";
        packageName = "pyrovelocity";

        mkPoetryEnvWithSource = packageName: src: groups:
          pkgs.poetry2nix.mkPoetryEnv (
            mkPoetryAttrs
            // {
              groups = groups;
              extraPackages = ps:
                with pkgs; [
                  python310Packages.pip
                ];
              editablePackageSources = {
                ${packageName} = src;
              };
            }
          );

        defaultPackages = import ./nix/pkgs {
          inherit
            system
            pkgs
            pkgs_unstable
            ;
        };
        sysPackages = defaultPackages.sysPackages;
        coreDevPackages = defaultPackages.coreDevPackages;
        devPackages = defaultPackages.devPackages;

        containerImageConfigs = import ./nix/containers {
          inherit
            pkgs
            mkPoetryEnvWithSource
            gitHubOrg
            packageName
            sysPackages
            devPackages
            ;
        };

        gcpProjectId = builtins.getEnv "GCP_PROJECT_ID";
        gcpSaEncodedJson = builtins.getEnv "ENCODED_GAR_SA_CREDS";

        # aarch64-linux may be disabled for more rapid image builds during
        # development setting NIX_IMAGE_SYSTEMS="x86_64-linux".
        # Note the usage of `preferWheels` as well.
        # NIX_IMAGE_SYSTEMS="x86_64-linux aarch64-linux"
        # will expand similar to the following:
        # images = with self.packages; [
        #   x86_64-linux.devcontainerImage
        #   aarch64-linux.devcontainerImage
        # ];
        includedSystems = let
          envVar = builtins.getEnv "NIX_IMAGE_SYSTEMS";
        in
          if envVar == ""
          then ["x86_64-linux" "aarch64-linux"]
          else builtins.filter (sys: sys != "") (builtins.split " " envVar);
      in {
        formatter = pkgs.alejandra;

        devShells = {
          default = pkgs.mkShell {
            name = packageName;
            nativeBuildInputs = with pkgs;
              [
                (mkPoetryEnvWithSource packageName ./src ["test" "docs" "workflows"])
              ]
              ++ devPackages;
            shellHook = ''
              export QUARTO_PYTHON=${pkgs.python310}/bin/python
            '';
          };
          devCore = pkgs.mkShell {
            name = packageName;
            nativeBuildInputs = with pkgs;
              [
                (mkPoetryEnvWithSource packageName ./src ["test" "docs" "workflows"])
              ]
              ++ coreDevPackages;
          };
        };

        packages = {
          default = pkgs.poetry2nix.mkPoetryApplication (
            mkPoetryAttrs
            // {
              buildInputs = appBuildInputs;
              checkInputs = appBuildInputs;
              nativeCheckInputs = appBuildInputs;

              preCheck = ''
                set -euo pipefail

                mkdir -p $TMPDIR/numba_cache
                export NUMBA_CACHE_DIR=$TMPDIR/numba_cache
                echo "NUMBA_CACHE_DIR: $NUMBA_CACHE_DIR"
              '';

              checkPhase = ''
                set -euo pipefail

                runHook preCheck

                # TODO: fix train_model test on Darwin
                # > src/pyrovelocity/train.py line 1564: 50991 Trace/BPT trap: 5
                pytest \
                -rA \
                -k "not workflows and not git" \
                --xdoc \
                --no-cov \
                --disable-warnings \
                --pyargs pyrovelocity

                runHook postCheck
              '';

              doCheck = true;

              pythonImportsCheck = ["pyrovelocity"];
            }
          );

          releaseEnv = pkgs.buildEnv {
            name = "release-env";
            paths = with pkgs; [poetry python310];
          };

          containerImage =
            pkgs.dockerTools.buildLayeredImage
            containerImageConfigs.containerImageConfig;
          containerStream =
            pkgs.dockerTools.streamLayeredImage
            containerImageConfigs.containerImageConfig;

          devcontainerImage =
            pkgs.dockerTools.buildLayeredImage
            containerImageConfigs.devcontainerImageConfig;
          devcontainerStream =
            pkgs.dockerTools.streamLayeredImage
            containerImageConfigs.devcontainerImageConfig;
        };

        legacyPackages.containerManifest = inputs.flocken.legacyPackages.${system}.mkDockerManifest {
          github = {
            enable = true;
            enableRegistry = false;
            token = builtins.getEnv "GH_TOKEN";
          };
          autoTags = {
            branch = false;
          };
          registries = {
            "ghcr.io" = {
              enable = true;
              repo = "${gitHubOrg}/${packageName}";
              username = builtins.getEnv "GITHUB_ACTOR";
              password = builtins.getEnv "GH_TOKEN";
            };
            "us-central1-docker.pkg.dev" = {
              enable = true;
              repo = "${gcpProjectId}/${packageName}/${packageName}";
              username = "_json_key_base64";
              password = "${gcpSaEncodedJson}";
            };
          };
          version = builtins.getEnv "VERSION";
          images = builtins.map (sys: self.packages.${sys}.containerImage) includedSystems;
          tags = [
            (builtins.getEnv "GIT_SHA_SHORT")
            (builtins.getEnv "GIT_SHA")
            (builtins.getEnv "GIT_REF")
          ];
        };

        legacyPackages.devcontainerManifest = inputs.flocken.legacyPackages.${system}.mkDockerManifest {
          github = {
            enable = true;
            enableRegistry = false;
            token = builtins.getEnv "GH_TOKEN";
          };
          autoTags = {
            branch = false;
          };
          registries = {
            "ghcr.io" = {
              enable = true;
              repo = "${gitHubOrg}/${packageName}dev";
              username = builtins.getEnv "GITHUB_ACTOR";
              password = builtins.getEnv "GH_TOKEN";
            };
            "us-central1-docker.pkg.dev" = {
              enable = true;
              repo = "${gcpProjectId}/${packageName}/${packageName}dev";
              username = "_json_key_base64";
              password = "${gcpSaEncodedJson}";
            };
          };
          version = builtins.getEnv "VERSION";
          images = builtins.map (sys: self.packages.${sys}.devcontainerImage) includedSystems;
          tags = [
            (builtins.getEnv "GIT_SHA_SHORT")
            (builtins.getEnv "GIT_SHA")
            (builtins.getEnv "GIT_REF")
          ];
        };
      };
    };
}
