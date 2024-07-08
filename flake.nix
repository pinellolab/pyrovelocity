{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    flake-parts.url = "github:hercules-ci/flake-parts";
    flake-utils.url = github:numtide/flake-utils;
    gitignore = {
      url = "github:hercules-ci/gitignore.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    poetry2nix = {
      url = github:nix-community/poetry2nix;
      inputs = {
        nixpkgs.follows = "nixpkgs";
        flake-utils.follows = "flake-utils";
        systems.follows = "systems";
      };
    };
    flocken = {
      url = "github:cameronraysmith/flocken/crane-tag";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-parts.follows = "flake-parts";
      inputs.systems.follows = "systems";
    };
    nixpod = {
      url = "github:cameronraysmith/nixpod";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-parts.follows = "flake-parts";
      inputs.flocken.follows = "flocken";
      inputs.systems.follows = "systems";
    };
  };

  nixConfig = {
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
      "poetry2nix.cachix.org-1:eXpeBJl0EQjO+vs9/1cUq19BH1LLKQT9HScbJDeeHaA="
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
        inputs',
        pkgs,
        system,
        config,
        ...
      }: let
        poetry2nixOverrides = import ./nix/poetry {
          inherit
            (pkgs)
            poetry2nix
            lib
            stdenv
            writeText
            autoPatchelfHook
            cudaPackages_12_1
            tbb_2021_11
            ;
        };

        appBuildInputs = with pkgs; [
        ];
        mkPoetryAttrs = {
          projectDir = ./.;
          overrides = poetry2nixOverrides;
          python = pkgs.python310;
          # aarch64 cross-compilation on x86_64 may be unusable if preferWheels
          # is disabled. If all of the individually contributing packages were
          # identified, it may be possible to use the library-specific overrides
          # in ./nix/poetry and disable the global usage of wheels
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
            ;
        };
        sysPackages = defaultPackages.sysPackages;
        extraSysPackages = defaultPackages.extraSysPackages;
        coreDevPackages = defaultPackages.coreDevPackages;
        devPackages = defaultPackages.devPackages;

        mkDevShell = env:
          pkgs.mkShell {
            name = "${packageName}-${env.python.version}";
            nativeBuildInputs = with pkgs;
              [
                env
              ]
              ++ devPackages;
            shellHook = ''
              export QUARTO_PYTHON=${pkgs.python310}/bin/python
              export LD_LIBRARY_PATH=${env}/lib:/usr/local/nvidia/lib64
            '';
          };

        containerImages = import ./nix/containers {
          inherit
            pkgs
            mkPoetryEnvWithSource
            gitHubOrg
            packageName
            sysPackages
            devPackages
            extraSysPackages
            ;
        };

        buildMultiUserNixImage = import ("${inputs.nixpod.outPath}" + "/containers/nix.nix");

        gcpProjectId = builtins.getEnv "GCP_PROJECT_ID";

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

        devShells = rec {
          pyrovelocity310 = mkDevShell pkgs.pyrovelocityDevEnv310;
          pyrovelocity311 = mkDevShell pkgs.pyrovelocityDevEnv311;
          pyrovelocity312 = mkDevShell pkgs.pyrovelocityDevEnv312;

          default = pyrovelocity310;
        };

        _module.args.pkgs = import inputs.nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
          overlays = [
            inputs.gitignore.overlay
            inputs.poetry2nix.overlays.default
            (import ./nix/pyrovelocity/overlay.nix {
              inherit poetry2nixOverrides;
              inherit (pkgs) stdenv;
            })
          ];
        };

        packages = {
          inherit (pkgs) pyrovelocity310 pyrovelocity311 pyrovelocity312;

          default = pkgs.pyrovelocity310;

          releaseEnv = pkgs.buildEnv {
            name = "release-env";
            paths = with pkgs; [poetry python310];
          };

          containerImage = containerImages.containerImage;

          devcontainerImage = containerImages.devcontainerImage;

          codeImage = import ./nix/containers/code.nix {
            inherit pkgs devPackages buildMultiUserNixImage;
            sudoImage = inputs'.nixpod.packages.sudoImage;
            homeActivationPackage = inputs'.nixpod.legacyPackages.homeConfigurations.jovyan.activationPackage;
            pythonPackageEnv = pkgs.pyrovelocityDevEnv310;
          };

          jupyterImage = import ./nix/containers/jupyter.nix {
            inherit pkgs devPackages buildMultiUserNixImage;
            sudoImage = inputs'.nixpod.packages.sudoImage;
            homeActivationPackage = inputs'.nixpod.legacyPackages.homeConfigurations.jovyan.activationPackage;
            pythonPackageEnv = pkgs.pyrovelocityDevEnv310;
          };
        };

        legacyPackages = {
          pyrovelocityManifest = inputs'.flocken.legacyPackages.mkDockerManifest {
            github = {
              enable = true;
              enableRegistry = false;
              token = "$GH_TOKEN";
            };
            autoTags = {
              branch = false;
            };
            registries = {
              "ghcr.io" = {
                enable = true;
                repo = "${gitHubOrg}/${packageName}";
                username = builtins.getEnv "GITHUB_ACTOR";
                password = "$GH_TOKEN";
              };
              # "cr.cluster.pyrovelocity.net" = {
              #   enable = true;
              #   repo = "${packageName}/${packageName}";
              #   username = "admin";
              #   password = "$ARTIFACT_REGISTRY_PASSWORD";
              # };
              "us-central1-docker.pkg.dev" = {
                enable = true;
                repo = "${gcpProjectId}/${packageName}/${packageName}";
                username = "_json_key_base64";
                password = "$ENCODED_GAR_SA_CREDS";
              };
            };
            version = builtins.getEnv "VERSION";
            images = builtins.map (sys: self.packages.${sys}.containerImage) includedSystems;
            tags = [
              # (builtins.getEnv "GIT_SHA_SHORT")
              (builtins.getEnv "GIT_SHA")
              # (builtins.getEnv "GIT_REF")
            ];
          };

          pyrovelocitydevManifest = inputs'.flocken.legacyPackages.mkDockerManifest {
            github = {
              enable = true;
              enableRegistry = false;
              token = "$GH_TOKEN";
            };
            autoTags = {
              branch = false;
            };
            registries = {
              "ghcr.io" = {
                enable = true;
                repo = "${gitHubOrg}/${packageName}dev";
                username = builtins.getEnv "GITHUB_ACTOR";
                password = "$GH_TOKEN";
              };
              # "cr.cluster.pyrovelocity.net" = {
              #   enable = true;
              #   repo = "${packageName}/${packageName}dev";
              #   username = "admin";
              #   password = "$ARTIFACT_REGISTRY_PASSWORD";
              # };
              "us-central1-docker.pkg.dev" = {
                enable = true;
                repo = "${gcpProjectId}/${packageName}/${packageName}dev";
                username = "_json_key_base64";
                password = "$ENCODED_GAR_SA_CREDS";
              };
            };
            version = builtins.getEnv "VERSION";
            images = builtins.map (sys: self.packages.${sys}.devcontainerImage) includedSystems;
            tags = [
              # (builtins.getEnv "GIT_SHA_SHORT")
              (builtins.getEnv "GIT_SHA")
              # (builtins.getEnv "GIT_REF")
            ];
          };

          pyrovelocitycodeManifest = inputs'.flocken.legacyPackages.mkDockerManifest {
            github = {
              enable = true;
              enableRegistry = false;
              token = "$GH_TOKEN";
            };
            autoTags = {
              branch = false;
            };
            registries = {
              "ghcr.io" = {
                enable = true;
                repo = "${gitHubOrg}/${packageName}code";
                username = builtins.getEnv "GITHUB_ACTOR";
                password = "$GH_TOKEN";
              };
              # "cr.cluster.pyrovelocity.net" = {
              #   enable = true;
              #   repo = "${packageName}/${packageName}code";
              #   username = "admin";
              #   password = "$ARTIFACT_REGISTRY_PASSWORD";
              # };
              "us-central1-docker.pkg.dev" = {
                enable = true;
                repo = "${gcpProjectId}/${packageName}/${packageName}code";
                username = "_json_key_base64";
                password = "$ENCODED_GAR_SA_CREDS";
              };
            };
            version = builtins.getEnv "VERSION";
            images = builtins.map (sys: self.packages.${sys}.codeImage) includedSystems;
            tags = [
              # (builtins.getEnv "GIT_SHA_SHORT")
              (builtins.getEnv "GIT_SHA")
              # (builtins.getEnv "GIT_REF")
            ];
          };

          pyrovelocityjupyterManifest = inputs'.flocken.legacyPackages.mkDockerManifest {
            github = {
              enable = true;
              enableRegistry = false;
              token = "$GH_TOKEN";
            };
            autoTags = {
              branch = false;
            };
            registries = {
              "ghcr.io" = {
                enable = true;
                repo = "${gitHubOrg}/${packageName}jupyter";
                username = builtins.getEnv "GITHUB_ACTOR";
                password = "$GH_TOKEN";
              };
              # "cr.cluster.pyrovelocity.net" = {
              #   enable = true;
              #   repo = "${packageName}/${packageName}jupyter";
              #   username = "admin";
              #   password = "$ARTIFACT_REGISTRY_PASSWORD";
              # };
              "us-central1-docker.pkg.dev" = {
                enable = true;
                repo = "${gcpProjectId}/${packageName}/${packageName}jupyter";
                username = "_json_key_base64";
                password = "$ENCODED_GAR_SA_CREDS";
              };
            };
            version = builtins.getEnv "VERSION";
            images = builtins.map (sys: self.packages.${sys}.jupyterImage) includedSystems;
            tags = [
              # (builtins.getEnv "GIT_SHA_SHORT")
              (builtins.getEnv "GIT_SHA")
              # (builtins.getEnv "GIT_REF")
            ];
          };
        };
      };
    };
}
