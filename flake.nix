{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
    nixpkgs-2305.url = "github:NixOS/nixpkgs/nixos-23.05";
    nixpkgs-unstable.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    flake-parts.url = "github:hercules-ci/flake-parts";
    # flake-utils.url = github:numtide/flake-utils;
    poetry2nix = {
      # url = github:nix-community/poetry2nix;
      url = github:cameronraysmith/poetry2nix/patch;
      inputs = {
        nixpkgs.follows = "nixpkgs";
        # flake-utils.follows = "flake-utils";
      };
    };
    flocken = {
      url = "github:mirkolenz/flocken/v2";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  nixConfig = {
    extra-trusted-public-keys = [
      "pyrovelocity.cachix.org-1:+aX2YY45ZywieTsD2CnXLedN8RfKuRl6vL7+rLTCgnc="
    ];
    extra-substituters = [
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

        pyPkgsBuildRequirements = {
          cloudpickle = ["flit-core"];
          feather-format = ["setuptools"];
          hydra-core = ["setuptools"];
          hydra-joblib-launcher = ["setuptools"];
          hydra-zen = ["setuptools"];
          marshmallow-jsonschema = ["setuptools"];
          numba = [pkgs.tbb_2021_8];
          session-info = ["setuptools"];
          xdoctest = ["setuptools"];
        };

        poetry2nixOverrides = pkgs.poetry2nix.overrides.withDefaults (
          self: super: let
            buildInputsOverrides =
              builtins.mapAttrs (
                package: buildRequirements:
                  (builtins.getAttr package super).overridePythonAttrs (old: {
                    buildInputs =
                      (old.buildInputs or [])
                      ++ (builtins.map (pkg:
                        if builtins.isString pkg
                        then builtins.getAttr pkg super
                        else pkg)
                      buildRequirements);
                  })
              )
              pyPkgsBuildRequirements;
            conditionalOverrides =
              if pkgs.stdenv.isDarwin
              then {
                grpcio = super.grpcio.override {preferWheel = false;};
              }
              else if pkgs.stdenv.isLinux
              then {
                torch = super.torch.overridePythonAttrs (attrs: {
                  nativeBuildInputs =
                    attrs.nativeBuildInputs
                    or []
                    ++ [
                      pkgs.autoPatchelfHook
                    ];
                  buildInputs =
                    attrs.buildInputs
                    or []
                    ++ [
                      self.nvidia-cudnn-cu12
                      self.nvidia-cuda-nvrtc-cu12
                      self.nvidia-cuda-runtime-cu12
                      self.nvidia-cublas-cu12
                      self.nvidia-cuda-cupti-cu12
                      self.nvidia-cufft-cu12
                      self.nvidia-curand-cu12
                      self.nvidia-cusolver-cu12
                      self.nvidia-cusparse-cu12
                      self.nvidia-nccl-cu12
                      self.nvidia-nvjitlink-cu12
                      self.nvidia-nvtx-cu12
                    ];
                  postInstall = ''
                    addAutoPatchelfSearchPath "${self.nvidia-cublas-cu12}/${self.python.sitePackages}/nvidia/cublas/lib"
                    addAutoPatchelfSearchPath "${self.nvidia-cudnn-cu12}/${self.python.sitePackages}/nvidia/cudnn/lib"
                    addAutoPatchelfSearchPath "${self.nvidia-cuda-nvrtc-cu12}/${self.python.sitePackages}/nvidia/cuda_nvrtc/lib"
                    addAutoPatchelfSearchPath "${self.nvidia-cusparse-cu12}/${self.python.sitePackages}/nvidia/cusparse/lib"
                    addAutoPatchelfSearchPath "${self.nvidia-cusolver-cu12}/${self.python.sitePackages}/nvidia/cusolver/lib"
                    addAutoPatchelfSearchPath "${self.nvidia-nvjitlink-cu12}/${self.python.sitePackages}/nvidia/nvjitlink/lib"
                  '';
                });
              }
              else {};
          in
            buildInputsOverrides
            // {
              h5py = super.h5py.override {preferWheel = true;};
              hydra-core = super.hydra-core.override {preferWheel = true;};
              hydra-joblib-launcher = super.hydra-joblib-launcher.override {preferWheel = true;};
              mkdocs-material = super.mkdocs-material.override {preferWheel = false;};
              nvidia-cublas-cu12 = super.nvidia-cublas-cu12.overridePythonAttrs (_: {
                postFixup = ''
                  rm -r $out/${self.python.sitePackages}/nvidia/{__pycache__,__init__.py}
                '';
              });
              nvidia-cudnn-cu12 = super.nvidia-cudnn-cu12.overridePythonAttrs (old: {
                nativeBuildInputs = old.nativeBuildInputs or [] ++ [pkgs.autoPatchelfHook];
                propagatedBuildInputs =
                  old.propagatedBuildInputs
                  or []
                  ++ [
                    self.nvidia-cublas-cu12
                    pkgs_unstable.cudaPackages_12_1.cudnn
                  ];
                postFixup = ''
                  rm -r $out/${self.python.sitePackages}/nvidia/{__pycache__,__init__.py}
                '';
              });
              nvidia-cufft-cu12 = super.nvidia-cufft-cu12.overridePythonAttrs (_: {
                postFixup = ''
                  rm -r $out/${self.python.sitePackages}/nvidia/{__pycache__,__init__.py}
                '';
              });
              nvidia-curand-cu12 = super.nvidia-curand-cu12.overridePythonAttrs (_: {
                postFixup = ''
                  rm -r $out/${self.python.sitePackages}/nvidia/{__pycache__,__init__.py}
                '';
              });
              nvidia-cusparse-cu12 = super.nvidia-cusparse-cu12.overridePythonAttrs (old: {
                propagatedBuildInputs =
                  old.propagatedBuildInputs
                  or []
                  ++ [
                    self.nvidia-nvjitlink-cu12
                    pkgs_unstable.cudaPackages_12_1.libnvjitlink
                  ];
                postFixup = ''
                  rm -r $out/${self.python.sitePackages}/nvidia/{__pycache__,__init__.py}
                '';
              });
              nvidia-cusolver-cu12 = super.nvidia-cusolver-cu12.overridePythonAttrs (old: {
                propagatedBuildInputs =
                  old.propagatedBuildInputs
                  or []
                  ++ [
                    self.nvidia-cublas-cu12
                    self.nvidia-cusparse-cu12
                    pkgs_unstable.cudaPackages_12_1.libcublas
                    pkgs_unstable.cudaPackages_12_1.libcusparse
                  ];
                postFixup = ''
                  rm -r $out/${self.python.sitePackages}/nvidia/{__pycache__,__init__.py}
                '';
              });
              nvidia-cuda-cupti-cu12 = super.nvidia-cuda-cupti-cu12.overridePythonAttrs (_: {
                postFixup = ''
                  rm -r $out/${self.python.sitePackages}/nvidia/{__pycache__,__init__.py}
                '';
              });
              nvidia-cuda-nvrtc-cu12 = super.nvidia-cuda-nvrtc-cu12.overridePythonAttrs (_: {
                postFixup = ''
                  rm -r $out/${self.python.sitePackages}/nvidia/{__pycache__,__init__.py}
                '';
              });
              nvidia-cuda-runtime-cu12 = super.nvidia-cuda-runtime-cu12.overridePythonAttrs (_: {
                postFixup = ''
                  rm -r $out/${self.python.sitePackages}/nvidia/{__pycache__,__init__.py}
                '';
              });
              nvidia-nccl-cu12 = super.nvidia-nccl-cu12.overridePythonAttrs (_: {
                postFixup = ''
                  rm -r $out/${self.python.sitePackages}/nvidia/{__pycache__,__init__.py}
                '';
              });
              nvidia-nvtx-cu12 = super.nvidia-nvtx-cu12.overridePythonAttrs (_: {
                postFixup = ''
                  rm -r $out/${self.python.sitePackages}/nvidia/{__pycache__,__init__.py}
                '';
              });
              pyarrow = super.pyarrow.override {preferWheel = true;};
              scipy = super.scipy.override {preferWheel = true;};
              yarl = super.yarl.override {preferWheel = true;};
              optax = super.optax.overridePythonAttrs (
                _old: {
                  postInstall = ''
                    rm -f $out/lib/python3.10/site-packages/docs/conf.py
                    rm -fr $out/lib/python3.10/site-packages/docs/__pycache__
                  '';
                }
              );
            }
            // conditionalOverrides
        );

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
        packageSrcPath = "/root/${packageName}/src";

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

        sysPackages = with pkgs;
          [
            bashInteractive
            coreutils
            cacert
            direnv
            file
            findutils
            gnutar
            gzip
            less
            libgcc
            nix
            procps
            which
          ]
          ++ lib.optional (lib.elem system pkgs.shadow.meta.platforms) shadow;

        mkRootNss = pkgs.runCommand "mkRootNss" {} ''
          mkdir -p $out/etc

          cat > $out/etc/passwd <<EOF
          root:x:0:0:root user:/var/empty:/bin/sh
          nobody:x:65534:65534:nobody:/var/empty:/bin/sh
          EOF

          cat > $out/etc/group <<EOF
          root:x:0:root
          nobody:x:65534:
          nixbld:x:30000:nobody
          EOF

          echo "hosts: files dns" > $out/etc/nsswitch.conf

          mkdir -p $out/tmp
          mkdir -p $out/root
        '';

        rcRoot = pkgs.runCommand "rcRoot" {} ''
          mkdir -p $out/root

          cat > $out/root/.zshrc <<EOF
          eval "\$(direnv hook zsh)"
          eval "\$(starship init zsh)"
          eval "\$(atuin init zsh)"
          EOF
        '';

        customNixConf = pkgs.runCommand "custom-nix-conf" {} ''
          mkdir -p $out/etc/nix
          cat > $out/etc/nix/nix.conf <<EOF
          build-users-group = nixbld
          experimental-features = nix-command flakes repl-flake
          bash-prompt-prefix = (nix:\$name)\040
          max-jobs = auto
          extra-nix-path = nixpkgs=flake:nixpkgs
          trusted-users = root
          EOF
        '';

        devPackages = with pkgs;
          [
            atuin
            bat
            bazelisk
            pkgs_unstable.cue
            gawk
            gh
            git
            gnugrep
            gnumake
            gnused
            gnupg
            helix
            htop
            jqp
            pkgs_unstable.k9s
            kubectl
            kubectx
            lazygit
            man-db
            man-pages
            neovim
            openvscode-server
            poetry
            poethepoet
            ripgrep
            skaffold
            starship
            pkgs_unstable.timoni
            tree
            yq-go
            zellij
            zsh
          ]
          ++ lib.optional (lib.elem system pkgs.nvitop.meta.platforms) nvitop;

        # The local path can be used instead of `builtins.fetchGit` applied to
        # the repository source url to be used in `packageGitRepoToContainer` to
        # place a copy of the local source in the devcontainer if it does not
        # exist on a ref+rev:
        # packageGitRepo = ./.;
        # OR
        # packageGitRepo = builtins.fetchGit ./.;
        # should also work as an alternative to directly copying the local repo
        # path, see https://github.com/NixOS/nix/pull/7706/files; however, the
        # explicit ref+rev should likely be preferred outside of development
        # experimentation
        packageGitRepo = builtins.fetchGit {
          name = "${packageName}-source";
          url = "https://github.com/${gitHubOrg}/${packageName}.git";
          # the ref is not strictly required when specifying a rev but it should
          # be included whenever possible or it may be necessary to include
          # allRefs = true;
          ref = builtins.getEnv "GIT_REF";
          # the rev can be omitted transiently in development to track the HEAD
          # of a ref but doing so requires `--impure` image builds (this may
          # already be required for other reasons, e.g. `builtins.getEnv`)
          rev = builtins.getEnv "GIT_SHA";
        };

        # The `chmod -R` command to make the root user's home directory writable
        # is only necessary to allow overwriting the source in the devcontainer
        # without rebuilding the image. Since it violates the intended
        # immutability of the image, it can be disabled in release images or
        # when image rebuilds are used to update the source during development.
        packageGitRepoToContainer = pkgs.runCommand "copy-package-git-repo" {} ''
          mkdir -p $out/root
          cp -r ${packageGitRepo} $out/root/${packageName}

          chmod -R 755 $out/root
        '';

        pythonPackages = [
          (mkPoetryEnvWithSource packageName packageSrcPath ["workflows"])
        ];
        devpythonPackages = [
          (mkPoetryEnvWithSource packageName packageSrcPath ["test" "docs" "workflows"])
        ];

        devcontainerContents = [
          # similar to pkgs.fakeNss
          mkRootNss
          (pkgs.buildEnv {
            name = "root";
            paths = sysPackages;
            pathsToLink = "/bin";
          })
          customNixConf
          rcRoot
          packageGitRepoToContainer
        ];

        makeContainerConfig = {
          pkgs,
          packageSrcPath,
          pythonPackages,
          containerPackages,
          cmd ? [],
          entrypoint ? [],
        }: {
          # Use default empty Entrypoint to completely defer to Cmd for flexible override
          Entrypoint = entrypoint;
          # but provide default Cmd to start zsh
          Cmd = cmd;
          User = "root";
          WorkingDir = "/root";
          Env = [
            "PATH=${pkgs.lib.makeBinPath containerPackages}:/root/.nix-profile/bin"
            "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
            "NIX_PAGER=cat"
            "USER=root"
            "HOME=/root"
            "GIT_REPO_NAME=${builtins.getEnv "GIT_REPO_NAME"}"
            "GIT_REF=${builtins.getEnv "GIT_REF"}"
            "GIT_SHA=${builtins.getEnv "GIT_SHA"}"
            "GIT_SHA_SHORT=${builtins.getEnv "GIT_SHA_SHORT"}"
            "PYTHONPATH=${packageSrcPath}:${pkgs.lib.strings.makeSearchPathOutput "" "lib/python3.10/site-packages" pythonPackages}"
            "LD_LIBRARY_PATH=/usr/local/nvidia/lib64"
            "NVIDIA_DRIVER_CAPABILITIES='compute,utility'"
            "NVIDIA_VISIBLE_DEVICES=all"
          ];
        };
        containerCmd = [
          "${pkgs.bashInteractive}/bin/bash"
        ];
        devcontainerCmd = [
          "${pkgs.bashInteractive}/bin/bash"
          "-c"
          "${pkgs.zsh}/bin/zsh"
        ];
        containerImageConfig = {
          name = "${packageName}";
          tag = "latest";
          created = "now";

          maxLayers = 123;

          contents = devcontainerContents;
          config = makeContainerConfig {
            pkgs = pkgs;
            packageSrcPath = packageSrcPath;
            pythonPackages = pythonPackages;
            containerPackages = sysPackages ++ pythonPackages;
            cmd = containerCmd;
          };
        };
        devcontainerImageConfig = {
          name = "${packageName}dev";
          # with mkDockerManifest, tags may be automatically generated from
          # git metadata
          tag = "latest";
          created = "now";

          # maxLayers <=127; defaults to 100
          maxLayers = 123;

          contents = devcontainerContents;
          config = makeContainerConfig {
            pkgs = pkgs;
            packageSrcPath = packageSrcPath;
            pythonPackages = devpythonPackages;
            containerPackages = sysPackages ++ devPackages ++ devpythonPackages;
            cmd = devcontainerCmd;
          };
        };

        gcpProjectId = builtins.getEnv "GCP_PROJECT_ID";

        # aarch64-linux may be disabled for more rapid image builds during
        # development setting NIX_IMAGE_SYSTEMS="x86_64-linux".
        # Note the usage of `preferWheels` as well.
        # images = with self.packages; [x86_64-linux.devcontainerImage aarch64-linux.devcontainerImage];
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
            buildInputs = with pkgs;
              [
                (mkPoetryEnvWithSource packageName ./src ["test" "docs" "workflows"])
              ]
              ++ devPackages;
          };
        };

        packages = {
          default = pkgs.poetry2nix.mkPoetryApplication (
            mkPoetryAttrs
            // {
              checkPhase = ''
                export NUMBA_CACHE_DIR=${builtins.getEnv "NUMBA_CACHE_DIR"}
                echo "NUMBA_CACHE_DIR: $NUMBA_CACHE_DIR"
                pytest
              '';
            }
          );

          releaseEnv = pkgs.buildEnv {
            name = "release-env";
            paths = with pkgs; [poetry python310];
          };

          containerImage = pkgs.dockerTools.buildLayeredImage containerImageConfig;
          containerStream = pkgs.dockerTools.streamLayeredImage containerImageConfig;

          devcontainerImage = pkgs.dockerTools.buildLayeredImage devcontainerImageConfig;
          devcontainerStream = pkgs.dockerTools.streamLayeredImage devcontainerImageConfig;
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
              username = "_json_key";
              password = builtins.getEnv "GOOGLE_APPLICATION_CREDENTIALS_DATA";
            };
          };
          version = builtins.getEnv "VERSION";
          images = builtins.map (sys: self.packages.${sys}.containerImage) includedSystems;
          tags = [(builtins.getEnv "GIT_SHA_SHORT") (builtins.getEnv "GIT_SHA") (builtins.getEnv "GIT_REF")];
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
              username = "_json_key";
              password = builtins.getEnv "GOOGLE_APPLICATION_CREDENTIALS_DATA";
            };
          };
          version = builtins.getEnv "VERSION";
          images = builtins.map (sys: self.packages.${sys}.devcontainerImage) includedSystems;
          tags = [(builtins.getEnv "GIT_SHA_SHORT") (builtins.getEnv "GIT_SHA") (builtins.getEnv "GIT_REF")];
        };
      };
    };
}
