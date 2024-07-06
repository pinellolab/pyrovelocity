{
  pkgs,
  mkPoetryEnvWithSource,
  gitHubOrg,
  packageName,
  sysPackages,
  devPackages,
  extraSysPackages,
}: let
  packageRootDir = "/root/${packageName}";
  packageSrcPath = "${packageRootDir}/src";

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

    mkdir -p $out/lib64
    ln -sf ${pkgs.glibc}/lib/ld-linux-x86-64.so.2 $out/lib64/ld-linux-x86-64.so.2
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
    # rev = "7f37d369baf6d5775f3c4b13b69208adbb93535e";
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

  pythonPackageEnv = mkPoetryEnvWithSource packageName packageSrcPath ["workflows"];
  pythonPackages = [
    pythonPackageEnv
  ];
  devpythonPackageEnv = mkPoetryEnvWithSource packageName packageSrcPath ["test" "docs" "workflows"];
  devpythonPackages = [
    devpythonPackageEnv
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
    # packageGitRepoToContainer
  ];

  makeContainerConfig = {
    pkgs,
    packageSrcPath,
    pythonPackageEnv,
    pythonPackages,
    containerPackages,
    cmd ? [],
    entrypoint ? [],
    extraEnv ? [],
  }: {
    # Use default empty Entrypoint to completely defer to Cmd for flexible override
    Entrypoint = entrypoint;
    # but provide default Cmd to start zsh
    Cmd = cmd;
    User = "root";
    WorkingDir = "/root";
    Env =
      [
        "PATH=${pkgs.lib.makeBinPath containerPackages}:/root/.nix-profile/bin"
        "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
        "NIX_PAGER=cat"
        "USER=root"
        "HOME=/root"
        # "GIT_REPO_NAME=${builtins.getEnv "GIT_REPO_NAME"}"
        # "GIT_REF=${builtins.getEnv "GIT_REF"}"
        # "GIT_SHA=${builtins.getEnv "GIT_SHA"}"
        # "GIT_SHA_SHORT=${builtins.getEnv "GIT_SHA_SHORT"}"
        "PYTHONPATH=${packageSrcPath}:${pkgs.lib.strings.makeSearchPathOutput "" "lib/python3.10/site-packages" pythonPackages}"
        "LD_LIBRARY_PATH=${pythonPackageEnv}/lib:/usr/local/nvidia/lib64"
        "NVIDIA_DRIVER_CAPABILITIES='compute,utility'"
        "NVIDIA_VISIBLE_DEVICES=all"
        "LANG=en_US.UTF-8"
        "LC_ALL=en_US.UTF-8"
        "LC_CTYPE=en_US.UTF-8"
      ]
      ++ extraEnv;
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
    name = "base-${packageName}";
    tag = "latest";
    # created = "now";

    maxLayers = 118;
    compressor = "none";

    contents = devcontainerContents;
    config = makeContainerConfig {
      pkgs = pkgs;
      packageSrcPath = packageSrcPath;
      pythonPackageEnv = pythonPackageEnv;
      pythonPackages = pythonPackages;
      containerPackages = sysPackages ++ pythonPackages;
      cmd = containerCmd;
    };
  };
  devcontainerImageConfig = {
    name = "base-${packageName}dev";
    tag = "latest";
    # created = "now";

    maxLayers = 118;
    compressor = "none";

    contents = devcontainerContents;
    # runAsRoot = ''
    #   #!${pkgs.runtimeShell}
    #   export PATH=${pkgs.lib.makeBinPath [ pkgs.gnumake pkgs.openvscode-server ]}:/bin:/usr/bin:$PATH
    #   cd ${packageRootDir}
    #   make vscode-install-extensions
    # '';
    config = makeContainerConfig {
      pkgs = pkgs;
      packageSrcPath = packageSrcPath;
      pythonPackageEnv = devpythonPackageEnv;
      pythonPackages = devpythonPackages;
      containerPackages = sysPackages ++ extraSysPackages ++ devPackages ++ devpythonPackages;
      cmd = devcontainerCmd;
      extraEnv = [
        "QUARTO_PYTHON=${pkgs.python310}/bin/python"
      ];
    };
  };
in rec {
  baseContainerImage = pkgs.dockerTools.buildLayeredImageWithNixDb containerImageConfig;
  baseDevContainerImage = pkgs.dockerTools.buildLayeredImageWithNixDb devcontainerImageConfig;
  containerImage = pkgs.dockerTools.buildLayeredImageWithNixDb {
    name = "${packageName}";
    tag = "latest";
    maxLayers = 121;
    compressor = "zstd";
    fromImage = baseContainerImage;
    contents = [packageGitRepoToContainer];
    config = {
      Env = [
        "GIT_REPO_NAME=${builtins.getEnv "GIT_REPO_NAME"}"
        "GIT_REF=${builtins.getEnv "GIT_REF"}"
        "GIT_SHA=${builtins.getEnv "GIT_SHA"}"
        "GIT_SHA_SHORT=${builtins.getEnv "GIT_SHA_SHORT"}"
      ];
    };
  };
  devcontainerImage = pkgs.dockerTools.buildLayeredImageWithNixDb {
    name = "${packageName}dev";
    tag = "latest";
    maxLayers = 121;
    compressor = "zstd";
    fromImage = baseDevContainerImage;
    contents = [packageGitRepoToContainer];
    config = {
      Env = [
        "GIT_REPO_NAME=${builtins.getEnv "GIT_REPO_NAME"}"
        "GIT_REF=${builtins.getEnv "GIT_REF"}"
        "GIT_SHA=${builtins.getEnv "GIT_SHA"}"
        "GIT_SHA_SHORT=${builtins.getEnv "GIT_SHA_SHORT"}"
      ];
    };
  };
}
