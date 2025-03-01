{
  system,
  pkgs,
}: let
  tex = pkgs.texlive.combine {
    inherit
      (pkgs.texlive)
      scheme-small
      algorithm2e
      algorithmicx
      algorithms
      algpseudocodex
      apacite
      appendix
      caption
      cm-super
      dvipng
      framed
      git-latexdiff
      latexdiff
      latexmk
      latexpand
      multirow
      ncctools
      placeins
      rsfs
      sttools
      threeparttable
      type1cm
      vruler
      wrapfig
      xurl
      ;
  };

  extraSysPackages = with pkgs; [
    atuin
    bat
    btop
    curl
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
    kubectl
    kubectx
    k9s
    lazygit
    man-db
    man-pages
    neovim
    openvscode-server
    ripgrep
    skaffold
    starship
    wget
    zellij
    zsh
  ];

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
      lmodern
      nix
      procps
      tex
      time
      which
    ]
    ++ pkgs.lib.optional (pkgs.lib.elem system pkgs.shadow.meta.platforms) shadow;

  coreDevPackages = with pkgs; [
    bazelisk
    bazel-buildtools
    cue
    poetry
    poethepoet
    timoni
    tree
    yq-go
  ];

  # TODO: all dvc packages are disabled due to
  # https://github.com/NixOS/nixpkgs/issues/338146
  dvcPackage = pkgs.python312Packages.dvc;

  dvcWithOptionalRemotes = pkgs.dvc.override {
    enableGoogle = true;
    enableAWS = true;
    enableAzure = true;
    enableSSH = true;
  };

  dvcWithGoogleAuth = (dvcPackage.overrideAttrs (oldAttrs: {
    buildInputs = oldAttrs.buildInputs ++ (with pkgs.python312Packages; [
      dvc-gs
      gcsfs
      dvc-objects
      aiohttp
      crcmod
      decorator
      fsspec
      google-auth
      google-auth-oauthlib
      google-cloud-storage
      requests
      ujson
    ]);
  })).overridePythonAttrs (oldAttrs: {
    dependencies = oldAttrs.dependencies ++ (with pkgs.python312Packages; [
      dvc-gs
      gcsfs
      dvc-objects
      aiohttp
      crcmod
      decorator
      fsspec
      google-auth
      google-auth-oauthlib
      google-cloud-storage
      requests
      ujson
    ]);
    pythonImportsCheck = [
      "dvc"
      "dvc.api"
      "dvc_gs"
      "google.auth"
    ];
  });

  extraDevPackages = with pkgs; [
    lmodern
    pandoc
    quarto
    tex
    yarn-berry
  ];
in {
  sysPackages = sysPackages;
  extraSysPackages = extraSysPackages;
  coreDevPackages = coreDevPackages;
  devPackages = coreDevPackages ++ extraDevPackages ++ pkgs.lib.optional (system == "x86_64-linux") pkgs.nvitop;
}
