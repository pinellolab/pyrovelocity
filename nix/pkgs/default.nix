{
  system,
  pkgs,
  pkgs_unstable,
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
      multirow
      ncctools
      rsfs
      sttools
      threeparttable
      vruler
      wrapfig
      xurl
      ;
  };

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
      time
      which
    ]
    ++ pkgs.lib.optional (pkgs.lib.elem system pkgs.shadow.meta.platforms) shadow;

  coreDevPackages = with pkgs; [
    atuin
    bat
    bazelisk
    pkgs_unstable.btop
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
    lmodern
    man-db
    man-pages
    neovim
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
  ];

  extDevPackages = with pkgs; [
    openvscode-server
    pandoc
    pkgs_unstable.quarto
    tex
  ];
in {
  sysPackages = sysPackages;
  coreDevPackages = coreDevPackages;
  devPackages = coreDevPackages ++ extDevPackages ++ pkgs.lib.optional (system == "x86_64-linux") pkgs.nvitop;
}
