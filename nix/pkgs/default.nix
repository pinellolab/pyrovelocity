{
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
in {
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
      lmodern
      man-db
      man-pages
      neovim
      openvscode-server
      pandoc
      poetry
      poethepoet
      pkgs_unstable.quarto
      ripgrep
      skaffold
      starship
      tex
      pkgs_unstable.timoni
      tree
      yq-go
      zellij
      zsh
    ]
    ++ pkgs.lib.optional (system == "x86_64-linux") nvitop;
}
