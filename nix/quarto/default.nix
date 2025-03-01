{
  stdenv,
  lib,
  esbuild,
  dart-sass,
  deno,
  pandoc,
  python3,
  fetchurl,
  makeWrapper,
  rWrapper,
  rPackages,
  autoPatchelfHook,
  extraRPackages ? [],
  extraPythonPackages ? ps: with ps; [],
}: let
  platforms = {
    "x86_64-linux" = "linux-amd64";
    "aarch64-linux" = "linux-arm64";
    "aarch64-darwin" = "macos";
  };
  shas = {
    "x86_64-linux" = "sha256-15fHlnE6V8FNgRX0mkXWJqFkeGlwlqBCHy0tmA5fnUo=";
    "aarch64-linux" = "sha256-yzzaMnKyeEGGI3Col7iD6FAF3a6bXlfsE8EHmNRu4LY=";
    "aarch64-darwin" = "sha256-W0IvOWdW7g7iaJcK6FF3X+1+EAWuqYUA1Zt/Es2aThY=";
  };
  inherit (stdenv.hostPlatform) system;
in
  stdenv.mkDerivation rec {
    pname = "quarto";
    version = "1.6.39";
    src = fetchurl {
      url = "https://github.com/quarto-dev/quarto-cli/releases/download/v${version}/quarto-${version}-${platforms.${system}}.tar.gz";
      sha256 = shas.${system};
    };

    preUnpack = lib.optionalString stdenv.isDarwin "mkdir ${sourceRoot}";
    sourceRoot = lib.optionalString stdenv.isDarwin "quarto-${version}";
    unpackCmd = lib.optionalString stdenv.isDarwin "tar xzf $curSrc --directory=$sourceRoot";

    nativeBuildInputs = lib.optionals stdenv.isLinux [autoPatchelfHook] ++ [makeWrapper];

    preFixup = ''
      wrapProgram $out/bin/quarto \
        --prefix QUARTO_ESBUILD : ${lib.getExe esbuild} \
        --prefix QUARTO_DART_SASS : ${lib.getExe dart-sass} \
        --prefix QUARTO_DENO : ${lib.getExe deno} \
        --prefix QUARTO_PANDOC : ${lib.getExe pandoc} \
        ${lib.optionalString (rWrapper != null) "--prefix QUARTO_R : ${rWrapper.override {packages = with rPackages; [dplyr reticulate rmarkdown tidyr] ++ extraRPackages;}}/bin/R"}
        # ${lib.optionalString (python3 != null) "--prefix QUARTO_PYTHON : ${python3.withPackages (ps: with ps; [jupyter jupyter-cache ipython] ++ (extraPythonPackages ps))}/bin/python3"}
    '';

    installPhase = ''
      runHook preInstall

      mkdir -p $out/bin $out/share

      rm -r bin/tools/*/deno*

      mv bin/* $out/bin
      mv share/* $out/share

      runHook postInstall
    '';

    meta = with lib; {
      description = "Open-source scientific and technical publishing system built on Pandoc";
      mainProgram = "quarto";
      longDescription = ''
        Quarto is an open-source scientific and technical publishing system built on Pandoc.
        Quarto documents are authored using markdown, an easy to write plain text format.
      '';
      homepage = "https://quarto.org/";
      changelog = "https://github.com/quarto-dev/quarto-cli/releases/tag/v${version}";
      license = licenses.gpl2Plus;
      maintainers = with maintainers; [minijackson mrtarantoga];
      platforms = ["x86_64-linux" "aarch64-linux" "aarch64-darwin"];
      sourceProvenance = with sourceTypes; [binaryNativeCode binaryBytecode];
    };
  }
