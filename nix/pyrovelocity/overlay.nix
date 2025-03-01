{
  poetry2nixOverrides,
  stdenv,
  ...
}: self: super: let
  mkPoetryEnv = {
    groups,
    python,
    extras ? ["*"],
  }:
    self.poetry2nix.mkPoetryEnv {
      inherit python groups extras;
      projectDir = self.gitignoreSource ../../.;
      editablePackageSources = {pyrovelocity = self.gitignoreSource ../../src;};
      overrides = poetry2nixOverrides;
      preferWheels = true;
    };

  mkPoetryDevEnv = python:
    mkPoetryEnv {
      inherit python;
      groups = ["main" "workflows" "docs" "test"];
    };
in {
  pyrovelocity310 = self.callPackage ./pyrovelocity.nix {
    python3 = self.python310;
    inherit poetry2nixOverrides stdenv;
  };
  pyrovelocity311 = self.callPackage ./pyrovelocity.nix {
    python3 = self.python311;
    inherit poetry2nixOverrides stdenv;
  };
  pyrovelocity312 = self.callPackage ./pyrovelocity.nix {
    python3 = self.python312;
    inherit poetry2nixOverrides stdenv;
  };
  # quarto = self.callPackage ../quarto {};

  pyrovelocityDevEnv310 = mkPoetryDevEnv self.python310;
  pyrovelocityDevEnv311 = mkPoetryDevEnv self.python311;
  pyrovelocityDevEnv312 = mkPoetryDevEnv self.python312;
}
