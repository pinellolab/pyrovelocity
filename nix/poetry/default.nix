{
  pkgs,
  pkgs_unstable,
  self,
}: let
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
in
  pkgs.poetry2nix.overrides.withDefaults (
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
        else if pkgs.stdenv.hostPlatform.system == "x86_64-linux"
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

      nvidiaPostFixup = ''
        rm -r $out/${self.python.sitePackages}/nvidia/{__pycache__,__init__.py}
      '';

      nvidiaPackagesPostFixup = name:
        super.${name}.overridePythonAttrs (old: {
          postFixup = old.postFixup or "" + nvidiaPostFixup;
        });

      nvidiaCudaPostFixupOnlyPackages =
        pkgs.lib.genAttrs [
          "nvidia-cublas-cu12"
          "nvidia-cufft-cu12"
          "nvidia-curand-cu12"
          "nvidia-cuda-cupti-cu12"
          "nvidia-cuda-nvrtc-cu12"
          "nvidia-cuda-runtime-cu12"
          "nvidia-nccl-cu12"
          "nvidia-nvtx-cu12"
        ]
        nvidiaPackagesPostFixup;
    in
      buildInputsOverrides
      // nvidiaCudaPostFixupOnlyPackages
      // {
        h5py = super.h5py.override {preferWheel = true;};
        hydra-core = super.hydra-core.override {preferWheel = true;};
        hydra-joblib-launcher = super.hydra-joblib-launcher.override {preferWheel = true;};
        mkdocs-material = super.mkdocs-material.override {preferWheel = false;};
        nvidia-cudnn-cu12 = super.nvidia-cudnn-cu12.overridePythonAttrs (old: {
          nativeBuildInputs = old.nativeBuildInputs or [] ++ [pkgs.autoPatchelfHook];
          propagatedBuildInputs =
            old.propagatedBuildInputs
            or []
            ++ [
              self.nvidia-cublas-cu12
              pkgs_unstable.cudaPackages_12_1.cudnn
            ];
          postFixup = nvidiaPostFixup;
        });
        nvidia-cusparse-cu12 = super.nvidia-cusparse-cu12.overridePythonAttrs (old: {
          propagatedBuildInputs =
            old.propagatedBuildInputs
            or []
            ++ [
              self.nvidia-nvjitlink-cu12
              pkgs_unstable.cudaPackages_12_1.libnvjitlink
            ];
          postFixup = nvidiaPostFixup;
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
          postFixup = nvidiaPostFixup;
        });
        pyarrow = super.pyarrow.override {preferWheel = true;};
        scipy = super.scipy.override {preferWheel = true;};
        tensorstore = super.tensorstore.override {preferWheel = true;};
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
  )