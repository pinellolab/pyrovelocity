{
  poetry2nix,
  lib,
  stdenv,
  writeText,
  autoPatchelfHook,
  cudaPackages_12_1,
  tbb_2021_11,
}:
poetry2nix.overrides.withDefaults (
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
      {
        cloudpickle = ["flit-core"];
        feather-format = ["setuptools"];
        hydra-core = ["setuptools"];
        hydra-joblib-launcher = ["setuptools"];
        hydra-zen = ["setuptools"];
        marshmallow-jsonschema = ["setuptools"];
        matplotlib-venn = ["setuptools"];
        memoized-property = ["setuptools"];
        numba = [tbb_2021_11];
        session-info = ["setuptools"];
        tinytimer = ["setuptools"];
        typechecks = ["setuptools"];
        xdoctest = ["setuptools"];
      };

    conditionalOverrides =
      if stdenv.isDarwin
      then {
        grpcio = super.grpcio.override {preferWheel = true;};
      }
      else if stdenv.hostPlatform.system == "x86_64-linux"
      then {
        torch = super.torch.overridePythonAttrs (attrs: {
          nativeBuildInputs =
            attrs.nativeBuildInputs
            or []
            ++ [
              autoPatchelfHook
            ];
          buildInputs =
            attrs.buildInputs
            or []
            ++ [
              self.nvidia-cublas-cu12
              self.nvidia-cuda-cupti-cu12
              self.nvidia-cuda-nvrtc-cu12
              self.nvidia-cuda-runtime-cu12
              self.nvidia-cudnn-cu12
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
            addAutoPatchelfSearchPath "${self.nvidia-cuda-cupti-cu12}/${self.python.sitePackages}/nvidia/cuda_cupti/lib"
            addAutoPatchelfSearchPath "${self.nvidia-cuda-nvrtc-cu12}/${self.python.sitePackages}/nvidia/cuda_nvrtc/lib"
            addAutoPatchelfSearchPath "${self.nvidia-cuda-runtime-cu12}/${self.python.sitePackages}/nvidia/cuda_runtime/lib"
            addAutoPatchelfSearchPath "${self.nvidia-cudnn-cu12}/${self.python.sitePackages}/nvidia/cudnn/lib"
            addAutoPatchelfSearchPath "${self.nvidia-cufft-cu12}/${self.python.sitePackages}/nvidia/cufft/lib"
            addAutoPatchelfSearchPath "${self.nvidia-curand-cu12}/${self.python.sitePackages}/nvidia/curand/lib"
            addAutoPatchelfSearchPath "${self.nvidia-cusolver-cu12}/${self.python.sitePackages}/nvidia/cusolver/lib"
            addAutoPatchelfSearchPath "${self.nvidia-cusparse-cu12}/${self.python.sitePackages}/nvidia/cusparse/lib"
            addAutoPatchelfSearchPath "${self.nvidia-nccl-cu12}/${self.python.sitePackages}/nvidia/nccl/lib"
            addAutoPatchelfSearchPath "${self.nvidia-nvjitlink-cu12}/${self.python.sitePackages}/nvidia/nvjitlink/lib"
            addAutoPatchelfSearchPath "${self.nvidia-nvtx-cu12}/${self.python.sitePackages}/nvidia/nvtx/lib"
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
      lib.genAttrs [
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
      dm-tree = super.dm-tree.override {preferWheel = true;};
      duckdb = super.duckdb.override {preferWheel = true;};
      h5py = super.h5py.override {preferWheel = true;};
      hydra-core = super.hydra-core.override {preferWheel = true;};
      hydra-joblib-launcher = super.hydra-joblib-launcher.override {preferWheel = true;};
      mkdocs-material = super.mkdocs-material.override {preferWheel = false;};
      nvidia-cudnn-cu12 = super.nvidia-cudnn-cu12.overridePythonAttrs (old: {
        nativeBuildInputs = old.nativeBuildInputs or [] ++ [autoPatchelfHook];
        propagatedBuildInputs =
          old.propagatedBuildInputs
          or []
          ++ [
            self.nvidia-cublas-cu12
            cudaPackages_12_1.cudnn
          ];
        postFixup = nvidiaPostFixup;
      });
      nvidia-cusparse-cu12 = super.nvidia-cusparse-cu12.overridePythonAttrs (old: {
        propagatedBuildInputs =
          old.propagatedBuildInputs
          or []
          ++ [
            self.nvidia-nvjitlink-cu12
            cudaPackages_12_1.libnvjitlink
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
            cudaPackages_12_1.libcublas
            cudaPackages_12_1.libcusparse
          ];
        postFixup = nvidiaPostFixup;
      });
      pyarrow = super.pyarrow.override {preferWheel = true;};
      scikit-learn = super.scikit-learn.override {preferWheel = true;};
      scipy = super.scipy.override {preferWheel = true;};
      tensorstore = super.tensorstore.override {preferWheel = true;};
      yarl = super.yarl.override {preferWheel = true;};
      gtfparse = super.gtfparse.overridePythonAttrs (
        _old: {
          postInstall = ''
            rm -f $out/lib/python3.10/site-packages/requirements.txt
            rm -f $out/lib/python3.11/site-packages/requirements.txt
            rm -f $out/lib/python3.12/site-packages/requirements.txt
          '';
        }
      );
      # matplotlib = super.matplotlib.overridePythonAttrs (
      #   old:
      #   let
      #     disable-macos-fonts = writeText "disable-macos-fonts.patch" ''
      #       --- a/lib/matplotlib/font_manager.py
      #       +++ b/lib/matplotlib/font_manager.py
      #       @@ -289,7 +289,6 @@
      #                 else:
      #                     installed_fonts = _get_fontconfig_fonts()
      #                     if sys.platform == 'darwin':
      #       -                installed_fonts += _get_macos_fonts()
      #                         fontpaths = [*X11FontDirectories, *OSXFontDirectories]
      #                     else:
      #                         fontpaths = X11FontDirectories
      #     '';
      #   in
      #   {
      #     preferWheel = false;
      #     patches = old.patches or [ ] ++ [
      #       # ./disable_macos_fonts.patch
      #       disable-macos-fonts
      #     ];
      #   }
      # );
      optax = super.optax.overridePythonAttrs (
        _old: {
          postInstall = ''
            rm -f $out/lib/python3.10/site-packages/docs/conf.py
            rm -fr $out/lib/python3.10/site-packages/docs/__pycache__
            rm -f $out/lib/python3.11/site-packages/docs/conf.py
            rm -fr $out/lib/python3.11/site-packages/docs/__pycache__
            rm -f $out/lib/python3.12/site-packages/docs/conf.py
            rm -fr $out/lib/python3.12/site-packages/docs/__pycache__
          '';
        }
      );
    }
    // conditionalOverrides
)
