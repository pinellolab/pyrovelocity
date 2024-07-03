{
  pkgs,
  devPackages,
  buildMultiUserNixImage,
  sudoImage,
  homeActivationPackage,
  pythonPackageEnv,
}: let
  username = "jovyan";
  storeOwner = {
    uid = 1000;
    gid = 0;
    uname = username;
    gname = "wheel";
  };
  activateUserHomeScript = pkgs.writeScript "activate-user-home-run" ''
    #!/command/with-contenv ${pkgs.runtimeShell}
    printf "activating home manager\n\n"
    /activate
    printf "home manager environment\n\n"
    printenv | sort
    printf "====================\n\n"
  '';
  activateUserHomeService = pkgs.runCommand "activate-user-home" {} ''
    mkdir -p $out/etc/cont-init.d
    ln -s ${activateUserHomeScript} $out/etc/cont-init.d/01-activate-user-home
  '';
  createJupyterLogScript = pkgs.writeScript "create-jupyter-log-run" ''
    #!/command/with-contenv ${pkgs.runtimeShell}
    /run/wrappers/bin/sudo mkdir -p /var/log/jupyterlab
    /run/wrappers/bin/sudo chown nobody:nobody /var/log/jupyterlab
    /run/wrappers/bin/sudo chmod 02777 /var/log/jupyterlab
  '';
  # createJupyterLogService will not be used if not added to extraContents
  # in buildMultiUserNixImage below
  createJupyterLogService = pkgs.runCommand "create-jupyter-log" {} ''
    mkdir -p $out/etc/cont-init.d
    ln -s ${createJupyterLogScript} $out/etc/cont-init.d/02-create-jupyter-log
  '';
  jupyterServerScript = pkgs.writeScript "jupyter-service-run" ''
    #!/command/with-contenv ${pkgs.bashInteractive}/bin/bash
    printf "jupyter environment\n\n"
    export JUPYTER_RUNTIME_DIR="/tmp/jupyter_runtime"
    export SHELL=zsh
    printenv | sort
    printf "====================\n\n"
    printf "Starting jupyterlab with NB_PREFIX=''${NB_PREFIX}\n\n"
    cd "''${HOME}"
    exec jupyter lab \
      --notebook-dir="''${HOME}" \
      --ip=0.0.0.0 \
      --no-browser \
      --allow-root \
      --port=8888 \
      --ServerApp.token="" \
      --ServerApp.password="" \
      --ServerApp.allow_origin="*" \
      --ServerApp.allow_remote_access=True \
      --ServerApp.terminado_settings="shell_command=['zsh']" \
      --ServerApp.authenticate_prometheus=False \
      --ServerApp.base_url="''${NB_PREFIX}"
  '';
  jupyterLog = pkgs.writeScript "jupyter-log" ''
    #!/command/with-contenv ${pkgs.runtimeShell}
    exec logutil-service /var/log/jupyterlab
  '';
  # # Add the following to redirect stdout logging and manage rotation
  # mkdir -p $out/etc/services.d/jupyterlab/log
  # ln -s ${jupyterLog} $out/etc/services.d/jupyterlab/log/run
  jupyterServerService = pkgs.runCommand "jupyter-service" {} ''
    mkdir -p $out/tmp/jupyter_runtime
    mkdir -p $out/etc/services.d/jupyterlab
    ln -s ${jupyterServerScript} $out/etc/services.d/jupyterlab/run
  '';
  atuinDaemonScript = pkgs.writeScript "atuin-daemon" ''
    #!/command/with-contenv ${pkgs.bashInteractive}/bin/bash
    printf "running atuin daemon\n\n"
    exec ${pkgs.atuin}/bin/atuin daemon
  '';
  atuinDaemonService = pkgs.runCommand "atuin-daemon" {} ''
    mkdir -p $out/etc/services.d/atuindaemon
    ln -s ${atuinDaemonScript} $out/etc/services.d/atuindaemon/run
  '';
in
  buildMultiUserNixImage {
    inherit pkgs storeOwner;
    name = "pyrovelocityjupyter";
    tag = "latest";
    maxLayers = 121;
    fromImage = sudoImage;
    extraPkgs = with pkgs;
      [
        ps
        su
        sudo
        zsh
      ]
      ++ [pythonPackageEnv]
      ++ devPackages;
    extraContents = [
      activateUserHomeService
      atuinDaemonService
      jupyterServerService
      homeActivationPackage
    ];
    extraFakeRootCommands = ''
      chown -R ${username}:wheel /nix
      chown -R ${username}:wheel /tmp/jupyter_runtime
    '';
    nixConf = {
      allowed-users = ["*"];
      experimental-features = ["nix-command" "flakes"];
      max-jobs = ["auto"];
      sandbox = "false";
      trusted-users = ["root" "jovyan" "runner"];
    };
    extraEnv = [
      "NB_USER=${username}"
      "NB_UID=1000"
      "NB_PREFIX=/"
      "LD_LIBRARY_PATH=${pythonPackageEnv}/lib:${pkgs.libgcc}/lib:/usr/local/nvidia/lib64"
      "NVIDIA_DRIVER_CAPABILITIES='compute,utility'"
      "NVIDIA_VISIBLE_DEVICES=all"
      "QUARTO_PYTHON=${pythonPackageEnv}/bin/python"
    ];
    extraConfig = {
      ExposedPorts = {
        "8888/tcp" = {};
      };
    };
  }
