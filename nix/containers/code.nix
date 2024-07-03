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
  # https://gist.github.com/hyperupcall/99e355405611be6c4e0a38b6e3e8aad0#file-settings-jsonc
  installCodeServerExtensionsScript = pkgs.writeScript "install-code-extensions-run" ''
    #!/command/with-contenv ${pkgs.runtimeShell}
    VSCODE_EXTENSIONS=(
      "alefragnani.project-manager"
      "Catppuccin.catppuccin-vsc"
      "charliermarsh.ruff"
      "christian-kohler.path-intellisense"
      "cweijan.vscode-database-client2"
      "donjayamanne.python-extension-pack"
      "eamodio.gitlens"
      "github.vscode-github-actions"
      "GitHub.vscode-pull-request-github"
      "ionutvmi.path-autocomplete"
      "jnoortheen.nix-ide"
      "ms-azuretools.vscode-docker"
      "ms-kubernetes-tools.vscode-kubernetes-tools"
      "ms-toolsai.jupyter"
      "ms-vsliveshare.vsliveshare"
      "njzy.stats-bar"
      "patbenatar.advanced-new-file"
      "rangav.vscode-thunder-client"
      "redhat.vscode-yaml"
      "sleistner.vscode-fileutils"
      "stkb.rewrap"
      "streetsidesoftware.code-spell-checker"
      "tamasfe.even-better-toml"
      "vscode-icons-team.vscode-icons"
      "vscodevim.vim"
      "richie5um2.vscode-sort-json"
    )

    printf "Listing currently installed extensions...\n\n"
    code-server --list-extensions --show-versions
    echo ""

    install_command="code-server"
    for extension in "''${VSCODE_EXTENSIONS[@]}"; do
        install_command+=" --install-extension \"''${extension}\""
    done

    eval "''${install_command} --force"

    printf "Listing extensions after installation...\n\n"
    code-server --list-extensions --show-versions

    settings_file="''${HOME}/.local/share/code-server/User/settings.json"
    mkdir -p "''${HOME}/.local/share/code-server/User"
    [ ! -s "''${settings_file}" ] && echo '{}' > "''${settings_file}"

    ${pkgs.jq}/bin/jq '{
      "gitlens.showWelcomeOnInstall": false,
      "gitlens.showWhatsNewAfterUpgrades": false,
      "update.showReleaseNotes": false,
      "workbench.iconTheme": "vscode-icons",
      "workbench.colorTheme": "Catppuccin Macchiato",
    } + . ' "''${settings_file}" > "''${settings_file}.tmp" && mv "''${settings_file}.tmp" "''${settings_file}"

    printf "Updated settings in %s\n\n" "''${settings_file}"
  '';
  installCodeServerExtensionsService = pkgs.runCommand "install-code-extensions" {} ''
    mkdir -p $out/etc/cont-init.d
    ln -s ${installCodeServerExtensionsScript} $out/etc/cont-init.d/02-install-code-extensions
  '';
  codeServerScript = pkgs.writeScript "code-service-run" ''
    #!/command/with-contenv ${pkgs.bashInteractive}/bin/bash
    printf "code environment\n\n"
    export SHELL=zsh
    printenv | sort
    printf "====================\n\n"
    printf "Starting code-server with NB_PREFIX=''${NB_PREFIX}\n\n"
    cd "''${HOME}"
    exec code-server \
      --bind-addr 0.0.0.0:8888 \
      --disable-telemetry \
      --disable-update-check \
      --disable-workspace-trust \
      --disable-getting-started-override \
      --auth none \
      "''${HOME}"
  '';
  codeServerService = pkgs.runCommand "code-service" {} ''
    mkdir -p $out/etc/services.d/codeserver
    ln -s ${codeServerScript} $out/etc/services.d/codeserver/run
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
    name = "pyrovelocitycode";
    tag = "latest";
    maxLayers = 121;
    fromImage = sudoImage;
    extraPkgs = with pkgs;
      [
        code-server
        ps
        su
        sudo
        zsh
      ]
      ++ [pythonPackageEnv]
      ++ devPackages;
    extraContents = [
      activateUserHomeService
      installCodeServerExtensionsService
      atuinDaemonService
      codeServerService
      homeActivationPackage
    ];
    extraFakeRootCommands = ''
      chown -R ${username}:wheel /nix
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
      "LD_LIBRARY_PATH=${pythonPackageEnv}/lib:/usr/local/nvidia/lib64"
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
