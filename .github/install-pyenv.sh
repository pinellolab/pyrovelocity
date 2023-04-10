#!/usr/bin/env bash

set -x

sudo apt-get update -qq && \
sudo apt-get install -y \
    zlib1g \
    zlib1g-dev \
    libssl-dev \
    libbz2-dev \
    libsqlite3-dev \
    libreadline-dev \
    libncursesw5 \
    libffi-dev \
    liblzma-dev
PYENV_PYTHON_VERSION=$1
function install_pyenv() {
    cd $HOME
    PYENV_PYTHON_VERSION=$1
    curl https://pyenv.run | bash
    .pyenv/bin/pyenv install --skip-existing $PYENV_PYTHON_VERSION
    printf '
# >>> pyenv >>>
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
# <<< pyenv <<<
' > initpyenv
    target=.bashrc
    touch .bashrc
    sed -i -e '$a\' "$target"
    while IFS= read -r line ; do
        if ! grep -Fqxe "$line" "$target" ; then
            printf "%s\n" "$line" >> "$target"
        fi
    done < initpyenv
    rm initpyenv
}
echo
bash -c \
"$(declare -f install_pyenv); install_pyenv $PYENV_PYTHON_VERSION"

set +x
