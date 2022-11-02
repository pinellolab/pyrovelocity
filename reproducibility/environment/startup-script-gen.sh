#!/usr/bin/env sh

#########
#
# source this file to set environment variables
# relating to the startup script:
#
#     source startup-script-gen.sh
# 
# and upload post-startup-script-*.sh to github gist
#
# - depends on [gh cli](https://cli.github.com/)
# - depends on [dotenv-gen.sh](dotenv-gen.sh)
# 
#
########## 

# set -xv

if ! command -v gh
then
    echo "github cli could not be found"
    echo "please see https://cli.github.com/ for installation instructions"
    echo "or set the following variables manually"
    echo "GITHUB_STARTUP_SCRIPT_GIST_ID to use a gist for startup script"
    echo "or TF_VAR_post_startup_script_url to use another publicly accessible url"
    echo "and remove or comment this check"
    exit 1
fi


set -a            
source .env
set +a

STARTUP_SCRIPT_NAME="post-startup-script-$TF_VAR_notebooks_name.sh"

cat template-post-startup-script.sh | \
envsubst '${GITHUB_ORG_NAME} ${GITHUB_REPO_NAME} ${GITHUB_BRANCH_NAME} ${GITHUB_REPO_CONDA_ENV_PATH_NAME}' > $STARTUP_SCRIPT_NAME

get_startup_script_gist_id () {
   echo "$(gh gist list | grep -m1 $STARTUP_SCRIPT_NAME | cut -f1)"
}

temp_gist_id=$(get_startup_script_gist_id)
if [ -z "$temp_gist_id" ]; then
   echo "$temp_gist_id"
   echo "no gist matching pattern $STARTUP_SCRIPT_NAME"
   echo "creating gist from $STARTUP_SCRIPT_NAME"
   gh gist create "$STARTUP_SCRIPT_NAME"
else
   echo "updating gist $temp_gist_id from local $STARTUP_SCRIPT_NAME"
   gh gist edit "$temp_gist_id" "$STARTUP_SCRIPT_NAME"
fi

GITHUB_STARTUP_SCRIPT_GIST_ID="$(get_startup_script_gist_id)"
echo "startup script gist id: $GITHUB_STARTUP_SCRIPT_GIST_ID"
TF_VAR_post_startup_script_url="https://gist.githubusercontent.com/$GITHUB_USERNAME/$GITHUB_STARTUP_SCRIPT_GIST_ID/raw/$STARTUP_SCRIPT_NAME"
export GITHUB_STARTUP_SCRIPT_GIST_ID TF_VAR_post_startup_script_url

url_status=$(curl -s -o /dev/null -w "%{http_code}" "$TF_VAR_post_startup_script_url")
echo "post startup script url: $TF_VAR_post_startup_script_url"
echo "view post startup script in stdout: gh gist view $GITHUB_STARTUP_SCRIPT_GIST_ID"
echo "check startup script gists with: gh gist list | grep '.*post-startup-script.*'"
echo "post startup script url status: $url_status"

# hardcode these values in the .env file if you don't want to use github gist 
{
    echo "TF_VAR_post_startup_script_url=$TF_VAR_post_startup_script_url";
    echo "GITHUB_STARTUP_SCRIPT_GIST_ID=$GITHUB_STARTUP_SCRIPT_GIST_ID";
} >> .env