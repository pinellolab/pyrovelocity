#!/usr/bin/env bash

#-------------#
# Uploads all pdf and png files to an rclone google drive remote
#
# Requirements
#
#   - rclone
#   - tree
#   - dvc
#   - git
#   - Google Drive account
#
# A shared Google Drive folder named "pyrovelocity/figures" must exist
# and an rclone remote name pyrovelocitydrive must be configured.
# Update the values of gdrive_name and prefix_path below accordingly
#
#   mamba install -y rclone
#   rclone config create "pyrovelocitydrive" drive user_acknowledge_abuse true
#
# The latter will redirect to the web browser to authenticate the rclone remote.
#-------------#

set -euxo pipefail

# Variables
github_username="pinellolab"
github_repo="pyrovelocity"
gdrive_name="pyrovelocitydrive"
prefix_path="pyrovelocity/figures"

timestamp=$(TZ=":America/New_York" date +%Y%m%d_%H%M%S)

git_branch=$(git rev-parse --abbrev-ref HEAD | tr '/' '_')
git_hash=$(git rev-parse --short HEAD)
git_full_hash=$(git rev-parse HEAD)

gdrive_folder="${prefix_path}/figures_${timestamp}_${git_branch}_${git_hash}"
tarball_name="${git_branch}_${git_hash}_${timestamp}.tar.gz"


# sync with remote
dvc status
dvc pull
dvc push
dvc status -q
if [ $? -ne 0 ]; then
    echo "dvc status returned a non-zero exit code."
    echo "Please check the status of the file synchronization before uploading figures."
    exit 1
fi

# create drive directory
rclone mkdir "${gdrive_name}:${gdrive_folder}"

# markdown file with repository snapshot
tree_output_file="figure_file_tree.md"
github_url="https://github.com/${github_username}/${github_repo}/tree/${git_full_hash}"
{
  echo "# Repository Snapshot: [${github_username}/${github_repo}@${git_hash}](${github_url})"
  tree --du -h -f -P '*.pdf|*.png' -I "outputs*|mlruns*|archive|*pycache*"
} > "$tree_output_file"
rclone copy "$tree_output_file" "${gdrive_name}:${gdrive_folder}/"
rm "$tree_output_file"

cp "dvc.lock" "dvc.lock.txt"
rclone copy "dvc.lock.txt" "${gdrive_name}:${gdrive_folder}/"
rm "dvc.lock.txt"

dvc_tracked_files=$(dvc list --dvc-only -R .)
for file in $dvc_tracked_files; do
  du -sh "$file"
done | grep -E ".*.png|.*.pdf" > dvc_tracked_files.txt
rclone copy "dvc_tracked_files.txt" "${gdrive_name}:${gdrive_folder}/"
rm "dvc_tracked_files.txt"

# create and upload tarball
echo "Uploading figure tarball to ${gdrive_name}:${gdrive_folder}/${tarball_name}"
find . -type f \( -name "*.pdf" -o -name "*.png" \) -print0 |\
tar -czvf - --null -T - |\
rclone rcat "${gdrive_name}:${gdrive_folder}/${tarball_name}"

# download and extract tarball to temporary directory
tmpdir=$(mktemp -d)
trap '[ -n "$tmpdir" ] && rm -rf "$tmpdir"' EXIT

echo "Temporary directory: ${tmpdir}/${tarball_name}"
rclone copy "${gdrive_name}:${gdrive_folder}/${tarball_name}" "${tmpdir}"

cd "${tmpdir}"
tar -xzvf "${tarball_name}"

# upload extracted files to drive
rclone copy --transfers=20 "./" "${gdrive_name}:${gdrive_folder}/" -P

echo "Figure upload completed."
