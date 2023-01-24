#!/usr/bin/env bash

df -h
docker info
echo "===stopping docker service==="
sudo service docker stop
docker info

echo "===moving /var/lib/docker==="
# sudo rsync -ah --info=progress2 --no-i-r /var/lib/docker /home/runner/work/pyrovelocity/pyrovelocity
sudo rsync -ah --no-i-r /var/lib/docker /home/runner/work/pyrovelocity/pyrovelocity
sudo rm -fr /var/lib/docker
df -h

echo "===linking back to original location==="
sudo ln -s /home/runner/work/pyrovelocity/pyrovelocity/docker /var/lib/docker
sudo find /var/lib -type l -ls

echo "===restarting docker==="
sudo service docker start
docker info
df -h

echo "===relocation of /var/lib/docker complete==="