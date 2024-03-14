ARG GIT_REF=520-sum
FROM ghcr.io/pinellolab/pyrovelocity:$GIT_REF

COPY . /root/pyrovelocity
