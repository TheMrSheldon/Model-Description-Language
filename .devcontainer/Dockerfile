FROM mcr.microsoft.com/devcontainers/cpp:ubuntu

RUN <<EOF
set -e
apt update

# Dev Tools
apt install -y cmake cppcheck clang-format-19 clang-tidy-19 openjdk-11-jre

# Dependencies
apt install -y nvidia-cuda-toolkit

EOF