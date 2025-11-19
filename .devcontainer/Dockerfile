FROM mcr.microsoft.com/devcontainers/cpp:1-ubuntu-24.04

RUN <<EOF
apt update

apt install -y cmake git cppcheck gdb clang-format openjdk-11-jre

apt install -y nvidia-cuda-toolkit

EOF