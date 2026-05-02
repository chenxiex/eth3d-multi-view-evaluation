FROM ubuntu:24.04

# Install dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        cmake \
        libboost-filesystem-dev \
        libboost-system-dev \
        libeigen3-dev \
        libpcl-dev && \
    rm -rf /var/lib/apt/lists/*

# Build the ETH3D binaries
WORKDIR /app
COPY . .
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release . && \
    cmake --build build --parallel
