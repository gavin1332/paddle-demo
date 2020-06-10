#!/bin/bash

set -eux

python3 gen_model_and_data.py

mkdir -p build
cd build
rm -rf *

wget --no-check-certificate https://paddle-inference-lib.bj.bcebos.com/1.6.0-cpu-avx-mkl/fluid_inference.tgz
tar zxvf fluid_inference.tgz

LIB_DIR=$PWD/fluid_inference
export CPLUS_INCLUDE_PATH=$LIB_DIR
for i in glog gflags; do
  export CPLUS_INCLUDE_PATH=$LIB_DIR/third_party/install/$i/include:$CPLUS_INCLUDE_PATH
done

DEMO_NAME=demo

cmake .. \
  -DPADDLE_LIB=$LIB_DIR \
  -DWITH_MKL=ON \
  -DDEMO_NAME=$DEMO_NAME \
  -DWITH_GPU=OFF \
  -DWITH_STATIC_LIB=OFF \
  -DWITH_ANAKIN=ON \
  -DUSE_TENSORRT=OFF \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo

make clean
make -j

./demo 2>&1 | tee log.log
