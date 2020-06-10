#!/bin/bash

set -exu

wget --no-check-certificate https://paddle-inference-lib.bj.bcebos.com/1.6.0-cpu-avx-mkl/fluid_inference.tgz
tar zxvf fluid_inference.tgz

rm fluid_inference.tgz

python gen_model_and_data.py
