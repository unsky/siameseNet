#!/usr/bin/env sh
set -e
export PYTHONPATH=$PYTHONPATH:/root/cly/caffe/python
export PYTHONPATH=$PYTHONPATH:/root/cly/caffe/siameseNet/lib
TOOLS=./../build/tools
#LOG=/home/cly/curve/mytrain.log

$TOOLS/caffe train --solver=models/siamese_solver.prototxt --gpu 0 
