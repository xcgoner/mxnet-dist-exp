#!/bin/bash 
# source activate python2
# cd /home/ubuntu/mxnet-dist-exp/example/ssd
# python train.py --gpus $OMPI_COMM_WORLD_RANK --batch-size 32 --kv-store dist_sync_allreduce --network resnet50
python train.py --gpus $OMPI_COMM_WORLD_RANK --batch-size 32 --lr 0.001 --kv-store dist_sync_allreduce
