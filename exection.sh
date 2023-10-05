#!/bin/bash

# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=12345 byol_main.py --cfg /home/acf15772rb/r2o-vit/config/train_imagenet_300_vit.yaml

# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=4 --node_rank=0 --master_addr="localhost" --master_port=12345 byol_main.py --cfg /home/acf15772rb/r2o-vit/config/train_imagenet_300_vit.yaml

node_rank=1
for slave_node in `cat $SGE_JOB_HOSTLIST | awk 'NR != 1 { print }'`; do
	qrsh -inherit -V -cwd $slave_node python -m torch.distributed.launch --nproc_per_node 2 --nnodes 4 --node_rank $node_rank --master_port 12345 --master_addr `hostname` byol_main.py --cfg /home/acf15772rb/r2o-vit/config/train_imagenet_300_vit.yaml & \
	node_rank=`expr $node_rank + 1`
done

# launch on master node
node_rank=0
python -m torch.distributed.launch --nproc_per_node 2 --nnodes 4 --node_rank $node_rank --master_port 12345 --master_addr `hostname` byol_main.py --cfg /home/acf15772rb/r2o-vit/config/train_imagenet_300_vit.yaml

