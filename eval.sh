


WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12345  knn_eval.py\
 --cfg train_coco_300.yaml\
 $@