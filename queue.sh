CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=19345 byol_main.py --cfg train_imagenet_300_detcol_fs_tune_0.1.yaml;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=19345 byol_main.py --cfg train_imagenet_300_detcol_fs_tune_0.05.yaml;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=19345 byol_main.py --cfg train_imagenet_300_detcol_fs_tune_0.01.yaml;

