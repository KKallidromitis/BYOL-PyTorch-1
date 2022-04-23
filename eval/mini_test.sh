CUDA_VISIBLE_DEVICES=5,6,7,8 python main_lincls.py \
  -a resnet50 \
  --batch-size 8 \
  --epochs 1\
  --pretrained /home/jacklishufan/resnet50_300.pth_pregather.tar \
  --dist-url 'tcp://localhost:10001' --mini-test  --multiprocessing-distributed --world-size 1 --rank 0 /shared/group/ilsvrc