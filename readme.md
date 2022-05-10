# Detectron Eval Code

### Commands

train:
```
MASTER_ADDR=127.0.0.1 MASTER_PORT=15248 python train.py
```
eval:
```
python -m torch.distributed.launch eval.py
```

### Checklist

1. you want to create a different config for each run following the example of `configdeepmind.yaml` which was used to run the official deep mind weights.

2.in each config, double check the following

```
OUTPUT_DIR: output file path
SOLVER->IMS_PER_BATCH: global batch size = 64
SOLVER->REFERENCE_WORLD_SIZE: number of gpus you plan to run the job, if you are running on 16 gpus, the value should set to exactly 16 to prevent unwanted autoscaling
MODEL->WEIGHTS: weight file path
```

3. check/change the following in `train.py`

```
if __name__ == '__main__':
    launch(main_train,8,num_machines=1, machine_rank=0, dist_url=None)
    # change 8 to num of gpus
```

