python train.py --config-file experiments/PTSEFormer_r50_2gpus.yaml


python -m torch.distributed.launch --nproc_per_node=2 train.py --config-file experiments/PTSEFormer_r50_2gpus.yaml