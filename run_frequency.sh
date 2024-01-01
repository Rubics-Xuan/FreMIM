# !/bin/Bash

python3 -m torch.distributed.launch --master_port=29500 --nproc_per_node=4 ./train_2d_FreMIM/train_frequency_pretrain_2D_fore.py


