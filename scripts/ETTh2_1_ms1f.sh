#########################################################################
# File Name: ETTh2_1.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: 2022年02月17日 11時43分51秒
#########################################################################
#!/bin/bash

# for debug only use "-m ipdb"
#python -m ipdb main_informer.py --use_gpu True --gpu 0 --model informer --data ETTh2ms1f2 --data_path ETTh2ms1f2.csv --freq ms --features ms --seq_len 336 --label_len 336 --pred_len 168 --e_layers 3 --d_layers 2 --attn prob --des 'Exp' --itr 5

python main_informer.py --use_gpu True --gpu 0 --model informer --data ETTh2ms1f2 --data_path ETTh2ms1f2.csv --freq ms --features ms --seq_len 336 --label_len 336 --pred_len 168 --e_layers 3 --d_layers 2 --attn prob --des 'Exp' --itr 5

