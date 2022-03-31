#########################################################################
# File Name: ETTh2_1.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: 2022年02月17日 11時43分51秒
#########################################################################
#!/bin/bash

# for debug only use "-m ipdb"
#python -m ipdb main_informer.py --use_gpu True --gpu 0 --model informer --data ETTh2ms1f2 --data_path ETTh2ms1f2.csv --freq ms --features ms --seq_len 336 --label_len 336 --pred_len 168 --e_layers 3 --d_layers 2 --attn prob --des 'Exp' --itr 5


slice_len=168  # 7*24 = one week as one slice. the number of points in one slice

in_slice_num=4 #2 # NOTE changable, can try other values >= 1 
out_slice_num=1 # NOTE changable, can try other values >= 1 

seq_len=`expr $slice_len \* $in_slice_num`
label_len=`expr $slice_len \* $in_slice_num`
pred_len=`expr $slice_len \* $out_slice_num`

inpath="./data/ETT/"
infn="mydata_f4t1.csv"
infn_full=$inpath/$infn

echo $seq_len
echo $label_len
echo $pred_len

train_ratio=0.8
dev_ratio=0.1
test_ratio=0.1

#python -m ipdb main_informer.py \
python main_informer.py \
	--use_gpu True \
	--gpu 0 \
	--model informer \
	--data ETTh2f4t1 \
	--data_path $infn \
	--train_ratio $train_ratio \
	--dev_ratio $dev_ratio \
	--test_ratio $test_ratio \
	--freq 'h' \
	--features 'M' \
	--seq_len $seq_len \
	--label_len $label_len \
	--pred_len $pred_len \
	--e_layers 3 \
	--d_layers 2 \
	--attn prob \
	--des 'Exp' \
	--itr 10 \
	--batch_size 16 #\
	#--debug

