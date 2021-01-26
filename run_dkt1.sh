#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
for lr in $(seq 0.001 0.009 0.01)
do
	for num_hid_layers in 1 2
	do
		for hid_size in 50 100 200
		do
			for drop_prob in $(seq 0.0 0.25 0.5)
			do
				python train_dkt1.py "--dataset=$2" "--lr=$lr" "--num_hid_layers=$num_hid_layers" "--hid_size=$hid_size" "--drop_prob=$drop_prob" "--num_epochs=300"
			done
		done
	done
done

