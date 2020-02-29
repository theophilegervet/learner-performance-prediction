trap "kill 0" EXIT

CUDA_VISIBLE_DEVICES=0 python train_dkt.py --dataset  &
CUDA_VISIBLE_DEVICES=1 python train_dkt.py --dataset MSMATH --seed 1 &
CUDA_VISIBLE_DEVICES=2 python train_dkt.py --dataset MSMATH --seed 2 &
wait
CUDA_VISIBLE_DEVICES=0 python train_dkt.py --dataset MSMATH --seed 3 &
CUDA_VISIBLE_DEVICES=1 python train_dkt.py --dataset MSMATH --seed 4 &
CUDA_VISIBLE_DEVICES=2 python train_dkt.py --dataset MSMATH --seed 5 &

wait