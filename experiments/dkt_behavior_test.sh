indices=(0 1 2 3 4 5 6 7 8)
datasets=("assistments09" "assistments12" "assistments15" "assistments17" "bridge_algebra06" "algebra05" "spanish" "statics" "ednet_small")
for i in "${indices[@]}"
do
    command1="python train_dkt2.py --dataset ${datasets[i]}"
    command2="python behavior_test.py --model dkt --load_dir save/dkt --filename ${datasets[i]} --dataset ${datasets[i]}"
    echo $command1
    eval $command1
    echo $command2
    eval $command2
done