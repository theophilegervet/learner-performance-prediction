indices=(2 3 4 5 6 7)
datasets=("assistments09" "assistments12" "assistments15" "assistments17" "bridge_algebra06" "algebra05" "spanish" "statics" "ednet_small")
for i in "${indices[@]}"
do
    command1="python train_saint.py --dataset ${datasets[i]} --gpu 0 --device cuda --val_check_steps 50 --name ${datasets[i]}"
    command2="python behavior_test_saint.py --model dkt --load_dir weight/${datasets[i]} --filename best_val_auc.ckpt --dataset ${datasets[i]}"
    echo $command1
    eval $command1
    echo $command2
    eval $command2
done