datasets=("restaurants14" "laptop14" "restaurants15" "restaurants16")
model_names=("HAN")
#model_names=("ATAE_LSTM" "ATAE_GRU" "ATAE_BiLSTM" "ATAE_BiGRU" "TC_LSTM" "GCAE" "CABASC" "LCRS" "RAM" "ContextAvg" "AEContextAvg" "TD_LSTM" "LSTM" "BiLSTM" "GRU" "BiGRU" "MemNet" "IAN")
batchsizes=(16 32)
dropouts=(0.5)
max_seq_lens=(-1)
optimizers=("Adam")
learning_rates=(0.001 0.0005)
devs=(0.1 0.2)
for dataset in ${datasets[@]}; do
	for model_name in ${model_names[@]}; do
	    for batchsize in ${batchsizes[@]}; do
            for optimizer in ${optimizers[@]}; do
                for dropout in ${dropouts[@]}; do
                    for max_seq_len in ${max_seq_lens[@]}; do
                        for learning_rate in ${learning_rates[@]}; do
                            echo $model_name $dataset $batchsize $optimizer $learning_rate $max_seq_len $dropout ${model_name}_${dataset}.log
                            python train_han.py --model_name $model_name --dataset $dataset --batch_size $batchsize --optimizer $optimizer --learning_rate $learning_rate --max_seq_len $max_seq_len --dropout $dropout >> result/log/${dataset}_${model_name}.log
                            # python train_han.py --softmax --model_name $model_name --dataset $dataset --batch_size $batchsize --optimizer $optimizer --learning_rate $learning_rate --max_seq_len $max_seq_len --dropout $dropout >> result/log/${dataset}_${model_name}.log
                        done
                    done
                done
            done
        done
	done
done
