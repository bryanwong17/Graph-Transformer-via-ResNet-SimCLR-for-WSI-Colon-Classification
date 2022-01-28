export CUDA_VISIBLE_DEVICES=0
python main.py \
--n_class 2 \
--data_path "../graphs/simclr_files" \
--train_set "../train_set.txt" \
--val_set "../val_set.txt" \
--model_path "../graph_transformer/saved_models/" \
--log_path "../graph_transformer/runs/" \
--task_name "GraphVIT" \
--batch_size 8 \
--train \
--log_interval_local 6 \
