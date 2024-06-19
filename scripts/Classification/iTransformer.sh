python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path 6_classes_netherlands_10days.csv \
  --model_id netherlands_10days_lr0001 \
  --model iTransformer_classification \
  --data crop \
  --features M \
  --seq_len 23 \
  --pred_len 9 \
  --e_layers 4 \
  --enc_in 15 \
  --dec_in 9 \
  --c_out 9 \
  --des 'Exp' \
  --d_model 2048 \
  --d_ff 512 \
  --itr 1\
  --devices 0\
  --train_epochs 50\
  --learning_rate 0.0001\

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path 6_classes_netherlands_10days.csv \
  --model_id netherlands_10days_lr0005 \
  --model iTransformer_classification \
  --data crop \
  --features M \
  --seq_len 23 \
  --pred_len 9 \
  --e_layers 4 \
  --enc_in 15 \
  --dec_in 9 \
  --c_out 9 \
  --des 'Exp' \
  --d_model 2048 \
  --d_ff 512 \
  --itr 1\
  --devices 0\
  --train_epochs 50\
  --learning_rate 0.0005\

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path 6_classes_netherlands_10days.csv \
  --model_id netherlands_10days_lr001 \
  --model iTransformer_classification \
  --data crop \
  --features M \
  --seq_len 23 \
  --pred_len 9 \
  --e_layers 4 \
  --enc_in 15 \
  --dec_in 9 \
  --c_out 9 \
  --des 'Exp' \
  --d_model 2048 \
  --d_ff 512 \
  --itr 1\
  --devices 0\
  --train_epochs 50\
  --learning_rate 0.001\


  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path 6_classes_netherlands_10days.csv \
  --model_id netherlands_10days_lr01 \
  --model iTransformer_classification \
  --data crop \
  --features M \
  --seq_len 23 \
  --pred_len 9 \
  --e_layers 4 \
  --enc_in 15 \
  --dec_in 9 \
  --c_out 9 \
  --des 'Exp' \
  --d_model 2048 \
  --d_ff 512 \
  --itr 1\
  --devices 0\
  --train_epochs 50\
  --learning_rate 0.01
#  --is_training 1   --root_path ./dataset/   --data_path 6_classes_netherlands_7days.csv    --model iTransformer_classification   --data crop   --features M   --seq_len 20   --pred_len 9   --train_epochs 150 --e_layers 4   --enc_in 15   --dec_in 9  --c_out 9   --des 'Exp'   --d_model 2048   --d_ff 512   --batch_size 32   --learning_rate 0.0001   --itr 1  --use_gpu 1  --target class_name_late --model_id netherlands_seq_end_4_7days_interval --devices 1 --is_training 1 --patience 3



#  python -u run.py   --is_training 1   --root_path ./dataset/   --data_path 6_classes_netherlands_10days.csv    --model iTransformer_classification   --data crop   --features M   --seq_len 23   --pred_len 9   --train_epochs 1 --e_layers 4   --enc_in 15   --dec_in 9  --c_out 9   --des 'Exp'   --d_model 2048   --d_ff 512   --batch_size 32   --learning_rate 0.0001   --itr 1  --use_gpu 1  --target class_name_late --model_id netherlands_pstats --devices 0 --is_training 1 --patience 3