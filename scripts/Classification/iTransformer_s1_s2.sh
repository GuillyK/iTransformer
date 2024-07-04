
# for i in {5..12}
# do
#     python -u run.py \
#   --is_training 0 \
#   --root_path ./dataset/ \
#   --data_path Netherlands_S1+2_${i}days_noMais.csv \
#   --model_id netherlands_s1_s2_${i}days_full_year_lr0.00001_noMais_d_model_1024_full_att \
#   --model iTransformer_classification \
#   --data crop \
#   --features M \
#   --seq_len 33 \
#   --pred_len 9 \
#   --e_layers 4 \
#   --enc_in 15 \
#   --dec_in 9 \
#   --c_out 9 \
#   --des 'Exp' \
#   --d_model 1024 \
#   --d_ff 512 \
#   --itr 1\
#   --devices 0\
#   --train_epochs 25\
#   --learning_rate 0.00001\
#   --patience 4\
#   --num_classes 5\
#   --output_attention \

  #     python -u run.py \
  # --is_training 0 \
  # --root_path ./dataset/ \
  # --data_path Netherlands_S1+2_5days_noMais.csv \
  # --model_id netherlands_s1_s2_5days_January_full_year_lr0.00001_noMais_lower_multihead_full_att \
  # --model iTransformer_classification \
  # --data crop \
  # --features M \
  # --seq_len 33 \
  # --pred_len 9 \
  # --e_layers 4 \
  # --enc_in 15 \
  # --dec_in 9 \
  # --c_out 9 \
  # --des 'Exp' \
  # --d_model 2048 \
  # --d_ff 512 \
  # --itr 1\
  # --devices 0\
  # --train_epochs 25\
  # --learning_rate 0.00001\
  # --patience 4\
  # --n_heads 4\
  # --num_classes 5\
  # --output_attention \

# done

# for i in {4..12}
# do
#     python -u run.py \
#   --is_training 0 \
#   --root_path ./dataset/ \
#   --data_path Netherlands_S1+2_${i}days_noMais.csv \
#   --model_id netherlands_s1_s2_${i}days_full_year_lr0.00001_noMais_d_model_512_2enc \
#   --model iTransformer_classification \
#   --data crop \
#   --features M \
#   --seq_len 33 \
#   --pred_len 9 \
#   --e_layers 2 \
#   --enc_in 15 \
#   --dec_in 9 \
#   --c_out 9 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --itr 1\
#   --devices 0\
#   --train_epochs 25\
#   --learning_rate 0.00001\
#   --patience 4\
#   --num_classes 5\
#   --output_attention \


#       python -u run.py \
#   --is_training 0 \
#   --root_path ./dataset/ \
#   --data_path Netherlands_S1+2_${i}days_noMais.csv \
#   --model_id netherlands_s1_s2_${i}days_full_year_lr0.000001_noMais_lower_multihead_d_model1024 \
#   --model iTransformer_classification \
#   --data crop \
#   --features M \
#   --seq_len 33 \
#   --pred_len 9 \
#   --e_layers 4 \
#   --enc_in 15 \
#   --dec_in 9 \
#   --c_out 9 \
#   --des 'Exp' \
#   --d_model 1024 \
#   --d_ff 512 \
#   --itr 1\
#   --devices 0\
#   --train_epochs 25\
#   --learning_rate 0.000001\
#   --patience 4\
#   --n_heads 4\
#   --num_classes 5\
#   --output_attention \


# done

Run for mais
      python -u run.py \
  --is_training 0 \
  --root_path ./dataset/ \
  --data_path Netherlands_S1+2_5days.csv \
  --model_id netherlands_s1_s2_5days_full_year_lr0.00001_lower_multihead_full_att \
  --model iTransformer_classification \
  --data crop \
  --features M \
  --seq_len 33 \
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
  --train_epochs 25\
  --learning_rate 0.00001\
  --patience 4\
  --n_heads 4\
  --num_classes 6\
  --output_attention \