

### M
model_name=missformer
dataset=ETTh1
gpuid=0
elayers=1
dlayers=1
lrdecay=0.5
learningrate=0.00001
patience=6
psurvival=1
d_ff=2048
d_model=512
pos_val_type=5
attn_dim=2048
for mp in 0.2 0.3 0.4 0.5 0.6; do
    python -u main_missformer.py \
    --model $model_name \
    --data $dataset \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 48 \
    --e_layers $elayers \
    --d_layers $dlayers \
    --batch_size 64 \
    --des 'Exp' \
    --itr 5 \
    --split True  \
    --mask_percentage $mp \
    --gpu $gpuid \
    --lr_decay $lrdecay \
    --learning_rate $learningrate \
    --prob_survival $psurvival \
    --patience $patience \
    --d_ff $d_ff \
    --d_model $d_model \
    --pos_val_type $pos_val_type \
    --attn_dim $attn_dim;
done