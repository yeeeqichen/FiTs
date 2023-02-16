#!/bin/bash
export TOKENIZERS_PARALLELISM=true
dt=`date '+%Y%m%d_%H%M%S'`


dataset='csqa'

align_mask=true
align_option=contrastive-cls
mlm=true
mlm_mask_percent=0.15
negative_strategy=random
span_num=1
neg_num=1
run_name='POST_TRAIN'
use_mean_pool=false

#shift
encoder='roberta-large'
#args=$@


elr="1e-5"
dlr="1e-3"
bs=128
mbs=4
unfreeze_epoch=4
k=5 #num of gnn layers
gnndim=200

# Existing arguments but changed for GreaseLM
encoder_layer=-1
max_node_num=200
seed=5
lr_schedule=fixed

if [ ${dataset} = obqa ]
then
  n_epochs=30
  max_epochs_before_stop=10
  ie_dim=400
else
  n_epochs=15
  max_epochs_before_stop=10
  ie_dim=400
fi

max_seq_len=100
ent_emb=tzw

# Added for GreaseLM
info_exchange=true
ie_layer_num=1
resume_checkpoint=None
resume_id=None
sep_ie_layers=false
random_ent_emb=false
debug=false
use_wandb=true

echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "enc_name: $encoder"
echo "batch_size: $bs mini_batch_size: $mbs"
echo "learning_rate: elr $elr dlr $dlr"
echo "gnn: dim $gnndim layer $k"
echo "iterative_ie_layer: ${iterative_ie_layer}"
echo "node classification: ${node_classification}"
echo "ie_dim: ${ie_dim}, info_exchange: ${info_exchange}"
echo "******************************"

save_dir_pref='runs'
mkdir -p $save_dir_pref

log=logs/train_${dataset}__${run_name}.log.txt

###### Training ######
python3 -u post_train.py \
    --dataset $dataset \
    --align_mask $align_mask \
    --align_option $align_option \
    --span_num $span_num \
    --neg_num $neg_num \
    --mlm $mlm \
    --use_mean_pool $use_mean_pool \
    --negative_strategy $negative_strategy \
    --mlm_mask_percent $mlm_mask_percent \
    --debug $debug \
    --use_wandb $use_wandb \
    --encoder $encoder -k $k --gnn_dim $gnndim -elr $elr -dlr $dlr -bs $bs --seed $seed -mbs ${mbs} --unfreeze_epoch ${unfreeze_epoch} --encoder_layer=${encoder_layer} -sl ${max_seq_len} --max_node_num ${max_node_num} \
    --n_epochs $n_epochs --max_epochs_before_stop ${max_epochs_before_stop} \
    --save_dir ${save_dir_pref}/${dataset}/${run_name} \
    --run_name ${run_name} \
    --ie_dim ${ie_dim} --info_exchange ${info_exchange} --ie_layer_num ${ie_layer_num} --resume_checkpoint ${resume_checkpoint} --resume_id ${resume_id} --sep_ie_layers ${sep_ie_layers} --random_ent_emb ${random_ent_emb} --ent_emb ${ent_emb//,/ } --lr_schedule ${lr_schedule}
# > ${log} 2>&1 &
# echo log: ${log}