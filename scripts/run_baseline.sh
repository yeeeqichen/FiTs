#!/bin/bash
export TOKENIZERS_PARALLELISM=true
dt=`date '+%Y%m%d_%H%M%S'`


dataset='csqa'
# Our hyper-params
iterative_ie_layer=false
node_classification=false
score_regression=false
node_regularization=false
node_regularization_num=20
noisy_node=false
noisy_node_num=10
random_choice=false
align_mask=false

name_prefix='BASELINE'

#shift
encoder='roberta-large'
#args=$@


elr="1e-5"
dlr="1e-3"
bs=128
mbs=8
unfreeze_epoch=4
k=5 #num of gnn layers
gnndim=200

# Existing arguments but changed for GreaseLM
encoder_layer=-1
max_node_num=210
seed=5
lr_schedule=fixed

if [ ${dataset} = obqa ]
then
  n_epochs=70
  max_epochs_before_stop=10
  ie_dim=400
else
  n_epochs=30
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

run_name=${name_prefix}_greaselm__ds_${dataset}__enc_${encoder}__k${k}__sd${seed}__iedim${ie_dim}__${dt}
log=logs/train_${dataset}__${run_name}.log.txt

###### Training ######
python3 -u finetune.py \
    --dataset $dataset \
    --iterative_ie_layer $iterative_ie_layer \
    --node_classification $node_classification \
    --score_regression $score_regression \
    --node_regularization $node_regularization \
    --node_regularization_num $node_regularization_num \
    --noisy_node $noisy_node \
    --noisy_node_num $noisy_node_num \
    --align_mask $align_mask \
    --random_choice $random_choice \
    --encoder $encoder -k $k --gnn_dim $gnndim -elr $elr -dlr $dlr -bs $bs --seed $seed -mbs ${mbs} --unfreeze_epoch ${unfreeze_epoch} --encoder_layer=${encoder_layer} -sl ${max_seq_len} --max_node_num ${max_node_num} \
    --n_epochs $n_epochs --max_epochs_before_stop ${max_epochs_before_stop} \
    --save_dir ${save_dir_pref}/${dataset}/${run_name} \
    --run_name ${run_name} \
    --ie_dim ${ie_dim} --info_exchange ${info_exchange} --ie_layer_num ${ie_layer_num} --resume_checkpoint ${resume_checkpoint} --resume_id ${resume_id} --sep_ie_layers ${sep_ie_layers} --random_ent_emb ${random_ent_emb} --ent_emb ${ent_emb//,/ } --lr_schedule ${lr_schedule}
# > ${log} 2>&1 &
# echo log: ${log}