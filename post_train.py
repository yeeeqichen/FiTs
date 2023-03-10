import argparse
import logging
import random
import shutil
import time

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import transformers
try:
    from transformers import (ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule)
except:
    from transformers import get_constant_schedule, get_constant_schedule_with_warmup,  get_linear_schedule_with_warmup
import wandb

from modeling import modeling_aligned_mask
from utils import data_utils
from utils import optimization_utils
from utils import parser_utils
from utils import utils


DECODER_DEFAULT_LR = {
    'csqa': 1e-3,
    'csqa-entity-label': 1e-3,
    'csqa-text-label': 1e-3,
    'obqa': 3e-4,
    'medqa_usmle': 1e-3,
}

import numpy as np

import socket, os, subprocess

logger = logging.getLogger(__name__)


def load_data(args, devices, kg):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)


    #########################################################
    # Construct the dataset
    #########################################################
    dataset = data_utils.GreaseLM_DataLoader(args.train_statements, args.train_adj,
                                             args.dev_statements, args.dev_adj,
                                             args.test_statements, args.test_adj,
                                             batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                             device=devices,
                                             model_name=args.encoder,
                                             max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
                                             is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                                             subsample=args.subsample, n_train=args.n_train, debug=args.debug,
                                             cxt_node_connects_all=args.cxt_node_connects_all, kg=kg,
                                             align_mask=args.align_mask,
                                             train_span_path=args.train_span_path,
                                             train_ids_path=args.train_ids_path,
                                             dev_span_path=args.dev_span_path,
                                             dev_ids_path=args.dev_ids_path,
                                             test_span_path=args.test_span_path,
                                             test_ids_path=args.test_ids_path)

    return dataset


def construct_model(args, kg):
    ########################################################
    #   Load pretrained concept embeddings
    ########################################################
    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = np.concatenate(cp_emb, 1)
    cp_emb = torch.tensor(cp_emb, dtype=torch.float)

    concept_num, concept_in_dim = cp_emb.size(0), cp_emb.size(1)
    print('| num_concepts: {} |'.format(concept_num))
    if args.random_ent_emb:
        cp_emb = None
        freeze_ent_emb = False
        concept_in_dim = args.gnn_dim
    else:
        freeze_ent_emb = args.freeze_ent_emb

    ##########################################################
    #   Build model
    ##########################################################

    if kg == "cpnet":
        n_ntype = 4 if not args.noisy_node else 5
        n_etype = 38
    elif kg == "ddb":
        n_ntype = 4 if not args.noisy_node else 5
        n_etype = 34
    else:
        raise ValueError("Invalid KG.")
    if args.cxt_node_connects_all:
        n_etype += 2

    assert args.align_mask
    cls = modeling_aligned_mask.MaskedGreaseLM
    model = cls(args, args.encoder, k=args.k, n_ntype=n_ntype, n_etype=n_etype, n_concept=concept_num,
        concept_dim=args.gnn_dim,
        concept_in_dim=concept_in_dim,
        n_attention_head=args.att_head_num, fc_dim=args.fc_dim, n_fc_layer=args.fc_layer_num,
        p_emb=args.dropouti, p_gnn=args.dropoutg, p_fc=args.dropoutf,
        pretrained_concept_emb=cp_emb, freeze_ent_emb=freeze_ent_emb,
        init_range=args.init_range, ie_dim=args.ie_dim, info_exchange=args.info_exchange, ie_layer_num=args.ie_layer_num, sep_ie_layers=args.sep_ie_layers, layer_id=args.encoder_layer)
    return model


def sep_params(model, loaded_roberta_keys):
    """Separate the parameters into loaded and not loaded."""
    loaded_params = dict()
    not_loaded_params = dict()
    params_to_freeze = []
    small_lr_params = dict()
    large_lr_params = dict()
    for n, p in model.named_parameters():
        if n in loaded_roberta_keys:
            loaded_params[n] = p
            params_to_freeze.append(p)
            small_lr_params[n] = p
        else:
            not_loaded_params[n] = p
            large_lr_params[n] = p

    return loaded_params, not_loaded_params, params_to_freeze, small_lr_params, large_lr_params


def count_parameters(loaded_params, not_loaded_params):
    num_params = sum(p.numel() for p in not_loaded_params.values() if p.requires_grad)
    num_fixed_params = sum(p.numel() for p in not_loaded_params.values() if not p.requires_grad)
    num_loaded_params = sum(p.numel() for p in loaded_params.values())
    print('num_trainable_params:', num_params)
    print('num_fixed_params:', num_fixed_params)
    print('num_loaded_params:', num_loaded_params)
    print('num_total_params:', num_params + num_fixed_params + num_loaded_params)


def train(args, resume, has_test_split, devices, kg):
    print("args: {}".format(args))

    if resume:
        args.save_dir = os.path.dirname(args.resume_checkpoint)
    if not args.debug:
        log_path = os.path.join(args.save_dir, 'log.csv')
        utils.check_path(log_path)

        # Set up tensorboard
        tb_dir = os.path.join(args.save_dir, "tb")
        if not resume:
            with open(log_path, 'w') as fout:
                fout.write('epoch,step,dev_acc,test_acc,best_dev_acc,final_test_acc,best_dev_epoch\n')

            if os.path.exists(tb_dir):
                shutil.rmtree(tb_dir)
        tb_writer = SummaryWriter(tb_dir)

        config_path = os.path.join(args.save_dir, 'config.json')
        utils.export_config(args, config_path)

        model_path = os.path.join(args.save_dir, 'model.pt')

    model = construct_model(args, kg)

    dataset = load_data(args, devices, kg)
    loader = dataset.train_dev_test()
    # print(len(loader) * args.batch_size)
    # exit('debug')
    model.lmgnn.mp.resize_token_embeddings(len(dataset.tokenizer))

    # Get the names of the loaded LM parameters
    loading_info = model.lmgnn.loading_info
    # loaded_roberta_keys = [k.replace("roberta.", "lmgnn.mp.") for k in loading_info["all_keys"]]
    def _rename_key(key):
        if key.startswith("roberta."):
            return key.replace("roberta.", "lmgnn.mp.")
        else:
            return "lmgnn.mp." + key

    loaded_roberta_keys = [_rename_key(k) for k in loading_info["all_keys"]]

    # Separate the parameters into loaded and not loaded
    loaded_params, not_loaded_params, params_to_freeze, small_lr_params, large_lr_params = sep_params(model, loaded_roberta_keys)

    # print non-loaded parameters
    print('Non-loaded parameters:')
    for name, param in not_loaded_params.items():
        if param.requires_grad:
            print('\t{:45}\ttrainable\t{}\tdevice:{}'.format(name, param.size(), param.device))
        else:
            print('\t{:45}\tfixed\t{}\tdevice:{}'.format(name, param.size(), param.device))

    # Count parameters
    count_parameters(loaded_params, not_loaded_params)

    if args.freeze_unimodal:
        param_names = [k for k, v in model.named_parameters()]
        del_params = []
        for k in param_names:
            for i in range(0, 24 - args.k):
                if 'lmgnn.mp.encoder.layer.{}.'.format(i) in k:
                    del_params.append(k)
                    break
        for k in del_params:
            if k in small_lr_params:
                del small_lr_params[k]
            if k in large_lr_params:
                del large_lr_params[k]
        print('small lr params:')
        print([k for k, v in small_lr_params.items()])
        print('large lr params:')
        print([k for k, v in large_lr_params.items()])

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    #########################################################
    # Create an optimizer
    #########################################################
    grouped_parameters = [
        {'params': [p for n, p in small_lr_params.items() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
        {'params': [p for n, p in small_lr_params.items() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.encoder_lr},
        {'params': [p for n, p in large_lr_params.items() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
        {'params': [p for n, p in large_lr_params.items() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.decoder_lr},
    ]
    optimizer = optimization_utils.OPTIMIZER_CLASSES[args.optim](grouped_parameters)

    #########################################################
    # Optionally loading from a checkpoint
    #########################################################
    if resume:
        print("loading from checkpoint: {}".format(args.resume_checkpoint))
        checkpoint = torch.load(args.resume_checkpoint, map_location='cpu')
        last_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        model.load_state_dict(checkpoint["model"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_dev_epoch = checkpoint["best_dev_epoch"]
        best_dev_acc = checkpoint["best_dev_acc"]
        final_test_acc = checkpoint["final_test_acc"]
        best_test_epoch = checkpoint["best_test_epoch"]
        best_test_acc = checkpoint["best_test_acc"]
        best_both_dev_test_acc = checkpoint["best_both_dev_test_acc"]
        best_both_dev_test_epoch = checkpoint["best_both_dev_test_epoch"]
    else:
        last_epoch = -1
        global_step = 0
        best_dev_epoch = best_dev_acc = final_test_acc = best_test_epoch =\
            best_test_acc = best_both_dev_test_epoch = 0
        best_both_dev_test_acc = [0, 0]


    #########################################################
    # Create a scheduler
    #########################################################
    if args.lr_schedule == 'fixed':
        try:
            scheduler = ConstantLRSchedule(optimizer)
        except:
            scheduler = get_constant_schedule(optimizer)
    elif args.lr_schedule == 'warmup_constant':
        try:
            scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps, last_epoch=last_epoch)
        except:
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, last_epoch=last_epoch)
    elif args.lr_schedule == 'warmup_linear':
        max_steps = int(args.n_epochs * (dataset.train_size() / args.batch_size))
        try:
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=max_steps, last_epoch=last_epoch)
        except:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max_steps, last_epoch=last_epoch)
    if resume:
        scheduler.load_state_dict(checkpoint["scheduler"])

    model.to(devices[1])
    model.lmgnn.concept_emb.to(devices[0])
    if args.masked_entity_modeling and args.masked_loss == 'classification':
        model.lmgnn.masked_entity_classifier.to(devices[0])
    model.lmgnn.set_device(devices[0])

    # Construct the loss function
    if args.loss == 'margin_rank':
        loss_func = nn.MarginRankingLoss(margin=0.1, reduction='mean')
    elif args.loss == 'cross_entropy':
        loss_func = nn.CrossEntropyLoss(reduction='mean')
    else:
        raise ValueError("Invalid value for args.loss.")

    #############################################################
    #   Training
    #############################################################

    print()
    print('-' * 71)

    total_loss_acm = 0.0
    n_samples_acm = 0
    total_time = 0
    model.train()
    # If all the parameters are frozen in the first few epochs, just skip those epochs.
    if len(params_to_freeze) >= len(list(model.parameters())) - 1:
        args.unfreeze_epoch = 0
    if last_epoch + 1 <= args.unfreeze_epoch:
        utils.freeze_params(params_to_freeze)
    for epoch_id in trange(last_epoch + 1, args.n_epochs, desc="Epoch"):
        if epoch_id == args.unfreeze_epoch:
            utils.unfreeze_params(params_to_freeze)
        if epoch_id == args.refreeze_epoch:
            utils.freeze_params(params_to_freeze)
        model.train()
        mini_batch_id = 0
        for qids, labels, *input_data in tqdm(loader, desc="Batch"):
            # labels: [bs]
            linked_span = None
            linked_ids = None
            if args.align_mask:
                linked_span = input_data[-2]
                linked_ids = input_data[-1]
                input_data = input_data[:-2]
            start_time = time.time()
            optimizer.zero_grad()
            bs = labels.size(0)
            for a in range(0, bs, args.mini_batch_size):
                b = min(a + args.mini_batch_size, bs)
                _align_data = None
                if args.align_mask:
                    _align_data = (linked_span[a:b], linked_ids[a:b])
                loss = model(*[x[a:b] for x in input_data], mini_batch_id=mini_batch_id, purpose='train',
                             labels=labels[a:b],
                             align_data=_align_data)
                # logits: [bs, nc]
                mini_batch_id += 1

                total_loss_acm += loss.item()
                loss = loss / bs
                loss.backward()
                n_samples_acm += (b - a)

            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scheduler.step()
            # Gradients are accumulated and not back-proped until a batch is processed (not a mini-batch).
            optimizer.step()

            total_time += (time.time() - start_time)

            if (global_step + 1) % args.log_interval == 0:
                ms_per_batch = 1000 * total_time / args.log_interval
                wandb.log({"lr": scheduler.get_lr()[0], "train_loss": total_loss_acm / n_samples_acm,
                           "ms_per_batch": ms_per_batch}, step=global_step)
                print('| step {:5} |  lr: {:9.7f} | total loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step, scheduler.get_lr()[0], total_loss_acm / n_samples_acm, ms_per_batch))

                total_loss_acm = 0.0
                n_samples_acm = 0
                total_time = 0
            global_step += 1  # Number of batches processed up to now

        # Save the model checkpoint
        if args.save_model:
            model_state_dict = model.state_dict()
            del model_state_dict["lmgnn.concept_emb.emb.weight"]
            checkpoint = {
                "model": model_state_dict,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch_id,
                "global_step": global_step,
                "config": args}
            if epoch_id % 5 == 0:
                print('Saving model to {}.{}'.format(model_path, epoch_id))
                torch.save(checkpoint, model_path + ".{}".format(epoch_id))
            torch.save(checkpoint, model_path + ".final")
        model.train()
        torch.cuda.empty_cache()

        if args.debug:
            break

    if not args.debug:
        tb_writer.close()


def get_devices(use_cuda):
    """Get the devices to put the data and the model based on whether to use GPUs and, if so, how many of them are available."""
    if torch.cuda.device_count() >= 2 and use_cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:1")
        print("device0: {}, device1: {}".format(device0, device1))
    elif torch.cuda.device_count() == 1 and use_cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:0")
    else:
        device0 = torch.device("cpu")
        device1 = torch.device("cpu")
    return device0, device1


def main(args):
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(name)s:%(funcName)s():%(lineno)d] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.WARNING)

    has_test_split = True
    devices = get_devices(args.cuda)
    kg = "cpnet"
    if args.dataset == "medqa_usmle":
        kg = "ddb"

    if not args.use_wandb:
        wandb_mode = "disabled"
    elif args.debug:
        wandb_mode = "offline"
    else:
        wandb_mode = "online"

    # We can optionally resume training from a checkpoint. If doing so, also set the `resume_id` so that you resume your previous wandb run instead of creating a new one.
    resume = args.resume_checkpoint is not None and args.resume_checkpoint != "None"
    wandb_id = args.resume_id if resume else wandb.util.generate_id()
    args.wandb_id = wandb_id

    args.hf_version = transformers.__version__

    with wandb.init(project="greaselm", config=args, name=args.run_name, resume="allow", id=wandb_id, settings=wandb.Settings(start_method="fork"), mode=wandb_mode):
        print(socket.gethostname())
        print ("pid:", os.getpid())
        print ("screen: %s" % subprocess.check_output('echo $STY', shell=True).decode('utf'))
        print ("gpu: %s" % subprocess.check_output('echo $CUDA_VISIBLE_DEVICES', shell=True).decode('utf'))
        utils.print_cuda_info()
        print("wandb id: ", wandb_id)

        if args.mode == 'train':
            train(args, resume, has_test_split, devices, kg)
        else:
            raise ValueError('Invalid mode')


if __name__ == '__main__':
    __spec__ = None

    parser = parser_utils.get_parser()
    args, _ = parser.parse_known_args()

    # General
    parser.add_argument('--mode', default='train', choices=['train', 'eval'], help='run training or evaluation')
    parser.add_argument('--save_dir', default=f'./saved_models/greaselm/', help='model output directory')
    parser.add_argument('--save_model', default=True, type=utils.bool_flag, help="Whether to save model checkpoints or not.")
    parser.add_argument('--load_model_path', default=None, help="The model checkpoint to load in the evaluation mode.")
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    parser.add_argument("--run_name", required=True, type=str, help="The name of this experiment run.")
    parser.add_argument("--resume_checkpoint", default=None, type=str,
                        help="The checkpoint to resume training from.")
    parser.add_argument('--use_wandb', default=False, type=utils.bool_flag, help="Whether to use wandb or not.")
    parser.add_argument("--resume_id", default=None, type=str, help="The wandb run id to resume if `resume_checkpoint` is not None or 'None'.")

    # Data
    parser.add_argument('--train_adj', default=f'{args.data_dir}/{args.dataset}/graph/train.graph.adj.pk', help="The path to the retrieved KG subgraphs of the training set.")
    parser.add_argument('--dev_adj', default=f'{args.data_dir}/{args.dataset}/graph/dev.graph.adj.pk', help="The path to the retrieved KG subgraphs of the dev set.")
    parser.add_argument('--test_adj', default=f'{args.data_dir}/{args.dataset}/graph/test.graph.adj.pk', help="The path to the retrieved KG subgraphs of the test set.")
    parser.add_argument('--max_node_num', default=200, type=int, help="Max number of nodes / the threshold used to prune nodes.")
    parser.add_argument('--subsample', default=1.0, type=float, help="The ratio to subsample the training set.")
    parser.add_argument('--n_train', default=-1, type=int, help="Number of training examples to use. Setting it to -1 means using the `subsample` argument to determine the training set size instead; otherwise it will override the `subsample` argument.")

    # Model architecture
    parser.add_argument('-k', '--k', default=5, type=int, help='The number of GreaseLM layers')
    parser.add_argument('--att_head_num', default=2, type=int, help='number of attention heads of the final graph nodes\' pooling')
    parser.add_argument('--gnn_dim', default=100, type=int, help='dimension of the GNN layers')
    parser.add_argument('--fc_dim', default=200, type=int, help='number of FC hidden units (except for the MInt operators)')
    parser.add_argument('--fc_layer_num', default=0, type=int, help='number of hidden layers of the final MLP')
    parser.add_argument('--freeze_ent_emb', default=True, type=utils.bool_flag, nargs='?', const=True, help='Whether to freeze the entity embedding layer.')
    parser.add_argument('--ie_dim', default=200, type=int, help='number of the hidden units of the MInt operator.')
    parser.add_argument('--info_exchange', default=True, choices=[True, False, "every-other-layer"], type=utils.bool_str_flag, help="Whether we have the MInt operator in every GreaseLM layer or every other GreaseLM layer or not at all.")
    parser.add_argument('--ie_layer_num', default=1, type=int, help='number of hidden layers in the MInt operator')
    parser.add_argument("--sep_ie_layers", default=False, type=utils.bool_flag, help="Whether to share parameters across the MInt ops across differernt GreaseLM layers or not. Setting it to `False` means sharing.")
    parser.add_argument('--random_ent_emb', default=False, type=utils.bool_flag, nargs='?', const=True, help='Whether to use randomly initialized learnable entity embeddings or not.')
    parser.add_argument("--cxt_node_connects_all", default=False, type=utils.bool_flag, help="Whether to connect the interaction node to all the retrieved KG nodes or only the linked nodes.")


    # Regularization
    parser.add_argument('--dropouti', type=float, default=0.2, help='dropout for embedding layer')
    parser.add_argument('--dropoutg', type=float, default=0.2, help='dropout for GNN layers')
    parser.add_argument('--dropoutf', type=float, default=0.2, help='dropout for fully-connected layers')

    # Optimization
    parser.add_argument('-dlr', '--decoder_lr', default=DECODER_DEFAULT_LR[args.dataset], type=float, help='Learning rate of parameters not in LM')
    parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=2, type=int)
    parser.add_argument('--unfreeze_epoch', default=4, type=int, help="Number of the first few epochs in which LM’s parameters are kept frozen.")
    parser.add_argument('--refreeze_epoch', default=10000, type=int)
    parser.add_argument('--init_range', default=0.02, type=float, help='stddev when initializing with normal distribution')

    parser.add_argument('--iterative_ie_layer', type=utils.bool_flag, default=False)
    parser.add_argument('--node_classification', type=utils.bool_flag, default=False)
    parser.add_argument('--score_regression', type=utils.bool_flag, default=False)
    parser.add_argument('--node_regularization', type=utils.bool_flag, default=False)
    parser.add_argument('--node_regularization_num', type=int, default=20)
    parser.add_argument('--random_choice', type=utils.bool_flag, default=False)
    parser.add_argument('--noisy_node', type=utils.bool_flag, default=False)
    parser.add_argument('--noisy_node_num', type=int, default=0)
    parser.add_argument('--masked_entity_modeling', type=utils.bool_flag, default=False)
    parser.add_argument('--mask_percent', type=float, default=0.15)
    parser.add_argument('--mask_change', type=utils.bool_flag, default=False)
    parser.add_argument('--change_percent', type=float, default=0.05)
    parser.add_argument('--masked_loss', type=str, default='new-mse')
    parser.add_argument('--freeze_unimodal', type=utils.bool_flag, default=False)
    parser.add_argument('--masked_language_modeling', type=utils.bool_flag, default=False)
    parser.add_argument('--use_lm_only', type=utils.bool_flag, default=False)
    parser.add_argument('--align_mask', type=utils.bool_flag, default=False)

    parser.add_argument('--align_option', choices=['contrastive-cls', 'semantic-mse',
                                                   'semantic-cos', 'no-align',
                                                   'mutual-info'],
                        default='contrastive-cls')
    parser.add_argument('--negative_strategy', choices=['random', 'in-batch'], default='random')
    parser.add_argument('--mlm', type=utils.bool_flag, default=False)
    parser.add_argument('--mlm_mask_percent', type=float, default=0.15)

    parser.add_argument('--use_mean_pool', type=utils.bool_flag, default=False)

    parser.add_argument('--mix_neg', type=utils.bool_flag, default=False)
    parser.add_argument('--span_num', type=int, default=1)
    parser.add_argument('--neg_num', type=int, default=1)
    args = parser.parse_args()
    main(args)
