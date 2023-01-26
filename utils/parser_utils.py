import argparse

from utils import utils

ENCODER_DEFAULT_LR = {
    'default': 1e-3,
    'csqa': {
        'lstm': 3e-4,
        'openai-gpt': 1e-4,
        'bert-base-uncased': 3e-5,
        'bert-large-uncased': 2e-5,
        'roberta-large': 1e-5,
    },
    'csqa-entity-label': {
        'lstm': 3e-4,
        'openai-gpt': 1e-4,
        'bert-base-uncased': 3e-5,
        'bert-large-uncased': 2e-5,
        'roberta-large': 1e-5,
    },
    'csqa-text-label': {
        'lstm': 3e-4,
        'openai-gpt': 1e-4,
        'bert-base-uncased': 3e-5,
        'bert-large-uncased': 2e-5,
        'roberta-large': 1e-5,
    },
    'obqa': {
        'lstm': 3e-4,
        'openai-gpt': 3e-5,
        'bert-base-cased': 1e-4,
        'bert-large-cased': 1e-4,
        'roberta-large': 1e-5,
    },
    'medqa_usmle': {
        'cambridgeltl/SapBERT-from-PubMedBERT-fulltext': 5e-5,
    },
}

DATASET_LIST = ['csqa', 'obqa', 'medqa_usmle', 'csqa-entity-label', 'csqa-text-label']

DATASET_SETTING = {
    'csqa': 'inhouse',
    'csqa-entity-label': 'inhouse',
    'csqa-text-label': 'inhouse',
    'obqa': 'official',
    'medqa_usmle': 'official',
}

DATASET_NO_TEST = []

EMB_PATHS = {
    'transe': 'data/cpnet/glove.transe.sgd.ent.npy',
    'numberbatch': 'data/cpnet/concept.nb.npy',
    'tzw': 'data/cpnet/tzw.ent.npy',
    'ddb': 'data/ddb/ent_emb.npy',
}


def add_data_arguments(parser):
    # arguments that all datasets share
    parser.add_argument('--ent_emb', default=['tzw'], nargs='+', help='sources for entity embeddings')
    # dataset specific
    parser.add_argument('-ds', '--dataset', default='csqa', choices=DATASET_LIST, help='dataset name')
    parser.add_argument('--data_dir', default='data', type=str, help='Path to the data directory')
    parser.add_argument('-ih', '--inhouse', type=utils.bool_flag, nargs='?', const=True, help='run in-house setting')
    parser.add_argument('--inhouse_train_qids', default='data/{dataset}/inhouse_split_qids.txt', help='qids of the in-house training set')
    # statements
    parser.add_argument('--train_statements', default='{data_dir}/{dataset}/statement/train.statement.jsonl')
    parser.add_argument('--dev_statements', default='{data_dir}/{dataset}/statement/dev.statement.jsonl')
    parser.add_argument('--test_statements', default='{data_dir}/{dataset}/statement/test.statement.jsonl')

    parser.add_argument('--train_span_path', default='{data_dir}/{dataset}/linking/{dataset}_linked_span_train.jsonl')
    parser.add_argument('--dev_span_path', default='{data_dir}/{dataset}/linking/{dataset}_linked_span_dev.jsonl')
    parser.add_argument('--test_span_path', default='{data_dir}/{dataset}/linking/{dataset}_linked_span_test.jsonl')

    parser.add_argument('--train_ids_path', default='{data_dir}/{dataset}/linking/{dataset}_linked_ids_train.jsonl')
    parser.add_argument('--dev_ids_path', default='{data_dir}/{dataset}/linking/{dataset}_linked_ids_dev.jsonl')
    parser.add_argument('--test_ids_path', default='{data_dir}/{dataset}/linking/{dataset}_linked_ids_test.jsonl')

    # preprocessing options
    parser.add_argument('-sl', '--max_seq_len', default=100, type=int)
    # set dataset defaults
    args, _ = parser.parse_known_args()
    parser.set_defaults(ent_emb_paths=[EMB_PATHS.get(s) for s in args.ent_emb],
                        inhouse=(DATASET_SETTING[args.dataset] == 'inhouse'),
                        inhouse_train_qids=args.inhouse_train_qids.format(dataset=args.dataset))
    data_splits = ('train', 'dev') if args.dataset in DATASET_NO_TEST else ('train', 'dev', 'test')
    for split in data_splits:
        for attribute in ('statements', 'span_path', 'ids_path'):
            attr_name = f'{split}_{attribute}'
            parser.set_defaults(**{attr_name: getattr(args, attr_name).format(dataset=args.dataset, data_dir=args.data_dir)})
    if 'test' not in data_splits:
        parser.set_defaults(test_statements=None)


def add_encoder_arguments(parser):
    parser.add_argument('-enc', '--encoder', default='bert-large-uncased', help='encoder type')
    parser.add_argument('--encoder_layer', default=-1, type=int, help='encoder layer ID to use as features (used only by non-LSTM encoders)')
    parser.add_argument('-elr', '--encoder_lr', default=2e-5, type=float, help='learning rate')
    args, _ = parser.parse_known_args()
    parser.set_defaults(encoder_lr=ENCODER_DEFAULT_LR[args.dataset].get(args.encoder, ENCODER_DEFAULT_LR['default']))


def add_optimization_arguments(parser):
    parser.add_argument('--loss', default='cross_entropy', choices=['margin_rank', 'cross_entropy'], help='model type')
    parser.add_argument('--optim', default='radam', choices=['sgd', 'adam', 'adamw', 'radam'], help='learning rate scheduler')
    parser.add_argument('--lr_schedule', default='fixed', choices=['fixed', 'warmup_linear', 'warmup_constant'], help='learning rate scheduler')
    parser.add_argument('-bs', '--batch_size', default=32, type=int)
    parser.add_argument('--warmup_steps', type=float, default=150)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad norm (0 to disable)')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='l2 weight decay strength')
    parser.add_argument('--n_epochs', default=100, type=int, help='total number of training epochs to perform.')
    parser.add_argument('-me', '--max_epochs_before_stop', default=10, type=int, help='stop training if dev does not increase for N epochs')


def add_additional_arguments(parser):
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--cuda', default=True, type=utils.bool_flag, nargs='?', const=True, help='use GPU')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--debug', default=False, type=utils.bool_flag, nargs='?', const=True, help='run in debug mode')
    args, _ = parser.parse_known_args()
    if args.debug:
        parser.set_defaults(batch_size=1, log_interval=1, eval_interval=5)


def get_parser():
    """A helper function that handles the arguments that all models share"""
    parser = argparse.ArgumentParser(add_help=False)
    add_data_arguments(parser)
    add_encoder_arguments(parser)
    add_optimization_arguments(parser)
    add_additional_arguments(parser)
    return parser
