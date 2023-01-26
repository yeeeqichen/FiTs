import argparse
from multiprocessing import cpu_count
from preprocess_utils.convert_csqa import convert_to_entailment
from preprocess_utils.convert_obqa import convert_to_obqa_statement
from preprocess_utils.conceptnet import extract_english, construct_graph
from preprocess_utils.grounding import create_matcher_patterns, ground, my_ground_entity_label, my_ground_text_label
from preprocess_utils.graph import generate_adj_data_from_grounded_concepts__use_LM

input_paths = {
    'csqa': {
        'train': './data/csqa/train_rand_split.jsonl',
        'dev': './data/csqa/dev_rand_split.jsonl',
        'test': './data/csqa/test_rand_split_no_answers.jsonl',
    },
    'csqa-entity-label': {
        'train': './data/csqa-entity-label/train_rand_split.jsonl',
        'dev': './data/csqa-entity-label/dev_rand_split.jsonl',
        'test': './data/csqa-entity-label/test_rand_split_no_answers.jsonl',
    },
    'csqa-text-label': {
        'train': './data/csqa-text-label/train_rand_split.jsonl',
        'dev': './data/csqa-text-label/dev_rand_split.jsonl',
        'test': './data/csqa-text-label/test_rand_split_no_answers.jsonl',
    },
    'obqa': {
        'train': './data/obqa/OpenBookQA-V1-Sep2018/Data/Additional/train_complete.jsonl',
        'dev': './data/obqa/OpenBookQA-V1-Sep2018/Data/Additional/dev_complete.jsonl',
        'test': './data/obqa/OpenBookQA-V1-Sep2018/Data/Additional/test_complete.jsonl',
    },
    'cpnet': {
        'csv': './data/cpnet/conceptnet-assertions-5.6.0.csv',
    },
}

output_paths = {
    'cpnet': {
        'csv': './data/cpnet/conceptnet.en.csv',
        'vocab': './data/cpnet/concept.txt',
        'patterns': './data/cpnet/matcher_patterns.json',
        'unpruned-graph': './data/cpnet/conceptnet.en.unpruned.graph-new',
        'pruned-graph': './data/cpnet/conceptnet.en.pruned.graph-new',
    },
    'csqa': {
        'statement': {
            'train': './data/csqa/statement/train.statement.jsonl',
            'dev': './data/csqa/statement/dev.statement.jsonl',
            'test': './data/csqa/statement/test.statement.jsonl',
        },
        'grounded': {
            'train': './data/csqa/grounded/train.grounded.jsonl',
            'dev': './data/csqa/grounded/dev.grounded.jsonl',
            'test': './data/csqa/grounded/test.grounded.jsonl',
        },
        'graph': {
            'adj-train': './data/csqa/graph/train.graph.adj.pk',
            'adj-dev': './data/csqa/graph/dev.graph.adj.pk',
            'adj-test': './data/csqa/graph/test.graph.adj.pk',
        },
    },
    'csqa-entity-label': {
        'statement': {
            'train': './data/csqa-entity-label/statement/train.statement.jsonl',
            'dev': './data/csqa-entity-label/statement/dev.statement.jsonl',
            'test': './data/csqa-entity-label/statement/test.statement.jsonl',
        },
        'grounded': {
            'train': './data/csqa-entity-label/grounded/train.grounded.jsonl',
            'dev': './data/csqa-entity-label/grounded/dev.grounded.jsonl',
            'test': './data/csqa-entity-label/grounded/test.grounded.jsonl',
        },
        'graph': {
            'adj-train': './data/csqa-entity-label/graph/train.graph.adj.pk',
            'adj-dev': './data/csqa-entity-label/graph/dev.graph.adj.pk',
            'adj-test': './data/csqa-entity-label/graph/test.graph.adj.pk',
        },
    },
    'csqa-text-label': {
        'statement': {
            'train': './data/csqa-text-label/statement/train.statement.jsonl',
            'dev': './data/csqa-text-label/statement/dev.statement.jsonl',
            'test': './data/csqa-text-label/statement/test.statement.jsonl',
        },
        'grounded': {
            'train': './data/csqa-text-label/grounded/train.grounded.jsonl',
            'dev': './data/csqa-text-label/grounded/dev.grounded.jsonl',
            'test': './data/csqa-text-label/grounded/test.grounded.jsonl',
        },
        'graph': {
            'adj-train': './data/csqa-text-label/graph/train.graph.adj.pk',
            'adj-dev': './data/csqa-text-label/graph/dev.graph.adj.pk',
            'adj-test': './data/csqa-text-label/graph/test.graph.adj.pk',
        },
    },
    'obqa': {
        'statement': {
            'train': './data/obqa/statement/train.statement.jsonl',
            'dev': './data/obqa/statement/dev.statement.jsonl',
            'test': './data/obqa/statement/test.statement.jsonl',
            'train-fairseq': './data/obqa/fairseq/official/train.jsonl',
            'dev-fairseq': './data/obqa/fairseq/official/valid.jsonl',
            'test-fairseq': './data/obqa/fairseq/official/test.jsonl',
        },
        'grounded': {
            'train': './data/obqa/grounded/train.grounded.jsonl',
            'dev': './data/obqa/grounded/dev.grounded.jsonl',
            'test': './data/obqa/grounded/test.grounded.jsonl',
        },
        'graph': {
            'adj-train': './data/obqa/graph/train.graph.adj.pk',
            'adj-dev': './data/obqa/graph/dev.graph.adj.pk',
            'adj-test': './data/obqa/graph/test.graph.adj.pk',
        },
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['common', 'csqa', 'obqa'],
                        choices=['common', 'csqa', 'obqa', 'csqa-entity-label', 'csqa-text-label'], nargs='+')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    routines = {
        'common': [
            {'func': extract_english, 'args': (input_paths['cpnet']['csv'], output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'])},
            {'func': construct_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
                                               output_paths['cpnet']['unpruned-graph'], False)},
            {'func': construct_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
                                               output_paths['cpnet']['pruned-graph'], True)},
            {'func': create_matcher_patterns, 'args': (output_paths['cpnet']['vocab'], output_paths['cpnet']['patterns'])},
        ],
        'csqa': [
            {'func': convert_to_entailment, 'args': (input_paths['csqa']['train'], output_paths['csqa']['statement']['train'])},
            {'func': convert_to_entailment, 'args': (input_paths['csqa']['dev'], output_paths['csqa']['statement']['dev'])},
            {'func': convert_to_entailment, 'args': (input_paths['csqa']['test'], output_paths['csqa']['statement']['test'])},
            {'func': ground, 'args': (output_paths['csqa']['statement']['train'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['train'], args.nprocs)},
            {'func': ground, 'args': (output_paths['csqa']['statement']['dev'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['dev'], args.nprocs)},
            {'func': ground, 'args': (output_paths['csqa']['statement']['test'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['test'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-test'], args.nprocs)},
        ],

        'csqa-entity-label': [
            # {'func': convert_to_entailment, 'args': (input_paths['csqa-entity-label']['train'], output_paths['csqa-entity-label']['statement']['train'])},
            # {'func': convert_to_entailment, 'args': (input_paths['csqa-entity-label']['dev'], output_paths['csqa-entity-label']['statement']['dev'])},
            # {'func': convert_to_entailment, 'args': (input_paths['csqa-entity-label']['test'], output_paths['csqa-entity-label']['statement']['test'])},
            # {'func': my_ground_entity_label, 'args': (output_paths['csqa-entity-label']['statement']['train'], output_paths['cpnet']['vocab'],
            #                           output_paths['cpnet']['patterns'], output_paths['csqa-entity-label']['grounded']['train'], args.nprocs)},
            # {'func': my_ground_entity_label, 'args': (output_paths['csqa-entity-label']['statement']['dev'], output_paths['cpnet']['vocab'],
            #                           output_paths['cpnet']['patterns'], output_paths['csqa-entity-label']['grounded']['dev'], args.nprocs)},
            {'func': my_ground_entity_label, 'args': (output_paths['csqa-entity-label']['statement']['test'], output_paths['cpnet']['vocab'],
                                                      output_paths['cpnet']['patterns'], output_paths['csqa-entity-label']['grounded']['test'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa-entity-label']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa-entity-label']['graph']['adj-train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa-entity-label']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa-entity-label']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa-entity-label']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa-entity-label']['graph']['adj-test'], args.nprocs)},
        ],

        'csqa-text-label': [
            {'func': convert_to_entailment, 'args': (input_paths['csqa-text-label']['train'], output_paths['csqa-text-label']['statement']['train'])},
            {'func': convert_to_entailment, 'args': (input_paths['csqa-text-label']['dev'], output_paths['csqa-text-label']['statement']['dev'])},
            {'func': convert_to_entailment, 'args': (input_paths['csqa-text-label']['test'], output_paths['csqa-text-label']['statement']['test'])},
            {'func': my_ground_text_label, 'args': (output_paths['csqa-text-label']['statement']['train'], output_paths['cpnet']['vocab'],
                                                    output_paths['cpnet']['patterns'], output_paths['csqa-text-label']['grounded']['train'], args.nprocs)},
            {'func': my_ground_text_label, 'args': (output_paths['csqa-text-label']['statement']['dev'], output_paths['cpnet']['vocab'],
                                                    output_paths['cpnet']['patterns'], output_paths['csqa-text-label']['grounded']['dev'], args.nprocs)},
            {'func': my_ground_text_label,
             'args': (output_paths['csqa-text-label']['statement']['test'], output_paths['cpnet']['vocab'],
                      output_paths['cpnet']['patterns'], output_paths['csqa-text-label']['grounded']['test'],
                      args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (
                output_paths['csqa-text-label']['grounded']['train'], output_paths['cpnet']['pruned-graph'],
                output_paths['cpnet']['vocab'], output_paths['csqa-text-label']['graph']['adj-train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (
                output_paths['csqa-text-label']['grounded']['dev'], output_paths['cpnet']['pruned-graph'],
                output_paths['cpnet']['vocab'], output_paths['csqa-text-label']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (
                output_paths['csqa-text-label']['grounded']['test'], output_paths['cpnet']['pruned-graph'],
                output_paths['cpnet']['vocab'], output_paths['csqa-text-label']['graph']['adj-test'], args.nprocs)},
        ],

        'obqa': [
            {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['train'], output_paths['obqa']['statement']['train'], output_paths['obqa']['statement']['train-fairseq'])},
            {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['dev'], output_paths['obqa']['statement']['dev'], output_paths['obqa']['statement']['dev-fairseq'])},
            {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['test'], output_paths['obqa']['statement']['test'], output_paths['obqa']['statement']['test-fairseq'])},
            {'func': ground, 'args': (output_paths['obqa']['statement']['train'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['train'], args.nprocs)},
            {'func': ground, 'args': (output_paths['obqa']['statement']['dev'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['dev'], args.nprocs)},
            {'func': ground, 'args': (output_paths['obqa']['statement']['test'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['test'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['obqa']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['obqa']['graph']['adj-train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['obqa']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['obqa']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['obqa']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['obqa']['graph']['adj-test'], args.nprocs)},
        ],
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()
    # pass
