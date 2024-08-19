import os
import sys
# Get the absolute path to the root_dir
current_dir = os.path.abspath(os.path.dirname(__file__))
print('RUN current_dir:',current_dir)
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(os.path.join(current_dir,'utils'))
sys.path.append(os.path.join(current_dir,'attack'))
sys.path.append(os.path.join(current_dir,'explain'))
sys.path.append(os.path.join(current_dir,'detection'))


from   utils.config import *
from   utils.data_utils import *
from   utils.general_utils import *
from   attack.backdoor_utils import *
from   explain.explainer_utils import *
from   detection.explainer_detection_metrics import *
import argparse
import subprocess
# import torch
# device=torch.device('mps')



''' DO NOT CHANGE'''
hyp_dict_backdoor = get_info('hyp_dict_backdoor')
hyp_dict_backdoor_adaptive= get_info('hyp_dict_backdoor_adaptive')
hyp_dict_clean    = get_info('hyp_dict_clean')
data_shape_dict   = get_info('data_shape_dict')
src_dir     = get_info('src_dir')
print('src_dir:',src_dir)
data_dir    = get_info('data_dir')
explain_dir = get_info('explain_dir')
train_dir   = get_info('train_dir')
train_dir_clean = get_info('train_dir_cln')
adapt_gen_dir = get_info('adapt_gen_dir')
adapt_benign_models = get_info('adapt_benign_models')
generator_hyperparam_dicts_v1 = get_info('generator_hyperparam_dicts_v1')
generator_hyperparam_dicts_v2 = get_info('generator_hyperparam_dicts_v2')
generator_hyperparam_dicts_iterative_exp = get_info('generator_hyperparam_dicts_iterative_exp')
surrogate_hyperparams_initial = get_info('surrogate_hyperparams_initial')
surrogate_hyperparams_looping = get_info('surrogate_hyperparams_looping')



''' CAN CUSTOMIZE / EXPERIMENT WITH THESE VALUES '''
def parse_args():
    parser=argparse.ArgumentParser(description="Input arguments")
    # General
    # parser.add_argument('--device',                     type=str,               default='cpu',          help='cpu, gpu, mps, etc.')
    parser.add_argument('--dataset',                    type=str,               default='MUTAG',        help='Dataset to attack and explain.')
    parser.add_argument('--plot',                       action='store_true',                            help='Include to save plots of results.')
    parser.add_argument('--regenerate_data',            action='store_true',                            help='Include to regenerate any previously-generated data.')
    parser.add_argument('--seed',                       type=int,               default=2575,           help='Makes randomness constant.')
    parser.add_argument('--model_hyp_set',              type=str,               default='A',            help='Your choice of pre-defined hyperparameter sets, as defined in /repo/src/config.py.')

    # Attack
    parser.add_argument('--run_attack',                 action='store_true',                            help='If included, conducts attack before proceeding to explanation/detection.')
    parser.add_argument('--attack_target_label',        type=int,               default=0,              help='Class targeted by backdoor attack.')
    parser.add_argument('--backdoor_type',              type=str,               default='random',       help='Valid values: "random","adaptive","clean_label"')
    parser.add_argument('--ER_graph_P',                 type=float,             default=1,              help='Probability of an edge between any two nodes in Erdos-Renyi graph generation.')
    parser.add_argument('--poison_rate',                type=float,             default=0.2,            help='Poison rate, expressed as a portion of training data size.')
    parser.add_argument('--graph_type',                 type=str,               default='ER',           help='Random graph synthesis method for producing the trigger.')
    parser.add_argument('--PA_graph_K',                 type=int,               default=0,              help='Number of neighbors in initial ring lattice for Small-World graph generation. If 0, will automatically compute default value as a function of trigger size.')
    parser.add_argument('--SW_graph_K',                 type=int,               default=0,              help='Number of edges to attach from a new node to existing nodes in Preferential-Attachment graph generation. If 0, will automatically compute default value as a function of trigger size.')
    parser.add_argument('--SW_graph_P',                 type=float,             default=1,              help='Probability of rewiring each edge in Small-World graph generation.')
    parser.add_argument('--trigger_size',               type=int,               default=6,              help='Subgraph trigger size.')

    # Adaptive generator training
    parser.add_argument('--contin_or_scratch',          type=str,               default='from_scratch', help='Set to "continuous" if you would like to continue refining a generator; otherwise, "from_scratch".')
    # parser.add_argument('--gen_alg_v',                  type=int,               default=3,              help='Hyperparameter set to use for training adaptive trigger generator.')
    parser.add_argument('--gen_rounds',                 type=int,               default=3,              help='Number of iterations to train adaptive trigger generator.')
    parser.add_argument('--load_or_train_benign_model', type=str,               default='train',        help='Valid values: "load","train"')
    parser.add_argument('--load_or_train_generator',    type=str,               default='train',        help='Valid values: "load","train"')

    # GNNExplainer
    parser.add_argument('--edge_reduction',             type=str,               default='sum',          help='Method for aggregating edges for computations by GNNExplainer.')
    parser.add_argument('--edge_size',                  type=float,             default=0.0001,         help='Coefficient on "edge size" term in GNNExplainer loss; larger value will reduce number of edges preserved by explanatory subgraph.')
    parser.add_argument('--edge_ent',                   type=float,             default=1,              help='Coefficient on "edge entropy" term in GNNExplainer loss; larger value will make edge mask weights more decisive / binary.')
    parser.add_argument('--run_explain',                action='store_true',                            help='Set to True if running explanation/detection process.')
    parser.add_argument('--explain_lr',                 type=float,             default=0.1,            help='The learning rate use by GNNExplainer as it trains.')
    parser.add_argument('--explainer_epochs',           type=int,               default=50,             help='The number of epochs over which GNNExplainer trains.')
    parser.add_argument('--explanation_type',           type=str,               default='phenomenon',   help='The style of explanation used by GNNExplainer. "Phenomenon" explains with respect to a target label; "Model" explains with respect to GNN output.')
    parser.add_argument('--node_feat_ent',              type=float,             default=1.0,            help='Coefficient on "node entropy" term in GNNExplainer loss; larger value will make node mask weights more decisive / binary.')
    parser.add_argument('--node_feat_reduction',        type=str,               default='sum',          help='Method for aggregating node features for computations by GNNExplainer.')
    parser.add_argument('--node_feat_size',             type=float,             default=0.0001,         help='Coefficient on "node size" term in GNNExplainer loss; larger value will reduce number of nodes preserved by explanatory subgraph.')
    parser.add_argument('--thresh_type',                type=str,               default='hard',         help='Method for post-processing mask; "hard" preserves all features with weights above a threshold, and "topk" preserves the top k features.')
    parser.add_argument('--thresh_val',                 type=float,             default=0.3,            help='Threshold for preserving GNNExplainer mask features.')

    # Detection
    parser.add_argument('--lower_thresh_percentile',    type=float,             default=0.25,           help='Percentile defining lower threshold for backdoor prediction (must range from 0 to 1)')
    parser.add_argument('--upper_thresh_percentile',    type=float,             default=0.75,           help='Percentile defining upper threshold for backdoor prediction (must range from 0 to 1)')
    
    args=parser.parse_args()
    return args



def main():
    args=parse_args()

    # General
    # device = str(args.device)
    dataset = str(args.dataset)
    seed = str(args.seed)
    model_hyp_set = str(args.model_hyp_set)
    plot = args.plot #boolean
    regenerate_data = args.regenerate_data #boolean
    
    # Attack
    run_attack = args.run_attack
    attack_target_label = str(args.attack_target_label)
    backdoor_type = str(args.backdoor_type)
    ER_graph_P = str(args.ER_graph_P)
    poison_rate = str(args.poison_rate)
    graph_type = str(args.graph_type)
    PA_graph_K = str(args.PA_graph_K)
    SW_graph_K = str(args.SW_graph_K)
    SW_graph_P = str(args.SW_graph_P)
    trigger_size = str(args.trigger_size)

    # Adaptive generator training
    contin_or_scratch = str(args.contin_or_scratch)
    # gen_alg_v = str(args.gen_alg_v)
    gen_rounds = str(args.gen_rounds)
    load_or_train_benign_model = str(args.load_or_train_benign_model)

    # GNNExplainer
    edge_reduction = str(args.edge_reduction)
    edge_size = str(args.edge_size)
    edge_ent = str(args.edge_ent)
    run_explain = args.run_explain
    explain_lr = str(args.explain_lr)
    explainer_epochs = str(args.explainer_epochs)
    explanation_type = str(args.explanation_type)
    node_feat_ent = str(args.node_feat_ent)
    node_feat_reduction = str(args.node_feat_reduction)
    node_feat_size = str(args.node_feat_size)
    thresh_type = str(args.thresh_type)
    thresh_val = str(args.thresh_val)

    # Detection
    lower_thresh_percentile = str(args.lower_thresh_percentile)
    upper_thresh_percentile = str(args.upper_thresh_percentile)

    if run_attack==True:
        if backdoor_type == 'adaptive':
            if args.load_or_train_generator == 'train':
                assert os.path.exists(f'{src_dir}/attack/run_build_adaptive_generator.py')
                arguments = ['python',f'{src_dir}/attack/run_build_adaptive_generator.py']
                arguments += ['--attack_target_label',attack_target_label,'--contin_or_scratch',contin_or_scratch, '--dataset',dataset, '--poison_rate',poison_rate,
                            '--gen_rounds',gen_rounds, '--load_or_train_benign_model',load_or_train_benign_model, '--seed',seed, 
                            '--trigger_size',trigger_size]#, '--device',device]
                subprocess.run(arguments)
        
        assert os.path.exists(f'{src_dir}/attack/run_attack.py')
        arguments = ['python',f'{src_dir}/attack/run_attack.py']
        arguments += ['--attack_target_label',attack_target_label, '--backdoor_type',backdoor_type, '--contin_or_scratch',contin_or_scratch, '--dataset',dataset,
                      '--ER_graph_P', ER_graph_P, '--poison_rate',poison_rate,# '--gen_alg_v', gen_alg_v, #'--gen_rounds', gen_rounds, 
                      '--graph_type',graph_type,
                      '--model_hyp_set',model_hyp_set, '--PA_graph_K',PA_graph_K,  '--SW_graph_K',SW_graph_K, '--SW_graph_P',SW_graph_P,
                      '--seed',seed, '--trigger_size',trigger_size]#, '--device',device]
        if plot==True:
            arguments += ['--plot']
        if regenerate_data==True:
            arguments += ['--regenerate_data']
        
        print(' '.join(arguments))
        subprocess.run(arguments)

    if run_explain==True:
        assert os.path.exists(f'{src_dir}/detection/run_explain_and_detect.py')
        arguments = ['python',f'{src_dir}/detection/run_explain_and_detect.py']
        arguments += ['--attack_target_label',attack_target_label, '--backdoor_type',backdoor_type, '--contin_or_scratch',contin_or_scratch, '--dataset',dataset,
                      '--edge_reduction',edge_reduction, '--edge_size',edge_size, '--edge_ent',edge_ent, '--ER_graph_P',ER_graph_P, '--explain_lr',explain_lr, 
                      '--explainer_epochs',explainer_epochs, '--explanation_type',explanation_type, '--poison_rate',poison_rate, #'--gen_alg_v',gen_alg_v, 
                      '--gen_rounds',gen_rounds, '--graph_type',graph_type, '--model_hyp_set',model_hyp_set, '--node_feat_ent',node_feat_ent, 
                      '--node_feat_reduction', node_feat_reduction, '--node_feat_size',node_feat_size, '--PA_graph_K',PA_graph_K,
                      '--SW_graph_K',SW_graph_K, '--SW_graph_P',SW_graph_P, '--seed',seed, '--thresh_type',thresh_type, 
                      '--thresh_val',thresh_val, '--trigger_size',trigger_size, '--lower_thresh_percentile',lower_thresh_percentile, 
                      '--upper_thresh_percentile',upper_thresh_percentile]#, '--device',device]
        if plot==True:
            arguments += ['--plot']
        if regenerate_data==True:
            arguments += ['--regenerate_data']

        print(' '.join(arguments))
        subprocess.run(arguments)



if __name__ == '__main__':
    main()


