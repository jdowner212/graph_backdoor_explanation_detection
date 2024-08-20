
import argparse
import os
import sys

current_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
root_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
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
import numpy as np
import pickle
import random
import copy


hyp_dict_backdoor = get_info('hyp_dict_backdoor')
hyp_dict_backdoor_adaptive = get_info('hyp_dict_backdoor_adaptive')
data_shape_dict   = get_info('data_shape_dict')
explain_dir = get_info('explain_dir')


def parse_args():
    parser=argparse.ArgumentParser(description="GNNExplainer and backdoor detection: input arguments")
    parser.add_argument('--attack_target_label',        type=int,               default=0,              help='Class targeted by backdoor attack.')
    parser.add_argument('--backdoor_type',              type=str,               default='random',       help='Valid values: "random","adaptive","clean_label"')
    # parser.add_argument('--contin_or_scratch',          type=str,               default='from_scratch', help='Set to "continuous" if you would like to continue refining a generator; otherwise, "from_scratch".')
    parser.add_argument('--dataset',                    type=str,               default='MUTAG',        help='Dataset to attack and explain.')
    parser.add_argument('--edge_reduction',             type=str,               default='sum',          help='Method for aggregating edges for computations by GNNExplainer.')
    parser.add_argument('--edge_size',                  type=float,             default=0.0001,         help='Coefficient on "edge size" term in GNNExplainer loss; larger value will reduce number of edges preserved by explanatory subgraph.')
    parser.add_argument('--edge_ent',                   type=float,             default=1,              help='Coefficient on "edge entropy" term in GNNExplainer loss; larger value will make edge mask weights more decisive / binary.')
    parser.add_argument('--ER_graph_P',                 type=float,             default=1,              help='Probability of an edge between any two nodes in Erdos-Renyi graph generation.')
    parser.add_argument('--explain_lr',                 type=float,             default=0.1,            help='The learning rate use by GNNExplainer as it trains.')
    parser.add_argument('--explainer_epochs',           type=int,               default=50,             help='The number of epochs over which GNNExplainer trains.')
    parser.add_argument('--explanation_type',           type=str,               default='phenomenon',   help='The style of explanation used by GNNExplainer. "Phenomenon" explains with respect to a target label; "Model" explains with respect to GNN output.')
    parser.add_argument('--poison_rate',                type=float,             default=0.2,            help='Poison rate, expressed as a portion of training data size.')
    parser.add_argument('--gen_rounds',                 type=int,               default=3,              help='Number of iterations to train adaptive trigger generator.')
    parser.add_argument('--graph_type',                 type=str,               default='ER',           help='Random graph synthesis method for producing the trigger.')
    parser.add_argument('--lower_thresh_percentile',    type=float,             default=0.25,           help='Percentile defining lower threshold for backdoor prediction (must range from 0 to 1)')
    parser.add_argument('--upper_thresh_percentile',    type=float,             default=0.75,           help='Percentile defining upper threshold for backdoor prediction (must range from 0 to 1)')
    parser.add_argument('--model_hyp_set',              type=str,               default='A',            help='Your choice of pre-defined hyperparameter sets, as defined in /repo/src/config.py.')
    parser.add_argument('--node_feat_ent',              type=float,             default=1.0,            help='Coefficient on "node entropy" term in GNNExplainer loss; larger value will make node mask weights more decisive / binary.')
    parser.add_argument('--node_feat_reduction',        type=str,               default='sum',          help='Method for aggregating node features for computations by GNNExplainer.')
    parser.add_argument('--node_feat_size',             type=float,             default=0.0001,         help='Coefficient on "node size" term in GNNExplainer loss; larger value will reduce number of nodes preserved by explanatory subgraph.')
    parser.add_argument('--PA_graph_K',                 type=int,               default=0,              help='Number of neighbors in initial ring lattice for Small-World graph generation. If 0, will automatically compute default value as a function of trigger size.')
    parser.add_argument('--plot',                       action='store_true',                            help='Include to save plots of results.')
    parser.add_argument('--SW_graph_K',                 type=int,               default=0,              help='Number of edges to attach from a new node to existing nodes in Preferential-Attachment graph generation. If 0, will automatically compute default value as a function of trigger size.')
    parser.add_argument('--SW_graph_P',                 type=float,             default=1,              help='Probability of rewiring each edge in Small-World graph generation.')
    parser.add_argument('--regenerate_data',            action='store_true',                            help='Include to regenerate any previously-generated data.')
    parser.add_argument('--seed',                       type=int,               default=2575,           help='Makes randomness constant.')
    parser.add_argument('--thresh_type',                type=str,               default='hard',         help='Method for post-processing mask; "hard" preserves all features with weights above a threshold, and "topk" preserves the top k features.')
    parser.add_argument('--thresh_val',                 type=float,             default=0.3,            help='Threshold for preserving GNNExplainer mask features.')
    parser.add_argument('--trigger_size',               type=int,               default=6,              help='Subgraph trigger size.')

    args=parser.parse_args()
    return args



def main():
    args=parse_args()
    dataset                = args.dataset
    model_hyp_set          = args.model_hyp_set
    attack_target_label    = args.attack_target_label
    attack_target_label    = attack_target_label
    trigger_size           = args.trigger_size
    graph_type             = args.graph_type
    if args.backdoor_type=='random' or args.backdoor_type=='clean_label':
        this_hyp_dict=hyp_dict_backdoor
    elif args.backdoor_type=='adaptive':
        this_hyp_dict=hyp_dict_backdoor_adaptive
    these_classifier_hyperparams = this_hyp_dict[dataset][attack_target_label][model_hyp_set]
    model_type    = these_classifier_hyperparams['model_type']

    num_classes            = data_shape_dict[dataset]['num_classes']
    # assert args.contin_or_scratch == 'continuous' or args.contin_or_scratch == 'from_scratch'
    
    these_explainer_hyperparams = build_explainer_hyperparams()
    these_explainer_hyperparams['threshold_config']['threshold_type'] = args.thresh_type
    these_explainer_hyperparams['threshold_config']['value']          = args.thresh_val
    these_explainer_hyperparams['coeffs']['edge_reduction']           = args.edge_reduction
    these_explainer_hyperparams['coeffs']['edge_size']                = args.edge_size
    these_explainer_hyperparams['coeffs']['edge_ent']                 = args.edge_ent
    these_explainer_hyperparams['coeffs']['node_feat_reduction']      = args.node_feat_reduction
    these_explainer_hyperparams['coeffs']['node_feat_size']           = args.node_feat_size
    these_explainer_hyperparams['coeffs']['node_feat_ent']            = args.node_feat_ent
    these_explainer_hyperparams['explain_lr']                         = args.explain_lr
    these_explainer_hyperparams['explainer_epochs']                   = args.explainer_epochs
    these_explainer_hyperparams['explanation_type']                   = args.explanation_type

    these_attack_specs = build_attack_specs()
    these_attack_specs['graph_type']          = args.graph_type
    these_attack_specs['backdoor_type']       = args.backdoor_type
    these_attack_specs['poison_rate']         = args.poison_rate
    these_attack_specs['attack_target_label'] = args.attack_target_label 
    these_attack_specs['trigger_size']        = trigger_size
    
    these_model_specs = build_model_specs()
    these_model_specs['clean_or_backdoor'] = 'backdoor'
    these_model_specs['model_hyp_set'] = model_hyp_set


    K,prob=None,None
    if args.backdoor_type == 'random' or args.backdoor_type == 'clean_label':
        if args.graph_type=='SW':
            K=args.SW_graph_K
            prob=args.SW_graph_P
        elif args.graph_type=='ER':
            prob=args.ER_graph_P
        elif args.graph_type=='PA':
            K=args.PA_graph_K
        if (args.graph_type=='SW' and args.SW_graph_K==0) or (args.graph_type=='PA' and args.PA_graph_K==0):
            K=trigger_size-1
            print('graph_type, K, trigger_size:',graph_type, K, trigger_size)
            assert validate_K(graph_type, K, trigger_size)
        if args.graph_type=='ER' and args.ER_graph_P==0:
            prob=1
        elif args.graph_type=='SW' and args.SW_graph_P==0:
            prob=1
    these_attack_specs['K']=K
    these_attack_specs['prob']=prob
    dataset_dict_clean = retrieve_data_process(args.regenerate_data, True, dataset, {}, seed=args.seed)
    print_attack_description(these_classifier_hyperparams, these_attack_specs, model_hyp_set)


    if args.backdoor_type == 'random' or args.backdoor_type == 'clean_label':
        ''''''''''''''''''''''''''''''
        '''      Load Dataset      '''
        ''''''''''''''''''''''''''''''
        these_attack_specs['adaptive']='unadaptive'
        dataset_dict_backdoor = retrieve_data_process(args.regenerate_data, False, dataset, these_attack_specs, seed=args.seed)



        ''''''''''''''''''''''''''''''
        '''   Load Backdoored GNN  '''
        ''''''''''''''''''''''''''''''
        model_path = get_model_path(dataset, these_classifier_hyperparams, these_attack_specs, these_model_specs['model_hyp_set'])
        assert os.path.exists(model_path)
        model, history = load_model(model_type, dataset, model_path, dataset_dict_backdoor, this_hyp_dict, attack_target_label, model_hyp_set, num_classes, False)
        backdoor_train_success_indices, clean_train_success_indices, clean_val_success_indices, asr = get_all_success_indices_plus_asr(model, dataset_dict_backdoor, dataset_dict_clean, 30, args.seed)
        if these_attack_specs['backdoor_type']=='clean_label':
            backdoor_train_success_indices = [i for i in range(len(dataset_dict_backdoor['train_backdoor_graphs'])) if dataset_dict_backdoor['train_backdoor_graphs'][i].pyg_graph.is_backdoored==True]
        print('ASR:',asr)
        fail=False
        issues = []
        if len(clean_train_success_indices)<=5:
            issues.append('clean')
            fail=True
        if len(clean_val_success_indices)<=5:
            issues.appen('clean')
            fail=True
        if len(backdoor_train_success_indices)<=5:
            issues.append('backdoor')
            fail=True
        issues = list(set(issues))
        if fail==True:
            print('Attack did not generate enough samples to test detection method.')
            if 'clean' in issues and 'backdoor' in issues:
                print('Both clean and backdoor accuracy are low. This suggests that the GNN did not train well. Try again with new GNN hyperparameters -- see config.py.')
            elif 'clean' in issues and 'backdoor' not in issues:
                print('Clean accuracy is low. GNN may be over-fitting to the trigger -- try new GNN hyperparameters (see config.py) or a different attack (relevant arguments: --attack_target_label, --graph type, --trigger size, --K, --prob).')
            elif 'clean' not in issues and 'backdoor' in issues:
                print('Backoor accuracy is low. Try new GNN hyperparameters (see config.py) or a different attack (relevant arguments: --attack_target_label, --graph type, --trigger size, --K, --prob).')



        ''''''''''''''''''''''''''''''
        '''   Explain and Detect   '''
        ''''''''''''''''''''''''''''''
        if fail==False:
            if these_attack_specs['backdoor_type']!='clean_label':
                df_title = f'metrics_df_{dataset}_{str(trigger_size)}_{graph_type}__thresh_{args.thresh_type}_{args.thresh_val}.pkl'
            else:
                df_title = f'clean_label_metrics_df_{dataset}_{str(trigger_size)}_{graph_type}__thresh_{args.thresh_type}_{args.thresh_val}.pkl'
            path = f'{explain_dir}/{dataset}/metrics_images'
            create_nested_folder(path)
            these_explainer_hyperparams = copy.copy(these_explainer_hyperparams)
            these_explainer_hyperparams['explainer_target_label'] = attack_target_label
            _, _, boxplot_path = parse_metric_image_path(dataset, these_attack_specs, these_classifier_hyperparams, these_explainer_hyperparams, these_model_specs)
            np.random.seed(args.seed)
            df =   try_metrics(dataset_dict_backdoor,   
                                    dataset_dict_clean, 
                                    dataset,    
                                    clean_val_success_indices,  
                                    clean_train_success_indices,    
                                    backdoor_train_success_indices,
                                    model,                   
                                    asr,        
                                    history,                    
                                    these_attack_specs,             
                                    these_explainer_hyperparams,
                                    these_model_specs,       
                                    True, 
                                    df_title,                   
                                    boxplot_path,
                                    save_explainer=False,
                                    relevant_metrics=['elbow_dist', 'curv_dist', 'es', 'connectivity', 'pred_conf', 'node_deg_var', 'mask_feat_var'],
                                    lower_thresh_percentile=args.lower_thresh_percentile,
                                    upper_thresh_percentile=args.upper_thresh_percentile)
            

    elif args.backdoor_type=='adaptive':
        these_attack_specs['adaptive']='adaptive'
        ''''''''''''''''''''''''''''''
        '''      Load Dataset      '''
        ''''''''''''''''''''''''''''''
        random.seed(args.seed)
        # gen_dataset_folder_ext = f'_{args.contin_or_scratch}'
        dataset_path = get_dataset_path(dataset, these_attack_specs, clean=False,gen_dataset_folder_ext='')
        with open(dataset_path,'rb') as f:
            dataset_dict_adaptive = pickle.load(f)


        ''''''''''''''''''''''''''''''
        '''  Load Backdoored GNN   '''
        ''''''''''''''''''''''''''''''
        model_path = get_model_path(dataset, these_classifier_hyperparams, these_attack_specs, these_model_specs['model_hyp_set'])
        assert os.path.exists(model_path)
        model, history = load_model(model_type, dataset, model_path, dataset_dict_adaptive, this_hyp_dict, attack_target_label, model_hyp_set, num_classes, False)
        backdoor_train_success_indices, clean_train_success_indices, clean_val_success_indices, asr = get_all_success_indices_plus_asr(model, dataset_dict_adaptive, dataset_dict_clean, 30, args.seed)
        print('ASR:',asr)
        fail=False
        if len(clean_train_success_indices)<=5:
            print(f'Fewer than 3 successful clean_train examples -- try again with different attack.'); fail=True
        if len(clean_val_success_indices)<=5:
            print(f'Fewer than 3 successful clean_val examples -- try again with different attack.'); fail=True
        if len(backdoor_train_success_indices)<=5:
            print(f'Fewer than 3 successful backdoor_train examples -- try again with different attack.'); fail=True


        ''''''''''''''''''''''''''''''
        '''   Explain and Detect   '''
        ''''''''''''''''''''''''''''''
        if fail==False:
            df_title = f'metrics_df_{dataset}_adaptive_{str(trigger_size)}_thresh_{args.thresh_type}_{args.thresh_val}_{str(attack_target_label)}.pkl'
            path = f'{explain_dir}/{dataset}/metrics_images'
            create_nested_folder(path)
            these_explainer_hyperparams = copy.copy(these_explainer_hyperparams)
            these_explainer_hyperparams['explainer_target_label'] = attack_target_label
            _, _, boxplot_path = parse_metric_image_path(dataset, these_attack_specs, these_classifier_hyperparams, these_explainer_hyperparams, these_model_specs)

            np.random.seed(args.seed)
            df =   try_metrics(dataset_dict_adaptive,   
                                    dataset_dict_clean, 
                                    dataset,    
                                    clean_val_success_indices,  
                                    clean_train_success_indices,    
                                    backdoor_train_success_indices,
                                    model,                   
                                    asr,        
                                    history,                    
                                    these_attack_specs,             
                                    these_explainer_hyperparams,
                                    these_model_specs,       
                                    True, 
                                    df_title,                                
                                    boxplot_path,
                                    save_explainer=False,
                                    relevant_metrics=['elbow_dist', 'curv_dist', 'es', 'connectivity', 'pred_conf', 'node_deg_var', 'mask_feat_var'],
                                    lower_thresh_percentile=args.lower_thresh_percentile,
                                    upper_thresh_percentile=args.upper_thresh_percentile
                                    )

    ''''''''''''''''''''''''''''''
    '''     Display Results    '''
    ''''''''''''''''''''''''''''''            
    relevant_metrics = ['elbow_dist', 'curv_dist',     'es',     'connectivity',     'pred_conf',     'node_deg_var',     'mask_feat_var']
    relevant_cols = [m+' clean_val' for m in relevant_metrics]
    print()
    for npmr in range(1,8):
        display_detection_results(df,npmr,relevant_cols)
    print(f'See ~/repo/src/explainer_results/{args.dataset}/metrics_images for metric distributions.')


if __name__ == '__main__':
    main()

