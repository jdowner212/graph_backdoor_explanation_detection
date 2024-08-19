
import argparse
import os
import sys

current_dir = os.getcwd()
root_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir,'utils'))
sys.path.append(os.path.join(current_dir,'attack'))
sys.path.append(os.path.join(current_dir,'explain'))
sys.path.append(os.path.join(current_dir,'detection'))

from   backdoor_utils import *
import pickle
import random
import torch
import torch_geometric
from   torch_geometric.transforms import Compose, OneHotDegree
# device=torch.device('cpu')

hyp_dict_backdoor = get_info('hyp_dict_backdoor')
hyp_dict_backdoor_adaptive= get_info('hyp_dict_backdoor_adaptive')
hyp_dict_clean    = get_info('hyp_dict_clean')
data_shape_dict   = get_info('data_shape_dict')
src_dir     = get_info('src_dir')
data_dir    = get_info('data_dir')
train_dir   = get_info('train_dir')
train_dir_clean = get_info('train_dir_cln')
adapt_gen_dir = get_info('adapt_gen_dir')
adapt_benign_models = get_info('adapt_benign_models')
# generator_hyperparam_dicts_v1 = get_info('generator_hyperparam_dicts_v1')
# generator_hyperparam_dicts_v2 = get_info('generator_hyperparam_dicts_v2')
generator_hyperparam_dicts = get_info('generator_hyperparam_dicts')
surrogate_hyperparams_initial = get_info('surrogate_hyperparams_initial')
surrogate_hyperparams_looping = get_info('surrogate_hyperparams_looping')


def parse_args():
    parser=argparse.ArgumentParser(description="Attack input arguments")
    parser.add_argument('--attack_target_label',        type=int,               default=0,              help='Class targeted by backdoor attack.')
    parser.add_argument('--backdoor_type',              type=str,               default='random',       help='Valid values: "random","adaptive","clean_label"')
    parser.add_argument('--contin_or_scratch',          type=str,               default='from_scratch', help='Set to "continuous" if you would like to continue refining a generator; otherwise, "from_scratch".')
    parser.add_argument('--dataset',                    type=str,               default='MUTAG',        help='Dataset to attack and explain.')
    parser.add_argument('--ER_graph_P',                 type=float,             default=1,              help='Probability of an edge between any two nodes in Erdos-Renyi graph generation.')
    parser.add_argument('--poison_rate',                type=float,             default=0.2,            help='Poison rate, expressed as a portion of training data size.')
    # parser.add_argument('--gen_alg_v',                  type=int,               default=3,              help='Hyperparameter set to use for training adaptive trigger generator.')
    parser.add_argument('--graph_type',                 type=str,               default='ER',           help='Random graph synthesis method for producing the trigger.')
    parser.add_argument('--model_hyp_set',              type=str,               default='A',            help='Your choice of pre-defined hyperparameter sets, as defined in /repo/src/config.py.')
    parser.add_argument('--PA_graph_K',                 type=int,               default=0,              help='Number of neighbors in initial ring lattice for Small-World graph generation. If 0, will automatically compute default value as a function of trigger size.')
    parser.add_argument('--plot',                       action='store_true',                            help='Include to save plots of results.')
    parser.add_argument('--SW_graph_K',                 type=int,               default=0,              help='Number of edges to attach from a new node to existing nodes in Preferential-Attachment graph generation. If 0, will automatically compute default value as a function of trigger size.')
    parser.add_argument('--SW_graph_P',                 type=float,             default=1,              help='Probability of rewiring each edge in Small-World graph generation.')
    parser.add_argument('--regenerate_data',            action='store_true',                            help='Include to regenerate any previously-generated data.')
    parser.add_argument('--seed',                       type=int,               default=2575,           help='Makes randomness constant.')
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
    num_classes            = data_shape_dict[dataset]['num_classes']
    num_node_features      = data_shape_dict[dataset]['num_node_features']
    assert args.contin_or_scratch == 'continuous' or args.contin_or_scratch == 'from_scratch'
    

    these_attack_specs = build_attack_specs()
    these_attack_specs['backdoor_type']       = args.backdoor_type
    these_attack_specs['graph_type']          = args.graph_type
    these_attack_specs['poison_rate']         = args.poison_rate
    these_attack_specs['attack_target_label'] = args.attack_target_label 
    these_attack_specs['trigger_size']        = trigger_size
    
    these_model_specs = build_model_specs()
    these_model_specs['clean_or_backdoor'] = 'backdoor'
    these_model_specs['model_hyp_set'] = model_hyp_set
    
    
    dataset_dict_clean = retrieve_data_process(args.regenerate_data, True, dataset, these_attack_specs, seed=args.seed)

    if args.backdoor_type == 'random' or args.backdoor_type == 'clean_label':
        ''''''''''''''''''''''''''''''
        ''' Set Attack Hyperparams '''
        ''''''''''''''''''''''''''''''
        these_attack_specs['adaptive']='unadaptive'
        K,prob=None,None
        if args.graph_type=='SW':
            K=args.SW_graph_K
            prob=args.SW_graph_P
        elif args.graph_type=='ER':
            prob=args.ER_graph_P
        elif args.graph_type=='PA':
            K=args.PA_graph_K
        if (args.graph_type=='SW' and args.SW_graph_K==0) or (args.graph_type=='PA' and args.PA_graph_K==0):
            K=trigger_size-1
            assert validate_K(graph_type, K, trigger_size)
        if args.graph_type=='ER' and args.ER_graph_P==0:
            prob=1
        elif args.graph_type=='SW' and args.SW_graph_P==0:
            prob=1
        these_attack_specs['K']=K
        these_attack_specs['prob']=prob
        print_attack_description(these_classifier_hyperparams, these_attack_specs, model_hyp_set)


        ''''''''''''''''''''''''''''''
        '''     Poison Dataset     '''
        ''''''''''''''''''''''''''''''
        dataset_dict_backdoor = retrieve_data_process(args.regenerate_data, False, dataset, these_attack_specs, seed=args.seed)


        ''''''''''''''''''''''''''''''
        '''       Attack GNN       '''
        ''''''''''''''''''''''''''''''
        model_path = get_model_path(dataset, these_classifier_hyperparams, these_attack_specs, these_model_specs['model_hyp_set'])
        class_weights   = get_class_weights(dataset_dict_backdoor, attack_target_label, 'backdoor', num_classes) if these_classifier_hyperparams['balanced']==True else None
        dataloader_dict = get_dataloader_dict(dataset_dict_backdoor, dataset_dict_clean, these_model_specs, these_classifier_hyperparams)
        _, _ = train_loop_backdoor(dataset,            dataloader_dict,                class_weights,  model_path,     these_attack_specs, 
                                these_model_specs,  these_classifier_hyperparams,   args.plot,           verbose=True)


    elif args.backdoor_type=='adaptive':
            
        ''''''''''''''''''''''''''''''
        '''        Load Data       '''
        ''''''''''''''''''''''''''''''
        os.makedirs( os.path.join(adapt_gen_dir,dataset), exist_ok=True)
        dataset_name = dataset
        max_degree_dict = {'MUTAG':5, 'AIDS':7, 'PROTEINS': 26, 'IMDB-BINARY': 136, 'COLLAB': 49, 'REDDIT-BINARY': 3063, 'DBLP': 36}
        max_degree = max_degree_dict[dataset_name]
        if dataset != 'DBLP':
            transform = Compose([DeduplicateEdges(), OneHotDegree(max_degree=max_degree-1,cat=False)])
            tu_dataset = torch_geometric.datasets.TUDataset(root=os.path.join(data_dir,'clean'), name=dataset_name, transform=transform)
        else:
            ''' need to apply de-duplicate edegs to DBLP data!'''
            with open(f'{data_dir}/clean/DBLP_data_list_random_5000.pkl', 'rb') as f:
                tu_dataset = pickle.load(f)
        graphs, tag2index = load_data(dataset, data_type='pyg', cleaned=True, use_edge_attr=False)
        labels = [graph.pyg_graph.y for graph in graphs]
        train_idx, test_idx = get_train_test_idx(labels)
        train_dataset = [tu_dataset[idx] for idx in train_idx]
        test_dataset  = [tu_dataset[idx] for idx in test_idx]
        '''uniform sampling for backdoored samples'''
        num_train_to_attack = int(args.poison_rate  * len(train_dataset))
        random.seed(args.seed)
        train_backdoor_indices = random.sample(range(len(train_idx)), k=num_train_to_attack)
        train_backdoor_indices = [idx for idx in train_backdoor_indices if len(get_possible_new_edges(train_dataset[idx])) > trigger_size and train_dataset[idx].x.shape[0] < 500]
        test_backdoor_indices  = [idx for idx in range(len(test_idx))   if len(get_possible_new_edges(test_dataset[idx]))  > trigger_size and test_dataset[idx].x.shape[0]  < 500]


        ''''''''''''''''''''''''''''''
        '''     Load Generator     '''
        ''''''''''''''''''''''''''''''
        generator_name_dict = {'regular': EdgeGenerator, 'heavy': EdgeGeneratorHeavy}
        # generator_hyperparam_dicts= generator_hyperparam_dicts_iterative_exp if args.gen_alg_v==3 else None
        #if args.gen_alg_v!=3:
        #generator_checkpoint_path = os.path.join(adapt_gen_dir, dataset_name, f"Trigger_Generator_{dataset_name}_target_label_{attack_target_label}.ckpt")
        #else:
        generator_checkpoint_path = os.path.join(adapt_gen_dir, dataset_name, f"Trigger_Generator_{dataset_name}_target_label_{attack_target_label}_{args.contin_or_scratch}_final.ckpt")
        generator_class_name, hidden_dim, depth, dropout_prob = unpack_kwargs(generator_hyperparam_dicts[dataset][attack_target_label], ['generator_class', 'hidden_dim', 'depth', 'dropout_prob'])
        generator_class = generator_name_dict[generator_class_name]
        try:
            trigger_generator = generator_class(num_node_features, hidden_dim=hidden_dim,  depth=depth, dropout_prob=dropout_prob)
        except:
            trigger_generator = generator_class(num_node_features)
        try:
            assert os.path.isfile(generator_checkpoint_path)
            print(f"Found pretrained edge generator {attack_target_label}, loading...")
            trigger_generator.load_state_dict(torch.load(generator_checkpoint_path))
        except:
            print(f'No generator found at path: {generator_checkpoint_path}.')

        ''''''''''''''''''''''''''''''
        '''      Poison Dataset    '''
        ''''''''''''''''''''''''''''''
        if args.regenerate_data==True:
            print('regenerate data')
            dataset_dict_adaptive = build_dataset_dict_adaptive(trigger_generator, dataset_dict_clean, train_backdoor_indices, test_backdoor_indices, trigger_size, attack_target_label)
            clean_labels = []
            for i, g in enumerate(dataset_dict_adaptive['train_backdoor_graphs']):
                g = dataset_dict_adaptive['train_backdoor_graphs'][i]
                if g.pyg_graph.is_backdoored:
                    g_clean = dataset_dict_clean['train_clean_graphs'][i]
                    clean_labels.append(g_clean.pyg_graph.y)#.item())
            if len(set(clean_labels)) == 1:
                print("Largest trigger size is too big for dataset -- try limiting to attacks with smaller triggers.")
                # break
            else:
                gen_dataset_folder_ext = f'_{args.contin_or_scratch}'
                dataset_path = get_dataset_path(dataset, these_attack_specs, clean=False, gen_dataset_folder_ext=gen_dataset_folder_ext)
                create_nested_folder(dataset_path)
                with open(dataset_path,'wb') as f:
                    pickle.dump(dataset_dict_adaptive,f)
        else:
            print('regenerate data')
            dataset_path = get_dataset_path(dataset, these_attack_specs, clean=False)
            with open(dataset_path,'rb') as f:
                dataset_dict_adaptive = pickle.load(f)

        ''''''''''''''''''''''''''''''
        '''        Attack GNN      '''
        ''''''''''''''''''''''''''''''
        model_path      = get_model_path(dataset, these_classifier_hyperparams, these_attack_specs, these_model_specs['model_hyp_set'])
        class_weights   = get_class_weights(dataset_dict_adaptive, attack_target_label, 'backdoor', num_classes) if these_classifier_hyperparams['balanced']==True else None
        dataloader_dict = get_dataloader_dict(dataset_dict_adaptive, dataset_dict_clean, these_model_specs, these_classifier_hyperparams)
        
        _, _ = train_loop_backdoor(dataset,             dataloader_dict,                class_weights,   model_path,     these_attack_specs, 
                                   these_model_specs,   these_classifier_hyperparams,   args.plot,       verbose=True)


if __name__ == '__main__':
    main()


