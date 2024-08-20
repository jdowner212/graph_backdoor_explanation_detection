
import argparse
import os
import sys

current_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
root_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir,'utils'))
sys.path.append(os.path.join(current_dir,'attack'))
sys.path.append(os.path.join(current_dir,'explain'))
sys.path.append(os.path.join(current_dir,'detect'))

from   attack.backdoor_utils import *
import pickle
import random
import torch
import torch_geometric
from   torch_geometric.transforms import Compose, OneHotDegree

data_shape_dict   = get_info('data_shape_dict')
data_dir    = get_info('data_dir')
adapt_gen_dir = get_info('adapt_gen_dir')
adapt_surrogate_models = get_info('adapt_surrogate_models')
generator_hyperparam_dicts = get_info('generator_hyperparam_dicts')
surrogate_hyperparams_initial = get_info('surrogate_hyperparams_initial')
surrogate_hyperparams_looping = get_info('surrogate_hyperparams_looping')



def parse_args():
    parser=argparse.ArgumentParser(description="Adaptive generator training: input arguments")
    parser.add_argument('--attack_target_label',        type=int,               default=0,              help='Class targeted by backdoor attack.')
    parser.add_argument('--dataset',                    type=str,               default='MUTAG',        help='Dataset to attack and explain.')
    parser.add_argument('--poison_rate',                type=float,             default=0.2,            help='Poison rate, expressed as a portion of training data size.')
    parser.add_argument('--gen_rounds',                 type=int,               default=3,              help='Number of iterations to train adaptive trigger generator.')
    parser.add_argument('--load_or_train_surrogate_GNN',type=str,               default='train',        help='Valid values: "load","train"')
    parser.add_argument('--seed',                       type=int,               default=2575,           help='Makes randomness constant.')
    parser.add_argument('--trigger_size',               type=int,               default=6,              help='Subgraph trigger size.')

    args=parser.parse_args()
    return args


def main():
    args=parse_args()
    dataset                = args.dataset
    attack_target_label    = args.attack_target_label
    trigger_size           = args.trigger_size
    num_classes            = data_shape_dict[dataset]['num_classes']
    num_node_features      = data_shape_dict[dataset]['num_node_features']

    these_attack_specs = build_attack_specs()
    these_attack_specs['attack_target_label']=args.attack_target_label
    these_attack_specs['backdoor_type']='adaptive'
    these_attack_specs['graph_type']=None
    these_attack_specs['trigger_size']=args.trigger_size
    these_attack_specs['poison_rate']=args.poison_rate

    ''''''''''''''''''''''''''''''
    '''        Load Data       '''
    ''''''''''''''''''''''''''''''
    os.makedirs( os.path.join(adapt_gen_dir,dataset), exist_ok=True)

    dataset_name = dataset
    dataset_dict_clean = get_clean_data(dataset, seed=args.seed, verbose=True, clean_pyg_process=True, use_edge_attr=False)
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
    data_labels = [tu_dataset[idx].y.item() for idx in range(len(tu_dataset))]
    '''uniform sampling for backdoored samples'''
    num_train_to_attack = int(args.poison_rate  * len(train_dataset))
    random.seed(args.seed)
    train_backdoor_indices = random.sample(range(len(train_idx)), k=num_train_to_attack)
    train_backdoor_indices = [idx for idx in train_backdoor_indices if len(get_possible_new_edges(train_dataset[idx])) > trigger_size and train_dataset[idx].x.shape[0] < 500]
    test_backdoor_indices  = [idx for idx in range(len(test_idx))   if len(get_possible_new_edges(test_dataset[idx]))  > trigger_size and test_dataset[idx].x.shape[0]  < 500]

    ''''''''''''''''''''''''''''''''''''
    '''  Load/Train Surrogate Model  '''
    ''''''''''''''''''''''''''''''''''''
    surrogate_kwargs_looping = surrogate_hyperparams_looping[dataset_name]
    surrogate_model_name   = f'GNN_{dataset_name}'
    surrogate_filename     = os.path.join(adapt_surrogate_models, dataset_name, f"GraphLevel_{surrogate_model_name}.ckpt")
    if args.load_or_train_surrogate_GNN == 'load' and os.path.isfile(surrogate_filename):
        print("Found pretrained surrogate GNN, loading...")
        num_node_features = max_degree
        num_classes = len(set(data_labels))
        surrogate_model = GraphLevelGNN_opt(c_in=num_node_features, c_out=num_classes, **surrogate_hyperparams_initial[dataset_name])
        surrogate_model.load_state_dict(torch.load(surrogate_filename))
    else:
        print("Training surrogate GNN...")
        retrain_surrogate = True if args.load_or_train_surrogate_GNN == 'train' else False
        surrogate_model = train_surrogate(dataset_name, train_dataset, test_dataset, surrogate_filename, retrain=retrain_surrogate, save=True, seed=args.seed, **surrogate_kwargs_looping)

    ''''''''''''''''''''''''''''''
    '''  Load/Train Generator  '''
    ''''''''''''''''''''''''''''''
    generator_name_dict = {'regular': EdgeGenerator, 'heavy': EdgeGeneratorHeavy}
    generator_class_name, epochs, T, lr_Ma, lr_gen, weight_decay, hidden_dim, depth, dropout_prob, batch_size, max_num_edges = unpack_kwargs(generator_hyperparam_dicts[dataset][attack_target_label], ['generator_class', 'epochs', 'T', 'lr_Ma', 'lr_gen', 'weight_decay', 'hidden_dim', 'depth', 'dropout_prob', 'batch_size', 'max_num_edges'])
    generator_class = generator_name_dict[generator_class_name]
    print('Training generator...')
    generator_kwargs = {'generator_class':generator_class,'T':T,'lr_Ma':lr_Ma, 'lr_gen':lr_gen, 'weight_decay':weight_decay,'hidden_dim':hidden_dim,'depth':depth,'dropout_prob':dropout_prob,'batch_size':batch_size,'max_num_edges':max_num_edges, 'epochs':epochs}
    generator_path   = os.path.join(adapt_gen_dir, dataset_name, f"Trigger_Generator_{dataset_name}_target_label_{attack_target_label}")
    trigger_generator = train_generator_iterative_loop(surrogate_model, 
                                      dataset_name,
                                      train_dataset,
                                      test_dataset,
                                      train_backdoor_indices,
                                      test_backdoor_indices,
                                      trigger_size,
                                      generator_path,
                                      attack_target_label,
                                      args.gen_rounds,
                                      args.seed,
                                      surrogate_kwargs_looping,
                                      generator_kwargs)
    

    ''''''''''''''''''''''''''''''''''''''''''
    '''   Use Generator to Poison Dataset  '''
    ''''''''''''''''''''''''''''''''''''''''''

    dataset_dict_adaptive = poison_data_adaptive_attack(trigger_generator, dataset_dict_clean, train_backdoor_indices, test_backdoor_indices, trigger_size, attack_target_label)
    clean_labels = []
    for i, g in enumerate(dataset_dict_adaptive['train_backdoor_graphs']):
        g = dataset_dict_adaptive['train_backdoor_graphs'][i]
        if g.pyg_graph.is_backdoored:
            g_clean = dataset_dict_clean['train_clean_graphs'][i]
            clean_labels.append(g_clean.pyg_graph.y)#.item())
    if len(set(clean_labels)) == 1:
        print("Largest trigger size is too big for dataset -- try limiting to attacks with smaller triggers.")
    else:
        dataset_path = get_dataset_path(dataset, these_attack_specs, clean=False, gen_dataset_folder_ext='')
        create_nested_folder(dataset_path)
        with open(dataset_path,'wb') as f:
                    pickle.dump(dataset_dict_adaptive,f)

if __name__ == '__main__':
    main()


