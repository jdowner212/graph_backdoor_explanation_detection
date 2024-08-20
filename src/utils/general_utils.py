from   config import *
import numpy as np
import os
from   sklearn.model_selection import StratifiedKFold
import torch

def soft_eq(a: torch.Tensor,b: torch.Tensor):
    positive_difference = torch.sqrt(torch.sum((a-b)**2))
    soft_equality = 1 - positive_difference
    return soft_equality

def get_random_indices(length):
    return np.random.permutation(length).tolist()


def get_train_test_idx(labels, seed=2575):
    fold_idx=0
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]
    return train_idx, test_idx


def separate_data(graph_list, seed=2575):
    labels = [graph.pyg_graph.y for graph in graph_list]
    train_idx, test_idx = get_train_test_idx(labels, seed=seed)
    train_graph_list=[]
    for i, idx in enumerate(train_idx):
        graph = graph_list[idx]
        graph.pyg_graph.original_index=i
        graph.pyg_graph.triggered_edges=[]
        train_graph_list.append(graph)
    test_graph_list=[]
    for i, idx in enumerate(test_idx):
        graph = graph_list[idx]
        graph.pyg_graph.original_index=i
        graph.pyg_graph.triggered_edges=[]
        test_graph_list.append(graph)
    return train_graph_list, test_graph_list


def count_class_samples(graph_list, num_classes):
    label_counts = []
    for c in range(num_classes):
        c_idxs = [i for i in range(len(graph_list)) if graph_list[i].pyg_graph.y == c]
        label_counts.append(len(c_idxs))
    return label_counts


def universal_bins(lists_of_values, num_bins=20):
    mins = [min(value_list) for value_list in lists_of_values]
    maxs = [max(value_list) for value_list in lists_of_values]
    smallest_min = min(mins)
    largest_max = max(maxs)
    bins = np.linspace(start=smallest_min, stop=largest_max, num=num_bins)
    return bins


def square_edge_mask(x, edge_mask_dict):
    edge_index_matrix = torch.zeros((len(x), len(x)), dtype=torch.float)
    for k,v in edge_mask_dict.items():
        try:
            row, col = k[0].item(), k[1].item()
        except:
            row, col = k[0], k[1]
        edge_index_matrix[row, col] = v
    return edge_index_matrix


def min_max_scaling(tensor, small_difference):
    min_ = torch.min(tensor)
    max_ = torch.max(tensor)
    scaled = None
    if max_ - min_ < small_difference:
        scaled = tensor - min_ + torch.tensor([1-small_difference])
    else:
        scaled = (tensor-min_)/((max_-min_)+1e-15)
    return scaled


def deepcopy_dict_with_tensors(original_dict):
    new_dict = {}
    for key, value in original_dict.items():
        if isinstance(value, dict):
            new_dict[key] = deepcopy_dict_with_tensors(value)
        elif torch.is_tensor(value):
            new_dict[key] = value.clone()
        else:
            new_dict[key] = value
    return new_dict


def replace_none_nan_with_average(values):
    assert len(values) > 0
    values = [np.nan if val is None else val for val in values]
    np_values = np.array(values)
    avg = np.nanmean(np_values)
    np_values[np.isnan(np_values)] = avg
    return np_values


def unpack_kwargs(kwargs, keys):
    return [kwargs[k] for k in keys]


def create_nested_folder(path):
    path_parts = path.split('/')
    current_path = '/'
    for path_part in path_parts:
        if path_part != '' and path_part[-4:] not in ['.png','.txt','.pth','.pkl']:
            current_path = os.path.join(current_path,path_part)
            if os.path.exists(current_path)==False:
                os.mkdir(current_path)
    return


def get_model_name(classifier_hyperparams, attack_specs, model_hyp_set):
    model_type, balanced, epochs = unpack_kwargs(classifier_hyperparams, ['model_type','balanced','epochs'])
    attack_target_label, trigger_size, poison_rate = unpack_kwargs(attack_specs, ['attack_target_label','trigger_size','poison_rate'])
    poison_rate_round = str(int(poison_rate)) if poison_rate > 0.99 else poison_rate
    backdoor_type = attack_specs['backdoor_type']
    if backdoor_type == 'random' or backdoor_type == 'clean_label':
        graph_type = attack_specs['graph_type']
        K_str, prob_str = get_K_str_prob_str(attack_specs)
        model_name = f'{model_type}_{backdoor_type}_{graph_type}_attack_target_{int(attack_target_label)}_trigger_size_{trigger_size}_poison_rate_{poison_rate_round}{prob_str}{K_str}_model_hyp_set_{model_hyp_set}_balanced_{balanced}_epochs_{epochs}'
    elif backdoor_type == 'adaptive':
        model_name = f'{model_type}_{backdoor_type}_target_{int(attack_target_label)}_trigger_size_{trigger_size}_poison_rate_{poison_rate_round}_model_hyp_set_{model_hyp_set}_balanced_{balanced}_epochs_{epochs}'
    return model_name

def print_attack_description(classifier_hyperparams, attack_specs, model_hyp_set):
    model_type, balanced, epochs = unpack_kwargs(classifier_hyperparams, ['model_type','balanced','epochs'])
    attack_target_label, trigger_size, poison_rate = unpack_kwargs(attack_specs, ['attack_target_label','trigger_size','poison_rate'])
    K_description = get_K_description(attack_specs)
    prob_description = get_prob_description(attack_specs)
    poison_rate_round = str(int(poison_rate)) if poison_rate > 0.99 else poison_rate
    backdoor_type = attack_specs['backdoor_type']
    if backdoor_type == 'random' or backdoor_type=='clean_label':
        graph_type = attack_specs['graph_type']
        if backdoor_type == 'random':
            attack_description = f'Model Type: {model_type}\nEpochs: {epochs}\nAttack Type: Random\nTrigger Graph Type: {graph_type}\nTrigger Size: {trigger_size}\nPoison Rate: {poison_rate_round}\nAttack Target Label: {int(attack_target_label)}{prob_description}{K_description}\nClass balance applied: {balanced}\nModel Hyperparameter Set: {model_hyp_set}'
        if backdoor_type == 'clean_label':
            attack_description = f'Model Type: {model_type}\nEpochs: {epochs}\nAttack Type: Clean Label\nTrigger Graph Type: {graph_type}\nTrigger Size: {trigger_size}\nPoison Rate: {poison_rate_round}\nAttack Target Label: {int(attack_target_label)}{prob_description}{K_description}\nClass balance applied: {balanced}\nModel Hyperparameter Set: {model_hyp_set}'
    elif backdoor_type == 'adaptive':
        attack_description = f'Model Type: {model_type}\nEpochs: {epochs}\nAttack Type: Adaptive\nTrigger Size: {trigger_size}\nPoison Rate: {poison_rate_round}\nAttack Target Label: {int(attack_target_label)}\nClass balance applied: {balanced}\nModel Hyperparameter Set: {model_hyp_set}'
    print(attack_description)


def get_model_path(dataset, classifier_hyperparams, attack_specs, model_hyp_set):
    model_name = get_model_name(classifier_hyperparams, attack_specs, model_hyp_set)
    model_path = f'{train_dir}/{dataset}/models/{model_name}.pth'
    if os.path.exists(model_path)==False:
        model_name_parts = model_name.split('_epochs')
        model_name = ''.join(model_name_parts[:-1])
    model_path = f'{train_dir}/{dataset}/models/{model_name}.pth'
    return model_path


def get_explanation_name(model_specs, attack_specs, explainer_hyperparams, classifier_hyperparams):
    model_hyp_set = model_specs['model_hyp_set']
    backdoor_type, graph_type, attack_target_label, poison_rate, trigger_size = unpack_kwargs(attack_specs,['backdoor_type','graph_type','attack_target_label','poison_rate','trigger_size'])
    explainer_target_label,explanation_type             = unpack_kwargs(explainer_hyperparams, ['explainer_target_label','explanation_type'])
    model_type, balanced                                = unpack_kwargs(classifier_hyperparams, ['model_type','balanced'])
    poison_rate_round = str(int(poison_rate)) if poison_rate > 0.99 else poison_rate
    if backdoor_type=='random' or backdoor_type=='clean_label':
        K_str, prob_str = get_K_str_prob_str(attack_specs)
        explanation_name  = f'{explanation_type}_explain_target_{explainer_target_label}_{model_type}_{backdoor_type}_{graph_type}_attack_target_{attack_target_label}_trigger_size_{trigger_size}_poison_rate_{poison_rate_round}{prob_str}{K_str}_model_hyp_set_{model_hyp_set}_balanced_{balanced}'
    elif backdoor_type=='adaptive':
        explanation_name  = f'{explanation_type}_explain_target_{explainer_target_label}_{model_type}_{backdoor_type}_attack_target_{attack_target_label}_trigger_size_{trigger_size}_poison_rate_{poison_rate_round}_model_hyp_set_{model_hyp_set}_balanced_{balanced}'
    return explanation_name


def get_explanation_path(dataset, category, original_index, model_specs, attack_specs, explainer_hyperparams, classifier_hyperparams):
    explanation_root = f'{explain_dir}/{dataset}/explanations/{category}/original_index_{int(original_index)}'
    explanation_name = get_explanation_name(model_specs, attack_specs, explainer_hyperparams, classifier_hyperparams)
    full_explanation_path = f'{explanation_root}/{explanation_name}.pkl'
    return full_explanation_path


def get_explainer_path(dataset, category, original_index, model_specs, attack_specs, explainer_hyperparams, classifier_hyperparams):
    explanation_root = f'{explain_dir}/{dataset}/explainers/{category}/original_index_{int(original_index)}'
    explanation_name = get_explanation_name(model_specs, attack_specs, explainer_hyperparams, classifier_hyperparams)
    full_explanation_path = f'{explanation_root}/{explanation_name}.pkl'
    return full_explanation_path


def get_dataset_subfolder(attack_specs):
    backdoor_type, graph_type, trigger_size, poison_rate = unpack_kwargs(attack_specs,['backdoor_type','graph_type','trigger_size','poison_rate'])
    K_str, prob_str = get_K_str_prob_str(attack_specs)
    dataset_subfolder = f'trigger_size_{trigger_size}_poison_rate_{poison_rate}{prob_str}{K_str}'
    prefices = []
    prefices.append(backdoor_type)
    if backdoor_type != 'adaptive' and graph_type is not None:
        prefices.append(graph_type)
    dataset_subfolder = '_'.join(prefices) + '_' + dataset_subfolder
    return dataset_subfolder


def get_dataset_path(dataset, attack_specs=None, clean=False, gen_dataset_folder_ext=''):
    if clean==True:
        dataset_path = f'{data_dir}/clean/{dataset}/data_dict_{dataset}.pth'
    else:
        assert attack_specs is not None
        dataset_subfolder = get_dataset_subfolder(attack_specs)
        dataset_subfolder += gen_dataset_folder_ext
        dataset_path = f'{data_dir}/poisoned/{dataset}/{dataset_subfolder}/target_label_{attack_specs["attack_target_label"]}.pth'
    return dataset_path


def validate_K(graph_type, K, trigger_size):
    cond_1 = graph_type == 'PA' and K > trigger_size
    cond_2 = graph_type == 'SW' and K<2
    if graph_type=='ER':
        # K doesn't apply for ER graphs
        return True
    elif cond_1 or cond_2:
        if cond_1:
            print('Invalid value for flag --PA_graph_K: for PA graphs, K must be less than or equal to trigger size.')
        if cond_2:
            print('Invalid value for flag --SW_graph_K: for SW graphs, K must be greater than 2.')
        return False
    else:
        return True


def get_K_str_prob_str(attack_specs):
    graph_type, trigger_size, K, prob = unpack_kwargs(attack_specs, ['graph_type','trigger_size','K','prob'])
    if attack_specs['backdoor_type']=='adaptive':
        K_str, prob_str = '',''
        return K_str, prob_str
    else:
        if validate_K(graph_type, K, trigger_size):
            K_str = '' if graph_type == 'ER' else f'_K_{str(int(K))}'
            prob_str = str(int(prob)) if prob is not None and float(prob)%1==0 else str(prob)
            prob_str = '' if graph_type == 'PA' else f'_prob_{prob_str}'
        return K_str, prob_str


def get_K_description(attack_specs):
    graph_type, trigger_size, K = unpack_kwargs(attack_specs, ['graph_type','trigger_size','K'])
    print('graph_type:',graph_type)
    print('K:',K)
    if attack_specs['backdoor_type']=='adaptive':
        return ''
    else:
        if validate_K(graph_type, K, trigger_size):
            K_description = '' if graph_type == 'ER' or K==None else f'\nK: {str(int(K))}'
        return K_description

def get_prob_description(attack_specs):
    graph_type, prob = unpack_kwargs(attack_specs, ['graph_type','prob'])
    if attack_specs['backdoor_type']=='adaptive':
        return ''
    else:
        prob_str = str(int(prob)) if prob is not None and float(prob)%1==0 else str(prob)
        prob_description = '' if graph_type == 'PA' else f'\nProb: {prob_str}'
        return prob_description

def update_kwargs(kwargs,keys,values):
    for (k,v) in zip(keys,values):
        kwargs[k] = v
    return kwargs


def build_explainer_hyperparams():
    explainer_hyperparams = {'hyp_dict':                None,
                            'explainer_target_label':    None,
                            'degree_as_tag':           True,
                            'coeffs':                  {'edge_reduction':       'sum',      
                                                        'node_feat_reduction':  'sum',
                                                        'node_feat_size':       1,       
                                                        'node_feat_ent':        0.001,           
                                                        'edge_size':            1,                   
                                                        'edge_ent':             0.001,    
                                                        'EPS':                  1e-15},      
                            'explainer_epochs':        50,
                            'explain_lr':              0.1,
                            'apply_sigmoid':           True,
                            'node_mask_type':          'attributes',
                            'edge_mask_type':          'object',
                            'return_type':             'raw',
                            'explanation_type':        'model',
                            'threshold_config':        {'threshold_type':      'hard',
                                                        'value':                0.3},
                            'disconnect_params':       {},
                            'min_max_scale':           True,
                            'use_common_node_mask':    False,
                            'remove_edge_disconnects': False,
                            'remove_node_disconnects': False,
                            'relative_vmin_vmax':      True}
    return explainer_hyperparams


def build_attack_specs():
    attack_specs    =  {'attack_target_label':     None,
                        'graph_type':       None,
                        'trigger_size':     None,
                        'prob':             None,
                        'poison_rate':      None,
                        'K':                None,
                        'backdoor_type':    None}
    return attack_specs


def build_model_specs():
    model_specs     =  {'hyp_dict':         None,
                        'model_hyp_set':    None,
                        'clean_or_backdoor': None}
    return model_specs



def parse_metric_image_path(dataset, these_attack_specs, these_classifier_hyperparams, these_explainer_hyperparams, these_model_specs):
    explanation_name = get_explanation_name(these_model_specs, these_attack_specs, these_explainer_hyperparams, these_classifier_hyperparams)
    raw_image_path = f'{explain_dir}/{dataset}/metrics_images/{explanation_name}_raw.png'
    dist_image_path = f'{explain_dir}/{dataset}/metrics_images/{explanation_name}_dist.png'
    boxplot_image_path = f'{explain_dir}/{dataset}/metrics_images/boxplot_{explanation_name}.png'
    return raw_image_path, dist_image_path, boxplot_image_path
