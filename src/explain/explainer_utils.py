'''
Functions for training explainer algorithms and returning explanations. In particular, GNNExplainer is important here, as it is used in our main experiments.
See "GNN Explanation" on page 3 of our paper for more details on this process.

Relevant funtions:

-- run_explain()
    Function for training explainer (by default, GNNExplainer) and obtaining explanation on data input.
    GNNExplainer algorithm described on page 3 in our paper, and used in detection metrics and experiments throughout. 

'''

from config import *
from attack.backdoor_utils import *
from utils.data_utils import *
from utils.models import *
from utils.plot_utils import *
from   utils.general_utils import *
import os
import numpy as np
import torch
import pickle
import sys
sys.path.append(os.getcwd())
from   torch_geometric.data import Data
from   torch_geometric.explain import Explainer, GNNExplainer, PGExplainer, CaptumExplainer
from   torch_geometric.explain.metric.faithfulness import unfaithfulness
from   torch_geometric.explain.metric.fidelity     import characterization_score
from   torch_geometric.explain.algorithm.gnn_explainer import *
from   torch_geometric.explain.algorithm.pg_explainer import *
from   torch_geometric.explain.config import ExplanationType

EPS = 1e-15
data_shape_dict = get_info('data_shape_dict')
src_dir     = get_info('src_dir')
data_dir    = get_info('data_dir')
explain_dir = get_info('explain_dir')
train_dir   = get_info('train_dir')
train_dir_cln = get_info('train_dir_cln')


def run_explain(dataset,        data_input,     subset_name,            original_index,         category,
                attack_specs,   model_specs,    explainer_hyperparams,  classifier_hyperparams, which_explainer = 'gnn', 
                model = None,   save=True):
    ''' 
    Function for training explainer (by default, GNNExplainer) and obtaining explanation on data input.
    Explainer algorithms referenced from torch_geometric (rather than implemented directly).
    '''
    ''' Setup '''
    explanation_type = explainer_hyperparams['explanation_type']
    graph_type = unpack_kwargs(attack_specs,['graph_type'])
    if 'backdoor' in category and attack_specs['backdoor_type']=='random':
        assert graph_type is not None
    if isinstance(data_input, dict):
        assert subset_name is not None, "If data_input provided is a dictionary, need to also provide subset name."
        potential_data = [graph for graph in data_input[subset_name] if graph.pyg_graph.original_index == original_index][0]
        if isinstance(potential_data, GraphObject):
            this_data = potential_data.pyg_graph
    elif isinstance(data_input, Data):
        this_data = data_input
    if explanation_type=='phenomenon':
        explainer_target_label = this_data.y
    elif explanation_type=='model':
        explainer_target_label = None
    mode = 'binary' if data_shape_dict[dataset]['num_classes']==2 else 'multiclass'
    explainer_epochs, explain_lr, coeffs, apply_sigmoid, edge_mask_type, return_type, threshold_config = unpack_kwargs(explainer_hyperparams,['explainer_epochs','explain_lr','coeffs','apply_sigmoid','edge_mask_type','return_type','threshold_config'])
    algorithm, explainer = None, None
    ''' By default, which_explainer is set to 'gnn' (representing GNNExplainer) '''
    if which_explainer=='gnn':
        algorithm_ = GNNExplainer
        node_mask_type = explainer_hyperparams['node_mask_type']
    elif which_explainer=='pg' or which_explainer=='pge':
        algorithm_ = PGExplainer
        node_mask_type = None
    elif which_explainer=='captum':
        algorithm_ = CaptumExplainer
        node_mask_type = explainer_hyperparams['node_mask_type']
    if which_explainer=='captum':
        SUPPORTED_METHODS = ['IntegratedGradients','Saliency','InputXGradient','Deconvolution','ShapleyValueSampling','GuidedBackprop',]
        algorithm = algorithm_(attribution_method = 'Deconvolution')
    else:
        algorithm = algorithm_(epochs = explainer_epochs, lr = explain_lr, coeffs_dict = coeffs)
    ''' Train explainer '''
    explainer = Explainer(model            = model,
                          algorithm        = algorithm,
                          save_masks       = True,
                          apply_sigmoid    = apply_sigmoid,
                          explanation_type = explanation_type,
                          node_mask_type   = node_mask_type,
                          edge_mask_type   = edge_mask_type,
                          model_config     = dict(mode=f'multiclass_classification',
                                                  task_level='graph',
                                                  return_type=return_type),
                          threshold_config = threshold_config)
    explainer_target_label = None if explanation_type=='model' else explainer_target_label
    if explainer_target_label is not None and mode == 'binary':
        explainer_target_label_int = int(copy.copy(explainer_target_label))
        explainer_target_label = torch.tensor([[0.0,0.0]])
        explainer_target_label[0][explainer_target_label_int]=1.0
    if which_explainer=='captum':
        explainer_target_label = torch.tensor([explainer_target_label_int])  # One-dimensional tensor
    ''' Obtain explanation '''
    explanation = explainer(x=this_data.x, edge_index=this_data.edge_index, target=explainer_target_label)
    assert explainer is not None
    assert explanation is not None
    explainer_path   = get_explainer_path(dataset, category, original_index, model_specs, attack_specs, explainer_hyperparams, classifier_hyperparams)
    explanation_path = get_explanation_path(dataset, category, original_index, model_specs, attack_specs, explainer_hyperparams, classifier_hyperparams)
    if save==True:
        create_nested_folder(explainer_path)
        create_nested_folder(explanation_path)
        with open(explainer_path, 'wb') as f:
            pickle.dump(explainer,f)
        with open(explanation_path, 'wb') as f:
            pickle.dump(explanation,f)
    return explainer, explanation


def explanation_dict():
    d = {'explainer': None,
         'explanation': None,
         'node_mask_over_time': None,
         'edge_mask_over_time': None,
         'node_grad_over_time': None,
         'edge_grad_over_time': None,
         'pos_fids': [],
         'neg_fids': [],
         'node_sparsities': [],
         'edge_sparsities': [],
         'nodes_with_edges': [],
         'char_scores': [],
         'unfaiths': [],
         'edge_mask_dict':[],
         'node_mask_dict':[],}
    return d


def update_explanation_dict(d, explainer, explanation):
    explanation = explanation.clone().detach()
    d['explainer'] = explainer
    d['explanation'] = explanation
    try:
        d['edge_mask_over_time'] = [m.clone().detach() for  m in explanation.edge_mask_over_time]
        d['edge_grad_over_time'] = [g.clone().detach() for  g in explanation.edge_grads_over_time]
    except:
        pass
    try:
        d['node_mask_over_time'] = [m.clone().detach() for  m in explanation.node_mask_over_time]
        d['node_grad_over_time'] = [g.clone().detach() for  g in explanation.node_grads_over_time]
    except:
        pass
    fid = my_fidelity(explanation,explainer)
    unfaith = unfaithfulness(explainer,explanation,verbose=False)
    char_score = characterization_score(fid[0]+1e-15,fid[1]+1e-15)
    try:
        pos_fid = np.round(fid[0].item(),4)
        neg_fid = np.round(fid[1].item(),4)
    except:
        pos_fid = np.round(fid[0],4)
        neg_fid = np.round(fid[1],4)
    d['pos_fids'].append(pos_fid)
    d['neg_fids'].append(neg_fid)
    d['char_scores'].append(int(np.round(char_score)))
    d['unfaiths'].append(unfaith)
    return d

def my_fidelity(explanation: Explanation, explainer: Explainer, num_classes = 2) -> Tuple[float, float]:
    if explainer.model_config.mode == ModelMode.regression:
        raise ValueError("Fidelity not defined for 'regression' models")
    node_mask = explanation.get('node_mask')
    edge_mask = explanation.get('edge_mask')
    kwargs = {key: explanation[key] for key in explanation._model_args}
    ''' vectorized '''
    y = torch.tensor([0]*num_classes)
    target_label = torch.argmax(explanation.target[0])
    y[target_label] = 1
    if explainer.explanation_type == ExplanationType.phenomenon:
        ''' vectorized '''
        y_hat = explainer.get_prediction(explanation.x, explanation.edge_index, **kwargs)
        y_hat = F.softmax(y_hat,dim=1)[0]
    ''' vectorized '''
    explain_y_hat = explainer.get_masked_prediction(explanation.x, explanation.edge_index, node_mask, edge_mask, **kwargs)
    explain_y_hat = F.softmax(explain_y_hat, dim=1)[0]
    ''' vectorized '''
    complement_y_hat = explainer.get_masked_prediction(explanation.x,   explanation.edge_index, 1. - node_mask, 1. - edge_mask, **kwargs)
    complement_y_hat = F.softmax(complement_y_hat, dim=1)[0]
    if explanation.get('index') is not None:
        y = y[explanation.index]
        if explainer.explanation_type == ExplanationType.phenomenon:
            y_hat = y_hat[explanation.index]
        explain_y_hat = explain_y_hat[explanation.index]
        complement_y_hat = complement_y_hat[explanation.index]
    if explainer.explanation_type == ExplanationType.model:
        pos_fidelity = 1 - soft_eq(complement_y_hat, y)
        neg_fidelity = 1 - soft_eq(explain_y_hat, y)
    else:
        pos_fidelity = (soft_eq(y_hat, y) - soft_eq(complement_y_hat,y)).float().abs()
        neg_fidelity = (soft_eq(y_hat, y) - soft_eq(explain_y_hat, y)).float().abs()
    return float(pos_fidelity), float(neg_fidelity)


def get_all_success_indices_plus_asr(model, dataset_dict_backdoor, dataset_dict_clean, min_to_use = 30, seed=2575):
    bd_data_list = [data.pyg_graph for data in dataset_dict_backdoor['train_backdoor_graphs']]
    [cln_data_list, cln_val_data_list] = [[data.pyg_graph for data in dataset_dict_clean[category]] for category in ['train_clean_graphs','test_clean_graphs']]
    [backdoor_train_data, clean_train_data, clean_test_data] = [Batch.from_data_list(data_list) for data_list in [bd_data_list, cln_data_list, cln_val_data_list]]
    out = model(backdoor_train_data.x, backdoor_train_data.edge_index, backdoor_train_data.batch)
    predicted_labels = out.argmax(dim=1) 
    asr, backdoor_train_success_indices = get_asr(model, backdoor_train_data, clean_train_data, predicted_labels)
    clean_train_success_indices         = get_clean_accurate_indices(model, backdoor_train_data)
    clean_val_success_indices           = get_clean_accurate_indices(model, clean_test_data)
    np.random.seed(seed)
    backdoor_train_success_indices  = np.random.choice(backdoor_train_success_indices, min(min_to_use, len(backdoor_train_success_indices)), replace=False)
    clean_train_success_indices     = np.random.choice(clean_train_success_indices, min(30, len(clean_train_success_indices)), replace=False)
    clean_val_success_indices       = np.random.choice(clean_val_success_indices, min(50, len(clean_val_success_indices)), replace=False)
    return backdoor_train_success_indices, clean_train_success_indices, clean_val_success_indices, asr
