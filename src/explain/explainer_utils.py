from attack.backdoor_utils import *
from utils.config import *
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


def run_explain(dataset,
                data_input, 
                subset_name, 
                original_index, 
                category,
                attack_specs, model_specs, explainer_hyperparams, classifier_hyperparams,
                which_explainer = 'gnn', 
                model = None,
                save=True):

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
    if which_explainer=='gnn':
        algorithm_ = GNNExplainer
        node_mask_type   = explainer_hyperparams['node_mask_type']
    elif which_explainer=='pg' or which_explainer=='pge':
        algorithm_ = PGExplainer
        node_mask_type   = None
    elif which_explainer=='captum':
        algorithm_ = CaptumExplainer
        node_mask_type   = explainer_hyperparams['node_mask_type']
    if which_explainer=='captum':
        SUPPORTED_METHODS = [
            'IntegratedGradients',
            'Saliency',
            'InputXGradient',
            'Deconvolution',
            'ShapleyValueSampling',
            'GuidedBackprop',
        ]
        algorithm =   algorithm_(attribution_method = 'Deconvolution')
    else:
        algorithm = algorithm_(epochs      = explainer_epochs, 
                                lr          = explain_lr,
                                coeffs_dict = coeffs)
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
    explanation = explainer(x=this_data.x, edge_index=this_data.edge_index, target=explainer_target_label)
    assert explainer is not None
    assert explanation is not None
    explainer_path = get_explainer_path(dataset, category, original_index, model_specs, attack_specs, explainer_hyperparams, classifier_hyperparams)
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

def print_explanation_scores(explanation_dict_backdoor, explanation_dict_clean = None, backdoor_label = 0):
    if explanation_dict_clean is not None:
        explanation  = explanation_dict_clean['explanation']
        explainer    = explanation_dict_clean['explainer']
        ground_truth = str(1-backdoor_label) 
        prediction        = str(torch.argmax(explainer.get_prediction(explanation.x,explanation.edge_index,)).item())
        masked_prediction = str(torch.argmax(explainer.get_masked_prediction(explanation.x,explanation.edge_index,explanation.node_mask,explanation.edge_mask)).item())
        clean_pos_fid     = str(explanation_dict_clean['pos_fids'][-1])
        clean_neg_fid     = str(explanation_dict_clean['neg_fids'][-1])
        clean_unfaith     = str(np.round(explanation_dict_clean['unfaiths'][-1],4))
        print('**Clean**')
        print('Ground Truth'.ljust(15),'Full Pred'.ljust(10),'Mask Pred'.ljust(10),'Pos Fid'.ljust(10),'Neg Fid'.ljust(10), 'Unfaith'.ljust(10))
        print(ground_truth.ljust(15), prediction.ljust(10), masked_prediction.ljust(10), clean_pos_fid.ljust(10), clean_neg_fid.ljust(10), clean_unfaith.ljust(10))
    explanation  = explanation_dict_backdoor['explanation']
    explainer    = explanation_dict_backdoor['explainer']
    ground_truth = str(backdoor_label)
    prediction        = str(torch.argmax(explainer.get_prediction(explanation.x,explanation.edge_index,)).item())
    masked_prediction = str(torch.argmax(explainer.get_masked_prediction(explanation.x,explanation.edge_index,explanation.node_mask,explanation.edge_mask)).item())
    backdoor_pos_fid    = str(explanation_dict_backdoor['pos_fids'][-1])
    backdoor_neg_fid    = str(explanation_dict_backdoor['neg_fids'][-1])
    backdoor_unfaith    = str(np.round(explanation_dict_backdoor['unfaiths'][-1],4))
    if explanation_dict_clean is not None:
        print('**Backdoor**')
    print('Ground Truth'.ljust(15),'Full Pred'.ljust(10),'Mask Pred'.ljust(10),'Pos Fid'.ljust(10),'Neg Fid'.ljust(10), 'Unfaith'.ljust(10))
    print(ground_truth.ljust(15), prediction.ljust(10), masked_prediction.ljust(10), backdoor_pos_fid.ljust(10), backdoor_neg_fid.ljust(10), backdoor_unfaith.ljust(10))


def explain_multiple(model,
                     dataset_dict_backdoor,
                     dataset_dict_clean,
                     dataset,
                     backdoor_subset, 
                     clean_subset, 
                     success_indices_backdoor,
                     success_indices_clean, 
                     explain_clean, 
                     attack_specs, 
                     model_specs, 
                     explanation_hyperparams, 
                     classifier_hyperparams,
                     repeat_explanation_n = 1,
                     which_explainer='gnn'):
    backdoor_dictionary = {original_index: None for original_index in success_indices_backdoor}
    clean_dictionary    = {original_index: None for original_index in success_indices_clean}
    attack_label = attack_specs['attack_target_label']
    assert len(success_indices_backdoor) == len(success_indices_clean)
    for (clean_original_index, backdoor_original_index) in zip(success_indices_clean, success_indices_backdoor):
        try:
            clean_original_index_int       = clean_original_index.item()
            backdoor_original_index_int    = backdoor_original_index.item()
        except:
            clean_original_index_int = clean_original_index
            backdoor_original_index_int = backdoor_original_index
        clean_explanation=None
        this_clean_dict    = explanation_dict()
        if explain_clean==True:
            clean_explainers, clean_explanations = [],[]
            for n in range(repeat_explanation_n):
                assert explanation_hyperparams['explanation_type'] in ['phenomenon','model']
                clean_explanation_hyperparams = copy.copy(explanation_hyperparams)
                if clean_explanation_hyperparams['explanation_type']=='phenomenon':
                    ground_truth = dataset_dict_clean[clean_subset][clean_original_index_int].pyg_graph.y
                    clean_explanation_hyperparams['explainer_target_label'] = ground_truth
                elif clean_explanation_hyperparams['explanation_type']=='model':
                    clean_explanation_hyperparams['explainer_target_label'] = None

                '''
                ^^^
                Currently doing a "phenomenon" explanation of the backdoor target class for both the clean and backdoor sample. This is consistent
                with the explanationk process from "try_metrics".
                To have it predict its actual label, use: torch.tensor(dataset_dict[clean_subset][0][clean_original_index].pyg_graph.y)
                '''
                clean_explainer, clean_explanation = run_explain(dataset,
                                                                dataset_dict_clean, 
                                                                clean_subset, 
                                                                clean_original_index, 
                                                                'clean',
                                                                attack_specs, model_specs, clean_explanation_hyperparams, classifier_hyperparams,
                                                                which_explainer = which_explainer, 
                                                                model = model)
                clean_explainers.append(clean_explainer)
                clean_explanations.append(clean_explanation)
            if repeat_explanation_n > 1:
                average_node_mask = sum([exp.node_mask for exp in clean_explanations])/len(clean_explanations)
                average_edge_mask = sum([exp.edge_mask for exp in clean_explanations])/len(clean_explanations)
                clean_explanation.node_mask = average_node_mask
                clean_explanation.edge_mask = average_edge_mask
            clean_dictionary[clean_original_index_int] = update_explanation_dict(this_clean_dict, clean_explainer, clean_explanation)
        this_backdoor_dict = explanation_dict()
        backdoor_explainers, backdoor_explanations = [],[]
        backdoor_explanation_hyperparams = copy.copy(explanation_hyperparams)
        backdoor_explanation_hyperparams['explainer_target_label'] = attack_specs['attack_target_label']
        for n in range(repeat_explanation_n):
            backdoor_explainer, backdoor_explanation  = run_explain(dataset,
                                                            dataset_dict_backdoor, 
                                                            backdoor_subset, 
                                                            backdoor_original_index, 
                                                            'backdoor',
                                                            attack_specs, model_specs, backdoor_explanation_hyperparams, classifier_hyperparams,
                                                            which_explainer = which_explainer, 
                                                            model = model)
            backdoor_explainers.append(backdoor_explainer)
            backdoor_explanations.append(backdoor_explanation)
        if repeat_explanation_n > 1:
            average_node_mask = sum([exp.node_mask for exp in backdoor_explanations])/len(backdoor_explanations)
            average_edge_mask = sum([exp.edge_mask for exp in backdoor_explanations])/len(backdoor_explanations)
            backdoor_explanation.node_mask = average_node_mask
            backdoor_explanation.edge_mask = average_edge_mask
        backdoor_dictionary[backdoor_original_index_int] = update_explanation_dict(this_backdoor_dict, backdoor_explainer, backdoor_explanation)
    return clean_dictionary, backdoor_dictionary


# def get_connectivity(dataset_dict, subset, explanation, graph_location):
#     data = dataset_dict[subset][graph_location]
#     data.pyg_graph = data.pyg_graph
#     inverse_indices = [0]* len(explanation.node_mask)
#     for i,idx in enumerate(data.nx_graph.nodes()):
#         inverse_indices[idx] = i
#     node_mask = explanation.node_mask.clone().detach()
#     node_mask = node_mask[inverse_indices]
#     preserved_nodes = torch.where(torch.sum(node_mask,dim=1)==1)[0]
#     if len(preserved_nodes) == 0:
#         return 0
#     else:
#         edges = data.pyg_graph.edge_index.T
#         connected = 0
#         for n1 in preserved_nodes:
#             for n2 in preserved_nodes:
#                 if torch.tensor([n1,n2]) in edges:
#                     connected += 1
#         connectivity = connected/len(preserved_nodes)
#         return connectivity
    

def load_saved_explainer(dataset,
                         original_index,
                         category,
                         attack_specs,
                         model_specs,
                         classifier_hyperparams,
                         explainer_hyperparams):
    explanation_path = get_explanation_path(dataset, category, original_index, model_specs, attack_specs, explainer_hyperparams, classifier_hyperparams)
    explainer_path   = get_explainer_path(dataset, category, original_index, model_specs, attack_specs, explainer_hyperparams, classifier_hyperparams)
    with open(explanation_path, 'rb') as f:
        explanation = pickle.load(f)
    with open(explainer_path, 'rb') as f:
        explainer = pickle.load(f)
    return explainer, explanation


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
        y_hat = explainer.get_prediction(
            explanation.x,
            explanation.edge_index,
            **kwargs)
        y_hat = F.softmax(y_hat,dim=1)[0]
    ''' vectorized '''
    explain_y_hat = explainer.get_masked_prediction(
        explanation.x,
        explanation.edge_index,
        node_mask,
        edge_mask,
        **kwargs)
    explain_y_hat = F.softmax(explain_y_hat, dim=1)[0]
    ''' vectorized '''
    complement_y_hat = explainer.get_masked_prediction(
        explanation.x,
        explanation.edge_index,
        1. - node_mask,
        1. - edge_mask,
        **kwargs)
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
