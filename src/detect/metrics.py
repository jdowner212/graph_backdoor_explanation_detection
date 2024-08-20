'''
This file contains functions pertaining to the 7 novel metrics introduced in our paper.
Relevant functions:

-- get_pred_confidence()
    Computes Metric 1: Prediction Confidence -- as defined in Section 4 (page 3) of our paper.

-- get_explainability_score()
    Computes Metric 2: Explainability -- as defined in Equation (3) of our paper (Section 4, page 3)
    
-- get_connectivity()
    Computes Metric 3: Connectivity -- as defined in Equation (5) of our paper (Section 4, page 4)

-- node_degree_variance()
    Computes Metric 4: Node Degree Variance -- as defined in Equation (6) of our paper (page 4)

-- subgraph_node_degree_variance():
    Computes Metric 5: Subgraph Node Degree Variance -- as defined in Equation (7) of our paper (Section 4, page 4)

-- get_elbow_curvature()
    Computes Metric 6 and Metric 7
    Metric 6: Elbow -- as defined in Equation (8) of our paper (Section 4, page 4)
    Metric 7: Curvature -- as defined in Equation (9) of our paper (Section 4, page 4)

-- get_boundary() and clean_val_boundary():
    Computes clean validation thresholds that are used to predict whether incoming samples are backdoor or clean.
    See "Clean Validation Extrema as Prediction Threshold" in our paper (Section 4, page 4)

-- get_distance_from_clean_val()
    Used to compute "distance" metrics, as defined in equation 10 on (Section 4, page 5) of our paper.
    This is particularly used for elbow and curvature metrics -- see "Caveat for Loss Curve Metrics" on page 4.

-- def display_detection_results():
    Computes the composite metric F1 score for detection for a provided NPMR (number of positive metrics required), 
    as described in Section 4 (page 5, under "Composite Metric") of paper.
'''

from   attack.backdoor_utils import *
from   utils.data_utils import *
from   utils.general_utils import *
from   utils.plot_utils import *
from   explain.explainer_utils import *
# from   detection.dataframe_utils import *

import copy
from   kneed import KneeLocator
import numpy as np
import numpy as np
import pickle
from   sklearn.cluster import KMeans
from   sklearn.mixture import GaussianMixture
import torch
from   torch_geometric.explain.algorithm.gnn_explainer import *


# from   detection.metrics import *
from   torch_geometric.explain.metric.faithfulness import unfaithfulness
from   torch_geometric.explain.algorithm.gnn_explainer import *


data_shape_dict = get_info('data_shape_dict')
src_dir     = get_info('src_dir')
data_dir    = get_info('data_dir')
explain_dir = get_info('explain_dir')
train_dir   = get_info('train_dir')
train_dir_cln = get_info('train_dir_cln')
metric_plot_info_dict = get_info('metric_plot_info_dict')



data_shape_dict = get_info('data_shape_dict')
src_dir     = get_info('src_dir')
data_dir    = get_info('data_dir')
explain_dir = get_info('explain_dir')
train_dir   = get_info('train_dir')
train_dir_cln = get_info('train_dir_cln')
metric_plot_info_dict = get_info('metric_plot_info_dict')


'''''''''''''''''''''''''''''
''''  Detection Metrics  ''''
'''''''''''''''''''''''''''''

def get_pred_confidence(explainer, explanation):
    ''' 
    Metric 1: Prediction Confidence -- as defined in Section 4 (page 3) of our paper.
    '''
    kwargs = {key: explanation[key] for key in explanation._model_args}
    out = explainer.get_prediction(
        explanation.x,
        explanation.edge_index,
        **kwargs)
    prob_outputs = F.softmax(out, dim=1).detach().tolist()
    confidence = np.max(prob_outputs[0])
    return confidence

def get_explainability_score(explanation, explainer):
    ''' 
    Metric 2: Explainability -- as defined in Equation (3) of our paper (Section 4, page 3)
    '''
    pos_fid, neg_fid = my_fidelity(explanation,explainer)
    _ES = (pos_fid-neg_fid)
    return _ES

def get_connectivity(dataset_dict, subset, explanation, graph_location):
    ''' 
    Metric 3: Connectivity -- as defined in Equation (5) of our paper (Section 4, page 4)
    '''
    data = dataset_dict[subset][graph_location]
    data.pyg_graph = data.pyg_graph
    inverse_indices = [0]* len(explanation.node_mask)
    for i,idx in enumerate(data.nx_graph.nodes()):
        inverse_indices[idx] = i
    node_mask = explanation.node_mask.clone().detach()
    node_mask = node_mask[inverse_indices]
    preserved_nodes = torch.where(torch.sum(node_mask,dim=1)==1)[0]
    if len(preserved_nodes) == 0:
        return 0
    else:
        edges = data.pyg_graph.edge_index.T
        connected = 0
        for n1 in preserved_nodes:
            for n2 in preserved_nodes:
                if torch.tensor([n1,n2]) in edges:
                    connected += 1
        connectivity = connected/len(preserved_nodes)
        return connectivity

def node_degree_variance(data):
    ''' 
    Metric 4: Node Degree Variance -- as defined in Equation (6) of our paper (page 4)
    '''
    return np.var(torch.argmax(data.x, dim=1).tolist())


def subgraph_node_degree_variance(node_mask):
    ''' 
    Metric 5: Subgraph Node Degree Variance -- as defined in Equation (7) of our paper (Section 4, page 4)
    '''
    return np.var(torch.argmax(node_mask, dim=1).tolist())
    
def get_elbow_curvature(losses):
    ''' 
    Metric 6: Elbow -- as defined in Equation (8) of our paper (Section 4, page 4)
    Metric 7: Curvature -- as defined in Equation (9) of our paper (Section 4, page 4)
    '''
    losses = np.asarray(losses) + 1e-15
    percent_change = np.abs(100*(losses[0] - losses[-1])/losses[0])
    losses_min_max = min_max_scaling(torch.tensor(losses), small_difference=0.0001)
    min_max_var = torch.var(losses_min_max).item()
    if percent_change < 2 and min_max_var < 0.3:
        elbow = 0
        curvature = 1
    else:
        kneedle = KneeLocator(range(len(losses)), losses, S=1, curve="convex", direction="decreasing")
        elbow = kneedle.elbow if kneedle.elbow is not None else len(losses)
        curvature = kneedle.norm_elbow_y if kneedle.norm_elbow_y is not None else 0
    return elbow, curvature


def get_explainer_metrics(model,            dataset_dict_backdoor,  dataset_dict_clean,     dataset,                subset, 
                          original_index,   graph_location,         this_data,              re_explain,             category, 
                          attack_specs,     model_specs,            classifier_hyperparams, explainer_hyperparams,  save=True):
    explanation=None
    if re_explain==False:
        try:
            explainer, explanation = load_saved_explainer(dataset, original_index, category, attack_specs, model_specs, classifier_hyperparams, explainer_hyperparams)
        except:
            re_explain=True
    if re_explain==True:
        explanation_root = f'{explain_dir}/{dataset}/explanations/{category}/original_index_{int(original_index)}'
        explainer_root = f'{explain_dir}/{dataset}/explainers/{category}/original_index_{int(original_index)}'
        create_nested_folder(explanation_root)
        create_nested_folder(explainer_root)
        explainer, explanation = run_explain(dataset, this_data, None, original_index, category,attack_specs, 
                                             model_specs, explainer_hyperparams, classifier_hyperparams, which_explainer = 'gnn', model = model, save=save)
    losses = explanation.clf_loss_over_time
    loss_max                    = np.max(losses)
    loss_min                    = np.min(losses)
    node_deg_variance           = node_degree_variance(this_data)
    elbow, curvature            = get_elbow_curvature(explanation['clf_loss_over_time'])
    es_score                    = get_explainability_score(explanation, explainer)
    unfaith                     = unfaithfulness(explainer, explanation)
    if 'clean' in category:
        connectivity                = get_connectivity(dataset_dict_clean, subset, explanation, graph_location)
    else:
        connectivity                = get_connectivity(dataset_dict_backdoor, subset, explanation, graph_location)
    pred_conf                   = get_pred_confidence(explainer,explanation)
    mask_feat_variance          = subgraph_node_degree_variance(explanation.node_mask)    
    return explanation, losses, loss_max, loss_min, elbow, curvature, es_score, unfaith, connectivity, pred_conf, node_deg_variance, mask_feat_variance


'''''''''''''''''''''''''''''
''''     Thresholding    ''''
'''''''''''''''''''''''''''''

def get_boundary(clean_values, backdoor_values, clean_validation_values, inequality, thresh_type='clean_val', lower_thresh_percentile=0.25, upper_thresh_percentile=0.75):
    cutoff, score = None, None
    clean_validation_values = replace_none_nan_with_average(clean_validation_values)
    if len(set(clean_validation_values))>= 4:
        clean_validation_values = sorted(clean_validation_values)
        lower_percentile_value    = np.quantile(clean_validation_values, q=lower_thresh_percentile)
        upper_percentile_value    = np.quantile(clean_validation_values, q=upper_thresh_percentile)
        if lower_percentile_value == upper_percentile_value:
            clean_validation_values_no_outliers = clean_validation_values
        else:
            low_outlier_indices  = [i for i in range(len(clean_validation_values)) if clean_validation_values[i]<=lower_percentile_value]
            high_outlier_indices = [i for i in range(len(clean_validation_values)) if clean_validation_values[i]>=upper_percentile_value]
            lower_p_i = 0 if len(low_outlier_indices) == 0 else int(min(low_outlier_indices))
            higher_p_i = -1 if len(high_outlier_indices) == 0 else int(min(high_outlier_indices))
            clean_validation_values_no_outliers = clean_validation_values[lower_p_i:higher_p_i]
    else:
        clean_validation_values_no_outliers = clean_validation_values
    if thresh_type=='clean_val':
        cutoff = clean_val_boundary(clean_validation_values_no_outliers, inequality)
    elif thresh_type=='kmeans':
        data = list(clean_values) + list(backdoor_values)
        cutoff = kmeans_boundary(data)
    elif thresh_type == 'gmm':
        data = list(clean_values) + list(backdoor_values)
        cutoff = gmm_boundary(data)
    elif 'optimal' in thresh_type:
        cutoff, best_score = 0, 0
        qs = np.linspace(0, 1, 100)
        for q in qs:
            this_cutoff = np.quantile(clean_validation_values, q)
            score_type = thresh_type[len('optimal '):]
            this_score = get_optimal_score(score_type,clean_values,backdoor_values,this_cutoff,inequality)
            if this_score >= best_score:
                best_score = this_score
                cutoff = this_cutoff
    return cutoff


def clean_val_boundary(clean_validation_values, inequality):
    clean_validation_values = replace_none_nan_with_average(clean_validation_values)
    boundary = None
    if inequality == 'more':
        boundary = np.max(clean_validation_values)
    elif inequality == 'less':
        boundary = np.min(clean_validation_values)
    return boundary


def kmeans_boundary(data):
    data = replace_none_nan_with_average(data)
    if len(set(list(data)))==1:
        group_A, group_B = data, []
    else:
        data_reshaped = np.array(data).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(data_reshaped)
        group_A, group_B = [], []
        for d, label in zip(data, kmeans.labels_):
            if label==0:
                group_A.append(d)
            else:
                group_B.append(d)
    lower_group = None
    if len(group_A) > 0 and len(group_B) > 0:
        lower_group = group_A if np.mean(group_A) <= np.mean(group_B) else group_B
        boundary = max(lower_group)
    else:
        boundary = max(group_A) if len(group_A) > len(group_B) else max(group_B)
    return boundary


def gmm_boundary(data):
    data = replace_none_nan_with_average(data)
    if len(set(data))==1:
        group_A, group_B = data, []
    else:
        gmm = GaussianMixture(n_components=2).fit(np.array(data).reshape(-1, 1))
        labels = gmm.predict(np.array(data).reshape(-1, 1))
        group_A, group_B = [], []
        for d, label in zip(data, labels):
            if label==0:
                group_A.append(d)
            else:
                group_B.append(d)
    lower_group = None
    if len(group_A) > 0 and len(group_B) > 0:
        lower_group = group_A if np.mean(group_A) <= np.mean(group_B) else group_B
        boundary = max(lower_group)
    else:
        boundary = max(group_A) if len(group_A) > len(group_B) else max(group_B)
    return boundary


def get_distance_from_clean_val(metrics_dict, metric, new_value):
    ''' 
    Used to compute "distance" metrics, as defined in equation 10 on (Section 4, page 5) of our paper.
    This is particularly used for elbow and curvature metrics, who are more likely to vary from
    clean metrics by their *distance* from the clean validation distribution rather than their
    relative value (higher/lower).
    (For more intuition, see "Caveat for Loss Curve Metric" on page 4.)
    '''
    clean_values = metrics_dict[metric]['clean_val']['values']
    clean_values = replace_none_nan_with_average(clean_values)
    small_random    = np.random.uniform(-1e-5,1e-5, len(clean_values)).astype(clean_values.dtype)
    clean_values    += small_random
    std_            = np.std(clean_values)+1e-5
    median_         = np.median(clean_values)
    distance = np.abs((new_value - median_)/std_)
    return distance


def get_tp_fp_tn_fn(clean_values, backdoor_values, threshold, inequality):
    if inequality=='more':
        tp = sum(value  >   threshold   for value in backdoor_values)
        fp = sum(value  >   threshold   for value in clean_values)
        tn = sum(value  <=  threshold   for value in clean_values)
        fn = sum(value  <=  threshold   for value in backdoor_values)
    elif inequality=='less':
        tp = sum(value  <=  threshold   for value in backdoor_values)
        fp = sum(value  <=  threshold   for value in clean_values)
        tn = sum(value  >   threshold   for value in clean_values)
        fn = sum(value  >   threshold   for value in backdoor_values)
    return tp,fp,tn,fn


def get_optimal_score(score_type,clean_values,backdoor_values,cutoff,inequality):
    clean_values = replace_none_nan_with_average(clean_values)
    backdoor_values = replace_none_nan_with_average(backdoor_values)
    tp, fp, tn, fn = get_tp_fp_tn_fn(clean_values, backdoor_values, cutoff, inequality)
    score=None
    if score_type == 'tpr-fpr':
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        score = tpr - fpr
    elif score_type == 'tpr-fnr':
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
        score = tpr - fnr
    elif score_type == 'f1':
        f1 = tp / (tp + 0.5 * (fp + fn))
        score = f1
    elif score_type == 'accuracy':
        acc = (tp + tn) / (tp + fp + tn + fn)
        score = acc
    return score


''''''''''''''''''
''' Other '''
''''''''''''''''''


def build_explainer_metrics_dict(dataset_dict_backdoor,     dataset_dict_clean, dataset,        model,          clean_success_indices_val,  clean_success_indices,
                                 backdoor_success_indices,  re_explain,         attack_specs,   model_specs,    classifier_hyperparams,     explainer_hyperparams,
                                 save_explainer=True):
    categories = ['clean_val', 'clean', 'backdoor']
    abs_keys   = [key for key in metric_plot_info_dict.keys() if '_dist' not in key]
    dist_keys  = [key for key in metric_plot_info_dict.keys() if '_dist' in key]       
    metric_value_dict = {key: {cat: {'values': []} for cat in categories}  for key in abs_keys+dist_keys}
    explanation_type = explainer_hyperparams['explanation_type']
    if len(clean_success_indices_val) > 5 and len(clean_success_indices) > 5 and len(backdoor_success_indices) > 5:
        for (subset, original_indices, category) in zip(['test_clean_graphs','train_clean_graphs','train_backdoor_graphs'],[clean_success_indices_val, clean_success_indices, backdoor_success_indices],['clean_val', 'clean','backdoor']):
            dataset_dict = dataset_dict_clean if 'clean' in subset else dataset_dict_backdoor
            explainer_hyperparams_this_round = copy.copy(explainer_hyperparams)
            for idx, graph in enumerate(dataset_dict[subset]):
                if graph.pyg_graph.original_index in set(original_indices):
                    if explanation_type=='phenomenon':
                        explainer_target_label = graph.pyg_graph.y
                    elif explanation_type=='model':
                        explainer_target_label = None
                    explainer_hyperparams_this_round['explainer_target_label'] = explainer_target_label
                    explainer_metric_values = get_explainer_metrics(model, dataset_dict_backdoor, dataset_dict_clean, dataset, subset, 
                                                                    graph.pyg_graph.original_index, idx, graph.pyg_graph, 
                                                                    re_explain, 
                                                                    category, 
                                                                    attack_specs,
                                                                    model_specs,
                                                                    classifier_hyperparams,
                                                                    explainer_hyperparams_this_round,
                                                                    save=save_explainer)
                    for key, value in zip(abs_keys, explainer_metric_values[2:]):
                        metric_value_dict[key][category]['values'].append(value)
        for (subset, original_indices, category) in zip(['test_clean_graphs','train_clean_graphs','train_backdoor_graphs'],[clean_success_indices_val, clean_success_indices, backdoor_success_indices], ['clean_val', 'clean','backdoor']):
            for key in abs_keys:
                for raw_value in metric_value_dict[key][category]['values']:
                    distance = get_distance_from_clean_val(metric_value_dict, key, raw_value)
                    metric_value_dict[key+'_dist'][category]['values'].append(distance)
        create_nested_folder(f'{explain_dir}/{dataset}/metric_value_dictionaries')
        full_explanation_path = get_explanation_path(dataset, category, graph.pyg_graph.original_index, model_specs, attack_specs, explainer_hyperparams_this_round, classifier_hyperparams)
        create_nested_folder(full_explanation_path)
        with open(full_explanation_path, 'wb') as f:
            pickle.dump(metric_value_dict,f)
    else:
        pass
    return metric_value_dict


def try_metrics(dataset_dict_backdoor,  dataset_dict_clean, dataset,            cln_success_idxs_val,   cln_success_idxs,       bkd_success_idxs,
                model,                  asr,                history,            attack_specs,           explainer_hyperparams,  model_specs,
                re_explain = False,     df_title = None,    boxplot_path = '',  save_explainer=True,
                relevant_metrics=['elbow_dist', 'curv_dist', 'es', 'connectivity', 'pred_conf', 'node_deg_var', 'mask_feat_var'],
                lower_thresh_percentile=0.25,   upper_thresh_percentile=0.75):
    attack_target_label = attack_specs['attack_target_label']
    model_hyp_set       = model_specs['model_hyp_set']
    classifier_hyperparams = hyp_dict_backdoor[dataset][attack_target_label][model_hyp_set]
    main_df = load_or_create_df(df_title)
    metric_value_dict   = build_explainer_metrics_dict(dataset_dict_backdoor,   dataset_dict_clean, dataset,        model,          cln_success_idxs_val,   cln_success_idxs,
                                                       bkd_success_idxs,        re_explain,     attack_specs,   model_specs,            classifier_hyperparams,
                                                       explainer_hyperparams,   save_explainer=save_explainer)
    graph_geometry_info_dict =  get_graph_geometry_info(dataset_dict_backdoor, dataset_dict_clean)
    config_name = get_model_name(classifier_hyperparams, attack_specs, model_hyp_set)
    addition_to_df = load_or_create_df()
    columns = addition_to_df.columns
    if len(bkd_success_idxs)>5 and len(cln_success_idxs)>5 and len(cln_success_idxs_val)>5:
        addition_to_df  = process_addition( bkd_success_idxs,   cln_success_idxs,   cln_success_idxs_val,   config_name,            dataset, 
                                            attack_specs,       model_specs,        explainer_hyperparams,  classifier_hyperparams, graph_geometry_info_dict,   
                                            history,            asr,                metric_value_dict,      columns,                main_df, 
                                            addition_to_df,     lower_thresh_percentile,            upper_thresh_percentile)
    explainer_metrics_boxplot(addition_to_df,relevant_metrics,plot=True,save_image=True, plot_path=boxplot_path,figsize=None, lower_q=lower_thresh_percentile, 
                              upper_q=upper_thresh_percentile, show_outliers=False, highlight_index=None, highlight_category=None, yy=None)
    main_df = update_main_df(main_df, addition_to_df,columns)
    if df_title is None:
        df_title = f'metrics_df.pkl'
    with open (f'{explain_dir}/{df_title}', 'wb') as f:
        pickle.dump(main_df, f)
    return main_df


def display_detection_results(df,NPMR,relevant_cols):
    ''' Computes the composite metric F1 score for detection for a provided NPMR (number of positive metrics required), as described in Section 4 (page 5, under "Composite Metric") of paper. '''
    print(f'NPMR={NPMR}:')
    this_df_backdoor = df[df['category'] == 'backdoor'][relevant_cols]
    this_df_clean = df[df['category'] == 'clean'][relevant_cols]
    tp = sum(this_df_backdoor.applymap(lambda x: str(x) == 'True Positive').sum(axis=1) >= NPMR)
    fn = len(this_df_backdoor) - tp
    fp = sum(this_df_clean.applymap(lambda x: str(x) == 'False Positive').sum(axis=1) >= NPMR)
    tn = len(this_df_clean) - fp
    if tp + 0.5 * (fp + fn) > 0:
        f1 = tp / (tp + 0.5 * (fp + fn))
    else:
        f1 = np.nan
    acc = (tp+tn)/(tp+tn+fn+fp)
    print(''.ljust(15) + 'Backdoor Prediction'.ljust(30) + 'Clean Prediction')
    print(''.ljust(10) + '_'*(len('Backdoor Prediction') + 10+len('Clean Prediction') + 10))
    print('Correct'.ljust(10) + '|'.ljust(5+len('Backdoor Prediction')//2) + str(tp).ljust(5+len('Backdoor Prediction')//2) + '|'.ljust(5+len('Clean Prediction')//2) + str(tn).ljust(6+len('Clean Prediction')//2) + '|')
    print(''.ljust(10) + '_'*(len('Backdoor Prediction') + 10+len('Clean Prediction') + 10))
    print('Incorrect'.ljust(10) + '|'.ljust(5+len('Backdoor Prediction')//2) + str(fp).ljust(5+len('Backdoor Prediction')//2) + '|'.ljust(5+len('Clean Prediction')//2) + str(fn).ljust(6+len('Clean Prediction')//2) + '|')
    print(''.ljust(10) + '_'*(len('Backdoor Prediction') + 10+len('Clean Prediction') + 10))
    print('F1:',np.round(f1,2))
    print('Acc:',np.round(acc,3))
    print()
    print()



''''''''''''''''''''''''''
'''   DATAFRAME UTILS  '''
''''''''''''''''''''''''''


def load_or_create_df(df_title=None):
    columns = [ 'config_name',              'Dataset',                  'category',                     'list_index',               'original_index',   
                'Avg Nodes Pre-Attack',     'Avg Edges Pre-Attack',     'Avg Edges/Nodes Pre-Attack',   'Edge Ratio',               'Node Ratio',               'Density Ratio',            'Graph Type',           
                'Attack Target Label',      'Trigger Size',             'Prob',                         'K',                        'Poison Rate',              'Balanced',                 'Model Hyp Set',
                'Explanation Type',         'Explanation Target Label', 'Explain LR',                   'Explain Epochs',           'Node Reduce',              'Node Ent',                 'Node Size',            'Edge Reduce',          'Edge Ent', 'Edge Size',
                'train_backdoor_acc_bal',   'train_backdoor_pred_conf', 'train_clean_acc_bal',          'train_clean_pred_conf',    'test_backdoor_acc_bal',    'test_backdoor_pred_conf',  'test_clean_acc_bal',   'test_clean_pred_conf',
                'ASR',                      'Test ASR']
    metric_columns = []
    for metric in list(metric_plot_info_dict.keys()):
        metric_columns.append(metric)
        for thresh_type in ['optimal tpr-fpr','optimal tpr-fnr','optimal f1','clean_val','kmeans','gmm']:
            metric_columns.append(f'{metric} {thresh_type}')
    if df_title is None:
        df_title = f'metrics_df.pkl'
    if os.path.exists(f'{explain_dir}/{df_title}'):
        with open (f'{explain_dir}/{df_title}', 'rb') as f:
            df = pickle.load(f)
            for col in columns+metric_columns:
                if col not in df.columns:
                    df[col] = [None]*len(df)
    elif not os.path.exists(f'{explain_dir}/{df_title}'):
       df = pd.DataFrame(columns=columns+metric_columns)
    return df


def create_row_index_values_for_metrics_df(model_file_name,         dataset,                    category,       i,                          
                                           graph_index,             attack_specs,               model_specs,    explainer_hyperparams,   
                                           classifier_hyperparams,  graph_geometry_info_dict,   history,        asr):
    model_hyp_set               = model_specs['model_hyp_set']
    balanced                    = unpack_kwargs(classifier_hyperparams,          ['balanced'])
    geometry_values             = unpack_kwargs(graph_geometry_info_dict,        ['avg_num_clean_nodes','avg_num_clean_edges','graph_density','edge_ratio','node_ratio','density_ratio'])
    attack_values               = unpack_kwargs(attack_specs,                    ['graph_type','attack_target_label','trigger_size','prob','K','poison_rate'])
    explainer_hyp_values        = unpack_kwargs(explainer_hyperparams,           ['explanation_type','explainer_target_label'])
    explainer_hyp_values       += [np.round(explainer_hyperparams['explain_lr'],4), explainer_hyperparams['explainer_epochs']]
    explainer_hyp_values       += unpack_kwargs(explainer_hyperparams['coeffs'], ['node_feat_reduction','node_feat_ent','node_feat_size','edge_reduction','edge_ent','edge_size'])
    model_performance_values    = unpack_kwargs(history,                         ['train_bd_acc_bal','train_bd_pred_conf','train_cln_acc_bal','train_cln_pred_conf','test_bd_acc_bal','test_bd_pred_conf','test_cln_acc_bal','test_cln_pred_conf'])
    this_row = [model_file_name, dataset,category,i,graph_index] + geometry_values + attack_values + [balanced, model_hyp_set] + explainer_hyp_values + model_performance_values + [asr] + [history['test_asrs'][-1]]
    return this_row


binary_outcome = lambda inequality, value, threshold, category: \
    'True Positive'  if ('backdoor' in category) and ((value >  threshold and inequality == 'more') or (value <= threshold and inequality == 'less')) else \
    'False Negative' if ('backdoor' in category) and ((value <= threshold and inequality == 'more') or (value >  threshold and inequality == 'less')) else \
    'False Positive' if ('clean'    in category) and ((value >  threshold and inequality == 'more') or (value <= threshold and inequality == 'less')) else \
    'True Negative'  if ('clean'    in category) and ((value <= threshold and inequality == 'more') or (value >  threshold and inequality == 'less')) else \
    None


def process_addition(bkd_success_idxs,  cln_success_idxs,   cln_success_idxs_val,   config_name,                dataset, 
                     attack_specs,      model_specs,        explainer_hyperparams,  classifier_hyperparams,     graph_geometry_info_dict,   
                     history,           asr,                metric_value_dict,      columns,                    main_df, 
                     addition_to_df,    
                     lower_thresh_percentile=0.25, 
                     upper_thresh_percentile=0.75):
    for category, success_indices in zip(['backdoor','clean','clean_val'],[bkd_success_idxs, cln_success_idxs, cln_success_idxs_val]):
        for i, graph_index in enumerate(success_indices):
            this_row = create_row_index_values_for_metrics_df(config_name,              dataset,                    category,       i,
                                                              graph_index,              attack_specs,               model_specs,    explainer_hyperparams,
                                                              classifier_hyperparams,   graph_geometry_info_dict,   history,         asr)
            for m in metric_value_dict.keys():
                value = metric_value_dict[m][category]['values'][i]
                this_row = this_row + [value]
                for thresh_type in ['optimal tpr-fpr','optimal tpr-fnr','optimal f1','clean_val','kmeans','gmm']:
                    this_row = this_row + [None]
            this_row_as_df = pd.DataFrame(dict(zip(columns, this_row)), index=[len(main_df) + len(addition_to_df)])
            for col in columns:
                addition_to_df = recast_column_types(addition_to_df,col)
                this_row_as_df = recast_column_types(this_row_as_df,col)
            addition_to_df = pd.concat([addition_to_df, this_row_as_df])
    [cln_addition, bkd_addition, cln_val_addition] = [addition_to_df[addition_to_df['category']==cat] for cat in ['clean','backdoor','clean_val']]
    for m in metric_value_dict.keys():
        inequality = metric_plot_info_dict[m]['inequality']
        for thresh_type in ['optimal tpr-fpr','optimal tpr-fnr','optimal f1','clean_val','kmeans','gmm']:
            cutoff = get_boundary(cln_addition[m], bkd_addition[m], cln_val_addition[m], inequality, thresh_type, lower_thresh_percentile, upper_thresh_percentile)
            values_for_computation = list(zip([inequality]*len(addition_to_df), addition_to_df[m],  [cutoff]*len(addition_to_df), addition_to_df['category']))
            addition_to_df[f'{m} {thresh_type}'] = list(map(binary_outcome,*zip(*values_for_computation)))
    return addition_to_df


def recast_column_types(df,col):
    if col == 'Balanced':
        df[col] = df[col].astype(bool)
    elif df[col].isin([True, False, 0, 1]).all() and ('Label' in col or 'Ent' in col):
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    return df


def update_main_df(main_df, addition_to_df, columns):
    for col in columns:
        main_df        = recast_column_types(main_df, col)
        addition_to_df = recast_column_types(addition_to_df, col)
    main_df = pd.concat([main_df, addition_to_df])
    return main_df


def change_thresholds(df, lower_q, upper_q, metrics):
    df_ = copy.copy(df)
    clean_val_mask = df_['category']=='clean_val'
    clean_mask     = df_['category']=='clean'
    backdoor_mask  = df_['category']=='backdoor'
    if list(df_.columns)[38]=='Test ASR':
        relevant_columns = list(df_.columns)[:39]
    else:
        relevant_columns = list(df_.columns)[:38]
    unique_config_names = list(set(df_['config_name']))
    for c, config_name in enumerate(unique_config_names):
        this_config_mask = df['config_name']==config_name
        this_config_df = df[this_config_mask]
        datasets_this_config = list(set(this_config_df['Dataset']))
        for dataset in datasets_this_config:
            print(f'{c}/{len(unique_config_names)}, {dataset}'.ljust(30),end='\r')
            this_dataset_mask = df['Dataset']==dataset
            this_subset_mask = this_config_mask & this_dataset_mask
            this_subset_df = df[this_subset_mask]
            if sum(this_subset_mask) > 0:
                for metric in metrics:
                    if metric not in relevant_columns:
                        relevant_columns.append(metric)
                    if f'{metric} clean_val' not in relevant_columns:
                        relevant_columns.append(f'{metric} clean_val')
                    clean_val_values = this_subset_df[metric]
                    if metric_plot_info_dict[metric]['inequality'] == 'less':
                        clean_val_cutoff = np.quantile(clean_val_values, lower_q)
                        pos_mask = df[metric]<=clean_val_cutoff
                        neg_mask = df[metric]>clean_val_cutoff
                    elif metric_plot_info_dict[metric]['inequality'] == 'more':
                        clean_val_cutoff = np.quantile(clean_val_values, upper_q)
                        pos_mask = df[metric]>=clean_val_cutoff
                        neg_mask = df[metric]<clean_val_cutoff
                    tp_mask = this_subset_mask & backdoor_mask & pos_mask
                    fn_mask = this_subset_mask & backdoor_mask & neg_mask
                    fp_mask = this_subset_mask & clean_mask & pos_mask
                    tn_mask = this_subset_mask & clean_mask & neg_mask
                    df_.loc[tp_mask,metric + ' clean_val']='True Positive'
                    df_.loc[fn_mask,metric + ' clean_val']='False Negative'
                    df_.loc[fp_mask,metric + ' clean_val']='False Positive'
                    df_.loc[tn_mask,metric + ' clean_val']='True Negative'
                    fp_mask_clean_val = this_subset_mask & clean_val_mask & pos_mask
                    tn_mask_clean_val = this_subset_mask & clean_val_mask & neg_mask
                    df_.loc[fp_mask_clean_val,metric + ' clean_val']='False Positive'
                    df_.loc[tn_mask_clean_val,metric + ' clean_val']='True Negative'
    df_ = df_[relevant_columns]
    return df_


def get_graph_geometry_info(dataset_dict_backdoor, dataset_dict_clean):
    backdoor_indices            = [i for i,g in enumerate(dataset_dict_backdoor['train_backdoor_graphs']) if g.pyg_graph.is_backdoored==True]
    corresponding_clean_indices = backdoor_indices
    try:
        num_trigger_nodes = len(dataset_dict_backdoor['trigger_graph'].nodes())
    except:
        num_trigger_nodes = None
    try:
        num_trigger_edges = len(dataset_dict_backdoor['trigger_graph'].edges())
    except:
        num_trigger_edges = None
    clean_graphs = [g for (i,g) in enumerate(dataset_dict_clean['train_clean_graphs']) if i in corresponding_clean_indices]
    avg_num_clean_nodes = np.mean([len(graph.nx_graph.nodes()) for graph in clean_graphs])
    avg_num_clean_edges = np.mean([len(graph.nx_graph.edges()) for graph in clean_graphs])
    edge_ratio      = None if num_trigger_edges is None else num_trigger_edges/avg_num_clean_edges
    node_ratio      = None if num_trigger_nodes is None else num_trigger_nodes/avg_num_clean_nodes
    trigger_density = None if num_trigger_nodes is None else num_trigger_edges/num_trigger_nodes
    graph_density   = avg_num_clean_edges/avg_num_clean_nodes
    density_ratio   = None if trigger_density is None else trigger_density/graph_density
    graph_geometry_info_dict = {'avg_num_clean_nodes': avg_num_clean_nodes, 'avg_num_clean_edges': avg_num_clean_edges, 'edge_ratio': edge_ratio, 
                                'node_ratio': node_ratio, 'trigger_density':trigger_density,'graph_density':graph_density,'density_ratio':density_ratio}
    return graph_geometry_info_dict

