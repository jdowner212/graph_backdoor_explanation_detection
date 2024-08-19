from   attack.backdoor_utils import *
from   utils.data_utils import *
from   utils.general_utils import *
from   utils.plot_utils import *
from   explain.explainer_utils import *
from   kneed import KneeLocator
import numpy as np
import torch
from   sklearn.cluster import KMeans
from   sklearn.mixture import GaussianMixture
from   torch_geometric.explain.algorithm.gnn_explainer import *


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

def node_degree_variance(data):
    return np.var(torch.argmax(data.x, dim=1).tolist())


def mask_feature_variance(node_mask):
    return np.var(torch.argmax(node_mask, dim=1).tolist())


def get_explainability_score(explanation, explainer):
    pos_fid, neg_fid = my_fidelity(explanation,explainer)
    _ES = (pos_fid-neg_fid)
    return _ES

def get_pred_confidence(explainer, explanation):
    kwargs = {key: explanation[key] for key in explanation._model_args}
    out = explainer.get_prediction(
        explanation.x,
        explanation.edge_index,
        **kwargs)
    prob_outputs = F.softmax(out, dim=1).detach().tolist()
    confidence = np.max(prob_outputs[0])
    return confidence

def get_connectivity(dataset_dict, subset, explanation, graph_location):
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
    
def get_elbow_curvature(losses):
    losses = np.asarray(losses) + 1e-15
    percent_change = np.abs(100*(losses[0] - losses[-1])/losses[0])
    losses_min_max = min_max_scaling(torch.tensor(losses), small_difference=0.0001)
    min_max_var = torch.var(losses_min_max).item()
    if percent_change < 2 and min_max_var < 0.3:
        elbow = 0
        curvature = 1
    else:
        kneedle = KneeLocator(range(len(losses)), losses, S=1, curve="convex", direction="decreasing") # regular convergence
        elbow = kneedle.elbow if kneedle.elbow is not None else len(losses)
        curvature = kneedle.norm_elbow_y if kneedle.norm_elbow_y is not None else 0
    return elbow, curvature


'''''''''''''''''''''''''''''
''''     Thresholding    ''''
'''''''''''''''''''''''''''''

def get_boundary(clean_values, backdoor_values, clean_validation_values, inequality, thresh_type='optimal tpr-fnr', lower_thresh_percentile=0.25, upper_thresh_percentile=0.75):
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
    if 'optimal' in thresh_type:
        cutoff, best_score = 0, 0
        qs = np.linspace(0, 1, 100)
        for q in qs:
            this_cutoff = np.quantile(clean_validation_values, q)
            score_type = thresh_type[len('optimal '):]
            this_score = get_optimal_score(score_type,clean_values,backdoor_values,this_cutoff,inequality)
            if this_score >= best_score:
                best_score = this_score
                cutoff = this_cutoff
    else:
        if thresh_type=='clean_val':
            cutoff = clean_val_boundary(clean_validation_values_no_outliers, inequality)
        elif thresh_type=='kmeans':
            data = list(clean_values) + list(backdoor_values)
            cutoff = kmeans_boundary(data)
        elif thresh_type == 'gmm':
            data = list(clean_values) + list(backdoor_values)
            cutoff = gmm_boundary(data)
    return cutoff



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


def clean_val_boundary(clean_validation_values, inequality):
    clean_validation_values = replace_none_nan_with_average(clean_validation_values)
    boundary = None
    if inequality == 'more':
        boundary = np.max(clean_validation_values)
    elif inequality == 'less':
        boundary = np.min(clean_validation_values)
    return boundary


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