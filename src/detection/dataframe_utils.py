from utils.data_utils import *
from utils.general_utils import *
from utils.plot_utils import *
from attack.backdoor_utils import *
from explain.explainer_utils import *
from detection.metrics import *
import pandas as pd
import numpy as np
import pickle
from torch_geometric.explain.algorithm.gnn_explainer import *
import numpy as np


data_shape_dict = get_info('data_shape_dict')
src_dir     = get_info('src_dir')
data_dir    = get_info('data_dir')
explain_dir = get_info('explain_dir')
train_dir   = get_info('train_dir')
train_dir_cln = get_info('train_dir_cln')
metric_plot_info_dict = get_info('metric_plot_info_dict')



def load_or_create_df(df_title=None):
    columns = [ 'config_name',
                'Dataset',
                'category',
                'list_index',
                'original_index',
                'Avg Nodes Pre-Attack', 'Avg Edges Pre-Attack', 'Avg Edges/Nodes Pre-Attack',
                'Edge Ratio',  'Node Ratio',       'Density Ratio',
                'Graph Type',  'Attack Target Label',     'Trigger Size',
                'Prob',        'K',                'Poison Rate',         'Balanced',
                'Model Hyp Set',
                'Explanation Type', 'Explanation Target Label', 'Explain LR',  'Explain Epochs',   'Node Reduce',  'Node Ent',         'Node Size',    'Edge Reduce',  'Edge Ent', 'Edge Size',
                'train_backdoor_acc_bal',
                'train_backdoor_pred_conf',
                'train_clean_acc_bal',
                'train_clean_pred_conf',
                'test_backdoor_acc_bal',
                'test_backdoor_pred_conf',
                'test_clean_acc_bal',
                'test_clean_pred_conf',
                'ASR',
                'Test ASR']
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


def create_row_index_values_for_metrics_df(model_file_name, 
                                           dataset, 
                                           category, 
                                           i, 
                                           graph_index, 
                                           attack_specs, 
                                           model_specs,
                                           explainer_hyperparams, 
                                           classifier_hyperparams,
                                           graph_geometry_info_dict, 
                                           history, 
                                           asr):
    model_hyp_set      = model_specs['model_hyp_set']
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


def process_addition(bkd_success_idxs, cln_success_idxs, cln_success_idxs_val, config_name, dataset, attack_specs,
                     model_specs, explainer_hyperparams, classifier_hyperparams,     graph_geometry_info_dict,   history,
                     asr, metric_value_dict, columns, main_df, addition_to_df, lower_thresh_percentile=0.25, upper_thresh_percentile=0.75):
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