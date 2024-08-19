from   utils.general_utils import *
from   attack.backdoor_utils import *
from   explain.explainer_utils import *
from   detection.explainer_detection_metrics import *
import copy
import numpy as np
import os
import pandas as pd
import pickle
from   sklearn.preprocessing import StandardScaler
from   sklearn.linear_model import LinearRegression
from   sklearn.model_selection import train_test_split
from   sklearn.metrics import mean_squared_error, r2_score
import torch
from   torch_geometric.explain.algorithm.gnn_explainer import *
from   torch_geometric.explain.algorithm.gnn_explainer import *



data_shape_dict = get_info('data_shape_dict')
src_dir     = get_info('src_dir')
data_dir    = get_info('data_dir')
explain_dir = get_info('explain_dir')
train_dir   = get_info('train_dir')
train_dir_cln = get_info('train_dir_cln')
metric_plot_info_dict = get_info('metric_plot_info_dict')
explanation_hyp_dict = get_info('explanation_hyp_dict')


def build_graph_geometry_dataframe(big_df=None, datasets=['AIDS','MUTAG','PROTEINS'],save_subset_dict=False, hyp_dict_backdoor=None):
    if big_df is None:
        big_df  = pd.DataFrame({'Dataset': [],'Model Name': [],'Target Label': [],'Graph Type': [],'Explanation Type': [],
                            'Apply Sigmoid': [],'Node Mask Type': [],'Edge Mask Type': [],'Return Type': [],'Min Max Scale': [],
                            'Type': [],'Node Reduce': [],'Node Ent': [],'Node Size': [],'Edge Reduce': [],'Edge Ent': [],'Edge Size': [],
                            'Threshold Type': [],'Threshold Value': [],'Explainer LR': [],'Explainer Epochs': [],'Graph Index': [],
                            'Num Nodes': [],'Num Edges': [],'Node to Edge Ratio': [],'Node Degree Variance': [],'Node Degree Mean': [],'Node Degree Median': [],
                            'Real Node Mask': [],'Ideal Node Mask': [],'Real Edge Mask': [],'Ideal Edge Mask': [],
                            'Node Mask Similarity':[],'Edge Mask Similarity':[],'Dictionary Path': [],'Graph Path': []
                            })
    else:
        pass

    count=0
    edge_r = 'sum'
    node_r = 'sum'

    min_max_scale = True

    apply_sigmoid=True
    node_mask_type = 'attributes'
    edge_mask_type = 'object'
    return_type = 'raw'
    num_samples = 5
    explanation_type='model'
    dict_path_root = f'{explain_dir}/hyperparam_search/explanation_dicts'
    graph_path_root = f'{explain_dir}/hyperparam_search/graphs'
    images_path_root = f'{explain_dir}/hyperparam_search/graph_images'
    for dataset in datasets:
        print(dataset,'\n')
        data = os.listdir(f'{train_dir}/{dataset}/models')
        random_indices   = list(np.random.choice(range(len(data)),40))
        data_subset      = [data[random_idx] for random_idx in random_indices]
        extracted_values = list(map(extract_values, data_subset))

        for i in range(len(extracted_values)):

            model_type, target_label, graph_type, trigger_size, poison_rate, prob, K, hyperparam_set, balanced = extracted_values[i]
            target_label, trigger_size = int(target_label), int(trigger_size)

            K_str, prob_str = get_K_str_prob_str(graph_type, trigger_size, K, prob)
            if K_str is not None and prob_str is not None:
                dataset_path = f'{data_dir}/poisoned_datasets/{dataset}/{graph_type}_trigger_size_{trigger_size}_poison_rate_{poison_rate}{prob_str}{K_str}'
                with open (dataset_path,'rb') as f:
                    dataset_dict_backdoor = pickle.load(f)



            kwargs = hyp_dict_backdoor[dataset][int(target_label)][hyperparam_set]


            with open (f'{train_dir}/{dataset}/models/{data_subset[i]}','rb') as f:
                state_dict = torch.load(f)
                model_type_dict = {'gcn_plain': PlainGCN, 'gcn': GCN3, 'gin': GIN, 'gin2': GIN2}
                model = model_type_dict[model_type](**kwargs)
                model.load_state_dict(state_dict['state_dict'])


            all_train_backdoor  = dataset_dict_backdoor['train_backdoor_graphs'][target_label]
            all_train_clean     = dataset_dict_backdoor['train_clean_graphs'][target_label]
            backdoored_indices = [g.pyg_graph.original_index for g in all_train_backdoor if g.pyg_graph.is_backdoored==True]
            if len(backdoored_indices) > 200:
                random_indices              = np.random.choice(backdoored_indices, 200)
                train_subset_for_backdoor   = [all_train_backdoor[idx] for idx in random_indices]
                backdoor_data_source        =  Batch.from_data_list([data_.pyg_graph for data_ in train_subset_for_backdoor])
                train_clean_subset_for_asr  = [all_train_clean[idx] for idx in random_indices]
                clean_data_for_asr          = Batch.from_data_list([data_.pyg_graph for data_ in train_clean_subset_for_asr])

                random_indices              = np.random.choice(backdoored_indices, 200)
                train_subset_for_clean      = [all_train_backdoor[idx] for idx in random_indices]
                clean_data_source           = Batch.from_data_list([data_.pyg_graph for data_ in train_subset_for_clean])



            else:
                backdoor_data_source = Batch.from_data_list([data.pyg_graph for data in dataset_dict_backdoor['train_backdoor_graphs'][target_label]])
                clean_data_source    = Batch.from_data_list([data.pyg_graph for data in dataset_dict_backdoor['train_clean_graphs'][target_label]])
                clean_data_for_asr   = Batch.from_data_list([data.pyg_graph for data in dataset_dict_backdoor['train_clean_graphs'][target_label]])



            out = model(backdoor_data_source.x, backdoor_data_source.edge_index, backdoor_data_source.batch)
            predicted_labels = out.argmax(dim=1) 

            _, backdoor_success_indices = get_asr(model, backdoor_data_source, clean_data_for_asr, 
                                                  #target_label, 
                                                  backdoor_preds=predicted_labels)
            clean_success_indices       = get_clean_accurate_indices(model, clean_data_source, target_label) 
            if num_samples==None or len(clean_success_indices)>num_samples:
                clean_success_indices = np.random.choice(clean_success_indices, num_samples, replace=False)
            backdoor_success_indices    = [j for j in list(range(len(backdoor_data_source))) if backdoor_data_source[j].is_backdoored==True]
            if num_samples==None or len(backdoor_success_indices)>num_samples:
                backdoor_success_indices = np.random.choice(backdoor_success_indices, num_samples, replace=False)


            model_info = [dataset, model_type, trigger_size, poison_rate, prob, K, hyperparam_set, balanced]

            
            explainer_epochs = np.random.choice([20])
            explain_lr = np.round(np.random.uniform(0.05,0.5),2)
            node_e = np.round(np.random.uniform(0.05,1.5),2)
            node_s = np.round(np.random.uniform(0.0001,0.008),4)
            edge_e = np.round(np.random.uniform(0.05,1.5),2)
            edge_s = np.round(np.random.uniform(0.0001,0.1),4)
            coeffs = {'edge_reduction': edge_r,    'node_feat_reduction': node_r,  'node_feat_size': node_s,   'node_feat_ent': node_e,    'edge_size': edge_s,    'edge_ent': edge_e, 'EPS': 1e-15}
            threshold_config = {}
            threshold_config['threshold_type'] = np.random.choice(['hard','topk'])
            threshold_config['value'] = int(np.random.choice(range(2,16))) if threshold_config['threshold_type']=='topk' else np.round(np.random.uniform(0.05,0.95),2)
            _, backdoor_dict = explain_multiple(model, 
                                                        dataset_dict_backdoor, [target_label], 'train_backdoor_graphs', 'train_clean_graphs', backdoor_success_indices,backdoor_success_indices, True, coeffs, explainer_epochs, explain_lr, apply_sigmoid, 
                                                        node_mask_type, edge_mask_type, return_type, explanation_type, model_info, graph_type, threshold_config, 
                                                        disconnect_params = {}, min_max_scale=min_max_scale, use_common_node_mask=False, remove_edge_disconnects=False, remove_node_disconnects=False, explain_clean=False)


            type_, dict_ = 'backdoor', backdoor_dict

            if save_subset_dict==True:
                dict_name_base = '_'.join([str(val) for val in  [dataset, data_subset[i][:-4],
                                                                'expl_epochs', explainer_epochs,   'expl_lr',      explain_lr,     'node_e',   node_e,         
                                                                'node_s',      node_s,             'edge_e',       edge_e, 'edge_s',   edge_s,         
                                                                'thresh_type', threshold_config['threshold_type'],        'thresh_val',   threshold_config['value']]])
                dict_path = f'{dict_path_root}/{dict_name_base}_{type_}.pkl'
                with open(dict_path,'wb') as f:
                    pickle.dump(dict_,f)

            for graph_index in dict_.keys():
                graph = dataset_dict_backdoor[f'train_{type_}_graphs'][target_label][graph_index]

                graph_name = f'{dataset}_{type_}_target_label_{target_label}_{graph_type}_trigger_size_{trigger_size}_poison_rate_{poison_rate}{prob_str}{K_str}_graph_index_{graph_index}'
                graph_path = f'{graph_path_root}/{graph_name}.pkl'
                with open (graph_path,'wb') as f:
                    pickle.dump(graph,f)

                num_nodes = len(graph.nx_graph.nodes())
                num_edges = len(graph.nx_graph.edges())
                node_edge_ratio = len(graph.nx_graph.nodes())/len(graph.nx_graph.edges())
                degrees = list(dict(graph.nx_graph.degree).values())
                node_degree_variance = np.var(degrees)
                node_degree_mean = np.mean(degrees)
                node_degree_median = np.median(degrees)


                triggered_edges = dataset_dict_backdoor[f'train_backdoor_graphs'][target_label][graph_index].pyg_graph.triggered_edges
                
                real_node_mask = dict_[graph_index]['explanation'].node_mask
                real_edge_mask = dict_[graph_index]['explanation'].edge_mask

                ideal_edge_mask = torch.tensor([1.0 if ((n0,n1) in triggered_edges or (n1,n0) in triggered_edges)else 0.0 for (n0,n1) in graph.nx_graph.edges()])
                ideal_node_mask = torch.zeros_like(real_node_mask)
                max_degree = real_node_mask.shape[1]
                for i_,node in enumerate(list(graph.nx_graph.nodes())):
                    if node in torch.tensor(triggered_edges).unique():
                        degree = min(graph.nx_graph.degree[node],max_degree)
                        ideal_node_mask[i_][degree-1]=1

                edge_mask_similarity = np.dot(real_edge_mask, ideal_edge_mask)
                node_mask_similarity = 0
                for i_ in range(len(real_node_mask)):
                    for j_ in range(len(real_node_mask[i_])):
                        node_mask_similarity += (real_node_mask[i_][j_] * ideal_node_mask[i_][j_]).item()

                edge_mask_similarity /= len(triggered_edges)
                node_mask_similarity /= len(torch.tensor(triggered_edges).unique())
                
                this_row = [dataset, 
                            f'{data_subset[i]}_{len(big_df)}',
                            target_label,graph_type,explanation_type,
                            apply_sigmoid,node_mask_type,edge_mask_type,return_type,min_max_scale,
                            type_,node_r,node_e,node_s,edge_r,edge_e,edge_s,
                            threshold_config['threshold_type'],threshold_config['value'],explain_lr,explainer_epochs,graph_index,
                            num_nodes,num_edges,node_edge_ratio,node_degree_variance,node_degree_mean,node_degree_median,
                            str(real_node_mask.tolist()),
                            str(ideal_node_mask.tolist()),
                            str(real_edge_mask.tolist()),
                            str(ideal_edge_mask.tolist()),
                            node_mask_similarity,edge_mask_similarity,dict_path,graph_path]
                this_row_as_df = pd.DataFrame(dict(zip(big_df.columns,this_row)),index=[len(big_df)])
                big_df = pd.concat([big_df,this_row_as_df])
                count += 1
                print(f'{count}/38400',end='\r')
    return big_df



def graph_geometry_metrics(graph_list, graph_metric):

    get_density = lambda g: nx.density(g.nx_graph)
    get_clustering_coef = lambda g: nx.clustering(g.nx_graph)
    get_avg_short_path = lambda g: nx.average_shortest_path_length(g.nx_graph)
    get_diameter = lambda g: nx.diameter(g.nx_graph)
    get_centrality = lambda g: nx.degree_centrality(g.nx_graph)
    get_btwn_centrality = lambda g: nx.betweenness_centrality(g.nx_graph)
    get_closeness_centrality = lambda g: nx.closeness_centrality(g.nx_graph)
    get_eigen_centrality = lambda g: nx.eigenvector_centrality(g.nx_graph)

    metric_function_dict = {'density':                  get_density,
                            'clustering_coef':          get_clustering_coef,
                            'avg_short_path':           get_avg_short_path,
                            'diameter':                 get_diameter,
                            'centrality':               get_centrality,
                            'btwn_centrality':          get_btwn_centrality,
                            'closeness_centrality':     get_closeness_centrality,
                            'eigen_centrality':         get_eigen_centrality}
    
    graph_metrics = list(map(metric_function_dict[graph_metric], graph_list))

    return graph_metrics

def regression_geometry_vs_explainer_hyperparameters(graph_geometry_dataframe, geometry_metric='Num Edges', explainer_score = 'Edge Mask Similarity'):
    geometry_metric = 'Num Edges'
    graph_geometry = list(copy.deepcopy(graph_geometry_dataframe[geometry_metric]))
    boundaries = np.linspace(min(graph_geometry),max(graph_geometry),50)
    bin_indices = np.digitize(graph_geometry, boundaries)
    bin_indices_unique = list(set(bin_indices))
    for bin_index in bin_indices_unique[:20]:
        df_indices = np.where(bin_indices==bin_index)[0]
        big_df_subset = graph_geometry_dataframe.iloc[df_indices,:]

        X = big_df_subset[['Node Ent', 'Node Size', 'Edge Ent', 'Edge Size', 
                            'Explainer LR', 'Explainer Epochs']]

        y = big_df_subset[explainer_score]

        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=2575)

        numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

        scaler = StandardScaler()
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])  

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print('Bin:',min(big_df_subset[geometry_metric]), max(big_df_subset[geometry_metric]))
        print('Coefficients: \n', model.coef_)
        print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
        print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))
        best_predictor_index = np.argmax(model.coef_)
        best_predictor = ['Node Ent', 'Node Size', 'Edge Ent', 'Edge Size', 
        'Explainer LR', 'Explainer Epochs'][best_predictor_index]
        print(best_predictor)