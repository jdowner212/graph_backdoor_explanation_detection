
'''
This file contains functions necessary for attacking datasets and training GNNs on the result. 
Relevant functions:

-- get_asr()
    Computes attack success rate (ASR), as mentioned in Section 1: Introduction of the paper (and referenced throughout).

-- train_generator_single_iter()
    Single iteration of training adaptive trigger generator, as described in detail in Sectin E of the appendix.
    
-- train_generator_iterative_loop()
    Iterative training of the adaptive trigger generator, as summarized on page 5 of our main paper, and detailed in Section E of the appendix.
    Alternates between retraining (1) the adaptive trigger generator and (2) the surrogate GNN it relies on for optimization.

-- poison_data_adaptive_attack()
    Data poisoning process by trained adaptive trigger generator.

-- random_trigger_synthesis()
    Random trigger generation, as described on page 3 of our paper (under "Threat model").

-- poison_data_random_attack()
    Data poisoning process corresponding to random backdoor attack (Zhang et al.), as described on pages 2 and 3 of our paper.

-- poison_data_clean_label_attack()
    Data poisoning process corresponding to clean-label attack (Xu and Picek, 2022) as described in Section E of the appendix.

-- train_loop_backdoor()
    Training of GNN on poisoned dataset
'''

from   config import *
from   utils.data_utils import *
from   utils.general_utils import *
from   utils.models import *
from   utils.plot_utils import *
import copy
import pandas as pd
import pytorch_lightning as pl
from   lightning.pytorch.callbacks import ModelCheckpoint
import matplotlib
import networkx as nx
import numpy as np
import os
import pickle
import pytorch_lightning as pl
from   pytorch_lightning.callbacks import ModelCheckpoint
import random
from   torch_geometric.data import Batch
from   torch_geometric.loader import DataLoader
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric
from   torch_geometric.data import Data
from   torch_geometric.transforms import Compose, OneHotDegree



hyp_dict_backdoor = get_info('hyp_dict_backdoor')
data_shape_dict   = get_info('data_shape_dict')
src_dir     = get_info('src_dir')
data_dir    = get_info('data_dir')
train_dir   = get_info('train_dir')
train_dir_cln = get_info('train_dir_cln')
adapt_surrogate_models = get_info('adapt_surrogate_models')




def get_asr(model,backdoor_data, clean_data, backdoor_preds=None):
    ''' Computing attack success rate (ASR), as mentioned in Section 1: Introduction of the paper (and referenced throughout). '''
    model.eval()
    successes_changed_label_indices = []
    total_changed_labels = 0
    backdoor_indices = [i for i in range(len(backdoor_data)) if backdoor_data[i].is_backdoored.item()==1]
    all_success_indices=[]
    for i in backdoor_indices:
        this_backdoor_data = backdoor_data[i]
        backdoor_out = model(this_backdoor_data.x, this_backdoor_data.edge_index, this_backdoor_data.batch)
        backdoor_pred = torch.argmax(backdoor_out, 1) # y_hat
        actual_backdoor = this_backdoor_data.y # y        
        this_clean_data = clean_data[i]
        assert this_backdoor_data.original_index == this_clean_data.original_index
        clean_out = model(this_clean_data.x, this_clean_data.edge_index, this_clean_data.batch)
        clean_pred = torch.argmax(clean_out, 1) # y_hat
        actual_clean = this_clean_data.y # y
        assert this_backdoor_data.is_backdoored==True
        assert this_clean_data.is_backdoored==False
        if actual_backdoor!=actual_clean:
            total_changed_labels += 1
            if backdoor_pred.item() == actual_backdoor and clean_pred.item() == actual_clean:
                backdoor_y = torch.tensor([0,0])
                backdoor_y[int(this_backdoor_data.y)]=1
                clean_y = torch.tensor([0,0])
                clean_y[int(this_clean_data.y)]=1
                successes_changed_label_indices.append(this_backdoor_data.original_index.item())
                all_success_indices.append(this_backdoor_data.original_index.item())
    success_changed_labels = len(successes_changed_label_indices)
    try:
        asr_changed = success_changed_labels/total_changed_labels
    except:
        asr_changed = None
    return asr_changed, successes_changed_label_indices


''''''''''''''''''''''''
'''  DATA POISONING  '''
''''''''''''''''''''''''

def poison_data_random_attack(dataset, attack_specs=None, seed=2575, verbose=False, clean_pyg_process=True, use_edge_attr=False):
    '''
    Data poisoning process corresponding to random backdoor attack (Zhang et al.), as described on pages 2 and 3 of our paper.

    Source:
        Zhang, Z.; Jia, J.; Wang, B.; and Gong, N. Z. 2021b. Backdoor attacks to graph neural networks. In Proceedings of the
        26th ACM Symposium on Access Control Models and Technologies, 15–26.
    '''
    
    dataset_dict = {'train_backdoor_graphs': [], 'test_backdoor_graphs': [], 'trigger_graph': []}
    trigger_graph = None
    graphs, tag2index = load_data(dataset, data_type='pyg', cleaned=clean_pyg_process, use_edge_attr=use_edge_attr)
    train_clean_graphs, test_clean_graphs = separate_data(graphs,seed=seed)

    poison_rate, trigger_size, attack_target_label, graph_type, prob, K = unpack_kwargs(attack_specs,['poison_rate','trigger_size','attack_target_label','graph_type','prob','K'])

    ''' Random trigger synthesis. '''
    if trigger_graph == None:
        trigger_graph = random_trigger_synthesis(trigger_size, graph_type, prob, K)
    num_classes = len(set([train_clean_graphs[idx].pyg_graph.y for idx in range(len(train_clean_graphs))]))
    random.seed(seed)
    
    ''' Select graphs to poison with uniform random sampling. '''
    train_indices_to_attack = get_train_indices_to_attack(train_clean_graphs, attack_specs)
    test_indices_to_attack = [idx for idx in range(len(test_clean_graphs))]
    assert poison_rate > 0

    ''' Inject triggers into specified graphs at randomly-selected nodes. '''
    train_backdoor_graphs = inject_backdoor_trigger(train_clean_graphs, trigger_graph, train_indices_to_attack, attack_target_label, tag2index)
    test_backdoor_graphs = inject_backdoor_trigger(test_clean_graphs, trigger_graph, test_indices_to_attack, attack_target_label, tag2index)

    ''' Save results in data structure. '''
    dataset_dict['train_backdoor_graphs'] = train_backdoor_graphs
    dataset_dict['test_backdoor_graphs'] = test_backdoor_graphs
    dataset_dict['trigger_graph'] = trigger_graph

    return dataset_dict


def poison_data_adaptive_attack(trigger_generator, data_dict_clean, train_backdoor_indices, test_backdoor_indices, trigger_size, target_label):
    '''
    Data poisoning process corresponding to our original adaptive attack, as summarized on page 5 of our main paper, and detailed in Section E of the appendix. 
    ** Note: This is not the adaptive attack itself, but rather, the process of using the adaptively-trained generator to poison the dataset.
    '''
    dataset_dict = {'train_backdoor_graphs': [], 'test_backdoor_graphs': [],'trigger_graph': []}
    dataset_dict['train_backdoor_graphs'] = copy.deepcopy(data_dict_clean['train_clean_graphs'])
    dataset_dict['test_backdoor_graphs']  = copy.deepcopy(data_dict_clean['test_clean_graphs'])
    for (group, indices) in [('train',train_backdoor_indices), ('test',test_backdoor_indices)]:
        for i in indices:
            g = copy.deepcopy(dataset_dict[f'{group}_backdoor_graphs'][i])
            triggered_edges = []
            possible_edge_list = get_all_possible_edges(g.pyg_graph)
            original_edges_set  = {tuple(edge) for edge in g.pyg_graph.edge_index.t().tolist()}

            ''' B: 'adjacency matrix' including all non-existing edges. Comprises edges that may be adaptively-added by generator. '''
            B = torch.zeros(len(possible_edge_list),dtype=int)
            for j in range(len(possible_edge_list)):
                e0,e1 = tuple(possible_edge_list[j])
                if (e0,e1) not in original_edges_set and (e1,e0) not in original_edges_set and e0!=e1:
                    B[j]=1
                    
            ''' Add one edge at a time. '''
            for t in range(trigger_size):
                edge_scores, possible_edges = trigger_generator(g.pyg_graph)
                edge_scores -= torch.min(edge_scores)
                ''' Choose edge from B that maximizes generator 'score'. '''
                edge_scores = edge_scores*B
                new_edge_index      = torch.as_tensor(possible_edges)[:, edge_scores.argmax()]
                edge_index_plus_new = torch.cat([g.pyg_graph.edge_index, new_edge_index.unsqueeze(1)], dim=1)
                new_x = update_x_(g.pyg_graph, edge_index_plus_new)

                [n0,n1] = new_edge_index.t().tolist()
                possible_edge_list = possible_edges.t().tolist()
                for j in range(len(possible_edges.t())):
                    e0,e1 = tuple(possible_edge_list[j])
                    if (e0,e1) == (n0,n1) or (e0,e1) == (n1,n0):
                        ''' Remove added edge from consideration. '''
                        B[j]=0

                ''' Add new edge to graph. '''
                triggered_edges.append((n0,n1))
                g.label = target_label
                g.pyg_graph.edge_index = edge_index_plus_new
                g.pyg_graph.x = new_x
                g.pyg_graph.y = target_label
                g.nx_graph.add_edge(n0,n1)

            ''' Mark graph as backdoored for experimental reference. '''
            g.pyg_graph.is_backdoored=True
            g.pyg_graph.triggered_edges = triggered_edges
            dataset_dict[f'{group}_backdoor_graphs'][i] = g
    return dataset_dict



def poison_data_clean_label_attack(dataset, attack_specs, verbose=True, clean=False, clean_pyg_process=False, use_edge_attr=False, seed=2575):
    '''
    Data poisoning process corresponding to clean-label attack (Xu and Picek, 2022) as described in Section E of the appendix.
    Triggers generated by choice of random synthesis algorithm (Erdos Renyi, Small World, and Preferential Attachments). 
    
    Source:
        Xu, J.; and Picek, S. 2022. Poster: Clean-label Backdoor Attack on Graph Neural Networks. In Proceedings of the 2022 
        ACM SIGSAC Conference on Computer and Communications Security, CCS ’22, 3491–3493. New York, NY, USA: Association for 
        Computing Machinery. ISBN 9781450394505.
    '''
    dataset_dict = {'train_backdoor_graphs': [], 'test_backdoor_graphs': [],'trigger_graph': []}
    trigger_graph = None
    graphs, tag2index = load_data(dataset, data_type='pyg', cleaned=clean_pyg_process, use_edge_attr=use_edge_attr)
    train_clean_graphs, test_clean_graphs = separate_data(graphs,seed=seed)
    poison_rate, trigger_size, attack_target_label, graph_type, prob, K = unpack_kwargs(attack_specs,['poison_rate','trigger_size','attack_target_label','graph_type','prob','K'])

    ''' Random trigger synthesis. '''
    if trigger_graph == None:
        trigger_graph = random_trigger_synthesis(trigger_size, graph_type, prob, K)
    num_train_to_attack = None
    num_classes = len(set([train_clean_graphs[idx].pyg_graph.y for idx in range(len(train_clean_graphs))]))
    num_train_to_attack = int(poison_rate * len(train_clean_graphs))

    ''' Select graphs to poison with uniform random sampling. For training graphs, filter by graphs correspoding to target label. '''
    random.seed(seed)
    train_backdoor_candidates = [idx for idx in range(len(train_clean_graphs)) if train_clean_graphs[idx].pyg_graph.y==attack_target_label]
    try:
        train_backdoor_candidates = [idx for idx in train_backdoor_candidates if is_fully_connected(train_clean_graphs[idx].nx_graph)==False]
        train_indices_to_attack = random.sample(train_backdoor_candidates,k=num_train_to_attack)
    except:
        print('Not enough non-fully-connected graphs to choose from, so some may be fully-connected/dense')
        try:
            train_indices_to_attack = random.sample([idx for idx in train_backdoor_candidates], k=num_train_to_attack)
        except:
            print("Desired sample size larger than available backdoor candidates; selecting entire sample")
            train_indices_to_attack = train_backdoor_candidates
    ''' Poison test graphs regardless of label -- will want to test effectiveness of attack on graphs whose ground truth labels do not equal target labels. '''
    test_indices_to_attack = [idx for idx in range(len(test_clean_graphs))]

    ''' Inject triggers into specified graphs at randomly-selected nodes. '''
    assert poison_rate > 0
    train_backdoor_graphs = inject_backdoor_trigger(train_clean_graphs, trigger_graph, train_indices_to_attack, attack_target_label, tag2index)
    test_backdoor_graphs = inject_backdoor_trigger(test_clean_graphs, trigger_graph, test_indices_to_attack, attack_target_label, tag2index)

    ''' Save in data structure. '''
    dataset_dict['train_backdoor_graphs'] = train_backdoor_graphs
    dataset_dict['test_backdoor_graphs'] = test_backdoor_graphs
    dataset_dict['trigger_graph'] = trigger_graph
    create_nested_folder(f'{data_dir}/poisoned/{dataset}')

    return dataset_dict


''''''''''''''''''''''''
'''   TRIGGER UTILS  '''
''''''''''''''''''''''''

def random_trigger_synthesis(trigger_size, graph_type, prob, K):
    '''
    Random trigger generation, as described on page 3 of our paper (under "Threat model").
    '''
    if trigger_size > 0:
        while True:
            K = int(prob * trigger_size) if K == None else K
            if graph_type == 'ER':
                trigger_graph_unsorted = nx.erdos_renyi_graph(trigger_size, prob)
            elif graph_type == 'SW':
                assert trigger_size >= K
                trigger_graph_unsorted = nx.watts_strogatz_graph(trigger_size, K, prob)
            elif graph_type == 'PA':
                trigger_graph_unsorted = nx.barabasi_albert_graph(trigger_size, K)
            sorted_edges = [(min(edge), max(edge)) for edge in trigger_graph_unsorted.edges()]
            trigger_graph = nx.Graph()
            trigger_graph.add_edges_from(sorted_edges)
            node_mapping = {trigger_node: main_node for trigger_node, main_node in zip(trigger_graph.nodes(), range(len(trigger_graph.nodes())))}
            trigger_graph = nx.relabel_nodes(trigger_graph, node_mapping)
            if nx.is_connected(trigger_graph):
                break
    return trigger_graph


def replace_whole_graph(graphs, idx, trigger_graph):
    max_feat = graphs[0].pyg_graph.x.shape[1] - 1
    graphs[idx].nx_graph = trigger_graph
    graphs[idx].pyg_graph.edge_index = torch.LongTensor(list(trigger_graph.edges())).T
    graphs[idx].pyg_graph.x = get_nx_graph_node_features(graphs[idx].nx_graph, max_feat)
    assert len(graphs[idx].pyg_graph.x) == len(graphs[idx].nx_graph.nodes())
    assert len(graphs[idx].nx_graph.nodes()) == len(
        torch.unique(graphs[idx].pyg_graph.edge_index)), f'assertion didnt pass at index {idx}'
    assert graphs[idx].nx_graph.edges() == trigger_graph.edges()
    assert graphs[idx].nx_graph.nodes() == trigger_graph.nodes()
    graphs[idx].pyg_graph.triggered_edges = [[e0,e1] for (e0,e1) in list(graphs[idx].nx_graph.edges())]
    return graphs[idx]


def replace_part_graph(graph_data, idx=None, trigger_graph=None):
    ''' Inserts trigger at random nodes in graph. '''
    np.random.seed(2575)
    if isinstance(graph_data, list):
        assert idx is not None, "When providing a graph list as argument to replace_part_graph(), must also provide an index."
        graph = graph_data[idx]
    elif isinstance(graph_data, GraphObject):
        graph = graph_data

    ''' Nodes randomly-selected for trigger insertion.'''
    rand_select_nodes = select_random_nodes(graph.nx_graph, len(trigger_graph.nodes()))
    node_mapping = {trigger_node: main_node for trigger_node, main_node in zip(trigger_graph.nodes(), rand_select_nodes)}
    edges = graph.pyg_graph.edge_index.transpose(1, 0).numpy().tolist()

    ''' Remove any existing connections between selected trigger nodes. '''
    for n0 in rand_select_nodes:
        for n1 in rand_select_nodes:
            if (n0, n1) in graph.nx_graph.edges():
                graph.nx_graph.remove_edge(n0, n1)
            if [n0, n1] in edges:
                edges.remove([n0, n1])

    ''' Add edges specified by trigger graph. '''
    triggered_edges = []
    for e in trigger_graph.edges():
        edge = [node_mapping[e[0]], node_mapping[e[1]]]
        (n0, n1) = (min(edge), max(edge))
        triggered_edges.append((n0,n1))
        edges.append((n0, n1))
        graph.nx_graph.add_edge(n0, n1)
    graph.pyg_graph.edge_index = torch.LongTensor(np.asarray(edges).transpose())
    graph.pyg_graph.triggered_edges = triggered_edges
    return graph


def inject_backdoor_trigger(graphs, trigger_graph, indices_to_attack, attack_target_label, tag2index):
    '''
    For each graph, injects predefined-trigger at random locations in provided graphs, and applies node degree as node feature.
    '''
    graphs = copy.deepcopy(graphs)
    for idx in indices_to_attack:
        num_nodes_in_graph = len(graphs[idx].nx_graph.nodes())
        if len(graphs[idx].nx_graph.nodes()) != len(graphs[idx].pyg_graph.x):
            print('before injecting trigger: trouble at index', idx)
        if len(trigger_graph.nodes()) >= num_nodes_in_graph:
            ''' If trigger is larger than or equal to size of original graph (extreme case -- not practical or recommended), replace graph with trigger. '''
            graphs[idx] = replace_whole_graph(graphs, idx, trigger_graph)
        elif len(trigger_graph.nodes()) < num_nodes_in_graph:
            ''' Inserts trigger at random nodes in graph. '''
            graphs[idx] = replace_part_graph(graphs, idx, trigger_graph)

        graphs[idx].pyg_graph.y = attack_target_label
        max_tag = graphs[idx].pyg_graph.x.shape[1] - 1
        node_tags = [tag if tag <= max_tag else max_tag for tag in list(dict(graphs[idx].nx_graph.degree).values())]

        ''' Sets node features to their degrees. '''
        graphs[idx].pyg_graph.x = torch.zeros(len(node_tags), len(tag2index))
        graphs[idx].pyg_graph.x[range(len(node_tags)), [tag2index[tag] for tag in node_tags]] = 1

        ''' Marks as backdoored for experimental reference. '''
        graphs[idx].pyg_graph.is_backdoored = 1
        if len(graphs[idx].nx_graph.nodes()) != len(graphs[idx].pyg_graph.x):
            print('after injecting trigger: trouble at index', idx)
        assert graphs[idx].pyg_graph.y is not None
    return graphs



''''''''''''''''''''''''''''''''''''
'''  ADAPTIVE GENERATOR TRAINING ''' 
''''''''''''''''''''''''''''''''''''

def train_surrogate(dataset_name, train_dataset, test_dataset, surrogate_filename, retrain=False, save=True, seed=2575, **model_kwargs):
    num_node_features = train_dataset[0].x.shape[1]
    num_classes = len(set([train_dataset[idx].y.item() for idx in range(len(train_dataset))]))
    if os.path.isfile(surrogate_filename) and retrain==False:
        print("Found pretrained surrogate model, loading...")
        model = GraphLevelGNN_opt(c_in=num_node_features, c_out=num_classes, **model_kwargs)
        model.load_state_dict(torch.load(surrogate_filename))
    else:
        graph_train_loader  = torch_geometric.loader.DataLoader(train_dataset, batch_size=model_kwargs['batch_size'], shuffle=True, num_workers=7, persistent_workers=True)
        graph_val_loader    = torch_geometric.loader.DataLoader(test_dataset, batch_size=model_kwargs['batch_size'], shuffle=False, num_workers=7, persistent_workers=True) # Additional loader if you want to change to a larger dataset
        graph_test_loader   = torch_geometric.loader.DataLoader(test_dataset, batch_size=model_kwargs['batch_size'], shuffle=False, num_workers=7, persistent_workers=True)
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        pl.seed_everything(seed)
        checkpoint_dir = os.path.join(adapt_surrogate_models, dataset_name, 'checkpoints')
        os.makedirs(root_dir, exist_ok=True)
        trainer = pl.Trainer(default_root_dir=checkpoint_dir,
                            callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                            accelerator="gpu" if str(device).startswith("cuda") else "mps" if str(device).startswith('mps') else "cpu",
                            devices=1,
                            max_epochs=model_kwargs['max_epochs'],
                            enable_progress_bar=True,
                            log_every_n_steps=1)
        trainer.logger._default_hp_metric = None 
        pl.seed_everything(seed)
        model = GraphLevelGNN_opt(c_in=num_node_features, c_out=num_classes, **model_kwargs)
        trainer.fit(model, graph_train_loader, graph_val_loader)
        model = GraphLevelGNN_opt.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        train_result = trainer.test(model, graph_train_loader, verbose=False)
        test_result = trainer.test(model, graph_test_loader, verbose=False)
        result = {"test": test_result[0]['test_acc'], "train": train_result[0]['test_acc']}
        print(result)
        if save==True:
            with open(surrogate_filename, 'wb') as f:
                torch.save(model.state_dict(),surrogate_filename)
    return model


def train_generator_iterative_loop(original_surrogate_gnn, dataset_name,   train_dataset,  test_dataset,   train_backdoor_indices, test_backdoor_indices,
                                   trigger_size,        generator_path, target_label,   rounds,         seed,                   model_kwargs_retrain,   generator_kwargs):
    ''' Alternates between retraining (1) the adaptive trigger generator and (2) the surrogate GNN it relies on for optimization '''
    clean_data = train_dataset, test_dataset
    current_surrogate_gnn = copy.deepcopy(original_surrogate_gnn)
    num_node_features = train_dataset[0].x.shape[1]
    generator_class, lr_gen, weight_decay, hidden_dim, depth, dropout_prob = unpack_kwargs(generator_kwargs, ['generator_class', 'lr_gen', 'weight_decay', 'hidden_dim', 'depth', 'dropout_prob'])
    print("Initializing trigger_generator & generator_optimizer...")
    try:
        trigger_generator = generator_class(num_node_features, hidden_dim=hidden_dim,  depth=depth, dropout_prob=dropout_prob)
    except:
        trigger_generator = generator_class(num_node_features)
    generator_optimizer = torch.optim.AdamW(trigger_generator.parameters(), lr=lr_gen, weight_decay=weight_decay)
    for round in range(rounds):
        if round >= 1:
            ''' Retrain surrogate GNN with most recent version of dataset -- if round>0, will be poisoned by trigger generator (pseudocode line 22) '''
            print('\nRetraining surrogate GNN...')
            current_surrogate_gnn = train_surrogate(dataset_name, train_dataset, test_dataset, 'should_not_save_intermediate_gnn', retrain=True, save=False, seed=seed, **model_kwargs_retrain)
        this_generator_path =  generator_path + '_intermediate.ckpt' if round!=rounds-1 else generator_path + '_final.ckpt'
        train_dataset_for_generator = [train_dataset[idx] for idx in train_backdoor_indices]
        prefix = '\nTraining' if round==0 else '\nRetraining'
        print(prefix, 'generator: iteration', round+1)
        ''' Retrain generator (pseudocode lines 6-19)'''
        trigger_generator,generator_optimizer = train_generator_single_iter(train_dataset_for_generator, 
                                                                         current_surrogate_gnn, this_generator_path, target_label, generator_kwargs, 
                                                                         trigger_generator=trigger_generator,generator_optimizer=generator_optimizer)
        if round != rounds-1:
            ''' Poison dataset wtih most recent generator (pseudocode line 21) '''
            train_dataset_graph_objects, test_dataset_graph_objects = generate_attacked_dataset(trigger_generator, train_dataset, test_dataset, train_backdoor_indices, test_backdoor_indices, trigger_size, target_label)
            clean_labels = []
            for i, g in enumerate(train_dataset_graph_objects):
                if g.pyg_graph.is_backdoored:
                    g_clean = clean_data[0][i]
                    clean_labels.append(g_clean.y.item())
            if len(set(clean_labels)) == 1:
                print("Largest trigger size is too big for dataset -- try limiting to attacks with smaller triggers.")
                break
            train_dataset = [g.pyg_graph for g in train_dataset_graph_objects]
            test_dataset  = [g.pyg_graph for g in test_dataset_graph_objects]
    return trigger_generator


def train_generator_single_iter(dataset_to_attack, surrogate_model, save_path, target_label, generator_kwargs, trigger_generator=None, generator_optimizer=None):
    ''' General setup '''
    epochs, T, lr_Ma, batch_size, max_num_edges = unpack_kwargs(generator_kwargs, ['epochs', 'T', 'lr_Ma', 'batch_size', 'max_num_edges'])
    dataset_indices = list(range(len(dataset_to_attack)))
    batch_size = approve_batch_size(dataset_to_attack,dataset_indices,batch_size)
    assert trigger_generator is not None
    assert generator_optimizer is not None
    batch_exp_mask_sum = 0
    min_loss = 1e8 
    epoch_losses = []

    ''' Freeze surrogate model parameters while generator is optimized '''
    freeze_parameters(surrogate_model)
    for epoch in range(epochs):
        random.shuffle(dataset_indices)
        epoch_exp_mask_sum = 0
        for c, i in enumerate(dataset_indices):
            new_edge_index  = torch.tensor([[],[]],dtype=torch.int64)
            data = copy.deepcopy(dataset_to_attack[i])
            original_edges_set  = {tuple(edge) for edge in data.edge_index.t().tolist()}
            all_scores, all_possible_edges = trigger_generator(data)
            all_scores=all_scores.sigmoid()
            all_possible_edges_list = all_possible_edges.t().tolist()
            ''' 
            A: clean edges of graph. 
            B: potential edges new edges, excluding existing clean ones. (Pseudocode line 9)
            '''
            A = torch.zeros(len(all_possible_edges_list))
            B = torch.zeros(len(all_possible_edges_list))
            for i in range(len(all_possible_edges_list)):
                e0,e1 = tuple(all_possible_edges_list[i])
                if (e0,e1) in original_edges_set or (e1,e0) in original_edges_set:
                    A[i]=1
                elif (e0,e1) not in original_edges_set and (e1,e0) not in original_edges_set and e0!=e1:
                    B[i]=1
            for b in range(B.shape[0]):
                if B[[b]]==1:
                    ''' Assert A and B are disjoint '''
                    assert A[[b]]==0, (b,'\n',A,'\n',B)
            scores_new_candidates_only = all_scores*B
            num_added = random.choice(list(range(1,max_num_edges))) if max_num_edges>1 else 1
            num_added = min(num_added, sum(A==0))
            max_indices = [scores_new_candidates_only.argmax()] if num_added==1 else torch.topk(scores_new_candidates_only, num_added)[1]
            
            ''' C: will be a mask for actually-added edges'''
            C = torch.zeros_like(A)
            for _, max_index in enumerate(max_indices):
                ''' Choose new edge where generator output ('score') is maximized (pseudocode line 10)'''
                assert A[max_index]==0, (max_index,'\n',A,'\n',B,'\n',scores_new_candidates_only,torch.topk(scores_new_candidates_only, num_added))
                C[max_index]=1
                add_edge_index = torch.as_tensor(all_possible_edges)[:, max_index]
                new_edge_index = torch.cat([new_edge_index, add_edge_index.unsqueeze(1)], dim=1)
            scores_original = all_scores*A
            scores_new      = all_scores*C
            scores_joint    = scores_original + scores_new

            ''' Update data to reflect new edges (pseudocode line 11) '''
            edge_index_plus_new = torch.cat([copy.deepcopy(data.edge_index), new_edge_index], dim=1)
            updated_x =  update_x_(data, edge_index_plus_new)

            ''' Simluate process by which GNNExplainer optimizes explanatory edge mask, Ma (pseudocode lines 12-15)'''
            Ma = scores_joint
            for t in range(T):
                Ma.retain_grad()
                Ma_clone = Ma.clone()
                surrogate_model.eval()
                ''' Compute Cross Entropy Loss on edges weighted by Ma (pseudocode line 14)'''
                out = surrogate_model.model(x=updated_x, edge_index=all_possible_edges, batch_idx=None, edge_weight=Ma_clone.sigmoid(), use_edge_weight=True)
                out = out.squeeze(dim=-1)
                L = torch.nn.CrossEntropyLoss()(out, torch.tensor([target_label]))
                L.backward(retain_graph=True)
                ''' Update Ma in the opposite direction of gradient (pseudocode line 15; encourages low classification loss on subgraph yielded by Ma -- i.e., a good explanation) '''
                with torch.no_grad():
                    Ma -= lr_Ma * Ma.grad
                    Ma.grad.zero_()
                    print(f'Epoch {epoch}: {c}/{len(dataset_indices)} samples processed', '-- mask loss:', np.round(L.clone().detach().numpy(),5), end='\r')
            
            ''' 
            Generator loss computed as summed Ma weights of non-clean edges. 
            This encourages the generator to produce a perturbed graph such that non-clean edges are not included explanatory subgraph. 
            (pseudocode lines 18-19)
            '''
            sample_mask_sum = torch.sum(Ma.sigmoid()*B)
            batch_exp_mask_sum += sample_mask_sum
            epoch_exp_mask_sum += sample_mask_sum
            if (c+1)%batch_size==0 or c==len(dataset_indices)-1:
                generator_optimizer.zero_grad()
                n = 1 if (c==0) else batch_size if ((c+1)%batch_size==0 and c!=0) else len(dataset_indices)%batch_size
                loss = batch_exp_mask_sum/n
                loss.backward()
                generator_optimizer.step()
                batch_exp_mask_sum = 0
        epoch_loss = epoch_exp_mask_sum
        epoch_losses.append(epoch_loss)
        min_loss, saved_this_epoch = save_best_generator(epoch, epoch_losses, save_path, trigger_generator, min_loss)
        gen_epoch_printout(epoch, ['epoch_exp_mask_sum','total'], [epoch_exp_mask_sum, epoch_loss], saved_this_epoch)
    return trigger_generator, generator_optimizer



def generate_attacked_dataset(trigger_generator_one_class, train_dataset, test_dataset, train_backdoor_indices, test_backdoor_indices, trigger_size, target_label):
    clean_train_graphs = []
    clean_test_graphs = []
    for (group, subset) in [('train',train_dataset), ('test',test_dataset)]:
        for i, data in enumerate(subset):
            g = GraphObject(None, data.y.long(), node_features=data.x, edge_features=[], edge_index=data.edge_index, is_backdoored=False, original_index=i, triggered_edges=[])
            g.pyg_graph.x = data.x
            g.pyg_graph.edge_index = data.edge_index
            g.nx_graph = nx.from_edgelist(data.edge_index.t().tolist())
            if group=='train':
                clean_train_graphs.append(g)
            elif group=='test':
                clean_test_graphs.append(g)
    backdoor_train_graphs = copy.deepcopy(clean_train_graphs)
    backdoor_test_graphs = copy.deepcopy(clean_test_graphs)
    print('# clean_train_graphs:',len(clean_train_graphs))
    print('# clean_test_graphs:',len(clean_test_graphs))
    for (group, indices) in [('train',train_backdoor_indices), ('test',test_backdoor_indices)]:
        for i in indices:
            triggered_edges=[]
            if group=='train':
                g = copy.deepcopy(backdoor_train_graphs[i])
            elif group=='test':
                g = copy.deepcopy(backdoor_test_graphs[i])

            for t in range(trigger_size):
                edge_scores, possible_edges = trigger_generator_one_class(g.pyg_graph)
                new_edge_index      = torch.as_tensor(possible_edges)[:, edge_scores.argmax()]
                edge_index_plus_new = torch.cat([g.pyg_graph.edge_index, new_edge_index.unsqueeze(1)], dim=1)
                new_x = update_x_(g.pyg_graph, edge_index_plus_new)
                [n0,n1] = new_edge_index.t().tolist()
                triggered_edges.append((n0,n1))
                g.label = target_label
                g.pyg_graph.edge_index = edge_index_plus_new
                g.pyg_graph.x = new_x
                g.pyg_graph.y = torch.tensor([target_label])
                g.nx_graph.add_edge(n0,n1)
            g.pyg_graph.is_backdoored=True
            g.pyg_graph.triggered_edges = triggered_edges
            if group=='train':
                backdoor_train_graphs[i] = g
            elif group=='test':
                backdoor_test_graphs[i] = g
    return backdoor_train_graphs, backdoor_test_graphs


def update_x_(data, new_full_edge_index):
    num_nodes, max_degree = data.x.shape[0], data.x.shape[1]
    degrees = torch_geometric.utils.degree(new_full_edge_index[0], num_nodes)
    new_x = torch.zeros((data.num_nodes, max_degree))
    for node, degree in enumerate(degrees):
        if degree >= max_degree:
            degree = torch.tensor(max_degree - 1, dtype=torch.float)
        new_x[node, int(degree.item())] = 1
    return new_x


def get_possible_new_edges(data):
    num_nodes = data.x.size(0)
    edge_index = data.edge_index
    all_edges = torch.combinations(torch.arange(num_nodes), r=2)
    existing_edges = set(map(tuple, edge_index.t().cpu().numpy()))
    existing_edges.update([(j, i) for i, j in existing_edges])
    possible_edges = [tuple(edge.numpy()) for edge in all_edges if tuple(edge.numpy()) not in existing_edges]
    possible_edges = [(i,j) for (i,j) in possible_edges if i!=j]
    return possible_edges



''''''''''''''''''''''''''''''''''''''''''
'''    TRAINING GNN ON POISONED DATA   '''
''''''''''''''''''''''''''''''''''''''''''


def train_loop_backdoor(dataset, dataloader_dict, class_weights, model_path, these_attack_specs, these_model_specs, these_classifier_hyperparams, plot=False, verbose=False):
    '''
    Training model on previously-attacked dataset. 
    For the most part, this is no different from a traditional GNN training process, although additional functionalities included to record backdoor attack accuracy and other metrics for experimental analysis.
    '''
    assert model_path is not None
    paired_train_loader, paired_test_loader = dataloader_dict['train_loader'], dataloader_dict['test_loader']
    epochs, lr, weight_decay, model_type = unpack_kwargs(these_classifier_hyperparams, ['epochs','lr','weight_decay','model_type'])
    kwargs = these_classifier_hyperparams
    num_node_features, num_classes       = unpack_kwargs(data_shape_dict[dataset],['num_node_features','num_classes'])
    attack_target_label =  these_attack_specs['attack_target_label']
    train_bd_acc_bal,   train_bd_pred_prob  = None, None
    train_cln_acc_bal,  train_cln_pred_prob = None, None
    test_bd_acc_bal,    test_bd_pred_prob   = None, None
    test_cln_acc_bal,   test_cln_pred_prob  = None, None
    if plot==False:
        matplotlib.use('Agg')
    else:
        matplotlib.use('nbAgg')
    kwargs['num_node_features'] = num_node_features
    kwargs['num_classes'] = num_classes
    model = None
    lr_schedule = False
    sam = False
    model = model_dict()[model_type](**kwargs)
    if sam==True:
        base_optimizer = optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, adaptive=False, rho=0.05, lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_schedule == True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.6, patience=3,min_lr=0.00005)
    criterion = torch.nn.CrossEntropyLoss() if class_weights is None else torch.nn.CrossEntropyLoss(weight=class_weights)
    train_backdoor_accs,    train_backdoor_losses,   train_asrs = [], [], []
    test_clean_accs,        test_clean_losses                   = [], []
    test_backdoor_accs,     test_backdoor_losses,    test_asrs  = [], [], []
    for epoch in range(epochs):
        model.train()
        for cd, data in paired_train_loader:
            if sam==True:
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y.long())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        if epoch%10 == 0:
            _,              _,                  train_bd_acc,   train_bd_loss,  train_cln_accs_per_class,  train_bd_accs_per_class,   _,  _,  train_asr = test(model, paired_train_loader, criterion,  attack_target_label, 'backdoor')
            test_clean_acc, test_clean_loss,    test_bd_acc,    test_bd_loss,   test_cln_accs_per_class,   test_bd_accs_per_class,    _,  _,  test_asr  = test(model, paired_test_loader,  criterion,  attack_target_label, 'backdoor')
            training_printout(epoch, verbose, train_cln_accs_per_class, train_bd_accs_per_class, test_cln_accs_per_class, test_bd_accs_per_class, train_bd_loss, test_clean_loss, test_bd_loss,train_asr, test_asr)
            train_cln_acc_bal, train_bd_acc_bal = np.mean(train_cln_accs_per_class), np.mean(train_bd_accs_per_class)
            test_cln_acc_bal, test_bd_acc_bal   = np.mean(test_cln_accs_per_class),  np.mean(test_bd_accs_per_class)
        else:
            train_bd_acc,   train_bd_loss,      train_asr   = None, None, None
            test_clean_acc, test_clean_loss                 = None, None
            test_bd_acc,    test_bd_loss,       test_asr    = None, None, None

        train_backdoor_accs.append(train_bd_acc);   train_backdoor_losses.append(train_bd_loss);    train_asrs.append(train_asr)
        test_clean_accs.append(test_clean_acc);     test_clean_losses.append(test_clean_loss);      test_backdoor_accs.append(test_bd_acc);
        test_backdoor_losses.append(test_bd_loss);  test_asrs.append(test_asr)

        if lr_schedule==True and epoch%10==0:
            scheduler.step(np.mean([test_clean_loss,test_bd_loss]))
            print(scheduler.optimizer.param_groups[0]['lr'])
    history = { 'train_accs':           list(pd.Series(train_backdoor_accs).interpolate()),
                'train_losses':         list(pd.Series(train_backdoor_losses).interpolate()),
                'train_asrs':           list(pd.Series(train_asrs).interpolate()),
                'test_clean_accs':      list(pd.Series(test_clean_accs).interpolate()),
                'test_clean_losses':    list(pd.Series(test_clean_losses).interpolate()),
                'test_backdoor_accs':   list(pd.Series(test_backdoor_accs).interpolate()),
                'test_backdoor_losses': list(pd.Series(test_backdoor_losses).interpolate()),
                'test_asrs':            list(pd.Series(test_asrs).interpolate()),
                
                'train_bd_acc_bal':     train_bd_acc_bal,   'train_bd_pred_conf':   train_bd_pred_prob,
                'train_cln_acc_bal':    train_cln_acc_bal,  'train_cln_pred_conf':  train_cln_pred_prob,
                'test_bd_acc_bal':      test_bd_acc_bal,    'test_bd_pred_conf':    test_bd_pred_prob,
                'test_cln_acc_bal':     test_cln_acc_bal,   'test_cln_pred_conf':   test_cln_pred_prob
                }
    state_dict = {'state_dict': model.state_dict(), 'hyperparameters': these_classifier_hyperparams, 'history': history}
    if plot == True:
        accs, losses, asrs = ([history['train_accs'], history['test_clean_accs'], history['test_backdoor_accs']], 
                              [history['train_backdoor_losses'], history['test_clean_losses'], history['test_backdoor_losses']], 
                              [history['train_asrs'], history['test_asrs']])
        plot_training_results(dataset, plot, accs, losses, asrs=asrs, attack_specs=these_attack_specs, classifier_hyperparams=these_classifier_hyperparams, model_specs=these_model_specs) 
    create_nested_folder(f'{train_dir}/{dataset}/models')
    torch.save(state_dict, model_path)
    return model, history

def test(model, loader, criterion=None, target_label=1, clean_or_backdoor='clean'):
    model.eval()
    if clean_or_backdoor == 'clean':
        clean_dataset = loader.dataset
    if clean_or_backdoor=='backdoor':
        clean_dataset = Batch.from_data_list(list(zip(*loader.dataset))[0])
        num_classes = len(torch.unique(clean_dataset.y))
    clean_losses, clean_correct = [],0
    backdoor_losses, backdoor_correct = [],0
    clean_pred_confs, backdoor_pred_confs = [],[]
    asrs = []
    acc_by_class = {'clean':    {i: {'correct': 0, 'total': 0} for i in range(num_classes)},
                    'backdoor': {i: {'correct': 0, 'total': 0} for i in range(num_classes)}}
    for batch in loader:
        if clean_or_backdoor == 'clean':
            clean_batch = batch
        else:
            clean_batch, backdoor_batch = batch[0], batch[1]
        out_c = model(clean_batch.x, clean_batch.edge_index, clean_batch.batch)
        out_c_probs = F.softmax(out_c,dim=1)
        pred_c_conf = [torch.max(prob).detach().numpy() for prob in out_c_probs]
        clean_pred_confs += pred_c_conf
        pred_c = out_c.argmax(dim=1)
        loss_c = (criterion(out_c, clean_batch.y.long()))
        clean_correct += int((pred_c == clean_batch.y).sum())
        clean_losses.append(loss_c.detach().cpu().numpy())
        for class_ in range(num_classes):
            this_class_indices = torch.where(clean_batch.y==class_)[0]
            acc_by_class['clean'][class_]['correct'] += int((pred_c[this_class_indices] == clean_batch.y[this_class_indices]).sum())
            acc_by_class['clean'][class_]['total']   += len(clean_batch.y[this_class_indices])
        if clean_or_backdoor=='backdoor':
            out_b = model(backdoor_batch.x, backdoor_batch.edge_index, backdoor_batch.batch)
            out_b_probs = F.softmax(out_b,dim=1)#[0]
            pred_b_conf = [torch.max(prob).detach().numpy() for prob in out_b_probs]
            backdoor_pred_confs += pred_b_conf
            pred_b = out_b.argmax(dim=1)
            backdoor_batch.y = backdoor_batch.y.type(torch.long)
            loss_b = (criterion(out_b, backdoor_batch.y.long()))
            backdoor_correct += int((pred_b == backdoor_batch.y).sum())
            backdoor_losses.append(loss_b.detach().cpu().numpy())
            for class_ in range(num_classes):
                this_class_indices = torch.where(backdoor_batch.y==class_)[0]
                acc_by_class['backdoor'][class_]['correct'] += int((pred_b[this_class_indices] == backdoor_batch.y[this_class_indices]).sum())
                acc_by_class['backdoor'][class_]['total']   += len(backdoor_batch.y[this_class_indices])
            clean_data_list     = batch_to_data_list(clean_batch,backdoor=False)
            backdoor_data_list  = batch_to_data_list(backdoor_batch,backdoor=True)
            clean_original_indices      = [graph.original_index.item() for graph in clean_data_list]
            backdoor_original_indices   = [graph.original_index.item() for graph in backdoor_data_list]
            assert clean_original_indices == backdoor_original_indices
            asr, _ = get_asr(model, backdoor_batch, clean_batch, backdoor_preds=pred_b)
            if asr is not None:
                asrs.append(asr)
    clean_accs      = []
    backdoor_accs   = []
    for class_ in range(num_classes):
        try:
            clean_accs.append(acc_by_class['clean'][class_]['correct']/acc_by_class['clean'][class_]['total'])
        except:
            pass
        try:
            backdoor_accs.append(acc_by_class['backdoor'][class_]['correct']/acc_by_class['backdoor'][class_]['total'])
        except:
            pass
    clean_accs_per_class    = [np.round(a,3) for a in clean_accs]
    backdoor_accs_per_class = [np.round(a,3) for a in backdoor_accs]
    clean_pred_conf,  backdoor_pred_conf      = np.mean(clean_pred_confs), np.mean(backdoor_pred_confs)
    clean_acc = clean_correct/len(clean_dataset)
    clean_loss = np.mean(clean_losses)
    if clean_or_backdoor=='backdoor':
        backdoor_dataset = Batch.from_data_list(list(zip(*loader.dataset))[1])
        backdoor_acc = backdoor_correct/len(backdoor_dataset)
        backdoor_loss = np.mean(backdoor_losses)
        try:
            asr = np.mean(asrs)
        except:
            asr = 0
        return clean_acc, clean_loss, backdoor_acc, backdoor_loss, clean_accs_per_class, backdoor_accs_per_class, clean_pred_conf, backdoor_pred_conf, asr
    else:
        return clean_acc, clean_loss, out_c_probs


def test_clean(model, loader, criterion=None):
    correct = 0
    losses = []
    model.eval()
    for i, batch in enumerate(loader):
        out = model(batch.x, batch.edge_index, batch.batch)
        pred = out.argmax(dim=1)
        loss = criterion(out, batch.y)
        correct += int((pred == batch.y).sum())
        losses.append(loss.detach().item())
    acc = correct / len(loader.dataset)
    loss = np.mean(losses)
    return acc, loss


def get_dataloader_dict(dataset_dict_backdoor, dataset_dict_clean, model_specs, classifier_hyperparams):
    clean_or_backdoor   = model_specs['clean_or_backdoor']
    batchsize, balanced = unpack_kwargs(classifier_hyperparams,['batchsize','balanced'])
    shuffled_train_indices = get_random_indices(len(dataset_dict_clean['train_clean_graphs']))
    shuffled_test_indices  = get_random_indices(len(dataset_dict_clean['test_clean_graphs']))
    clean_train_data = [dataset_dict_clean['train_clean_graphs'][i] for i in shuffled_train_indices]
    clean_test_data  = [dataset_dict_clean['test_clean_graphs'][i] for i in shuffled_test_indices]
    clean_train_graphs = [g.pyg_graph for g in clean_train_data]
    clean_test_graphs  = [g.pyg_graph for g in clean_test_data]
    if clean_or_backdoor=='clean':
        if not (check_graph_data(clean_train_graphs) and check_graph_data(clean_test_graphs)):
            raise ValueError("Invalid graph data in clean dataset")
        train_loader_clean = DataLoader(clean_train_graphs, batch_size=batchsize,shuffle=True,  num_workers=7, persistent_workers=True)
        test_loader_clean  = DataLoader(clean_test_graphs, batch_size=batchsize, shuffle=False, num_workers=7, persistent_workers=True)
        dataloader_dict = {'train_loader': train_loader_clean, 'test_loader': test_loader_clean}
    if clean_or_backdoor=='backdoor':
        assert dataset_dict_backdoor is not None
        backdoor_train_data = [dataset_dict_backdoor['train_backdoor_graphs'][i] for i in shuffled_train_indices]
        if balanced==True:
            num_0_backdoor, num_1_backdoor = [len([g for g in backdoor_train_data if g.pyg_graph.y == label]) for label in [0, 1]]
            smaller_class = np.argmin([num_0_backdoor, num_1_backdoor])
            size_diff = np.abs(num_0_backdoor - num_1_backdoor)
            add_to_oversample = list(np.random.choice([g for g in backdoor_train_data if g.pyg_graph.y == smaller_class],size_diff))
            backdoor_train_data += add_to_oversample
        backdoor_test_data  = [dataset_dict_backdoor['test_backdoor_graphs'][i] for i in shuffled_test_indices]
        backdoor_train_graphs = [g.pyg_graph for g in backdoor_train_data]
        backdoor_test_graphs = [g.pyg_graph for g in backdoor_test_data]
        if not (check_graph_data(backdoor_train_graphs) and check_graph_data(backdoor_test_graphs)):
            raise ValueError("Invalid graph data in backdoor dataset")
        paired_train_data = [(clean,backdoor) for (clean,backdoor) in zip(clean_train_graphs,backdoor_train_graphs)]
        paired_test_data  = [(clean,backdoor) for (clean,backdoor) in zip(clean_test_graphs,backdoor_test_graphs)]
        paired_train_loader = DataLoader(paired_train_data, batch_size = batchsize, shuffle=True,  num_workers=7, persistent_workers=True)
        paired_test_loader  = DataLoader(paired_test_data,  batch_size = batchsize, shuffle=False, num_workers=7, persistent_workers=True)
        dataloader_dict = {'train_loader': paired_train_loader,'test_loader':    paired_test_loader,'paired': True}
    return dataloader_dict



''''''''''''''''''''''''
'''   MISCELLANEOUS  '''
''''''''''''''''''''''''

def retrieve_data_process(regenerate_data, clean_train, dataset, these_attack_specs, trigger_generator=None, seed=2575):
    dataset_path = get_dataset_path(dataset, these_attack_specs, clean_train)
    create_nested_folder(dataset_path)
    if regenerate_data == False and os.path.exists(dataset_path)==True:
        with open(dataset_path, 'rb') as f:
            dataset_dict = pickle.load(f)
    else:
        works = False
        while works == False:
            if clean_train==True:
                dataset_dict = get_clean_data(dataset, seed=seed, verbose=True, clean_pyg_process=True, use_edge_attr=False)
            else:
                if these_attack_specs['backdoor_type']=='random':
                    dataset_dict = poison_data_random_attack(dataset, attack_specs=these_attack_specs, seed=seed, verbose=True, clean_pyg_process=True, use_edge_attr=False)
                elif these_attack_specs['backdoor_type']=='clean_label':
                    dataset_dict = poison_data_clean_label_attack(dataset, these_attack_specs, verbose=True, clean=clean_train, clean_pyg_process=True, use_edge_attr=False,seed=seed)
                elif these_attack_specs['backdoor_type']=='adaptive':
                    assert trigger_generator is not None
                    data_dict_clean = get_clean_data(dataset, seed=seed, verbose=True, clean_pyg_process=True, use_edge_attr=False)
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
                    num_train_to_attack = int(these_attack_specs['poison_rate']  * len(train_dataset))
                    random.seed(seed)
                    train_backdoor_indices = random.sample(range(len(train_idx)), k=num_train_to_attack)
                    train_backdoor_indices = [idx for idx in train_backdoor_indices if len(get_possible_new_edges(train_dataset[idx])) > these_attack_specs['trigger_size'] and train_dataset[idx].x.shape[0] < 500]
                    test_backdoor_indices  = [idx for idx in range(len(test_idx))   if len(get_possible_new_edges(test_dataset[idx]))  > these_attack_specs['trigger_size'] and test_dataset[idx].x.shape[0]  < 500]
                    dataset_dict = poison_data_adaptive_attack(trigger_generator, data_dict_clean, train_backdoor_indices, test_backdoor_indices, these_attack_specs['trigger_size'], these_attack_specs['attack_target_label'])
            works = True
    if clean_train==False:
        create_nested_folder(dataset_path)
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset_dict, f)
    return dataset_dict


def gen_epoch_printout(epoch, loss_names, loss_values, saved_this_epoch):
    str_ = f'Epoch {epoch} -- '
    for i, (loss_name, loss_value) in enumerate(zip(loss_names, loss_values)):
        str_ += f'{loss_name}: '
        loss_value_str = str(np.round(loss_value.detach().numpy(),3))
        if i != len(loss_names)-1:
            loss_value_str += ','
        str_ += loss_value_str.ljust(12)
    if saved_this_epoch==True:
        str_ += ' -- saved'
    print(str_)


def save_best_generator(epoch, epoch_losses, save_path, trigger_generator, min_loss):
    if epoch_losses[-1] < min_loss:
        min_loss = epoch_losses[-1]
        with open(save_path, 'wb') as f:
            torch.save(trigger_generator.state_dict(),f)
        return min_loss, True
    else:
        return min_loss, False


def load_model(model_type, dataset, model_path, dataset_dict, hyp_dict, targ, model_hyp_set, num_classes=2, clean=False):
    kwargs  = hyp_dict[dataset][targ][model_hyp_set] if clean==False else hyp_dict[dataset][model_hyp_set]
    kwargs['num_node_features'] = data_shape_dict[dataset]['num_node_features']
    kwargs['num_classes'] = data_shape_dict[dataset]['num_classes']
    loaded_state_backdoor = torch.load(model_path)
    model = model_dict()[model_type](**kwargs)
    model.load_state_dict(loaded_state_backdoor['state_dict'])
    history = loaded_state_backdoor['history']
    return model, history


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


def approve_batch_size(dataset_to_attack,dataset_indices,batch_size):
    max_num_nodes = 0
    for i in range(len(dataset_to_attack)):
        num_nodes_in_graph_i = dataset_to_attack[i].x.shape[0]
        if num_nodes_in_graph_i > max_num_nodes:
            max_num_nodes = num_nodes_in_graph_i
    if batch_size>=len(dataset_indices):
        print(f"Provided batchsize too large -- defaulting to # backdoor samples ({len(dataset_indices)})")
        batch_size = len(dataset_indices)
    return batch_size


def get_class_weights(dataset_dict, clean_or_backdoor='backdoor', num_classes=2):
    if clean_or_backdoor=='clean':
        train_data = Batch.from_data_list([graph.pyg_graph for graph in dataset_dict[f'train_{clean_or_backdoor}_graphs']])
    else:
        train_data = Batch.from_data_list([graph.pyg_graph for graph in dataset_dict[f'train_{clean_or_backdoor}_graphs']])
    count_per_class = [len(torch.where(train_data.y==c)[0]) for c in range(num_classes)]
    count_first_class = count_per_class[0]
    class_weights = torch.tensor([1] + [count_first_class/count_per_class[c] for c in range(1,num_classes)])
    return class_weights


def training_printout(epoch, verbose, train_cln_accs_per_class, train_bd_accs_per_class, test_cln_accs_per_class, test_bd_accs_per_class, train_bd_loss, test_clean_loss, test_bd_loss, train_asr, test_asr):
    if epoch % 10 == 0 and verbose == True:
        acc_printout = f'trn cln/bkd acc: {np.round(np.mean(train_cln_accs_per_class),3)}, {np.round(np.mean(train_bd_accs_per_class),3)},\t'\
                       f'test cln/bkd acc: {np.round(np.mean(test_cln_accs_per_class),3)}, {np.round(np.mean(test_bd_accs_per_class),3)},\t'
        train_asr_str = '' if train_asr is None else str(np.round(train_asr, 3))
        test_asr_str = '' if test_asr is None else str(np.round(test_asr, 3))
        asr_printout = f'ASRs trn/tst: {train_asr_str}, {test_asr_str}'
        print(f'Epoch: {epoch:03d} -- {acc_printout}\t{asr_printout})')


def training_printout_clean(epoch, verbose, train_clean_acc, test_clean_acc, train_clean_loss, test_clean_loss):
    if epoch % 10 == 0 and verbose == True:
        acc_printout = f'Accuracies (train clean, test clean): {train_clean_acc:.3f}, {test_clean_acc:.3f}'
        loss_printout = f'Losses (train clean, test clean): {train_clean_loss:.3f}, {test_clean_loss:.3f}'
        print(f'Epoch: {epoch:03d} -- {acc_printout}\t{loss_printout}')


def get_clean_accurate_indices(model, backdoor_data):
    clean_success_indices = []
    for i in range(len(backdoor_data)):
        g = backdoor_data[i]
        if not g.is_backdoored:
            data = g
            out = model(data.x, data.edge_index, data.batch)
            pred = torch.argmax(out, dim=1)
            if pred == data.y:
                original_index = g.original_index if not isinstance(g.original_index, torch.Tensor) else g.original_index.item()
                clean_success_indices.append(original_index)
    return clean_success_indices


def batch_to_data_list(batch, backdoor=False):
    data_list = []
    for i in range(len(batch.y)):
        if backdoor:
            data = Data(x=batch[i].x,
                        edge_index=batch[i].edge_index,
                        edge_attr=batch[i].edge_attr,
                        is_backdoored = batch[i].is_backdoored,
                        original_index = batch[i].original_index,
                        triggered_edges = batch[i].triggered_edges,
                        y=batch[i].y)
        else:
            data = Data(x=batch[i].x,
                        edge_index=batch[i].edge_index,
                        is_backdoored = batch[i].is_backdoored,
                        original_index = batch[i].original_index,
                        y=batch[i].y)
        data_list.append(data)
    return data_list


def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False


def check_graph_data(graph_data):
    for data in graph_data:
        assert isinstance(data.x, torch.Tensor), f"data.x is not a tensor"
        assert isinstance(data.edge_index, torch.Tensor), f"data.edge_index is not a tensor"
    return True


def get_clean_data(dataset, seed=2575, verbose=False, clean_pyg_process=True, use_edge_attr=False):
    graphs, _ = load_data(dataset, data_type='pyg', cleaned=clean_pyg_process, use_edge_attr=use_edge_attr)
    train_clean_graphs, test_clean_graphs = separate_data(graphs,seed=seed)
    if verbose:
        num_classes = len(set([train_clean_graphs[idx].pyg_graph.y for idx in range(len(train_clean_graphs))]))
        train_clean_label_counts = count_class_samples(train_clean_graphs, num_classes)
        print("label distribution in clean train data:", train_clean_label_counts)
    dataset_dict = {'train_clean_graphs': train_clean_graphs, 'test_clean_graphs': test_clean_graphs}
    create_nested_folder(f'{data_dir}/clean')
    return dataset_dict


def get_nx_graph_node_features(graph, max_features):
    x = torch.zeros((len(graph.nodes()), max_features + 1))
    for i, node in enumerate(graph.nodes()):
        node_degree = min(graph.degree[node], max_features)
        x[i, node_degree] = 1
    return x

def get_train_test_backdoor_indices(all_train, all_test, num_train_to_attack, seed=2575):
    random.seed(seed)
    '''uniform sampling for backdoored samples'''
    train_indices_to_attack = random.sample([idx for idx in range(len(all_train))], k=num_train_to_attack)
    test_indices_to_attack = [idx for idx in range(len(all_test))]
    return train_indices_to_attack, test_indices_to_attack


def get_train_indices_to_attack(train_clean_graphs, attack_specs):
    poison_rate = attack_specs['poison_rate']
    num_train_to_attack = int(poison_rate * len(train_clean_graphs))
    random.seed(2575)
    '''uniform sampling for backdoored samples'''
    train_backdoor_candidates = [idx for idx in range(len(train_clean_graphs)) if is_fully_connected(train_clean_graphs[idx].nx_graph)==False]

    try:
        train_indices_to_attack = random.sample(train_backdoor_candidates, k=num_train_to_attack)
    except:
        train_indices_to_attack = random.sample([idx for idx in range(len(train_clean_graphs))], k=num_train_to_attack)
    return train_indices_to_attack


def udpate_data_from_graph(data, new_graph):
    data = copy.deepcopy(data)
    max_features = data.x.shape[1] - 1
    data.x = get_nx_graph_node_features(new_graph, max_features)
    data.edge_attr = None  
    data.edge_index = torch.Tensor(list(new_graph.edges())).T.int()
    return data


def select_random_nodes(graph, num_nodes):
    if num_nodes > len(graph.nodes()):
        raise ValueError("Number of nodes to select is greater than the number of nodes in the graph.")
    selected_nodes = []
    remaining_nodes = list(graph.nodes())
    random.seed(2575)
    for _ in range(num_nodes):
        if selected_nodes:
            no_edge_nodes = [n for n in remaining_nodes if all(not graph.has_edge(n, m) for m in selected_nodes)]
            if no_edge_nodes:
                new_node = random.choice(no_edge_nodes)
            else:
                new_node = random.choice(remaining_nodes)
        else:
            new_node = random.choice(remaining_nodes)
        selected_nodes.append(new_node)
        remaining_nodes.remove(new_node)
    return selected_nodes


