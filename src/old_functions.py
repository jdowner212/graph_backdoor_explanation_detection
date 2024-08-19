# def n_graphs(dataset_dict, subset='train'):
#     dataset_dict_subset = dataset_dict[f'{subset}_clean_graphs']
#     key_sample = list(dataset_dict_subset.keys())[0]
#     if isinstance(dataset_dict_subset[key_sample], list):
#         return len(dataset_dict_subset[key_sample])
#     else:
#         return len(dataset_dict_subset)


# def n_classes(dataset_dict):
#     train_subset = dataset_dict['train_clean_graphs']
#     test_subset = dataset_dict['test_clean_graphs']
#     key_sample = list(train_subset.keys())[0]
#     if isinstance(train_subset[key_sample], list):
#         train_subset, test_subset = train_subset[key_sample], test_subset[key_sample]
#     all_graphs = train_subset + test_subset
#     if 'train_backdoor_graphs' in dataset_dict.keys():
#         target_label_keys = list(dataset_dict['train_backdoor_graphs'].keys())
#         for t in target_label_keys:
#             all_graphs += dataset_dict['train_backdoor_graphs'][t]
#             all_graphs += dataset_dict['test_backdoor_graphs'][t]
#     ys = [all_graphs[i].pyg_graph.y for i in range(len(all_graphs))]
#     num_classes = len(set(list(ys)))
#     return num_classes


# def n_node_features(dataset_dict):
#     train_subset = dataset_dict['train_clean_graphs']
#     key_sample = list(train_subset.keys())[0]
#     if isinstance(train_subset[key_sample], list):
#         num_node_features = train_subset[key_sample][0].pyg_graph.x.shape[1]
#     else:
#         num_node_features = train_subset[key_sample].pyg_graph.x.shape[1]
#     return num_node_features


# def get_nodemax(dataset_dict):
#     train_subset = dataset_dict['train_clean_graphs']
#     test_subset = dataset_dict['test_clean_graphs']
#     key_sample = list(train_subset.keys())[0]
#     if isinstance(train_subset[key_sample], list):
#         train_subset, test_subset = train_subset[0], test_subset[0]
#     all_graphs = train_subset + test_subset
#     if 'train_backdoor_graphs' in dataset_dict.keys():
#         target_label_keys = list(dataset_dict['train_backdoor_graphs'].keys())
#         for t in target_label_keys:
#             all_graphs += dataset_dict['train_backdoor_graphs'][t]
#             all_graphs += dataset_dict['test_backdoor_graphs'][t]
#     x_shapes = [all_graphs[i].pyg_graph.x.shape[0] for i in range(len(all_graphs))]
#     nodemax = max(x_shapes)
#     return nodemax


# def reindex_nodes(data):
#     unique_nodes = torch.unique(data.edge_index)
#     mapping = {node.item(): i for i, node in enumerate(unique_nodes)}
#     for i in range(data.edge_index.size(1)):
#         data.edge_index[0, i] = mapping[data.edge_index[0, i].item()]
#         data.edge_index[1, i] = mapping[data.edge_index[1, i].item()]
#     return data


# class EdgePerturbation(object):
#     def __init__(self, edge_drop_prob=0.2, edge_add_prob=0.2):
#         self.edge_drop_prob = edge_drop_prob
#         self.edge_add_prob = edge_add_prob
#     def __call__(self, data):
#         edge_mask = torch.rand(data.edge_index.size(1)) > self.edge_drop_prob
#         data.edge_index = data.edge_index[:, edge_mask]
#         num_nodes = data.num_nodes
#         num_edges_add = int(self.edge_add_prob * data.edge_index.size(1))
#         random_edges = torch.randint(0, num_nodes, (2, num_edges_add))
#         data.edge_index = torch.cat([data.edge_index, random_edges], dim=1)
#         return data



# class PrintCallback(L.Callback):
#     def __init__(self,verbose):
#         self.verbose=verbose
#     def on_epoch_end(self, trainer, pl_module):
#         if self.verbose==True:
#             epoch = trainer.current_epoch
#             train_acc,  test_acc  = trainer.train_accs[-1],   trainer.test_accs[-1]
#             train_loss, test_loss = trainer.train_losses[-1], trainer.test_losses[-1]
#             training_printout_clean(epoch, True, train_acc, test_acc, train_loss, test_loss)
#         else:
#             pass


# def train_graph_classifier(model_name, dataset, max_epochs, graph_train_loader, graph_val_loader, graph_test_loader, verbose=True, **kwargs):
#     num_node_features = kwargs['num_node_features']
#     num_classes = kwargs['num_classes']
#     L.seed_everything(42)
#     callbacks = [ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
#                  PrintCallback(verbose=verbose)]
#     trainer = L.Trainer(
#         default_root_dir=root_dir,
#         callbacks=callbacks,
#         accelerator="gpu",
#         max_epochs=max_epochs,
#         enable_progress_bar=False,
#         log_every_n_steps=1
#     )
#     trainer.logger._default_hp_metric = None
#     L.seed_everything(42)
#     sub_model = GraphLevelGNN(
#         c_in=num_node_features,
#         c_out=2 if num_classes == 2 else num_classes,
#         **kwargs,
#     )
#     trainer.fit(sub_model, graph_train_loader, graph_val_loader)
#     sub_model = GraphLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
#     # Test best model on validation and test set
#     train_result = trainer.test(sub_model, dataloaders=graph_train_loader, verbose=False)
#     test_result = trainer.test(sub_model, dataloaders=graph_test_loader, verbose=False)
#     result = {"test": test_result[0]["test_acc"], "train": train_result[0]["test_acc"]}
#     train_losses,   train_accs = sub_model.train_losses,  sub_model.train_accs    
#     test_losses,    test_accs  = sub_model.test_losses,   sub_model.test_accs    
#     model = sub_model.model
#     history = {'train_accs': train_accs,
#                'train_losses': train_losses,
#                'test_clean_accs': test_accs,
#                'test_clean_losses': test_losses}
#     state_dict = {'state_dict': model.state_dict(), 'hyperparameters': {}, 'history': history}
#     return model, state_dict


# def train_loop_clean_2(dataset,these_model_specs, plot, verbose=True, model_path=None, **kwargs):
#     print('kwargs:',kwargs)
#     if plot==False:
#         matplotlib.use('Agg')
#     else:
#         matplotlib.use('nbAgg')
#     batch_size=200
#     max_epochs = 60
#     data = build_dataset_dict(dataset,attack_specs={},clean=True)
#     train_data = [g.pyg_graph for g in data['train_clean_graphs']]
#     test_data = [g.pyg_graph for g in data['test_clean_graphs']]
#     graph_train_loader = geom_data.DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=7,persistent_workers=True)
#     graph_val_loader = geom_data.DataLoader(test_data, batch_size=batch_size,shuffle=False,num_workers=7,persistent_workers=True)  # Additional loader for a larger datasets
#     graph_test_loader = geom_data.DataLoader(test_data, batch_size=batch_size,shuffle=False,num_workers=7,persistent_workers=True)
#     model, state_dict = train_graph_classifier("GraphConv", dataset, max_epochs, graph_train_loader, graph_val_loader, graph_test_loader,
#                            verbose=verbose, **kwargs)
#     if plot == True:
#         accs, losses = [model.train_accs, model.test_accs], [model.train_losses, model.test_losses]
#         plot_training_results(dataset, plot, accs, losses, asrs=None, classifier_hyperparams={}, model_specs=these_model_specs, attack_sepcs=None) 
#     create_nested_folder(f'{train_dir}/{dataset}/models')
#     model.eval()
#     print('\nSaving benign model to',model_path,'\n')
#     torch.save(state_dict, model_path)
#     return model


# def update_best(test_clean_acc, test_bd_acc, test_bd_loss, train_acc, model, epoch, best_val_score):
#     if 0.4 * test_clean_acc + 0.6 * test_bd_acc + 1 / np.sqrt(test_bd_loss) > best_val_score:
#         best_train_acc = train_acc
#         best_test_clean_acc = test_clean_acc
#         best_test_backdoor_acc = test_bd_acc
#         best_val_score = 0.4 * test_clean_acc + 0.6 * test_bd_acc + 1 / np.sqrt(test_bd_loss)
#         best_model = copy.deepcopy(model)
#         best_epoch = epoch
#     return best_model, best_epoch, best_train_acc, best_test_clean_acc, best_test_backdoor_acc


# def train_test_subplot(scores, metric_name, epochs, ax=None, y_lims=[-0.05, 1.05], asrs=None,
#                        clean_or_backdoor='backdoor'):
#     if clean_or_backdoor == 'backdoor':
#         subplot_labels = ['Train', 'Test (clean)', 'Test (backdoor)']
#     elif clean_or_backdoor == 'clean':
#         subplot_labels = ['Train', 'Test']
#     if ax == None:
#         use_ax = False
#         ax = plt
#     else:
#         use_ax = True
#         ax = ax
#     ax.plot(range(epochs), scores[0], label=subplot_labels[0])
#     ax.plot(range(epochs), scores[1], label=subplot_labels[1])
#     if clean_or_backdoor == 'backdoor':
#         ax.plot(range(epochs), scores[2], label=subplot_labels[2])
#         if asrs is not None:
#             ax.plot(range(epochs), asrs[0], label='Train ASR', linestyle='--')
#             ax.plot(range(epochs), asrs[1], label='Test ASR', linestyle='--')
#     if y_lims is None:
#         pass
#     elif isinstance(y_lims, list):
#         if use_ax == True:
#             ax.set_ylim(y_lims[0], y_lims[1])
#         else:
#             ax.ylim(y_lims[0], y_lims[1])
#     if use_ax == True:
#         ax.set_xlabel('Epochs')
#         ax.set_ylabel('Score')
#         ax.set_title(metric_name)
#     else:
#         ax.xlabel('Epochs')
#         ax.ylabel('Score')
#         ax.title(metric_name)
#     ax.legend()


# def plot_training_results(dataset, 
#                           plot,
#                           accs,
#                           losses, 
#                           asrs,
#                           classifier_hyperparams,
#                           model_specs,
#                           attack_specs=None):
 
#     assert dataset is not None
#     assert accs is not None
#     assert losses is not None

#     model_hyp_set, clean_or_backdoor = unpack_kwargs(model_specs, ['model_hyp_set','clean_or_backdoor'])
#     model_type, balanced = unpack_kwargs(classifier_hyperparams,['model_type','balanced'])
#     trigger_size, frac, prob, K, graph_type = unpack_kwargs(attack_specs,['trigger_size','frac','prob','K','graph_type'])
#     epochs = len(accs[0])
#     fig, axs = plt.subplots(nrows=1, ncols=3, sharey=False, figsize=(25, 5))
#     if clean_or_backdoor == 'backdoor':
#         title = f"Backdoor Attack - {dataset}, {graph_type}, trigger_size={trigger_size}, frac={frac}, prob={prob}, K={K}, set {model_hyp_set}"
#         train_test_subplot(accs, 'Accuracy', epochs, ax=axs[0], asrs=asrs, clean_or_backdoor=clean_or_backdoor)
#     else:
#         title = f"Benign Model - {dataset}"
#         train_test_subplot(accs, 'Accuracy', epochs, ax=axs[0], clean_or_backdoor=clean_or_backdoor)
#     train_test_subplot(losses, 'Loss', epochs, ax=axs[1], y_lims=None, clean_or_backdoor=clean_or_backdoor)
#     fig.suptitle(title)
#     if clean_or_backdoor == 'backdoor':
#         image_path = get_training_curve_image_path(dataset, classifier_hyperparams, attack_specs, model_hyp_set)
#     else:
#         plots_folder = f'{train_dir_cln}/{dataset}/plots'
#         create_nested_folder(plots_folder)
#         image_path = f'{plots_folder}/model_type_{model_type}_model_hyp_set_{model_hyp_set}_balanced_{balanced}.png'
#     plt.savefig(image_path)
#     if plot==True:
#         plt.show()
#     else:
#         plt.close()


# def get_training_curve_image_path(dataset, classifier_hyperparams, attack_specs, model_hyp_set):
#     model_name = get_model_name(classifier_hyperparams, attack_specs, model_hyp_set)
#     image_path = f'{train_dir}/{dataset}/plots/{model_name}.png'
#     return image_path


# def train_generator_iterative_loop_continue_after_fail(original_benign_gnn, 
#                                                         path_to_intermediate_generator,
#                                                         generator_optimizer,
#                                                         dataset,
#                                                         attack_target_label,
#                                                         dataset_name,
#                                                         train_dataset,
#                                                         test_dataset,
#                                                         train_backdoor_indices,
#                                                         test_backdoor_indices,
#                                                         trigger_size,
#                                                         generator_path,
#                                                         target_label,
#                                                         remaining_full_rounds,
#                                                         model_kwargs_retrain,
#                                                         generator_hyperparam_dicts,
#                                                         generator_kwargs
#                                                         ):
#     generator_hyperparam_dicts=generator_hyperparam_dicts_iterative_exp
#     ## load generator
#     lr_gen, weight_decay, hidden_dim, depth, dropout_prob = unpack_kwargs(generator_hyperparam_dicts[dataset][attack_target_label], ['lr_gen', 'weight_decay', 'hidden_dim', 'depth', 'dropout_prob'])
#     trigger_generator = EdgeGeneratorHeavy(num_node_features, hidden_dim=hidden_dim,  depth=depth, dropout_prob=dropout_prob)
#     trigger_generator.load_state_dict(torch.load(path_to_intermediate_generator))
#     # Given: clean data, untrained generaor, original benign GNN
#     clean_data = train_dataset, test_dataset
#     current_benign_gnn = copy.deepcopy(original_benign_gnn)
#     num_node_features = train_dataset[0].x.shape[1]
#     generator_optimizer = torch.optim.AdamW(trigger_generator.parameters(), lr=lr_gen, weight_decay=weight_decay)   
#     ## finish current round
#     current_benign_gnn = train_gnn(dataset_name, train_dataset, test_dataset, 'should_not_save_intermediate_gnn', retrain=True, save=False, **model_kwargs_retrain)
#     for round in range(remaining_full_rounds):
#         ''' update M '''
#         if round >= 1:
#             current_benign_gnn = train_gnn(dataset_name, train_dataset, test_dataset, 'should_not_save_intermediate_gnn', retrain=True, save=False, **model_kwargs_retrain)
#         this_generator_path =  generator_path + '_intermediate.ckpt' if round!=remaining_full_rounds-1 else generator_path + '_final.ckpt'
#         # filter dataset for part that we want to generate triggers for
#         train_dataset_for_generator = [train_dataset[idx] for idx in train_backdoor_indices]
#         trigger_generator,generator_optimizer = train_generator_exp_only(train_dataset_for_generator, 
#                                                                          current_benign_gnn, this_generator_path, target_label, generator_kwargs, 
#                                                                          contin_or_scratch='continuous',trigger_generator=trigger_generator,generator_optimizer=generator_optimizer)   
#         ''' get T using current G '''
#         if round != remaining_full_rounds-1:
#             train_dataset_graph_objects, test_dataset_graph_objects = generate_attacked_dataset(trigger_generator, train_dataset, test_dataset, train_backdoor_indices, test_backdoor_indices, trigger_size, target_label)         
#             # check that the data is valid
#             clean_labels = []
#             for i, g in enumerate(train_dataset_graph_objects):
#                 if g.pyg_graph.is_backdoored:
#                     g_clean = clean_data[0][i]
#                     clean_labels.append(g_clean.y.item())
#             if len(set(clean_labels)) == 1:
#                 print("Largest trigger size is too big for dataset -- try limiting to attacks with smaller triggers.")
#                 break
#             train_dataset = [g.pyg_graph for g in train_dataset_graph_objects]
#             test_dataset  = [g.pyg_graph for g in test_dataset_graph_objects]
#     return trigger_generator


# def get_edge_index_plus_new_2(data, all_possible_edges, scores_all_possible_edges, max_num_edges):
#     num_edges_forward = random.choice(list(range(1,max_num_edges)))
#     original_edges_set  = {tuple(edge) for edge in data.edge_index.t().tolist()}
#     original_edges_mask = torch.as_tensor([1 if tuple(all_possible_edges.t().tolist()[i]) in original_edges_set else 0 for i in range(len(all_possible_edges.t().tolist()))])
#     B = torch.ones_like(scores_all_possible_edges).int() - original_edges_mask
#     scores_new_candidates_only = scores_all_possible_edges[B]
#     max_indices     = [scores_new_candidates_only.argmax()] if num_edges_forward==1 else torch.topk(scores_new_candidates_only, num_edges_forward)[1]
#     new_edge_index  = torch.tensor([[],[]],dtype=torch.int64)
#     for max_index in max_indices:
#         add_edge_index = torch.as_tensor(all_possible_edges)[:, max_index]
#         new_edge_index = torch.cat([new_edge_index, add_edge_index.unsqueeze(1)], dim=1)
#     edge_index_plus_new = torch.cat([copy.deepcopy(data.edge_index), new_edge_index], dim=1)
#     return edge_index_plus_new, num_edges_forward, None


# def gumbel_softmax_sample(logits, temperature):
#     gumbel_noise = -torch.log(-torch.log(torch.rand(logits.shape, device=logits.device)))
#     # return torch.nn.functional.softmax((logits + gumbel_noise) / temperature, dim=0)
#     return torch.nn.functional.softmax((logits) / temperature, dim=0)


# def train_generator_2(generator_class, dataset_to_attack, benign_model, save_path, epochs, T, lr_Ma, lr_gen=0.01, lambda_=0.2, weight_decay=0, 
#                     hidden_dim=64, depth=2, dropout_prob=0, batch_size=500, max_num_edges=1, target_label=0):
#     num_node_features = dataset_to_attack[0].x.shape[1]
#     ''' train generator '''
#     max_num_nodes = 0
#     for i in range(len(dataset_to_attack)):
#         num_nodes_in_graph_i = dataset_to_attack[i].x.shape[0]
#         if num_nodes_in_graph_i > max_num_nodes:
#             max_num_nodes = num_nodes_in_graph_i
#     try:
#         trigger_generator = generator_class(num_node_features, hidden_dim=hidden_dim,  depth=depth, dropout_prob=dropout_prob)
#     except:
#         trigger_generator = generator_class(num_node_features)
#     generator_optimizer = torch.optim.AdamW(trigger_generator.parameters(), lr=lr_gen, weight_decay=weight_decay)
#     batch_loss_gnn, batch_exp_loss, min_loss, epoch_losses = 0, 0, 1e8, []
#     freeze_parameters(benign_model)
#     for epoch in range(epochs):
#         dataset_indices = random.sample(range(len(dataset_to_attack)), len(dataset_to_attack))
#         epoch_loss_gnn, epoch_exp_loss = 0,0
#         for c, i in enumerate(dataset_indices):
#             if (c+1)%batch_size==0 or c==dataset_indices[-1]:
#                 batch_loss_gnn, batch_exp_loss = 0, 0          
#             data = copy.deepcopy(dataset_to_attack[i])
#             edge_scores, possible_edges = trigger_generator(data)
#             edge_index_plus_new     = add_to_edge_index(data, edge_scores, possible_edges, max_num_edges)
#             new_edge_probabilities  = gumbel_softmax(edge_scores, temperature=0.5, hard=True)
#             existing_edges_mask     = get_existing_edge_mask(data.edge_index, possible_edges)
#             edge_weights            = new_edge_probabilities + existing_edges_mask
#             updated_x               = update_x_(data, edge_index_plus_new)
#             data.x, data.edge_index = updated_x, edge_index_plus_new
#             # Obtain explanation given trigger
#             Ma = edge_scores + torch.randn_like(edge_scores)
#             for t in range(T):
#                 Ma.retain_grad()
#                 Ma_clone = Ma.clone()
#                 L = explainer_forward_adaptive(benign_model, updated_x, edge_index_plus_new, target_label, Ma_clone, possible_edges, algorithm_version=2)
#                 L.backward(retain_graph=True)
#                 with torch.no_grad():
#                     Ma -= lr_Ma * Ma.grad
#                     Ma.grad.zero_()
#                 print(f'Epoch {epoch}: {c}/{len(dataset_indices)} samples processed',t,np.round(L.item(),5), end='\r')
#             # Explainer performance
#             sample_loss_exp = explainer_forward_adaptive(benign_model, updated_x, edge_index_plus_new, target_label, Ma_clone, possible_edges, algorithm_version=2)
#             batch_exp_loss += sample_loss_exp
#             epoch_exp_loss += sample_loss_exp
#             # Classifier loss
#             benign_model.eval()
#             out = benign_model.model(x=updated_x, edge_index=possible_edges, batch_idx=None, edge_weight=edge_weights, use_edge_weight=True).squeeze(dim=-1)
#             sample_loss_gnn = torch.nn.CrossEntropyLoss()(out, torch.tensor([target_label]))
#             batch_loss_gnn += sample_loss_gnn
#             epoch_loss_gnn += sample_loss_gnn
#             if (c+1)%batch_size==0 or c==dataset_indices[-1]:
#                 generator_optimizer.zero_grad()
#                 n = 1 if (c==0) else batch_size if ((c+1)%batch_size==0 and c!=0) else len(dataset_indices)%batch_size
#                 batch_loss = batch_loss_gnn/n - lambda_*batch_exp_loss/n
#                 batch_loss.backward()
#                 generator_optimizer.step()
#                 epoch_losses.append(batch_loss)     
#         gen_epoch_printout(epoch, ['loss_gnn','-neg_exp_loss','total'], [epoch_loss_gnn, -lambda_*epoch_exp_loss, epoch_loss_gnn-lambda_*epoch_exp_loss])
#         save_best_generator(epoch, epoch_losses, save_path, trigger_generator, min_loss)   
#     return trigger_generator


# def train_generator_2_new(generator_class, dataset_to_attack, benign_model, save_path, epochs, T, lr_Ma, lr_gen=0.01, lambda_=0.2, weight_decay=0, 
#                     hidden_dim=64, depth=2, dropout_prob=0, batch_size=500, max_num_edges=1, target_label=0):
#     # torch.autograd.set_detect_anomaly(True)
#     num_node_features = dataset_to_attack[0].x.shape[1]
#     ''' train generator '''
#     max_num_nodes = 0
#     for i in range(len(dataset_to_attack)):
#         num_nodes_in_graph_i = dataset_to_attack[i].x.shape[0]
#         if num_nodes_in_graph_i > max_num_nodes:
#             max_num_nodes = num_nodes_in_graph_i
#     dataset_indices = list(range(len(dataset_to_attack)))
#     if batch_size>=len(dataset_indices):
#         print(f"Provided batchsize too large -- defaulting to # backdoor samples ({len(dataset_indices)})")
#         batch_size = len(dataset_indices)
#     try:
#         trigger_generator = generator_class(num_node_features, hidden_dim=hidden_dim,  depth=depth, dropout_prob=dropout_prob)
#     except:
#         trigger_generator = generator_class(num_node_features)
#     generator_optimizer = torch.optim.AdamW(trigger_generator.parameters(), lr=lr_gen, weight_decay=weight_decay)
#     batch_loss_gnn, batch_loss_exp, min_loss = 0, 0, 1e8
#     freeze_parameters(benign_model)
#     epoch_losses = []
#     for epoch in range(epochs):
#         random.shuffle(dataset_indices)
#         epoch_loss_gnn, epoch_loss_exp = 0,0
#         for c, i in enumerate(dataset_indices):
#             num_added = random.choice(list(range(1,max_num_edges))) if max_num_edges>1 else 1
#             new_edge_index  = torch.tensor([[],[]],dtype=torch.int64)
#             if (c+1)%batch_size==0 or i==dataset_indices[-1]:
#                 batch_loss_gnn, batch_loss_exp = 0, 0
#             data = copy.deepcopy(dataset_to_attack[i])
#             original_edges_set  = {tuple(edge) for edge in data.edge_index.t().tolist()}
#             all_scores, all_possible_edges = trigger_generator(data)
#             all_scores=all_scores.sigmoid()
#             all_possible_edges_list = all_possible_edges.t().tolist()
#             A = torch.zeros(len(all_possible_edges_list))
#             B = torch.zeros(len(all_possible_edges_list))
#             for i in range(len(all_possible_edges_list)):
#                 e0,e1 = tuple(all_possible_edges_list[i])
#                 if (e0,e1) in original_edges_set or (e1,e0) in original_edges_set:
#                     A[i]=1
#                 elif (e0,e1) not in original_edges_set and (e1,e0) not in original_edges_set and e0!=e1:
#                     B[i]=1
#             C = torch.zeros_like(A)    # mask for actually added edges
#             scores_new_candidates_only = all_scores*B
#             num_added = min(num_added, sum(A==0))
#             max_indices = [scores_new_candidates_only.argmax()] if num_added==1 else torch.topk(scores_new_candidates_only, num_added)[1]
#             for max_index in max_indices:
#                 assert A[max_index]==0
#                 C[max_index]=1
#                 add_edge_index = torch.as_tensor(all_possible_edges)[:, max_index]
#                 new_edge_index = torch.cat([new_edge_index, add_edge_index.unsqueeze(1)], dim=1)
#             scores_original = all_scores*A
#             scores_new      = all_scores*C
#             scores_joint    = scores_original + scores_new
#             edge_index_plus_new = torch.cat([copy.deepcopy(data.edge_index), new_edge_index], dim=1)
#             updated_x           = update_x_2(data.x, edge_index_plus_new)
#             edge_mask_adversarial = gumbel_softmax_2(scores_joint,  sum(A==1)+sum(C==1),    temperature=0.5, hard=True)
#             Ma = scores_joint
#             for t in range(T):
#                 Ma.retain_grad()
#                 Ma_clone = Ma.clone()
#                 benign_model.eval()
#                 out = benign_model.model(x=updated_x, edge_index=all_possible_edges, batch_idx=None, edge_weight=Ma_clone.sigmoid()*B, use_edge_weight=True)
#                 out = out.squeeze(dim=-1)
#                 L_exp = torch.nn.CrossEntropyLoss()(out, torch.tensor([target_label]))
#                 L_exp.backward(retain_graph=True)
#                 with torch.no_grad():
#                     Ma -= lr_Ma * Ma.grad
#                     Ma.grad.zero_()
#                 print(f'Epoch {epoch}: {c}/{len(dataset_indices)} samples processed',t,np.round(L_exp.item(),5), end='\r')
#             # Explainer performance
#             benign_model.eval()
#             out = benign_model.model(x=updated_x, edge_index=all_possible_edges, batch_idx=None, edge_weight=Ma_clone.sigmoid()*B, use_edge_weight=True)
#             out = out.squeeze(dim=-1)
#             sample_loss_exp = torch.nn.CrossEntropyLoss()(out, torch.tensor([target_label]))
#             batch_loss_exp += sample_loss_exp
#             epoch_loss_exp += sample_loss_exp
#             # Classifier loss
#             benign_model.eval()
#             out = benign_model.model(x=updated_x, edge_index=all_possible_edges, batch_idx=None, edge_weight=edge_mask_adversarial, use_edge_weight=True)
#             out = out.squeeze(dim=-1)
#             sample_loss_gnn = torch.nn.CrossEntropyLoss()(out, torch.tensor([target_label]))
#             batch_loss_gnn += sample_loss_gnn
#             epoch_loss_gnn += sample_loss_gnn
#             if (c+1)%batch_size==0 or i==dataset_indices[-1]:
#                 generator_optimizer.zero_grad()
#                 n = 1 if (c==0) else batch_size if ((c+1)%batch_size==0 and c!=0) else len(dataset_indices)%batch_size
#                 loss = batch_loss_gnn/n - lambda_*batch_loss_exp/n
#                 loss.backward()
#                 generator_optimizer.step()
#         epoch_loss = epoch_loss_gnn - lambda_*epoch_loss_exp
#         epoch_losses.append(epoch_loss)
#         min_loss, saved_this_epoch = save_best_generator(epoch, epoch_losses, save_path, trigger_generator, min_loss)
#         gen_epoch_printout(epoch, ['loss_gnn','neg_loss_exp','total'], [epoch_loss_gnn, -lambda_*epoch_loss_exp, epoch_loss], saved_this_epoch)
#     return trigger_generator


# def train_generator(generator_class, dataset_to_attack, benign_model, save_path, epochs, T, lr_Ma, lr_gen=0.01, lambda_=0.2, weight_decay=0, 
#                     hidden_dim=64, depth=2, dropout_prob=0, batch_size=500, max_num_edges=1, target_label=0, algorithm_version=1, get_degree_change=False):
#     num_node_features = dataset_to_attack[0].x.shape[1]
#     switch_to_explainer_epoch=20
#     # ''' train generator '''
#     max_num_nodes = 0
#     for i in range(len(dataset_to_attack)):
#         num_nodes_in_graph_i = dataset_to_attack[i].x.shape[0]
#         if num_nodes_in_graph_i > max_num_nodes:
#             max_num_nodes = num_nodes_in_graph_i
#     try:
#         trigger_generator = generator_class(num_node_features, hidden_dim=hidden_dim,  depth=depth, dropout_prob=dropout_prob)
#     except:
#         trigger_generator = generator_class(num_node_features)
#     generator_optimizer = torch.optim.AdamW(trigger_generator.parameters(), lr=lr_gen, weight_decay=weight_decay)
#     batch_loss_gnn, batch_exp_mask_sum, batch_exp_loss = 0, 0, 0
#     min_gnn_loss, min_exp_loss = 1e8, 1e8
#     freeze_parameters(benign_model)
#     gnn_epoch_losses, exp_epoch_losses = [],[]
#     best_model = None
#     for epoch in range(epochs):
#         switch_to_explainer_condition = epoch > 1
#         if switch_to_explainer_condition==True:
#             trigger_generator = best_model
#         dataset_indices = list(range(len(dataset_to_attack)))
#         random.shuffle(dataset_indices)
#         epoch_loss_gnn, epoch_exp_mask_sum, epoch_exp_loss = 0,0,0
#         generator_optimizer.zero_grad()
#         for c, i in enumerate(dataset_indices):
#             data = copy.deepcopy(dataset_to_attack[i])
#             edge_scores, possible_edges = trigger_generator(data)
#             if max_num_edges==1:
#                 new_edge_index = torch.as_tensor(possible_edges)[:, edge_scores.argmax()]
#                 edge_index_plus_new = torch.cat([data.edge_index, new_edge_index.unsqueeze(1)], dim=1)
#             elif max_num_edges>1:
#                 num_edges_forward = random.choice(list(range(1,max_num_edges)))
#                 _, indices = torch.topk(edge_scores, num_edges_forward)
#                 for index in indices:
#                     new_edge_index = torch.as_tensor(possible_edges)[:, index]
#                     data.edge_index = torch.cat([data.edge_index, new_edge_index.unsqueeze(1)], dim=1)
#                 edge_index_plus_new = data.edge_index
#             new_edge_probabilities  = gumbel_softmax(edge_scores, temperature=0.5, hard=True)
#             existing_edges_mask     = get_existing_edge_mask(data.edge_index, possible_edges)
#             edge_weights            = new_edge_probabilities + existing_edges_mask
#             updated_x               = update_x_(data, edge_index_plus_new)
#             data.x, data.edge_index = updated_x, edge_index_plus_new
#             benign_model.eval()
#             out = benign_model.model(x=updated_x, edge_index=possible_edges, batch_idx=None, edge_weight=edge_weights, use_edge_weight=True)
#             out = out.squeeze(dim=-1)
#             sample_loss_gnn = torch.nn.CrossEntropyLoss()(out, torch.tensor([target_label]))
#             batch_loss_gnn += sample_loss_gnn
#             epoch_loss_gnn += sample_loss_gnn
#             if (c+1)%batch_size==0 or c==dataset_indices[-1]:
#                 n = 1 if (c==0) else batch_size if ((c+1)%batch_size==0 and c!=0) else len(dataset_indices)%batch_size
#                 loss_gnn_avg = batch_loss_gnn/n
#                 gnn_epoch_losses.append(loss_gnn_avg)
#             if switch_to_explainer_condition == False: 
#                 # Classification
#                 if (c+1)%batch_size==0 or c==dataset_indices[-1]:
#                     n = 1 if (c==0) else batch_size if ((c+1)%batch_size==0 and c!=0) else len(dataset_indices)%batch_size
#                     loss=loss_gnn_avg
#                     loss.backward()
#                     generator_optimizer.step()
#                     generator_optimizer.zero_grad()
#                     batch_loss_gnn, batch_exp_mask_sum, batch_exp_loss = 0, 0, 0
#             elif switch_to_explainer_condition == True:
#                 # Explanation
#                 Ma = edge_scores + torch.randn_like(edge_scores)
#                 for t in range(T):
#                     Ma.retain_grad()
#                     Ma_clone = Ma.clone()
#                     L = explainer_forward_adaptive(benign_model, updated_x, edge_index_plus_new, target_label, Ma_clone, possible_edges, algorithm_version=algorithm_version)
#                     L.backward(retain_graph=True)
#                     with torch.no_grad():
#                         Ma -= lr_Ma * Ma.grad
#                         Ma.grad.zero_()
#                     print(f'Epoch {epoch}: {c}/{len(dataset_indices)} samples processed',t,np.round(L.item(),5), end='\r')
#                 if algorithm_version==1:
#                     sigmoid_Ma = torch.sigmoid(Ma)
#                     Ma_times_B = get_Ma_times_B(edge_index_plus_new, sigmoid_Ma, possible_edges.t())
#                     sample_mask_sum = torch.sum(Ma_times_B)
#                     batch_exp_mask_sum += sample_mask_sum
#                     epoch_exp_mask_sum += sample_mask_sum
#                     exp_metric, epoch_exp_performance = ('exp_mask_sum',epoch_exp_mask_sum)
#                 elif algorithm_version==2:
#                     sample_loss_exp = explainer_forward_adaptive(benign_model, updated_x, edge_index_plus_new, target_label, Ma_clone, possible_edges, algorithm_version=algorithm_version)
#                     batch_exp_loss += sample_loss_exp
#                     epoch_exp_loss += sample_loss_exp
#                     exp_metric, epoch_exp_performance = ('neg_exp_loss',-epoch_exp_loss)
#                 if (c+1)%batch_size==0 or c==dataset_indices[-1]:
#                     n = 1 if (c==0) else batch_size if ((c+1)%batch_size==0 and c!=0) else len(dataset_indices)%batch_size
#                     if algorithm_version==1:
#                         mask_sum_avg = lambda_ * batch_exp_mask_sum/n
#                         loss = mask_sum_avg
#                         loss.backward()
#                         generator_optimizer.step()
#                         exp_epoch_losses.append(loss)
#                     elif algorithm_version==2:
#                         neg_loss_exp_avg = - lambda_ * batch_exp_loss/n
#                         loss = neg_loss_exp_avg
#                         loss.backward()
#                         generator_optimizer.step()
#                         exp_epoch_losses.append(neg_loss_exp_avg)
#                     batch_loss_gnn, batch_exp_mask_sum, batch_exp_loss = 0, 0, 0
#                     generator_optimizer.zero_grad()
#         if switch_to_explainer_condition == False:
#             epoch_printout=f'GNN Epoch {epoch}, loss_gnn: {epoch_loss_gnn}'
#             print(epoch_printout)
#             gnn_loss_last_3_epochs = torch.mean(torch.as_tensor(gnn_epoch_losses[-3:]))
#             if gnn_loss_last_3_epochs < min_gnn_loss:
#                 min_gnn_loss = gnn_loss_last_3_epochs
#                 best_model = trigger_generator
#         if switch_to_explainer_condition == True:
#             save=False
#             exp_loss_last_3_epochs = torch.mean(torch.as_tensor(exp_epoch_losses[-3:]))
#             gnn_loss_last_3_epochs = torch.mean(torch.as_tensor(gnn_epoch_losses[-3:]))
#             loss_last_3_epochs = gnn_loss_last_3_epochs + exp_loss_last_3_epochs
#             if loss_last_3_epochs < min_exp_loss:
#                 save=True
#                 min_exp_loss = loss_last_3_epochs
#                 with open(save_path, 'wb') as f:
#                     torch.save(trigger_generator.state_dict(),f)
#             epoch_printout=f'GNN Epoch {epoch}, loss_gnn: {epoch_loss_gnn}, Exp Epoch {epoch}, {exp_metric}: {epoch_exp_performance}'
#             if save==True:
#                 epoch_printout += ' -- save' 
#             print(epoch_printout)
#     return trigger_generator


# def inject_optimized_backdoor_trigger(trigger_generator, graphs, trigger_size, indices_to_attack, attack_target_label, tag2index):
#     graphs = copy.deepcopy(graphs)
#     for idx in indices_to_attack:
#         triggered_edges = []
#         for t in trigger_size:
#             edge_scores, possible_edges = trigger_generator(graphs[idx].pyg_graph)
#             new_edge_index      = torch.as_tensor(possible_edges)[:, edge_scores.argmax()]
#             edge_index_plus_new = torch.cat([graphs[idx].pyg_graph.edge_index, new_edge_index.unsqueeze(1)], dim=1)
#             new_x = update_x_(graphs[idx].pyg_graph, edge_index_plus_new)
#             [n0,n1] = edge_index_plus_new.t().tolist()
#             triggered_edges.append((n0,n1))
#             graphs[idx].pyg_graph.edge_index = new_edge_index
#             graphs[idx].pyg_graph.x = new_x
#             graphs[idx].nx_graph.add_edge(n0,n1)
#         graphs[idx].pyg_graph.triggered_edges = triggered_edges

#         ''' defaulting degree as tag to True'''
#         graphs[idx].pyg_graph.y = attack_target_label#.type(torch.long)
#         max_tag = graphs[idx].pyg_graph.x.shape[1] - 1
#         node_tags = [tag if tag <= max_tag else max_tag for tag in list(dict(graphs[idx].nx_graph.degree).values())]
#         graphs[idx].pyg_graph.x = torch.zeros(len(node_tags), len(tag2index))
#         graphs[idx].pyg_graph.x[range(len(node_tags)), [tag2index[tag] for tag in node_tags]] = 1
#         graphs[idx].pyg_graph.is_backdoored = 1
#         if len(graphs[idx].nx_graph.nodes()) != len(graphs[idx].pyg_graph.x):
#             print('after injecting trigger: trouble at index', idx)

#         assert graphs[idx].pyg_graph.y is not None
#     return graphs


# def build_dataset_dict_optimized(trigger_generator_dict, train_dataset, test_dataset, train_backdoor_indices, test_backdoor_indices, trigger_size):
#     print('build_dataset_dict_optimized')
#     num_classes = len(trigger_generator_dict.keys())
#     dataset_dict = {'train_backdoor_graphs': [], 'train_clean_graphs': [], 'test_backdoor_graphs': [], 'test_clean_graphs': [], 'trigger_graphs': []}
#     # clean
#     for target_label in range(num_classes):#[0,1]:
#         for (group, subset) in [('train',train_dataset), ('test',test_dataset)]:
#             graphs = []
#             for i, data in enumerate(subset):
#                 g = GraphObject(None, data.y.long(), node_features=data.x, edge_features=[], edge_index=data.edge_index, is_backdoored=False, original_index=i, triggered_edges=[])
#                 g.pyg_graph.x = data.x
#                 g.pyg_graph.edge_index = data.edge_index
#                 g.nx_graph = nx.from_edgelist(data.edge_index.t().tolist())
#                 graphs.append(g)
#             dataset_dict[f'{group}_clean_graphs'].append(graphs)
#     dataset_dict['train_backdoor_graphs'] = copy.deepcopy(dataset_dict['train_clean_graphs'])
#     dataset_dict['test_backdoor_graphs']  = copy.deepcopy(dataset_dict['test_clean_graphs'])
#     print('len train_backdoor_indices:',len(train_backdoor_indices))
#     target_labels = list(range(num_classes))
#     for target_label in target_labels:
#         print(f'len all train {target_labels}:',len([d for d in range(len(dataset_dict['train_clean_graphs'][target_label])) if dataset_dict['train_clean_graphs'][target_label][d].pyg_graph.y==target_label]))
#         # backdoor
#         trigger_generator = trigger_generator_dict[f'generator_{target_label}']
#         for (group, indices) in [('train',train_backdoor_indices), ('test',test_backdoor_indices)]:
#             graphs = []
#             for i in indices:
#                 g = copy.deepcopy(dataset_dict[f'{group}_backdoor_graphs'][target_label][i])
#                 triggered_edges = []
#                 for t in range(trigger_size):
#                     edge_scores, possible_edges = trigger_generator(g.pyg_graph)
#                     new_edge_index      = torch.as_tensor(possible_edges)[:, edge_scores.argmax()]
#                     edge_index_plus_new = torch.cat([g.pyg_graph.edge_index, new_edge_index.unsqueeze(1)], dim=1)
#                     new_x = update_x_(g.pyg_graph, edge_index_plus_new)
#                     [n0,n1] = new_edge_index.t().tolist()
#                     triggered_edges.append((n0,n1))
#                     g.label = target_label
#                     g.pyg_graph.edge_index = edge_index_plus_new
#                     g.pyg_graph.x = new_x
#                     g.pyg_graph.y = torch.tensor([target_label])
#                     g.nx_graph.add_edge(n0,n1)
#                 g.pyg_graph.is_backdoored=True
#                 g.pyg_graph.triggered_edges = triggered_edges
#                 dataset_dict[f'{group}_backdoor_graphs'][target_label][i] = g
#     return dataset_dict


# def build_dataset_dict_adaptive_start_to_end(dataset, attack_target_labels, trigger_size):
#     trigger_sizes = [trigger_size]
#     generator_class_name = 'heavy'
#     generator_hyperparam_dicts_iterative_exp = get_info('generator_hyperparam_dicts_iterative_exp')
#     gen_alg_v = 3
#     contin_or_scratch = 'continuous'
#     these_attack_specs = build_attack_specs()
#     these_attack_specs['backdoor_type'] = 'optimized'
#     these_attack_specs['optimized']='optimized'
#     these_attack_specs['frac'] = 0.2
#     num_classes       = data_shape_dict[dataset]['num_classes']
#     num_node_features = data_shape_dict[dataset]['num_node_features']
#     adapt_gen_dir = get_info('adapt_gen_dir')
#     dataset_name = dataset
#     max_degree_dict = {'MUTAG':5, 'AIDS':7, 'PROTEINS': 26, 'IMDB-BINARY': 136, 'COLLAB': 49, 'REDDIT-BINARY': 3063, 'DBLP': 36}
#     max_degree = max_degree_dict[dataset_name]
#     if dataset != 'DBLP':
#         transform = Compose([DeduplicateEdges(), OneHotDegree(max_degree=max_degree-1,cat=False)])
#         tu_dataset = torch_geometric.datasets.TUDataset(root=os.path.join(data_dir,'clean'), name=dataset_name, transform=transform)
#     else:
#         ''' need to apply de-duplicate edegs to DBLP data!'''
#         with open(f'{data_dir}/clean/DBLP_data_list_random_5000.pkl', 'rb') as f:
#             tu_dataset = pickle.load(f)
#     labels = [tu_dataset[i].y.item() for i in range(len(tu_dataset))]
#     train_idx, test_idx = get_train_test_idx(labels,seed=2575,fold_idx=0)
#     train_dataset = [tu_dataset[idx] for idx in train_idx]
#     test_dataset  = [tu_dataset[idx] for idx in test_idx]
#     print("Length:", len(tu_dataset))
#     data_labels = [tu_dataset[idx].y.item() for idx in range(len(tu_dataset))]
#     print(f"Average label: {np.mean(data_labels):4.2f}")
#     '''uniform sampling for backdoored samples'''
#     num_train_to_attack = int(these_attack_specs['frac']  * len(train_dataset))
#     random.seed(2575)
#     train_backdoor_indices = random.sample(range(len(train_idx)), k=num_train_to_attack)
#     train_backdoor_indices = [idx for idx in train_backdoor_indices if len(get_possible_new_edges(train_dataset[idx])) > max(trigger_sizes) and train_dataset[idx].x.shape[0] < 500]
#     test_backdoor_indices  = [idx for idx in range(len(test_idx))   if len(get_possible_new_edges(test_dataset[idx]))  > max(trigger_sizes) and test_dataset[idx].x.shape[0]  < 500]
#     generator_name_dict = {'regular': EdgeGenerator, 'heavy': EdgeGeneratorHeavy}
#     generator_hyperparam_dicts= generator_hyperparam_dicts_iterative_exp
#     generator_dict = {f'generator_{i}': None for i in range(num_classes)}
#     for attack_target_label in attack_target_labels:
#         print('attack target label',attack_target_label)
#         generator_checkpoint_path = os.path.join(adapt_gen_dir, dataset_name, f"Trigger_Generator_{dataset_name}_target_label_{attack_target_label}_alg_v_{gen_alg_v}_exp_only_{contin_or_scratch}_final.ckpt")
#         generator_class_name,  hidden_dim, depth, dropout_prob = unpack_kwargs(generator_hyperparam_dicts[dataset][attack_target_label], ['generator_class', 'hidden_dim', 'depth', 'dropout_prob'])
#         generator_class = generator_name_dict[generator_class_name]
#         trigger_generator = generator_class(num_node_features, hidden_dim=hidden_dim,  depth=depth, dropout_prob=dropout_prob)
#         print('loading generator')
#         ''' load pre-trained generator '''
#         assert os.path.isfile(generator_checkpoint_path)
#         print(f"Found pretrained edge generator {attack_target_label}, loading...")
#         trigger_generator.load_state_dict(torch.load(generator_checkpoint_path))
#         print('saving generator to dictionary')
#         generator_dict[f'generator_{attack_target_label}']=trigger_generator
#     dataset_dict_optimized = build_dataset_dict_optimized(generator_dict, train_dataset, test_dataset, train_backdoor_indices, test_backdoor_indices, trigger_size)
#     return dataset_dict_optimized


# def find_motifs(graph, motif_size=3):
#     """Find all motifs of a given size in the graph."""
#     motifs = []
#     nodes = list(graph.nodes())
#     for combo in combinations(nodes, motif_size):
#         subgraph = graph.subgraph(combo)
#         if nx.is_connected(subgraph):
#             motifs.append(subgraph)
#     return motifs


# def select_motif(graphs, motif_size=3):
#     """Select a rare motif to use as a trigger."""
#     motif_counts = {}
#     for graph in graphs:
#         motifs = find_motifs(graph.nx_graph, motif_size)
#         for motif in motifs:
#             motif_tuple = tuple(sorted(motif.edges()))
#             if motif_tuple not in motif_counts:
#                 motif_counts[motif_tuple] = 0
#             motif_counts[motif_tuple] += 1
#     sorted_motifs = sorted(motif_counts.items(), key=lambda item: item[1])
#     selected_motif_edges = sorted_motifs[0][0]
#     selected_motif = nx.Graph()
#     selected_motif.add_edges_from(selected_motif_edges)
#     return selected_motif


# def calculate_degree_centrality(graph):
#     """Calculate degree centrality for all nodes in the graph."""
#     return nx.degree_centrality(graph)


# def filter_based_on_graph_structure(graphs, k):
#     """Filter and select k most important nodes based on degree centrality."""
#     candidate_nodes = {}
#     for graph in graphs:
#         dc = calculate_degree_centrality(graph.nx_graph)
#         sorted_nodes = sorted(dc, key=dc.get, reverse=True)[:k]
#         candidate_nodes[graph] = sorted_nodes
#     return candidate_nodes


# def calculate_subscore(graph, shadow_model, candidate_nodes):
#     """Calculate the importance of each candidate node by dropping them and measuring the impact on the model output."""
#     subscores = {}
#     for node in candidate_nodes:
#         modified_graph = graph.nx_graph.copy()
#         modified_graph.remove_node(node)
#         data = Data(x=graph.pyg_graph.x, edge_index=torch.tensor(list(modified_graph.edges())).T)
#         original_output = shadow_model(graph.pyg_graph.x, graph.pyg_graph.edge_index)
#         modified_output = shadow_model(data.x, data.edge_index)
#         subscore = torch.abs(original_output - modified_output).sum().item()
#         subscores[node] = subscore
#     return subscores


# def filter_based_on_gnn_model(graphs, shadow_model, candidate_nodes):
#     """Filter and select the most influential nodes based on the shadow model feedback."""
#     trigger_nodes = {}
#     for graph in graphs:
#         subscores = calculate_subscore(graph, shadow_model, candidate_nodes[graph])
#         sorted_nodes = sorted(subscores, key=subscores.get, reverse=True)
#         trigger_nodes[graph] = sorted_nodes
#     return trigger_nodes


# def attack_dataset_with_optimized_motif(dataset,
#                                         graphs, motif, attack_target_label, num_features, num_classes, 
#                                         tag2index,
#                                         these_classifier_hyperparams,
#                                         these_attack_specs,
#                                         these_model_specs,
#                                         hyp_dict,
#                                         regenerate_data,
#                                         verbose=True):
#     # Filter based on graph structure
#     candidate_nodes = filter_based_on_graph_structure(graphs, k=10)
#     # Construct shadow model
#     dataset_dict_clean = retrieve_data_process(regenerate_data, True, dataset, None, False)
#     class_weights = get_class_weights(dataset_dict_clean, attack_target_label, 'clean', num_classes) if these_classifier_hyperparams['balanced'] else None
#     dataloader_dict = get_dataloader_dict(None, dataset_dict_clean, these_model_specs, these_classifier_hyperparams)
#     shadow_model = train_loop_clean(dataset, dataloader_dict, hyp_dict, num_features, class_weights, these_model_specs, num_classes, False, model_hyp_set=these_model_specs['model_hyp_set'], verbose=True, model_path=None)
#     # Filter based on GNN model
#     trigger_nodes = filter_based_on_gnn_model(graphs, shadow_model, candidate_nodes)
#     # Inject motif into the selected nodes
#     poisoned_graphs = []
#     for graph in graphs:
#         graph_copy = copy.deepcopy(graph)
#         for node in trigger_nodes[graph]:
#             if len(motif.nodes()) >= len(graph_copy.nx_graph.nodes()):
#                 graph_copy = replace_whole_graph([graph_copy], node, motif)
#             else:
#                 graph_copy = replace_part_graph([graph_copy], node, motif)
#         graph_copy.pyg_graph.y = attack_target_label
#         poisoned_graphs.append(graph_copy)
#     # Separate data into train and test sets
#     train_clean_graphs, test_clean_graphs = separate_data(graphs)
#     train_indices_to_attack = get_train_indices_to_attack(train_clean_graphs, these_attack_specs)
#     test_indices_to_attack = [idx for idx in range(len(test_clean_graphs))]
#     train_backdoor_graphs, test_backdoor_graphs = set_backdoor(motif, train_clean_graphs, test_clean_graphs, train_indices_to_attack, test_indices_to_attack, attack_target_label, these_attack_specs['frac'], tag2index, verbose)
#     train_backdoor_label_counts = count_class_samples(train_backdoor_graphs, num_classes)
#     if verbose:
#         print("Label distribution in backdoor train data:", train_backdoor_label_counts)
#         print("# train data triggers:", len(train_indices_to_attack))
#         print("# test data triggers:", len(test_indices_to_attack))
#     return train_backdoor_graphs, train_clean_graphs, test_backdoor_graphs, test_clean_graphs, motif


# class EdgeGeneratorHeavy(torch.nn.Module):
#     def __init__(self, node_features, hidden_dim=64, depth=2, dropout_prob=0):
#         super(EdgeGeneratorHeavy, self).__init__()
#         self.fc_input = torch.nn.Linear(node_features * 2, hidden_dim)
#         self.fc_hidden = torch.nn.ModuleList([
#             torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(depth - 1)
#         ])
#         self.fc_output = torch.nn.Linear(hidden_dim, 1)
#         self.activation = torch.nn.ReLU()
#         self.dropout = torch.nn.Dropout(dropout_prob)
#         self.batch_norms = torch.nn.ModuleList([
#             torch.nn.BatchNorm1d(hidden_dim) for _ in range(depth - 1)
#         ])
#     def forward(self, data, batch=None):
#         x = data.x
#         possible_edges = get_possible_new_edges(data)
#         for edge in data.edge_index.t().tolist():
#             assert (edge[0],edge[1]) not in possible_edges
#         possible_edges = torch.tensor(possible_edges, dtype=torch.long).t().contiguous()
#         edge_features  = torch.cat((x[possible_edges[0]], x[possible_edges[1]]), dim=1)
#         edge_features = self.fc_input(edge_features)
#         edge_features = self.activation(edge_features)
#         edge_features = self.dropout(edge_features)
#         for fc_hidden, batch_norm in zip(self.fc_hidden, self.batch_norms):
#             edge_features = fc_hidden(edge_features)
#             edge_features = batch_norm(edge_features)
#             edge_features = self.activation(edge_features)
#             edge_features = self.dropout(edge_features)
#         edge_scores = self.fc_output(edge_features).squeeze()
#         return edge_scores, possible_edges


# def evaluate_and_log(history, model, train_loader, test_loader, criterion, attack_target_label, attack_type, epoch):
#     print(f"Epoch {epoch}:")
#     train_results = test(model, train_loader, criterion, attack_target_label, attack_type)
#     test_results = test(model, test_loader, criterion, attack_target_label, attack_type)
#     for key, result in zip(history.keys(), train_results + test_results):
#         history[key].append(result)
#     return history


# def update_history_with_none(history):
#     for key in history.keys():
#         history[key].append(None)
#     return history


# def interpolate_history(history):
#     for key, values in history.items():
#         history[key] = list(pd.Series(values).interpolate())
#     return history


# def train_gnn(dataset_name, train_dataset, test_dataset, benign_filename, retrain=False, save=True,**model_kwargs):
#     model = train_benign(dataset_name, train_dataset, test_dataset, benign_filename, retrain=retrain, save=save,**model_kwargs)
#     return model


# def get_existing_edge_mask(edge_index, possible_edges):
#     existing_edges_set = {tuple(edge) for edge in edge_index.t().tolist()}
#     existing_edges_mask = torch.tensor([1 if tuple(edge) in existing_edges_set else 0 for edge in possible_edges.t().tolist()])
#     return existing_edges_mask


# def gumbel_softmax(logits, temperature, hard=False):
#     y_soft = torch.nn.functional.softmax((logits) / temperature, dim=0)
#     if hard:
#         y_hard = torch.zeros_like(logits).scatter_(0, y_soft.argmax(dim=0, keepdim=True), 1.0)
#         y = y_hard - y_soft.detach() + y_soft
#     else:
#         y = y_soft
#     return y


# def explainer_forward_adaptive(benign_model, updated_x, edge_index_plus_new, target_label, Ma, possible_edges, algorithm_version=1):
#     combined_edge_index, combined_edge_weights = apply_mask_to_edges(edge_index_plus_new, Ma, possible_edges, algorithm_version=algorithm_version)
#     benign_model.eval()
#     out = benign_model.model(x=updated_x, edge_index=combined_edge_index, batch_idx=None, edge_weight=combined_edge_weights, use_edge_weight=True)
#     out = out.squeeze(dim=-1)
#     L = torch.nn.CrossEntropyLoss()(out, torch.tensor([target_label]))
#     return L


# def explainer_forward_adaptive_2(benign_model, updated_x, edge_index_plus_new, target_label, Ma, possible_new_edges, algorithm_version=1):
#     weights_from_mask = apply_mask_to_edges_2(edge_index_plus_new, Ma, possible_new_edges, algorithm_version=algorithm_version)
#     benign_model.eval()
#     out = benign_model.model(x=updated_x, edge_index=possible_new_edges, batch_idx=None, edge_weight=weights_from_mask, use_edge_weight=True)
#     out = out.squeeze(dim=-1)
#     L = torch.nn.CrossEntropyLoss()(out, torch.tensor([target_label]))
#     return L


# def explainer_forward_adaptive_3(benign_model, updated_x, edge_index_plus_new, target_label, Ma, possible_new_edges, algorithm_version=1):
#     weights_from_mask = apply_mask_to_edges_3(edge_index_plus_new, Ma, possible_new_edges, algorithm_version=algorithm_version)
#     benign_model.eval()
#     out = benign_model.model(x=updated_x, edge_index=possible_new_edges, batch_idx=None, edge_weight=weights_from_mask, use_edge_weight=True)
#     out = out.squeeze(dim=-1)
#     L = torch.nn.CrossEntropyLoss()(out, torch.tensor([target_label]))
#     return L


# def pretty_print(tensor):
#     as_np = tensor.detach().numpy()
#     rounded = np.round(as_np,2)
#     print(rounded)


# def train_generator_1(generator_class, dataset_to_attack, benign_model, save_path, epochs, T, lr_Ma, lr_gen=0.01, lambda_=0.2, weight_decay=0, 
#                     hidden_dim=64, depth=2, dropout_prob=0, batch_size=500, max_num_edges=1, target_label=0):
#     # torch.autograd.set_detect_anomaly(True)
#     num_node_features = dataset_to_attack[0].x.shape[1]
#     ''' train generator '''
#     max_num_nodes = 0
#     for i in range(len(dataset_to_attack)):
#         num_nodes_in_graph_i = dataset_to_attack[i].x.shape[0]
#         if num_nodes_in_graph_i > max_num_nodes:
#             max_num_nodes = num_nodes_in_graph_i
#     dataset_indices = list(range(len(dataset_to_attack)))[:5]
#     if batch_size>=len(dataset_indices):
#         print(f"Provided batchsize too large -- defaulting to # backdoor samples ({len(dataset_indices)})")
#         batch_size = len(dataset_indices)
#     try:
#         trigger_generator = generator_class(num_node_features, hidden_dim=hidden_dim,  depth=depth, dropout_prob=dropout_prob)
#     except:
#         trigger_generator = generator_class(num_node_features)
#     generator_optimizer = torch.optim.AdamW(trigger_generator.parameters(), lr=lr_gen, weight_decay=weight_decay)
#     batch_loss_gnn, batch_exp_mask_sum, min_loss = 0, 0, 1e8
#     freeze_parameters(benign_model)
#     epoch_losses = []
#     for epoch in range(epochs):
#         random.shuffle(dataset_indices)
#         epoch_loss_gnn, epoch_exp_mask_sum = 0,0
#         for c, i in enumerate(dataset_indices):
#             num_added = random.choice(list(range(1,max_num_edges))) if max_num_edges>1 else 1
#             new_edge_index  = torch.tensor([[],[]],dtype=torch.int64)
#             if (c+1)%batch_size==0 or i==dataset_indices[-1]:
#                 batch_loss_gnn, batch_exp_mask_sum = 0, 0
#             data = copy.deepcopy(dataset_to_attack[i])
#             original_edges_set  = {tuple(edge) for edge in data.edge_index.t().tolist()}
#             all_scores, all_possible_edges = trigger_generator(data)
#             all_scores=all_scores.sigmoid()      
#             all_possible_edges_list = all_possible_edges.t().tolist()
#             A = torch.zeros(len(all_possible_edges_list))
#             B = torch.zeros(len(all_possible_edges_list))
#             for i in range(len(all_possible_edges_list)):
#                 e0,e1 = tuple(all_possible_edges_list[i])
#                 if (e0,e1) in original_edges_set or (e1,e0) in original_edges_set:
#                     A[i]=1
#                 elif (e0,e1) not in original_edges_set and (e1,e0) not in original_edges_set and e0!=e1:
#                     B[i]=1
#             C = torch.zeros_like(A) 
#             scores_new_candidates_only = all_scores*B
#             num_added = min(num_added, sum(A==0))
#             max_indices = [scores_new_candidates_only.argmax()] if num_added==1 else torch.topk(scores_new_candidates_only, num_added)[1]
#             for max_index in max_indices:
#                 assert A[max_index]==0
#                 C[max_index]=1
#                 add_edge_index = torch.as_tensor(all_possible_edges)[:, max_index]
#                 new_edge_index = torch.cat([new_edge_index, add_edge_index.unsqueeze(1)], dim=1)
#             scores_original = all_scores*A
#             scores_new      = all_scores*C
#             scores_joint    = scores_original + scores_new
#             edge_index_plus_new = torch.cat([copy.deepcopy(data.edge_index), new_edge_index], dim=1)
#             updated_x           = update_x_2(data.x, edge_index_plus_new)
#             edge_mask_adversarial   = gumbel_softmax_2(scores_joint,    sum(A==1)+sum(C==1),    temperature=0.5, hard=True)
#             Ma = scores_joint
#             for t in range(T):
#                 Ma.retain_grad()
#                 Ma_clone = Ma.clone()
#                 benign_model.eval()
#                 out = benign_model.model(x=updated_x, edge_index=all_possible_edges, batch_idx=None, edge_weight=Ma_clone.sigmoid(), use_edge_weight=True)
#                 out = out.squeeze(dim=-1)
#                 L = torch.nn.CrossEntropyLoss()(out, torch.tensor([target_label]))
#                 L.backward(retain_graph=True)
#                 with torch.no_grad():
#                     Ma -= lr_Ma * Ma.grad
#                     Ma.grad.zero_()
#                 print(f'Epoch {epoch}: {c}/{len(dataset_indices)} samples processed',t,np.round(L.item(),5), end='\r')
#             sample_mask_sum = torch.sum(Ma.sigmoid()*B)
#             batch_exp_mask_sum += sample_mask_sum
#             epoch_exp_mask_sum += sample_mask_sum
#             # Classifier loss
#             benign_model.eval()
#             out = benign_model.model(x=updated_x, edge_index=all_possible_edges, batch_idx=None, edge_weight=edge_mask_adversarial, use_edge_weight=True)
#             out = out.squeeze(dim=-1)
#             sample_loss_gnn = torch.nn.CrossEntropyLoss()(out, torch.tensor([target_label]))
#             batch_loss_gnn += sample_loss_gnn
#             epoch_loss_gnn += sample_loss_gnn
#             if (c+1)%batch_size==0 or i==dataset_indices[-1]:
#                 generator_optimizer.zero_grad()
#                 n = 1 if (c==0) else batch_size if ((c+1)%batch_size==0 and c!=0) else len(dataset_indices)%batch_size
#                 loss = lambda_*batch_exp_mask_sum/n
#                 loss.backward()
#                 generator_optimizer.step()
#         epoch_loss = epoch_loss_gnn+lambda_*epoch_exp_mask_sum
#         epoch_losses.append(epoch_loss)
#         min_loss, saved_this_epoch = save_best_generator(epoch, epoch_losses, save_path, trigger_generator, min_loss)
#         gen_epoch_printout(epoch, ['loss_gnn','epoch_exp_mask_sum','total'], [epoch_loss_gnn, lambda_*epoch_exp_mask_sum, epoch_loss], saved_this_epoch)
#     return trigger_generator


# def add_to_edge_index(data, edge_scores, possible_edges, max_num_edges):
#     if max_num_edges==1:
#         new_edge_index = torch.as_tensor(possible_edges)[:, edge_scores.argmax()]
#         edge_index_plus_new = torch.cat([data.edge_index, new_edge_index.unsqueeze(1)], dim=1)
#     elif max_num_edges>1:
#         num_edges_forward = random.choice(list(range(1,max_num_edges)))
#         _, indices = torch.topk(edge_scores, num_edges_forward)
#         for index in indices:
#             new_edge_index = torch.as_tensor(possible_edges)[:, index]
#             data.edge_index = torch.cat([data.edge_index, new_edge_index.unsqueeze(1)], dim=1)
#         edge_index_plus_new = data.edge_index
#     return edge_index_plus_new

# def get_edge_index_plus_new(data, possible_edges, edge_scores, max_num_edges):
#     num_edges_forward = random.choice(list(range(1,max_num_edges)))
#     edge_index_plus_new = copy.deepcopy(data.edge_index)
#     scores_only_preserved_edges = torch.zeros_like(edge_scores)
#     if num_edges_forward==1:
#         max_indices = [edge_scores.argmax()]
#     else:
#         _, max_indices = torch.topk(edge_scores, num_edges_forward)
#     scores_only_preserved_edges[max_indices]=edge_scores[max_indices]
#     new_edge_index = torch.tensor([[],[]],dtype=torch.int64)
#     for max_index in max_indices:
#         add_edge_index = torch.as_tensor(possible_edges)[:, max_index]
#         new_edge_index = torch.cat([new_edge_index, add_edge_index.unsqueeze(1)], dim=1)
#     edge_index_plus_new = torch.cat([copy.deepcopy(data.edge_index), new_edge_index], dim=1)
#     return edge_index_plus_new, num_edges_forward, scores_only_preserved_edges


# def gumbel_softmax_2(logits, num_edges_forward, temperature, hard=False):
#     y_soft = torch.nn.functional.softmax((logits) / temperature, dim=0)
#     if hard:
#         # Instead of creating a one-hot vector for the maximum value, create a tensor that has ones for the top num_edges_forward values
#         _, top_indices = torch.topk(y_soft, num_edges_forward, dim=0)
#         y_hard = torch.zeros_like(logits)
#         y_hard.scatter_(0, top_indices, 1.0)
#         # Use straight-through estimator for backprop
#         y = y_hard - y_soft.detach() + y_soft
#     else:
#         y = y_soft
#     return y


# def data_list_from_text_files(root, dataset_name, sample_size=None, save=False):
#     def calculate_degrees(edge_index, num_nodes):
#         degree = torch.zeros(num_nodes, dtype=torch.int32)
#         for node in edge_index[0]:
#             degree[node] += 1
#         return degree
#     edges = pd.read_csv(f'{root}/{dataset_name}_A.txt', header=None, sep=',')
#     graph_indicator = pd.read_csv(f'{root}/{dataset_name}_graph_indicator.txt', header=None)
#     graph_labels = pd.read_csv(f'{root}/{dataset_name}_graph_labels.txt', header=None)
#     node_labels = pd.read_csv(f'{root}/{dataset_name}_node_labels.txt', header=None)
#     data_list = []
#     N = graph_labels.shape[0]
#     indices_to_use = range(N)
#     max_degree = 0
#     all_degrees = []
#     if sample_size is not None:
#         print('Taking random sample of size',sample_size)
#         indices_to_use = random.sample(range(N),sample_size)
#     for c, i in enumerate(indices_to_use):
#         print(f'Processing graph {c}/{len(indices_to_use)}', end='\r')
#         node_indices = graph_indicator[graph_indicator[0] == i].index
#         node_idx_map = {idx: i for i, idx in enumerate(node_indices)}
#         graph_edges = edges[edges[0].isin(node_indices + 1) & edges[1].isin(node_indices + 1)]
#         graph_edges = graph_edges.applymap(lambda x: node_idx_map[x - 1])
#         edge_index = torch.tensor(graph_edges.values, dtype=torch.long).t().contiguous()
#         num_nodes = len(node_indices)
#         degrees = calculate_degrees(edge_index, num_nodes)
#         all_degrees.append(degrees)
#         max_degree = max(max_degree, degrees.max().item())
#         y = torch.tensor(graph_labels.iloc[i - 1].values, dtype=torch.long)
#         data = Data(edge_index=edge_index, y=y)
#         data_list.append(data)
#     # One-hot degrees as features
#     for data, degrees in zip(data_list, all_degrees):
#         degrees_int = degrees.to(torch.int32)
#         data.x = torch.nn.functional.one_hot(degrees_int, num_classes=max_degree + 1).float()
#     if save==True:
#         print(f"Saving data list at {src_dir}/DBLP_data_list_random_{sample_size}.pkl")
#         with open(f'{src_dir}/DBLP_data_list_random_{sample_size}.pkl', 'wb') as f:
#             pickle.dump(data_list,f)
#     return data_list


# def is_subgraph(graph, subgraph):
#     GM = isomorphism.GraphMatcher(graph, subgraph)
#     return GM.subgraph_is_isomorphic()


# def num_graph_nodes(graph):
#     return torch.max(graph.edge_mat).numpy() + 1


# def augment_data_object(data_object, x_rate, edge_rate,samples_to_augment):
#     data_object = augment_edge_indices(data_object, edge_rate,samples_to_augment)
#     data_object = augment_x(data_object, x_rate,samples_to_augment)
#     return data_object


# def augment_edge_indices(data_object, rate, samples_to_augment):
#     indices_to_augment = random.sample(range(len(data_object)),k=samples_to_augment)
#     indices_to_augment.sort()
#     for i in indices_to_augment:
#         edges = data_object.edge_index
#         # add
#         num_edges = edges.shape[1]
#         high = int(rate*num_edges) if int(rate*num_edges) > 0 else 1
#         number_to_add = np.random.randint(low=0,high=high)
#         edges = add_edges(edges, number_to_add)
#         # remove
#         num_edges = edges.shape[1]
#         high = int(rate*num_edges) if int(rate*num_edges) > 0 else 1
#         number_to_remove = np.random.randint(low=0,high=high)
#         edges = remove_edges(edges, number_to_remove)
#         # perturb
#         num_edges = edges.shape[1]
#         high = int(rate*num_edges) if int(rate*num_edges) > 0 else 1
#         number_to_perturb = np.random.randint(low=0,high=high)
#         edges = perturb_edges(edges, number_to_perturb)
#         data_object[i].edge_index = edges
#     return data_object


# def augment_x(data_object, rate,samples_to_augment):
#     indices_to_augment = random.sample(range(len(data_object)),k=samples_to_augment)
#     indices_to_augment.sort()
#     for i in indices_to_augment:
#         noise = torch.randn_like(data_object.x[i]) * rate
#         data_object.x[i] += noise
#     return data_object


# def add_edges(edge_index, num_edges_to_add):
#     num_nodes = len(torch.unique(edge_index))
#     """Randomly adds edges to the edge_index tensor."""
#     new_edges = torch.randint(0, num_nodes, (2, num_edges_to_add))
#     return torch.cat([edge_index, new_edges], dim=1)


# def remove_edges(edge_index, num_edges_to_remove):
#     """Randomly removes edges from the edge_index tensor."""
#     num_edges = edge_index.size(1)
#     indices_to_keep = torch.randperm(num_edges)[:num_edges - num_edges_to_remove].tolist()
#     indices_to_keep.sort()
#     return edge_index[:, indices_to_keep]


# def perturb_edges(edge_index, num_edges_to_perturb):
#     """Randomly perturbs edges in the edge_index tensor."""
#     num_nodes = len(torch.unique(edge_index))
#     num_edges = edge_index.size(1)
#     indices_to_perturb = torch.randperm(num_edges)[:num_edges_to_perturb].tolist()
#     indices_to_perturb.sort()
#     new_edges = torch.randint(0, num_nodes, (2, num_edges_to_perturb))
#     edge_index[:, indices_to_perturb] = new_edges
#     return edge_index


# def get_all_nx_subgraphs(nx_graph, max_subgraph_size=None):
#     if max_subgraph_size == None:
#         max_subgraph_size = len(nx_graph.nodes)
#     G = nx_graph.copy()
#     subgraphs = []
#     for i in range(1, len(max_subgraph_size) + 1):
#         for subset in itertools.combinations(max_subgraph_size, i):
#             subgraph = G.subgraph(subset)
#             subgraphs.append(subgraph.copy())
#     return subgraphs


# def parse_settings_from_df_index(df, row_i):
#     df = copy.copy(df).reset_index()    
#     attack_columns = ['Attack Target Label', 'Graph Type', 'Trigger Size','Prob', 'Frac', 'K']
#     attack_column_numbers = [list(df.columns).index(index_name) for index_name in attack_columns]
#     attack_values = df.iloc[row_i,attack_column_numbers]
#     attack_specs = build_attack_specs()
#     attack_specs = update_kwargs(attack_specs,['attack_target_label','graph_type','trigger_size','prob','frac','K'], attack_values)
#     explanation_columns_A = ['Explanation Type', 'Explanation Target Label', 'Explain LR','Explain Epochs']
#     explanation_column_numbers_A = [list(df.columns).index(index_name) for index_name in explanation_columns_A]
#     explanation_values_A = df.iloc[row_i, explanation_column_numbers_A]
#     explanation_columns_B = ['Node Reduce', 'Node Ent', 'Node Size', 'Edge Reduce', 'Edge Ent', 'Edge Size']
#     explanation_column_numbers_B = [list(df.columns).index(index_name) for index_name in explanation_columns_B]
#     explanation_values_B = df.iloc[row_i, explanation_column_numbers_B]
#     coeffs = dict(zip(['node_feat_reduction','node_feat_ent','node_feat_size','edge_reduction','edge_ent','edge_size'],explanation_values_B))
#     explainer_hyperparams = build_explainer_hyperparams()
#     explainer_hyperparams = update_kwargs(explainer_hyperparams,['explanation_type','explanation_target_label','explain_lr','explainer_epochs'], explanation_values_A)
#     explainer_hyperparams['coeffs'] = coeffs
#     model_hyp_set = df.loc[row_i,'Model Hyp Set']
#     balanced = df.loc[row_i, 'Balanced']
#     model_specs = build_model_specs()
#     model_specs = update_kwargs(model_specs,['model_hyp_set','balanced'],[model_hyp_set, balanced])
#     classifier_hyperparams = hyp_dict_backdoor[df.loc[row_i,'Dataset']][attack_specs['attack_target_label']][model_hyp_set]
#     return attack_specs, explainer_hyperparams, model_specs, classifier_hyperparams


# def extract_values(model_name):
#     terms = ['model_type','target_class', 'graph_type', 'trigger_size', 'frac', 'prob', 'K', 'model_hyp_set', 'balanced']
#     pattern = r'(?:' + '|'.join(terms) + r')_(.*?)(?=' + '|'.join(['_' + t for t in terms]) + '_|.pth|$)'
#     values = re.findall(pattern, model_name)
#     if 'prob' not in model_name:
#         values = values[:5] + [None] + values[5:]
#     if 'frac' not in model_name:
#         values = values[:4] + [None] + values[4:]
#     if 'K' not in model_name:
#         values = values[:6] + [None] + values[6:]
#     return values


# def n_unique_graph_perturbations(g, n, verbose=False):
#     new_graphs = []
#     past_perturbations = []
#     for i in range(n):
#         already_exists = True
#         n_attempts = 0
#         while already_exists == True and n_attempts < 50:
#             new_graph, action, feature_type, feat = randomly_perturb_graph(g)
#             perturbation = f'{action} {feature_type} {feat}'
#             already_exists = True if perturbation in past_perturbations else False
#             n_attempts += 1
#         if already_exists == True:
#             print(f'Graph too small to obtain {n} unique perturbations.')
#             return new_graphs
#         else:
#             if verbose == True:
#                 print(f'{action} {feature_type} {feat}')
#             new_graphs.append(new_graph)
#             past_perturbations.append(perturbation)
#     return new_graphs


# def randomly_perturb_graph(graph):
#     graph = copy.deepcopy(graph)
#     feature_type = 'edge' if np.random.uniform() > 0.5 else 'node'
#     drop_or_add = 'add' if np.random.uniform() > 0.5 else 'drop'
#     if drop_or_add == 'drop':
#         if feature_type == 'edge':
#             index_to_drop = np.random.choice(range(len(graph.edges())))
#             (n0, n1) = list(graph.edges())[index_to_drop]
#             feat = (n0, n1)
#             graph.remove_edge(n0, n1)
#         elif feature_type == 'node':
#             node_to_drop = np.random.choice(graph.nodes())
#             feat = node_to_drop
#             graph.remove_node(node_to_drop)
#     elif drop_or_add == 'add':
#         if feature_type == 'edge':
#             edge_already_exists = True
#             while edge_already_exists == True:
#                 nodes = copy.deepcopy(list(graph.nodes()))
#                 node_1 = np.random.choice(nodes)
#                 nodes.remove(node_1)
#                 node_2 = np.random.choice(nodes)
#                 if (node_1, node_2) in graph.edges() or (node_2, node_1) in graph.edges():
#                     edge_already_exists = True
#                 else:
#                     edge_already_exists = False
#             feat = (node_1, node_2)
#             graph.add_edge(node_1, node_2)
#         elif feature_type == 'node':
#             max_node = max(graph.nodes())
#             new_node = max_node + 1
#             feat = new_node
#             graph.add_node(new_node)
#     mapping = {node: i for i, node in enumerate(graph.nodes())}
#     graph = nx.relabel_nodes(graph, mapping)
#     action = 'Added' if drop_or_add == 'add' else 'Dropped'
#     return graph, action, feature_type, feat

# def scale_node_mask(node_mask, feature_scale):
#     node_mask = node_mask.clone().detach()
#     for j in range(len(feature_scale)):
#         node_mask[:,j] *= feature_scale[j]
#     node_mask = min_max_scaling(node_mask, small_difference=0)
#     node_mask[node_mask<0.9]=0
#     return 


# def difference_node_feature_weights(backdoor_dict, clean_dict):
#     backdoor_node_masks = []
#     for original_index in backdoor_dict.keys():
#         node_mask = backdoor_dict[original_index]['explanation'].node_mask.clone().detach()
#         for row in node_mask:
#             backdoor_node_masks.append(row.tolist())
#     backdoor_node_masks = torch.tensor(backdoor_node_masks).T
#     backdoor_node_masks = torch.sum(backdoor_node_masks, dim=1)
#     backdoor_node_masks = torch.tensor([torch.sum(row) for row in backdoor_node_masks])
#     backdoor_node_masks = backdoor_node_masks/sum(backdoor_node_masks)
#     clean_node_masks = []
#     for k in clean_dict.keys():
#         node_mask = clean_dict[k]['explanation'].node_mask.clone().detach()
#         for row in node_mask:
#             clean_node_masks.append(row.tolist())
#     clean_node_masks = torch.tensor(clean_node_masks).T
#     clean_node_masks = torch.tensor([torch.sum(row) for row in clean_node_masks])
#     clean_node_masks = clean_node_masks/sum(clean_node_masks)
#     difference = torch.tensor([b/c for (b,c) in zip(backdoor_node_masks, clean_node_masks)])


# def aggregate_node_feature_weights(subset, data_dict, explanation_dictionary, target_label, original_indices, min_max_scale=False):
#     original_indices = list(explanation_dictionary.keys()) if original_indices is None else original_indices
#     t = int(target_label)
#     n_features        =  data_dict[subset][t][0].pyg_graph.x.shape[1]
#     all_node_masks    = torch.cat([explanation_dictionary[original_index]['explanation'].node_mask.clone().detach() for original_index in original_indices], dim=0)
#     agg_feat_weights  = torch.sum(all_node_masks,dim=0)
#     print('agg_feat_weights before loop:',agg_feat_weights)
#     min_ = 1e15
#     for j in range(n_features):
#         nonzero=  torch.nonzero(all_node_masks[:, j])
#         num_nonzero_entries = len(nonzero)
#         print('num_nonzero_entries:',num_nonzero_entries)
#         agg_feat_weights[j] = agg_feat_weights[j]/num_nonzero_entries
#         if torch.isnan(agg_feat_weights[j])==False and agg_feat_weights[j] < min_:
#             min_ = agg_feat_weights[j]
#             print("passes -- min_ =",min_)
#     agg_feat_weights = torch.nan_to_num(agg_feat_weights, min_)
#     if min_max_scale==True:
#        agg_feat_weights = min_max_scaling(agg_feat_weights, small_difference=0)
#     return agg_feat_weights, explanation_dictionary


# def plot_multiple_explanations(dataset,
#                                dataset_dict_backdoor,
#                                dataset_dict_clean,                  
#                                backdoor_model,     
#                                use_indices_clean,      
#                                use_indices_backdoor,    
#                                model_specs, 
#                                attack_specs, 
#                                explainer_hyperparams,
#                                classifier_hyperparams,          
#                                show_clean=True,    
#                                do_loss=False,      
#                                loss_types=[],              
#                                print_scores=True,               
#                                clean_subset = 'train_clean_graphs', 
#                                backdoor_subset = 'train_backdoor_graphs', 
#                                show_curvature=True,   
#                                show_connectivity=True,              
#                                title=None,
#                                which_explainer='gnn',
#                                repeat_explanation_n=1):
#     loss_config = [do_loss, loss_types]
#     clean_dictionary, backdoor_dictionary =  explain_multiple(backdoor_model, 
#                                                               dataset_dict_backdoor, 
#                                                               dataset_dict_clean,
#                                                               dataset, 
#                                                               backdoor_subset, 
#                                                               clean_subset, 
#                                                               use_indices_backdoor, 
#                                                               use_indices_clean, 
#                                                               show_clean, 
#                                                               attack_specs, 
#                                                               model_specs, 
#                                                               explainer_hyperparams, 
#                                                               classifier_hyperparams,
#                                                               which_explainer=which_explainer,
#                                                               repeat_explanation_n=repeat_explanation_n)
#     attack_target_label = attack_specs['attack_target_label']
#     ''' To view successful clean explanation '''
#     clean_locations_and_indices     = [(i, g.pyg_graph.original_index) for (i,g) in enumerate(dataset_dict_clean[clean_subset])    if g.pyg_graph.original_index in use_indices_clean]
#     ''' To view clean counterpart to successful backdoor explanation '''
#     backdoor_locations_and_indices  = [(i, g.pyg_graph.original_index) for (i,g) in enumerate(dataset_dict_backdoor[backdoor_subset]) if g.pyg_graph.original_index in use_indices_backdoor]
#     for locations_and_indices in zip(clean_locations_and_indices, backdoor_locations_and_indices):
#         clean_explanation,clean_elbow,clean_curvature = None, None,None
#         clean_location,     clean_original_index    = locations_and_indices[0][0], locations_and_indices[0][1]
#         backdoor_location,  backdoor_original_index = locations_and_indices[1][0], locations_and_indices[1][1]
#         clean_graph     = dataset_dict_clean[clean_subset][clean_location]
#         backdoor_graph  = dataset_dict_backdoor[backdoor_subset][backdoor_location]
#         clean_inputs = [None,None,None]
#         if show_clean==True:
#             clean_explanation  =  clean_dictionary[clean_original_index]['explanation'].clone().detach()
#             clean_inputs       = [clean_explanation, clean_graph, None]
#         backdoor_explanation   =  backdoor_dictionary[backdoor_original_index]['explanation'].clone().detach()
#         backdoor_inputs        = [backdoor_explanation, backdoor_graph, dataset_dict_backdoor['trigger_graphs']]
#         if print_scores==True:
#             print_explanation_scores(backdoor_dictionary[backdoor_original_index], clean_dictionary[clean_original_index], backdoor_label = attack_target_label)
#         if show_curvature==True:
#             if clean_explanation is not None:
#                 clean_elbow, clean_curvature = get_elbow_curvature(clean_explanation['clf_loss_over_time'])
#             backdoor_elbow, backdoor_curvature = get_elbow_curvature(backdoor_explanation['clf_loss_over_time'])
#         clean_connectivity, backdoor_connectivity = None, None
#         if show_connectivity==True:
#             if show_clean==True:
#                 clean_connectivity = get_connectivity(dataset_dict_clean, 'train_clean_graphs', attack_target_label, clean_explanation, clean_location)
#             backdoor_connectivity = get_connectivity(dataset_dict_backdoor, 'train_backdoor_graphs',   attack_target_label, backdoor_explanation, backdoor_location)
#         plot_explanation_results(explainer_hyperparams, backdoor_inputs, clean_inputs, loss_config, backdoor_elbow, backdoor_curvature, clean_elbow, clean_curvature, clean_connectivity, backdoor_connectivity, title)
#     return clean_dictionary, backdoor_dictionary


# def pie_bar_plots(relevant_metrics, dfs,legend=False,titles=None):
#     relevant_metrics = get_metric_names() if relevant_metrics is None else relevant_metrics
#     cols   = relevant_metrics
#     names = [metric_plot_info_dict[m]['Title'] + ' Distance' if 'dist' in m else metric_plot_info_dict[m]['Title'] for m in relevant_metrics]
#     palette = {'F1 >= 0.9': 'g', '0.7 <= F1 < 0.9': 'lime', '0.4 <= F1 < 0.7': 'orange', 'F1 < 0.4': 'r'}
#     fig = plt.figure(figsize=(18, 2.45 * len(dfs)), tight_layout=True) 
#     outer_gs = gridspec.GridSpec(len(dfs), 1, figure=fig)
#     for idx,df in enumerate(dfs):
#         inner_gs = gridspec.GridSpecFromSubplotSpec(3, 8, subplot_spec=outer_gs[idx], height_ratios=[0.5, 0.5, 0.5],hspace=1.7) 
#         title_ax = fig.add_subplot(inner_gs[0, :])
#         title_ax.set_title(titles[idx], y=-1,fontsize=20)
#         title_ax.axis('off')
#         count_dict = {i: {0.9: 0, 0.7: 0, 0.4: 0, -100: 0} for i in range(len(cols))}
#         ax1 = fig.add_subplot(inner_gs[1,0:2])
#         ax2 = fig.add_subplot(inner_gs[1,2:4])
#         ax3 = fig.add_subplot(inner_gs[1,4:6])
#         ax4 = fig.add_subplot(inner_gs[1,6:8])
#         ax5 = fig.add_subplot(inner_gs[2,1:3])
#         ax6 = fig.add_subplot(inner_gs[2,3:5])
#         ax7 = fig.add_subplot(inner_gs[2,5:7])
#         axs = [ax1,ax2,ax3,ax4,ax5,ax6,ax7]
#         for config_name in set(df['config_name']):
#             df_subset = df[df['config_name']==config_name]
#             if len(df_subset) == 0:
#                 pass
#             else:
#                 for j, (col,title_) in enumerate(zip(relevant_metrics, names)):
#                     tp = sum(df_subset[col + ' clean_val'] =='True Positive')
#                     fp = sum(df_subset[col + ' clean_val'] =='False Positive')
#                     tn = sum(df_subset[col + ' clean_val'] =='True Negative')
#                     fn = sum(df_subset[col + ' clean_val'] =='False Negative')
#                     if tp+0.5*(tp+fn)>0:
#                         f1 = tp/(tp+0.5*(fp+fn))
#                     else:
#                         f1 = 0
#                     for thresh in count_dict[j].keys():
#                         if f1 >= thresh:
#                             count_dict[j][thresh] += 1
#                             break
#         for j, (col,title_) in enumerate(zip(relevant_metrics, names)):
#             if sum(count_dict[j].values())==0:
#                 pass
#             else:
#                 counts = list(count_dict[j].values())
#                 ax      = axs[j]
#                 ax.set_title(title_, fontsize=16)
#                 total = sum(counts)
#                 percentages = [count / total for count in counts]
#                 left = 0
#                 for k, percentage in enumerate(percentages):
#                     color = ['g','lime','orange','red'][k]
#                     ax.barh(y=0, width=percentage, left=left, color=color, edgecolor='white')
#                     left += percentage
#                     if percentage > 0:#10:
#                         ha = 'center'#'left' if percentage < 1 and k==0 else 'center'
#                         x = left - (percentage / 2) - 0.03 if percentage < 1 and k==0 else left - (percentage / 2)+0.01
#                         text = f'{percentage * 100:.0f}%' if percentage *100 > 1 else '<1%'
#                         ax.text(x, -0.8, text, ha=ha, va='center', color='black', fontsize=14)
#                 ax.set_xlim(0, 1)
#                 ax.axis('off')

#         if legend==True and idx == len(dfs)-1:
#             legend_labels   = list(palette.keys())
#             legend_handles  = [Patch(facecolor=palette[key], label=key) for key in legend_labels]
#             fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=[0.5,-0.1], fancybox=True, shadow=True, ncol=4,fontsize=16)#, title='TPR-FPR')
#     plt.tight_layout()
#     plt.subplots_adjust(hspace=0.1)
#     plt.show()


# def get_connectivity_alt(dataset_dict_backdoor, subset, target_label, explanation, graph_location):
#     # data = dataset_dict_backdoor[subset][target_label][graph_location]
#     data = dataset_dict_backdoor[subset][graph_location]
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
#         preserved_node_pairs = list(itertools.combinations(preserved_nodes, 2))
#         edges = data.pyg_graph.edge_index.T
#         connected_preserved_node_pairs = [(n1,n2) for (n1,n2) in preserved_node_pairs if torch.tensor([n1,n2]) in edges or torch.tensor([n2,n1]) in edges]
#         num_connected_nodes = len(np.unique(connected_preserved_node_pairs).tolist())
#         connectivity = num_connected_nodes/len(preserved_nodes)
#         return connectivity


# def get_pred_confidence_old(explanation):
#     out = explanation.predictions_over_time[-1]
#     prob_outputs = F.softmax(out, dim=1).detach().tolist()
#     confidence = np.max(prob_outputs[0])
#     return confidence




# def get_explainer_param_dicts(node_r, node_e, node_s, edge_r, edge_e, edge_s):
#     coeffs = {'edge_reduction': edge_r, 'node_feat_reduction': node_r,  'node_feat_size': node_s,   'node_feat_ent': node_e,    'edge_size': edge_s,    'edge_ent': edge_e,     'EPS': 1e-15}            
#     return coeffs


# def parse_model_from_backdoor_settings(dataset, 
#                                        classifier_hyperparams,
#                                        attack_specs,
#                                        model_hyp_set,
#                                        clean_label_attack=False):
#     hyp_dict = get_info('hyp_dict_backdoor')
#     attack_target_label = attack_specs['attack_target_label']
#     model_type = hyp_dict[dataset][attack_target_label][model_hyp_set]['model_type']
#     model_path = get_model_path(dataset, classifier_hyperparams, attack_specs, model_hyp_set)
#     loaded_state_backdoor = torch.load(model_path)
#     state_dict = loaded_state_backdoor['state_dict']
#     history = loaded_state_backdoor['history']
#     kwargs = hyp_dict[dataset][int(attack_target_label)][model_hyp_set]
#     kwargs['num_classes']       = data_shape_dict[dataset]['num_classes']
#     kwargs['num_node_features'] = data_shape_dict[dataset]['num_node_features']
#     model_type_dict = {'gcn': GCN3, 'gin': GIN, 'gin2': GIN2,
#                                         'gin3': GIN3, 'gin4': GIN4, 'gcn_plain': PlainGCN,
#                                         'carate': Net, 'diffpool': DiffPool, 'topkpool': Topkpool,
#                                         'sage': GraphSAGE, 'graphgnn': GraphGNNModel, 'graphlevelgnn': GraphLevelGNN}
#     backdoor_model = model_type_dict[model_type](**kwargs)
#     backdoor_model.load_state_dict(state_dict)
#     return backdoor_model, history


# def plot_metrics_from_dataframe(df, row_i): 
#     _, explainer_hyperparams, model_specs, classifier_hyperparams = parse_settings_from_df_index(df, row_i)
#     dataset = df.loc[row_i,'Dataset']
#     raw_image_path, dist_image_path = parse_metric_image_path(dataset, classifier_hyperparams, explainer_hyperparams, model_specs)
#     img1 = Image.open(raw_image_path)
#     img2 = Image.open(dist_image_path)
#     display(img1)
#     display(img2)


# def plot_from_dataframe(df,     
#                         dataset,
#                         row_i,  
#                         loss_types, 
#                         num_samples=None,
#                         original_index = None,
#                         title=None, 
#                         show_clean=True, 
#                         do_loss=True,
#                         show_connectivity=True,
#                         backdoor_subset = 'train_backdoor_graphs',
#                         clean_subset    = 'train_clean_graphs',
#                         which_explainer = 'gnn',
#                         regenerate_data = False,
#                         explanation_type='phenomenon',
#                         clean_type = 'counterpart',
#                         clean_label_attack=False,
#                         seed=2575):
#     assert clean_type in ['counterpart','success']
#     attack_specs, explainer_hyperparams, model_specs, classifier_hyperparams = parse_settings_from_df_index(df, row_i)
#     model_hyp_set       = model_specs['model_hyp_set']
#     attack_target_label = attack_specs['attack_target_label']
#     explainer_hyperparams['explanation_type'] = explanation_type 
#     dataset_path_backdoor = get_dataset_path(dataset, attack_specs)
#     if os.path.exists(dataset_path_backdoor)==True and regenerate_data==False:
#         with open (dataset_path_backdoor, 'rb') as f: 
#             dataset_dict_backdoor = pickle.load(f)
#     else:
#         create_nested_folder(dataset_path_backdoor)
#         dataset_dict_backdoor = build_dataset_dict(dataset, attack_specs, seed=seed, fold_idx=0, clean_pyg_process=True, use_edge_attr=False, verbose=False)
#         with open(dataset_path_backdoor, 'wb') as f: 
#             pickle.dump(dataset_dict_backdoor, f)
#     dataset_path_clean = get_dataset_path(dataset,clean=True)
#     if os.path.exists(dataset_path_clean)==True and regenerate_data==False:
#         with open (dataset_path_clean, 'rb') as f: 
#             dataset_dict_clean= pickle.load(f)
#     else:
#         create_nested_folder(dataset_path_clean)
#         dataset_dict_backdoor = build_dataset_dict(dataset, clean=True, seed=seed, fold_idx=0, clean_pyg_process=True, use_edge_attr=False, verbose=False)
#         with open(dataset_path_clean, 'wb') as f: 
#             pickle.dump(dataset_path_clean, f)
#     model, _   = parse_model_from_backdoor_settings(dataset, classifier_hyperparams, attack_specs, model_hyp_set,clean_label_attack=clean_label_attack)
#     clean_subset_for_asr = 'train_clean_graphs'
#     # backdoor_data_source = Batch.from_data_list([data.pyg_graph for data in dataset_dict_backdoor[backdoor_subset][attack_target_label]])
#     # clean_data_source    = Batch.from_data_list([data.pyg_graph for data in dataset_dict_backdoor[clean_subset][attack_target_label]])
#     # clean_data_for_asr   = Batch.from_data_list([data.pyg_graph for data in dataset_dict_backdoor[clean_subset_for_asr][attack_target_label]])
#     backdoor_data_source = Batch.from_data_list([data.pyg_graph for data in dataset_dict_backdoor[backdoor_subset]])
#     clean_data_source    = Batch.from_data_list([data.pyg_graph for data in dataset_dict_clean[clean_subset]])
#     clean_data_for_asr   = Batch.from_data_list([data.pyg_graph for data in dataset_dict_clean[clean_subset_for_asr]])
#     out = model(backdoor_data_source.x, backdoor_data_source.edge_index, backdoor_data_source.batch)
#     predicted_labels = out.argmax(dim=1) 
#     _, backdoor_success_indices = get_asr(model, backdoor_data_source, clean_data_for_asr, backdoor_preds=predicted_labels)
#     use_backdoor_indices=None
#     if num_samples is None and original_index is not None:
#         use_backdoor_indices = [original_index]
#     elif num_samples is None and original_index is None:
#         use_backdoor_indices    = backdoor_success_indices
#     elif num_samples is not None:
#         use_backdoor_indices = np.random.choice(backdoor_success_indices, num_samples, replace=False)
#     if clean_type == 'success':
#         clean_success_indices = get_clean_accurate_indices(model, backdoor_data_source)#, attack_target_label)
#         use_clean_indices = np.random.choice(clean_success_indices, num_samples, replace=False)
#     elif clean_type == 'counterpart':
#         use_clean_indices = use_backdoor_indices
#     print('using backdoor indices:',use_backdoor_indices)
#     print('using clean indices:',use_clean_indices)
#     assert int(df.reset_index().loc[row_i,'Attack Target Label']) == attack_target_label
#     clean_dict, backdoor_dict = plot_multiple_explanations(dataset,             dataset_dict_backdoor,  dataset_dict_clean, model,                  use_clean_indices,      use_backdoor_indices,    
#                                                            model_specs,         attack_specs,           explainer_hyperparams,  classifier_hyperparams, show_clean,    
#                                                            do_loss,             loss_types,             True,                   clean_subset,           backdoor_subset,
#                                                            show_curvature=True, show_connectivity=show_connectivity, title=title, which_explainer = which_explainer)
#     # trigger_graph = dataset_dict_backdoor['trigger_graphs'][attack_target_label]
#     # trigger_graph = dataset_dict_backdoor['trigger_graphs']
#     trigger_graph = dataset_dict_backdoor['trigger_graph']
#     return clean_dict, backdoor_dict, backdoor_data_source, clean_data_source, trigger_graph, dataset_dict_backdoor, model


# def get_metric_f1_score_bars(df_tf,model_confidence_dict,relevant_columns):
#     relevant_metrics = get_metric_names()
#     for trigger_size in [2,4,6,8,10,12]:
#         cols   = relevant_columns
#         titles = [metric_plot_info_dict[m]['Title'] + ' Distance' if 'dist' in m else metric_plot_info_dict[m]['Title'] for m in relevant_metrics]
#         palette = {'>=0.9': 'g', '0.7-0.9': 'lime', '0.4-0.7': 'orange', '<0.4': 'r'}
#         count_dict = {i: {0.9: 0, 0.7: 0, 0.4: 0, -100: 0} for i in range(len(cols))}
#         df_ = df_tf.reset_index()
#         df_ = df_[df_['config_name'].str.contains(f'trigger_size_{trigger_size}')]
#         config_names = [k for (k,v) in model_confidence_dict.items() if np.mean(v)>0.7 and k in list(df_['config_name'])]
#         big_title = f'F1 Score Distributions by Metric\nAIDS, MUTAG, PROTEINS | Graph Types ER, SW, PA | Trigger Size {trigger_size}'
#         fig, axs = plt.subplots(4,5, figsize=(15, 3))
#         for config_name in config_names:
#             df_subset = df_[df_['config_name']==config_name]
#             df_subset_backdoor = df_subset[df_subset['category']=='backdoor']
#             df_subset_clean = df_subset[df_subset['category']=='clean']
#             for j, (col,title) in enumerate(zip(cols, titles)):
#                 tp = sum(df_subset_backdoor[col].str.contains('True'))
#                 fn = len(df_subset_backdoor[col])-tp
#                 tn = sum(df_subset_clean[col].str.contains('True'))
#                 fp = len(df_subset_clean[col])-tn
#                 if tp+0.5*(fn+fp)==0:
#                     pass
#                 else:
#                     f1 = tp/(tp+0.5*(fn+fp))
#                     for thresh in count_dict[j].keys():
#                         if f1 >= thresh:
#                             count_dict[j][thresh] += 1
#                             break
#         for j, (col,title) in enumerate(zip(cols, titles)):
#             counts = list(count_dict[j].values())
#             ax      = axs.flatten()[j]
#             ax.set_title(title, fontsize=10)
#             total = sum(counts)
#             percentages = [count / total for count in counts]
#             left = 0
#             for k, percentage in enumerate(percentages):
#                 color = ['g','lime','orange','red'][k]
#                 ax.barh(y=0, width=percentage, left=left, color=color, edgecolor='white')
#                 left += percentage
#                 if percentage > 0:
#                     ax.text(left - (percentage / 2), -0.1, f'{percentage * 100:.1f}%', ha='center', va='center', color='black', fontsize=7)
#             ax.set_xlim(0, 1)
#             ax.axis('off')
#         legend_labels   = list(palette.keys())
#         legend_handles  = [Patch(facecolor=palette[key], label=key) for key in legend_labels]
#         fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=[0.5,-0.2], fancybox=True, shadow=True, ncol=4)#, title='TPR-FPR')
#         plt.suptitle(big_title, y=1.05)
#         plt.tight_layout()
#         plt.show()


# def display_average_metric_acc(this_df, relevant_metrics, threshold_type):
#     relevant_true_false_columns = [c + f'_{threshold_type}' for c in relevant_metrics]
#     metric_score_dict = {}
#     for metric in relevant_true_false_columns:
#         metric_name = metric[:-len(f'_{threshold_type}')]
#         metric_score_dict[metric_name] = np.mean(this_df[metric].str.contains('True'))
#     metric_score_dict = dict(sorted(metric_score_dict.items(), key=lambda item: item[1], reverse=True))
#     plt.axis('off')
#     plt.text(0.1, 0.9, f"Average Metric Accuracy\n({threshold_type})",fontsize=8)
#     for idx, (k,v) in enumerate(metric_score_dict.items()):
#         plt.text(0.1,0.8-idx*0.05, f'{k}: {np.round(v,2)}%',fontsize=8)
#     plt.show()


# def show_saves(datasets, categories, graph_types, trigger_sizes, this_df, relevant_true_false_columns, threshold_types=['optimal tpr-fpr','optimal tpr-fnr','optimal f1','clean_val','kmeans','gmm']):
#     fig,axs = plt.subplots(1,3,figsize=(14,4),width_ratios=[1,1,2])
#     raw_cols  = [col for col in relevant_true_false_columns if 'dist' not in col]
#     dist_cols = [col for col in relevant_true_false_columns if 'dist' in col]
#     for i, (type_,cols) in enumerate(zip(['Raw Only','Dist Only','All'],[raw_cols,dist_cols,relevant_true_false_columns])):
#         ax = axs[i]
#         percentage_no_success = np.round((this_df[cols].applymap(lambda x: 'True' in str(x)).sum(axis=1) == 0).mean() * 100,1)
#         ranked_num_saves = count_saves_by_metric(this_df, threshold_types=threshold_types,relevant_columns=cols,existing_dict_to_add_to = None)
#         as_percentages = np.array(list(ranked_num_saves.values()))/len(ranked_num_saves.values())
#         ax.bar(height=as_percentages,x=list(ranked_num_saves.keys()))
#         ax.xaxis.set_major_locator(FixedLocator(range(len((ranked_num_saves.keys())))))
#         ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'Label {int(val)}'))
#         ax.set_xticklabels(labels=list(ranked_num_saves.keys()),rotation=45,ha='right',fontsize=8)
#         ax.set_ylim(0,100)
#         ax.set_ylabel('% of All Observations')
#         ax.set_title(type_,fontsize=8)
#         ax.text(0.95, 0.95, f'{percentage_no_success}% of all observations not\nclassified by any metric', verticalalignment='top', horizontalalignment='right',transform=ax.transAxes, fontsize=7,color='red') 
#     fig.suptitle(f'Percentages of Individual Observations "Saved" by Single Metrics\n\n{", ".join(datasets)} | {", ".join(categories)} | {", ".join(graph_types)} | trigger size {", ".join([str(ts) for ts in trigger_sizes])} | threshold type: {", ".join(threshold_types)}', y=1.1,fontsize=8)
#     plt.tight_layout()
#     plt.show()


# def show_postive_metric_composite_score(this_df, datasets, relevant_metrics):
#     relevant_true_false_columns_ = [c + f' clean_val' for c in relevant_metrics]
#     fig,axs = plt.subplots(1,4,figsize=(18,4), sharey=False)
#     handles_lists, labels_lists = [],[]
#     for i, graph_type in enumerate(['All','ER','SW','PA']):
#         this_ax = axs.flatten()[i]
#         for dataset in datasets:
#             scores = []
#             this_df_  = this_df[this_df['config_name'].str.contains(dataset)]
#             if graph_type=='All':
#                 this_df_ = this_df_[this_df_['config_name'].str.contains('|'.join(['ER','SW','PA']))]
#             else:
#                 this_df_ = this_df_[this_df_['config_name'].str.contains(graph_type)]
#             this_df_backdoor = this_df_[this_df_['category']=='backdoor'][relevant_true_false_columns_]
#             this_df_clean    = this_df_[this_df_['category']=='clean'][relevant_true_false_columns_]
#             for thresh in range(len(relevant_true_false_columns_)):
#                 this_df_backdoor_values = this_df_backdoor.applymap(lambda x: str(x)=='True Positive').sum(axis=1) >= thresh
#                 tp  = sum(this_df_backdoor_values)
#                 fn  = len(this_df_backdoor_values)-tp
#                 this_df_clean_values = this_df_clean.applymap(lambda x: str(x)=='False Positive').sum(axis=1) >= thresh
#                 fp  = sum(this_df_clean_values)
#                 tn  = len(this_df_clean_values)-fp
#                 f1 = tp/(tp+0.5*(fp+fn))
#                 scores.append(f1)  
#             scatter_plot = this_ax.scatter(range(len(scores)),scores, label=dataset, s=5)
#             if dataset not in labels_lists:
#                 handles_lists.append(scatter_plot)
#                 labels_lists.append(dataset)
#         this_ax.set_xticks(range(len(relevant_true_false_columns_)));   plt.yticks(np.linspace(0,1,11),fontsize=6)
#         this_ax.xaxis.set_major_locator(FixedLocator(range(len(relevant_true_false_columns_))))
#         this_ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'Label {int(val)}'))
#         this_ax.set_xticklabels(range(1,len(relevant_true_false_columns_)+1), fontsize=7)
#         this_ax.set_xlabel('Number of Positive Metrics Required',fontsize=10)
#         this_ax.yaxis.set_major_locator(FixedLocator([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]))
#         this_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'Label {int(val)}'))
#         this_ax.set_yticklabels([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], fontsize=7)
#         if i==0:
#             this_ax.set_ylabel('Score',fontsize=10)
#         this_ax.set_ylim(0,1.1)
#         this_ax.set_title(graph_type)
#     fig.legend(handles=handles_lists, labels=labels_lists, loc='lower center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=len(datasets))
#     fig.suptitle(f'F1 Score with rule "Predict Backdoor When # Positive Metrics > x"', y=1.05, fontsize=12)
#     plt.show()


# def show_postive_metric_composite_score(this_df, datasets, relevant_metrics):
#     relevant_true_false_columns_ = [c + f' clean_val' for c in relevant_metrics]
#     fig,axs = plt.subplots(1,4,figsize=(18,4), sharey=False)
#     handles_lists, labels_lists = [],[]
#     for i, graph_type in enumerate(['All','ER','SW','PA']):
#         this_ax = axs.flatten()[i]
#         for dataset in datasets:
#             scores = []
#             this_df_ = this_df[this_df['Dataset'] == dataset]
#             if graph_type=='All':
#                 this_df_ = this_df_[this_df_['config_name'].str.contains('|'.join(['ER','SW','PA']))]
#             else:
#                 this_df_ = this_df_[this_df_['config_name'].str.contains(graph_type)]
#             this_df_backdoor = this_df_[this_df_['category']=='backdoor'][relevant_true_false_columns_]
#             this_df_clean    = this_df_[this_df_['category']=='clean'][relevant_true_false_columns_]
#             for thresh in range(len(relevant_true_false_columns_)):
#                 this_df_backdoor_values = this_df_backdoor.applymap(lambda x: str(x)=='True Positive').sum(axis=1) >= thresh
#                 tp  = sum(this_df_backdoor_values)
#                 fn  = len(this_df_backdoor_values)-tp
#                 this_df_clean_values = this_df_clean.applymap(lambda x: str(x)=='False Positive').sum(axis=1) >= thresh
#                 fp  = sum(this_df_clean_values)
#                 tn  = len(this_df_clean_values)-fp
#                 if tp+0.5*(fp+fn) > 0:
#                     f1 = tp/(tp+0.5*(fp+fn))
#                     scores.append(f1)  
#             scatter_plot = this_ax.scatter(range(len(scores)),scores, label=dataset, s=5)
#             if dataset not in labels_lists:
#                 handles_lists.append(scatter_plot)
#                 labels_lists.append(dataset)
#         this_ax.set_xticks(range(20));   plt.yticks(np.linspace(0,1,11),fontsize=6)
#         this_ax.xaxis.set_major_locator(FixedLocator(range(len(relevant_true_false_columns_))))
#         this_ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'Label {int(val)}'))
#         this_ax.set_xticklabels(range(1,len(relevant_true_false_columns_)+1), fontsize=7)
#         this_ax.set_xlabel('Number of Positive Metrics Required',fontsize=10)
#         this_ax.yaxis.set_major_locator(FixedLocator([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]))
#         this_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'Label {int(val)}'))
#         this_ax.set_yticklabels([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], fontsize=7)
#         if i==0:
#             this_ax.set_ylabel('Score',fontsize=10)
#         this_ax.set_ylim(0,1.1)
#         this_ax.set_title(graph_type)
#     fig.legend(handles=handles_lists, labels=labels_lists, loc='lower center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=len(datasets))
#     fig.suptitle(f'F1 Score with rule "Predict Backdoor When # Positive Metrics > x"', y=1.05, fontsize=12)
#     plt.show()


# def metric_co_occurrences(this_df, relevant_true_false_columns):
#     raw_cols  = [col for col in relevant_true_false_columns if 'dist' not in col]
#     dist_cols = [col for col in relevant_true_false_columns if 'dist' in col]
#     co_occurrence_matrix_1 = pd.DataFrame(0, index=relevant_true_false_columns, columns=relevant_true_false_columns)
#     for i, col1 in enumerate(relevant_true_false_columns):
#         for j, col2 in enumerate(relevant_true_false_columns):
#             if i > j:
#                 co_occurrence_matrix_1.loc[col1, col2] = (this_df[col1] == this_df[col2]).sum()
#     if len(dist_cols)>0 and len(raw_cols)>0:
#         co_occurrence_matrix_2 = pd.DataFrame(0, index=dist_cols, columns=raw_cols)
#         for i, col1 in enumerate(dist_cols):
#             for j, col2 in enumerate(raw_cols):
#                 if i == j:
#                     co_occurrence_matrix_2.loc[col1, col2] = (this_df[col1] == this_df[col2]).sum()
#         fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
#         sns.heatmap(co_occurrence_matrix_1, ax=axs[0], annot=False, cmap='RdYlGn', cbar=True, square=True, linewidths=.5, cbar_kws={'label': '# co-occurrences', 'shrink': 0.5})
#         axs[0].set_title("Metric Co-Occurrence", fontsize=6)
#         axs[0].tick_params(axis='both', which='major', labelsize=6)
#         axs[0].xaxis.set_major_locator(FixedLocator([idx+0.5 for idx in range(len(relevant_true_false_columns))]))
#         axs[0].set_xticklabels(relevant_true_false_columns, rotation=60, ha='right')
#         sns.heatmap(co_occurrence_matrix_2, ax=axs[1], annot=False, cmap='RdYlGn', cbar=True, square=True, linewidths=.5, cbar_kws={'label': '# co-occurrences', 'shrink': 0.5})
#         axs[1].set_title("Metric Co-Occurrence:\nRaw vs. Dist Counterparts", fontsize=6)
#         axs[1].tick_params(axis='both', which='major', labelsize=6)
#         axs[1].xaxis.set_major_locator(FixedLocator([idx+0.5 for idx in range(len(raw_cols))]))
#         axs[1].set_xticklabels(raw_cols, rotation=60, ha='right')
#         for ax in axs:
#             cbar = ax.collections[0].colorbar
#             cbar.set_label(cbar.ax.get_ylabel(), fontsize=6)
#             cbar.ax.tick_params(labelsize=6)
#         plt.tight_layout()
#         plt.show()


# def top_n_sized_metric_subset(this_df, relevant_true_false_columns):
#     this_df_ = this_df[relevant_true_false_columns]
#     table_df = pd.DataFrame(columns=['# Positive'] + relevant_true_false_columns)
#     for N in range(21):
#         def get_true_columns(row):
#             true_cols = [col for col in row.index if 'True' in row[col]]
#             return tuple(true_cols) if len(true_cols) == N else None
#         true_column_sets = this_df_.apply(get_true_columns, axis=1)
#         true_column_sets = true_column_sets.dropna().tolist()
#         counter = Counter(true_column_sets)
#         if len(counter.most_common()) > 0:
#             col_set, _ = counter.most_common(1)[0]
#             row = pd.DataFrame({col: '✓' if col in col_set else '' for col in relevant_true_false_columns},index=[0])
#             row['# Positive'] = N
#             table_df = pd.DataFrame(pd.concat([table_df, row], ignore_index=True))
#     table_html = table_df.to_html(index=False)
#     style = '''
#     <style>
#         th {
#             writing-mode: vertical-rl;
#             transform: rotate(180deg);
#             transform-origin: center;
#             text-align: right;
#             white-space: nowrap;
#             padding: 15px 15px 15px;  /* Added bottom padding */
#         }
#         td {
#             text-align: center;
#             padding: 2px 2px;
#         }
#         table {
#             border-collapse: collapse;
#         }
#     </style>
#     '''
#     display(HTML(style + table_html))


# def get_df_subset(df_big, trigger_sizes=[2,4,6,8,10,12],datasets=['MUTAG','AIDS','PROTEINS'],graph_types=['ER','SW','PA'],train_backdoor_pred_cutoff=None,asr_cutoff=None,clean_val_threshold=None,restrict_datasets=['PROTEINS']):
#     this_df = copy.copy(df_big)
#     this_df = this_df[this_df['Trigger Size'].isin(trigger_sizes)]
#     this_df = this_df[this_df['Graph Type'].isin(graph_types)]
#     if train_backdoor_pred_cutoff is not None:
#         this_df = df_big[df_big['train_backdoor_pred_conf']>train_backdoor_pred_cutoff]
#     if asr_cutoff is not None:
#         this_df = this_df[this_df['ASR']>asr_cutoff]
#     restricted_dfs = []
#     for dataset in restrict_datasets:
#         this_df_ = this_df[this_df['Dataset']==dataset]
#         restricted_dfs.append(this_df_)
#     remaining_datasets = list(set.difference(set(datasets),set(restrict_datasets)))
#     remaining_dfs = []
#     for dataset in remaining_datasets:
#         this_df_ = this_df[this_df['Dataset']==dataset]
#         remaining_dfs.append(this_df_)
#     df_all = pd.concat(restricted_dfs + remaining_dfs)
#     if clean_val_threshold is not None:
#         df_all = change_thresholds(df_all, 1-clean_val_threshold, clean_val_threshold, get_metric_names())
#     return df_all


# def get_postive_metric_composite_score(this_df, datasets, relevant_metrics, plot=True):
#     relevant_true_false_columns_ = [c + f' clean_val' for c in relevant_metrics]
#     graph_types = ['All','ER','SW','PA']
#     symbol_dict = dict(zip(datasets,['d','s','o']))
#     if plot==True:
#         fig,axs = plt.subplots(1,4,figsize=(18,4), sharey=False)
#         handles_lists, labels_lists = [],[]
#     graph_type_composite_scores = {graph_type: {} for graph_type in graph_types}
#     for i, graph_type in enumerate(graph_types):
#         if plot==True:
#             this_ax = axs.flatten()[i]
#         graph_type_composite_scores[graph_type] = {dataset: [] for dataset in datasets}
#         for dataset,color in zip(datasets,['#f79165','#5e99f7','#ff3061']):
#             this_df_ = this_df[this_df['Dataset'] == dataset]
#             if graph_type=='All':
#                 this_df_ = this_df_[this_df_['config_name'].str.contains('|'.join(graph_types))]
#             else:
#                 this_df_ = this_df_[this_df_['config_name'].str.contains(graph_type)]
#             this_df_backdoor = this_df_[this_df_['category']=='backdoor'][relevant_true_false_columns_]
#             this_df_clean    = this_df_[this_df_['category']=='clean'][relevant_true_false_columns_]
#             for thresh in range(1,len(relevant_true_false_columns_)+1):
#                 this_df_backdoor_values = this_df_backdoor.applymap(lambda x: str(x)=='True Positive').sum(axis=1) >= thresh
#                 tp  = sum(this_df_backdoor_values)
#                 fn  = len(this_df_backdoor_values)-tp
#                 this_df_clean_values = this_df_clean.applymap(lambda x: str(x)=='False Positive').sum(axis=1) >= thresh
#                 fp  = sum(this_df_clean_values)
#                 tn  = len(this_df_clean_values)-fp
#                 if tp+0.5*(fp+fn) > 0:
#                     f1 = tp/(tp+0.5*(fp+fn))
#                     graph_type_composite_scores[graph_type][dataset].append(f1)  
#             if plot==True:
#                 this_ax.plot(range(len(graph_type_composite_scores[graph_type][dataset])),graph_type_composite_scores[graph_type][dataset], label=dataset,color=color,lw=0.3)#, s=5)
#                 scatter = this_ax.scatter(range(len(graph_type_composite_scores[graph_type][dataset])),graph_type_composite_scores[graph_type][dataset], label=dataset,color=color, marker = symbol_dict[dataset], s=20)
#                 if dataset not in labels_lists:
#                     handles_lists.append(scatter)
#                     labels_lists.append(dataset)
#         if plot==True:
#             this_ax.set_xticks(range(len(relevant_metrics)),labels=[str(i) for i in range(1,len(relevant_metrics)+1)])
#             this_ax.set_xlabel('Number of Positive Metrics Required',fontsize=10)
#             this_ax.yaxis.set_major_locator(FixedLocator([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]))
#             this_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'Label {int(val)}'))
#             this_ax.set_yticklabels([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], fontsize=7)
#             if i==0:
#                 this_ax.set_ylabel('Score',fontsize=10)
#             this_ax.set_ylim(0,1.1)
#             this_ax.set_title(graph_type)
#     if plot==True:
#         fig.legend(handles=handles_lists, labels=labels_lists, loc='lower center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=len(datasets))
#     plt.show()
#     return graph_type_composite_scores


# def variables_vs_F1_per_metric(df, relevant_cols, vars_=None):
#     if relevant_cols is None:
#         metrics = ['elbow_dist', 'curv_dist', 'es', 'connectivity', 'pred_conf', 'node_deg_var', 'mask_feat_var']
#         relevant_cols = [m + ' clean_val' for m in metrics]
#     if vars_ is None:
#         vars_ = ['Node Ratio','Edge Ratio','Density Ratio','Number of Nodes']
#     fig,axs = plt.subplots(1,len(vars_),figsize=(12,4))
#     handles, labels = [],[]
#     markers = ['o', 'X', 'D', '*', '+']
#     marker_size = [60, 75, 60, 75, 75]
#     marker_weight = [1, 1, 1, 1, 3]
#     for v,var in enumerate(vars_):
#         ax = axs[v]
#         sub_df = copy.copy(df)
#         additional_columns = ['Dataset','category', 'original_index',var] if var in df.columns else ['category', 'original_index']
#         sub_df = sub_df[relevant_cols + additional_columns]
#         if var != 'Number of Nodes':
#             values = sorted(set(v for v in sub_df[var] if v is not None))
#             selected_sizes=None
#         ''' Need to adjust this because what about if we want to consider all datasets'''
#         if var not in ['Prob', 'Trigger Size', 'Avg Nodes Pre-Attack', 'Number of Nodes']:
#             num_bins = 10
#             values = np.linspace(min(values), max(values), num_bins)
#         values = selected_sizes if var == 'Number of Nodes' else values
#         f1s_this_column = []
#         for i, num_pos_metrics_requirement in enumerate([2, 3, 4]):
#             f1s = calculate_f1s(sub_df, values, num_pos_metrics_requirement, var, relevant_cols)
#             f1s_this_column.append(f1s)
#             scat = ax.scatter(values, f1s, marker=markers[i], s=marker_size[i], linewidths=marker_weight[i])
#             ax.plot(values, f1s, lw=0.5)
#             if v==0:
#                 handles.append(scat)
#                 labels.append(num_pos_metrics_requirement)
#         if var=='Trigger Size':
#             ax.set_xlabel('Trigger Size',fontsize=20)
#         elif var=='ASR':
#             ax.set_xlabel('Attack Success Rate',fontsize=20)
#         if v==0:
#             ax.set_ylabel('F1 Score',fontsize=20)
#         ax.set_ylim(0.5,1)
#         ax.tick_params(axis='both', which='major', labelsize=20)
#     print(set(sub_df['Dataset']))
#     fig.subplots_adjust(hspace=0.1)
#     leg = fig.legend(handles, labels,
#             title='Number of Required Positive Metrics',
#             loc='lower center',
#             bbox_to_anchor=(0.515, -0.22), fancybox=True, shadow=True, ncol=5, fontsize=18,
#             title_fontproperties={'size': 18},handletextpad=0,columnspacing=0.7)
#     plt.tight_layout()
#     plt.show()



# def get_binned_geometry_variable_values(dataset, sub_df, geometry_var, geometry_variable_function=None, seed=2575):
#     geometry_variable_function_dict = {'Number of Nodes': num_node_g, 'Number of Edges': num_edge_g, 'Node Degree Variance': var_g}
#     if geometry_var not in geometry_variable_function_dict.keys():
#         assert geometry_variable_function is not None, 'Undefined geometry variable -- Must Provide Function to Obtain this Value from Graph Object'
#         f = geometry_variable_function
#     else:
#         f = geometry_variable_function_dict[geometry_var]
#     datasets_=[dataset] if dataset!='All' else ['MUTAG','AIDS','PROTEINS']
#     values = []
#     for dataset in datasets_:
#         dataset_dict_clean = build_dataset_dict(dataset, attack_specs=None, seed=seed, fold_idx=0, clean=True, clean_pyg_process=True, use_edge_attr=False, verbose=False)
#         graph_sizes_indices = [(f(g), g.pyg_graph.original_index) for g in dataset_dict_clean['train_clean_graphs']]
#         graph_sizes = [size for size, _ in graph_sizes_indices]
#         Q1 = np.percentile(graph_sizes, 25)
#         Q3 = np.percentile(graph_sizes, 75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR
#         filtered_sizes = [size for size in graph_sizes if lower_bound <= size <= upper_bound]
#         sorted_filtered_sizes_indices = sorted([(size, index) for size, index in graph_sizes_indices if lower_bound <= size <= upper_bound], key=lambda x: x[0])
#         graph_size_map = {index: size for size, index in sorted_filtered_sizes_indices}
#         sub_df.loc[:,geometry_var] = sub_df['original_index'].map(graph_size_map)
#         values += list(set(filtered_sizes))
#     values = sorted(values)
#     values = np.linspace(values[0], values[-1], 4)  
#     return values
# var_g = lambda g: np.var(list(dict(g.nx_graph.degree).values()))
# num_node_g = lambda g: len(g.nx_graph.nodes())
# num_edge_g = lambda g: len(g.nx_graph.edges())
# attack_strength_variables   = ['Trigger Size', 'ASR']
# backdoor_geometry_variables = ['Node Ratio','Edge Ratio','Density Ratio']
# clean_geometry_variables    = ['Number of Nodes','Number of Edges','Node Degree Variance','all of the above']


# def trigger_size_vs_F1_per_individual_metric_per_dataset(df_big_25_75, metrics, var_='Trigger Size',columnspacing=1,handletextpad=-1,bbox=(0.475,-0.15),patch_h=20,patch_w=20, trigger_sizes = [2,4,6,8,10,12],title=None):
#     df = df_big_25_75
#     markers=['o','v','s','P','*','X','D']
#     fig,axs= plt.subplots(1,5,figsize=(17,4))
#     for d,dataset_ in enumerate(['MUTAG','AIDS','PROTEINS','IMDB-BINARY','DBLP']):
#         average_metric_performance_across_trigger_sizes = {}
#         handles = []
#         labels = []
#         for metric,marker in zip(metrics,markers):
#             scores = []
#             for trigger_size in trigger_sizes:
#                 ts = df[df[var_]==trigger_size]
#                 ts = ts[ts['Dataset']==dataset_]
#                 if len(ts) == 0:
#                     scores.append(np.nan)
#                 else:
#                     f1 = get_f1_individual_metric(ts,metric)
#                     scores.append(f1)
#             label = metric_plot_info_dict[metric]['Title']
#             label = label+' Distance' if 'dist' in metric else label
#             axs[d].plot(scores, label=label, lw=1)
#             s=50 if marker in ['*','v','P','X'] else 30
#             scatter = axs[d].scatter(range(len(scores)),scores, label=metric, marker=marker, s=s)
#             handles.append(scatter)
#             labels.append(label)
#             average_metric_performance_across_trigger_sizes[metric] = scores
#         y_labels = [np.round(val,1) for val in np.arange(0,1,0.1)]
#         axs[d].set_xticks([0,1,2,3,4,5],labels=[2,4,6,8,10,12],fontsize=15)
#         axs[d].set_yticks(np.arange(0,1,0.1),labels=y_labels,fontsize=15)
#         axs[d].set_xlabel(var_,fontsize=15)
#         axs[d].set_ylabel('F1 Score',fontsize=15)
#         axs[d].set_xlim(-0.5,5.5)
#         axs[d].set_ylim(-0.1,1)
#         axs[d].set_title(dataset_,fontsize=16)
#     plt.tight_layout()
#     ncols = 4
#     nlines=7
#     kw = dict(framealpha=1, bbox_to_anchor=bbox,
#             fancybox=True, 
#             shadow=True,
#             )
#     leg1 = fig.legend(handles=handles[:nlines//ncols*ncols], labels=labels[:nlines//ncols*ncols], ncol=ncols, loc="lower center", **kw,fontsize=16,handletextpad=handletextpad,columnspacing=columnspacing)
#     plt.gca().add_artist(leg1)
#     leg2 = fig.legend(handles=handles[nlines//ncols*ncols:],labels=labels[nlines//ncols*ncols:], ncol=nlines-nlines//ncols*ncols,fontsize=16,handletextpad=handletextpad,columnspacing=columnspacing)
#     leg2.remove()
#     leg1._legend_box._children.append(leg2._legend_handle_box)
#     leg1._legend_box.stale = True
#     for patch in leg1.get_patches():
#         patch.set_height(patch_h)
#         patch.set_width(patch_w)
#     for patch in leg2.get_patches():
#         patch.set_height(patch_h)
#         patch.set_width(patch_w)
#     if title is not None:
#         fig.suptitle(title, x=bbox[0],y=1.07,fontsize=20)
#     plt.show()


# def raw_vs_distance(df, metrics, variable_values, variable_name, separate_by_dataset=False,title=None):
#     if metrics==None:
#         metrics = ['elbow','curv','es','connectivity','pred_conf','node_deg_var','mask_feat_var']
#     assert(len(metrics))==7, 'Modify "raw_vs_distance()" function to allow for more or fewer than 7 subplots'
#     assert variable_name=='Trigger Size', f'Modify "raw_vs_distance()" function to allow for x-axis variables other than "Trigger Size" '\
#                                         + f'(received "{variable_name}" as input). Focus on creating masks to bin values and handling x_ticks.'
#     datasets_ = ['MUTAG','AIDS','PROTEINS']
#     lists_ = {d:{} for d in datasets_}
#     labels, colors = ['raw', 'distance'], ['blue','red']
#     for dataset in datasets_:
#         if separate_by_dataset==True:
#             figsize=(18, 8)
#             fig = plt.figure(figsize=figsize,tight_layout=True)
#             axes = get_7_centered_axes(fig)
#         for i, metric in enumerate(metrics):
#             for j, m in enumerate([metric, metric + '_dist']):
#                 for val in variable_values:
#                     this_df_ = df[(df[variable_name] > val-2) & (df[variable_name] <= val) & (df['Dataset']==dataset)]
#                     f1 = get_f1_individual_metric(this_df_,m)
#                     if m in lists_[dataset].keys():
#                         lists_[dataset][m].append(f1)
#                     else:
#                         lists_[dataset][m] = [f1]
#                 if separate_by_dataset==True:
#                     f1s = lists_[dataset][m]
#                     axes[i].plot(variable_values, f1s, label=labels[j], lw=1,color=colors[j])
#                     axes[i].set_xlim(1,13)
#                     axes[i].set_xticks([0,2,4,6,8,10,12])
#                     axes[i].set_xlabel(variable_name,fontsize=18)
#                     axes[i].set_ylim(0, 1)
#                     if i==0 or i==4:
#                         axes[i].set_ylabel('F1',fontsize=20)
#                     subplot_title = 'SNDV' if metric=='mask_feat_var' else 'NDV' if metric=='node_deg_var' else metric_plot_info_dict[metric]['Title']
#                     axes[i].set_title(subplot_title,y=1.05,fontsize=18)
#                     axes[i].tick_params(axis='both', which='major', labelsize=18)
#         if separate_by_dataset==True:   
#             patch_1 = line_patch(linestyle='-',color='blue',ax=axes[0],lw=1)
#             patch_2 = line_patch(linestyle='--',color='red',ax=axes[0],lw=1)
#             legend_title_plot_procedure(fig=fig,handles=[patch_1,patch_2],labels=['Raw','Distance'],ncol=2,bbox=[0.555,-0.08], hspace=0.6,loc='lower center',title=dataset)
#     if separate_by_dataset == False:
#         figsize=(18, 8)
#         fig = plt.figure(figsize=figsize,tight_layout=True)
#         axes = get_7_centered_axes(fig)
#         for i,metric in enumerate(metrics):
#             for j, m in enumerate([metric,metric + '_dist']):
#                 f1s = [np.mean([lists_[d][m][k] for d in datasets_]) for k in range(len(variable_values))]
#                 linestyle='-' if j==0 else '--'
#                 axes[i].plot(variable_values, f1s, label=labels[j], linestyle=linestyle,lw=2,color=colors[j])
#                 axes[i].set_xlim(1,13)
#                 axes[i].set_xticks([0,2,4,6,8,10,12])
#                 axes[i].set_xlabel(variable_name,fontsize=18)
#                 axes[i].set_ylim(0, 1)
#                 if i==0 or i==4:
#                     axes[i].set_ylabel('F1',fontsize=18)
#                 subplot_title = 'SNDV' if metric=='mask_feat_var' else 'NDV' if metric=='node_deg_var' else metric_plot_info_dict[metric]['Title']
#                 axes[i].set_title(subplot_title,y=1.05,fontsize=18)
#                 axes[i].tick_params(axis='both', which='major', labelsize=18)
#         patch_1 = line_patch(linestyle='-',color='blue',ax=axes[0],lw=2)
#         patch_2 = line_patch(linestyle='--',color='red',ax=axes[0],lw=2)
#         legend_title_plot_procedure(fig=fig,handles=[patch_1,patch_2],labels=['Raw','Distance'],ncol=2,bbox=[0.555,-0.08], hspace=0.6,loc='lower center', title=title)


# def vote_to_numeric(vote):
#     if vote in ['True Positive', 'False Positive']:
#         return 1
#     elif vote in ['True Negative', 'False Negative']:
#         return 0
#     else:
#         raise ValueError(f"Unknown vote type: {vote}")


# def get_big_df_random():
#     with open('/Users/janedowner/Desktop/Desktop/IDEAL/Project_1/repo/explainer_results/metrics_df_IMDB-BINARY_26 28 30 32 34 36_ER SW__thresh_hard_0.3.pkl','rb') as f:
#         df_IMDB_ER_SW = pickle.load(f)
#     df_ER_4 = df_IMDB_ER_SW[df_IMDB_ER_SW['Graph Type']=='ER']
#     df_SW_4 = df_IMDB_ER_SW[df_IMDB_ER_SW['Graph Type']=='SW']
#     with open(f'{explain_dir}/metrics_df_MUTAG_2 4 6 8 10 12_ER__thresh_hard_0.3.pkl','rb') as f:
#         df_ER_1 = pickle.load(f)
#     with open(f'{explain_dir}/metrics_df_MUTAG_2 4 6 8 10 12_SW__thresh_hard_0.3.pkl','rb') as f:
#         df_SW_1 = pickle.load(f)
#     with open(f'{explain_dir}/metrics_df_MUTAG_2 4 6 8 10 12_PA__thresh_hard_0.3.pkl','rb') as f:
#         df_PA_1 = pickle.load(f)
#     with open(f'{explain_dir}/metrics_df_AIDS_2 4 6 8 10 12_ER__thresh_hard_0.3.pkl','rb') as f:
#         df_ER_2 = pickle.load(f)
#     with open(f'{explain_dir}/metrics_df_AIDS_2 4 6 8 10 12_SW__thresh_hard_0.3.pkl','rb') as f:
#         df_SW_2 = pickle.load(f)
#     with open(f'{explain_dir}/metrics_df_AIDS_2 4 6 8 10 12_PA__thresh_hard_0.3.pkl','rb') as f:
#         df_PA_2 = pickle.load(f)
#     with open(f'{explain_dir}/metrics_df_PROTEINS_2 4 6 8 10 12_ER__thresh_hard_0.3.pkl','rb') as f:
#         df_ER_3 = pickle.load(f)
#     with open(f'{explain_dir}/metrics_df_PROTEINS_2 4 6 8 10 12_SW__thresh_hard_0.3.pkl','rb') as f:
#         df_SW_3 = pickle.load(f)
#     with open(f'{explain_dir}/metrics_df_PROTEINS_2 4 6 8 10 12_PA__thresh_hard_0.3.pkl','rb') as f:
#         df_PA_3 = pickle.load(f)
#     with open(f'{explain_dir}/metrics_df_IMDB-BINARY_26 28 30 32 34 36_PA__thresh_hard_0.3.pkl','rb') as f:
#         df_PA_4 = pickle.load(f)
#     with open(f'{explain_dir}/metrics_df_DBLP_2 4 6 8 12_ER__thresh_hard_0.3.pkl','rb') as f:
#         df_ER_5 = pickle.load(f)
#     with open(f'{explain_dir}/metrics_df_DBLP_2 4 6 8 12_SW__thresh_hard_0.3.pkl','rb') as f:
#         df_SW_5 = pickle.load(f)
#     with open(f'{explain_dir}/metrics_df_DBLP_2 4 6 8 12_PA__thresh_hard_0.3.pkl','rb') as f:
#         df_PA_5 = pickle.load(f)
#     df_big = pd.concat([df_ER_1, df_ER_2, df_ER_3, df_ER_4, df_ER_5,
#                         df_SW_1, df_SW_2, df_SW_3, df_SW_4, df_SW_5,
#                         df_PA_1, df_PA_2, df_PA_3, df_PA_4, df_PA_5])
#     return df_big


# def get_big_df_adaptive_CVPR():
#     ''' versions used in CVPR rebuttal '''
#     df_MUTAG_adaptive    = pickle.load(open(f'{explain_dir}/metrics_df_MUTAG_adaptive_1 5 10 15 20 25 30 35 40 45 50__thresh_hard_0.3_0 1.pkl','rb'))
#     df_AIDS_adaptive     = pickle.load(open(f'{explain_dir}/metrics_df_AIDS_adaptive_1 5 10 15 20 25 30 35 40 45 50__thresh_hard_0.3_0 1.pkl','rb'))
#     df_PROTEINS_adaptive = pickle.load(open(f'{explain_dir}/metrics_df_PROTEINS_adaptive_1 5 10 15 20 25 30 35 40 45 50__thresh_hard_0.3_0 1.pkl','rb'))
#     df_IMDB_adaptive     = pickle.load(open(f'{explain_dir}/metrics_df_IMDB-BINARY_adaptive_1 5 10 15 20 25 30 35 40 45 50__thresh_hard_0.3_0 1.pkl','rb'))
#     df_DBLP_adaptive     = pickle.load(open(f'{explain_dir}/metrics_df_DBLP_adaptive_1 5 10 15 20 25 30 35 40 45 50__thresh_hard_0.3_0 1.pkl','rb'))
#     df_big_adaptive_cvpr = pd.concat([df_MUTAG_adaptive, df_AIDS_adaptive, df_PROTEINS_adaptive, df_IMDB_adaptive, df_DBLP_adaptive])
#     return df_big_adaptive_cvpr


# def get_big_df_adaptive_ECCV():
#     ''' v3 algorithm (iterative loop, explainer only)'''
#     df_MUTAG_adaptive_v3    = pickle.load(open(f'{explain_dir}/metrics_df_MUTAG_adaptive_2 4 6 8 10 12__gen_alg_v_3_thresh_hard_0.3_0 1.pkl','rb'))
#     df_AIDS_adaptive_v3     = pickle.load(open(f'{explain_dir}/metrics_df_AIDS_adaptive_2 4 6 8 10 12__gen_alg_v_3_thresh_hard_0.3_0 1.pkl','rb'))
#     df_PROTEINS_adaptive_v3 = pickle.load(open(f'{explain_dir}/metrics_df_PROTEINS_adaptive_2 4 6 8 10 12__gen_alg_v_3_thresh_hard_0.3_0 1.pkl','rb'))
#     df_DBLP_adaptive_v3     = pickle.load(open(f'{explain_dir}/metrics_df_DBLP_adaptive_2 4 6 8 10 12__gen_alg_v_3_thresh_hard_0.3_0 1.pkl','rb'))
#     df_IMDB_adaptive_v3     = pickle.load(open(f'{explain_dir}/metrics_df_IMDB-BINARY_adaptive_26 28 30 32 34 36__gen_alg_v_3_thresh_hard_0.3_0 1.pkl','rb'))
#     df_big_adaptive_eccv     = pd.concat([df_MUTAG_adaptive_v3, df_AIDS_adaptive_v3, df_PROTEINS_adaptive_v3, df_DBLP_adaptive_v3, df_IMDB_adaptive_v3])
#     return df_big_adaptive_eccv


# def num_metrics_in_k_sized_sets(df):
#     relevant_true_false_columns_ = ['curv_dist clean_val', 'connectivity clean_val', 'elbow_dist clean_val','es clean_val',
#                                     'mask_feat_var clean_val','node_deg_var clean_val','pred_conf clean_val']
#     key_names = {'es clean_val': 'Explainability', 'connectivity clean_val': 'Connectivity',  'pred_conf clean_val': 'Prediction Conf.',
#                 'node_deg_var clean_val': 'NDV','mask_feat_var clean_val': 'SNDV','elbow_dist clean_val': 'Elbow Dist.', 'curv_dist clean_val': 'Curvature Dist.'}
#     handles, labels = [],[]
#     additional_columns = ['Dataset','category', 'original_index']
#     sub_df = copy.copy(df)
#     sub_df = sub_df[relevant_true_false_columns_ + additional_columns]
#     sub_df[sub_df['category']!='clean_val']
#     sub_df = sub_df.reset_index()
#     handles = []
#     labels= []
#     count_dict = {i:{m:0 for m in relevant_true_false_columns_} for i in [1,2,3,4,5,6,7]}
#     fig, axs = plt.subplots(7,1, figsize=(10, 3))
#     for row in range(len(sub_df)):
#         row_portion = list(sub_df.loc[row,relevant_true_false_columns_])
#         positive_metrics = [1 if val=='True Positive' or val=='False Positive' else 0 for val in row_portion]
#         for col,val in zip(relevant_true_false_columns_,positive_metrics):
#             if val==1:
#                 count_dict[sum(positive_metrics)][col]+=1
#     for i in count_dict.keys():
#         if i==7:
#             ax.text(-0.04,-0.55,'Number of Metrics',fontsize=15,rotation='vertical')
#         counts = list(count_dict[i].values())
#         ax = axs.flatten()[i-1]
#         total = sum(counts)
#         percentages = [count/total for count in counts]
#         left=0
#         for k, percentage in enumerate(percentages):
#             color = ['red','blue','green','violet','orange','yellow','brown'][k]
#             handle = ax.barh(y=0, width=percentage, left=left, color=color, edgecolor='white',alpha=0.55)
#             if i==1:
#                 handles.append(handle)
#                 labels.append(key_names[relevant_true_false_columns_[k]])
#             left += percentage
#             if percentage > 0:
#                 ax.text(left - (percentage / 2), -0.1, f'{percentage * 100:.1f}%', ha='center', va='center', color='black', fontsize=12)
#         ax.set_xlim(0, 1)
#         ax.set_ylabel(i)
#         ax.axis('off')
#         ax.text(-0.015,-0.2,i,fontsize=12)
#     ncols = 4
#     nlines=7
#     handletextpad=-1
#     columnspacing=1
#     kw = dict(framealpha=1, bbox_to_anchor=(0.515, -0.16),
#             fancybox=True, 
#             shadow=True,
#             )
#     leg1 = fig.legend(handles=handles[:nlines//ncols*ncols], labels=labels[:nlines//ncols*ncols], ncol=ncols, loc="lower center", **kw,fontsize=14,handletextpad=handletextpad,columnspacing=columnspacing)
#     plt.gca().add_artist(leg1)
#     leg2 = fig.legend(handles=handles[nlines//ncols*ncols:],labels=labels[nlines//ncols*ncols:], ncol=nlines-nlines//ncols*ncols,fontsize=14,handletextpad=handletextpad,columnspacing=columnspacing)
#     leg2.remove()
#     leg1._legend_box._children.append(leg2._legend_handle_box)
#     leg1._legend_box.stale = True
#     patch_h,patch_w = 10,10
#     for patch in leg1.get_patches():
#         patch.set_height(patch_h)
#         patch.set_width(patch_w)
#     for patch in leg2.get_patches():
#         patch.set_height(patch_h)
#         patch.set_width(patch_w)


# def load_varied_cutoff_df(attack_type='random'):
#     print('Note: all trigger sizes adjusted to 2-12 range')
#     df_varied_cutoffs = {}
#     for cutoffs in ['50_50', '45_55', '40_60', '35_65', '30_70', '25_75',
#                     '20_80', '15_85', '10_90', '05_95', '0_100']:
#         with open(f'{explain_dir}/df_big_{cutoffs}_{attack_type}', 'rb') as f:
#             df_this_cutoff = pickle.load(f)
#         df_varied_cutoffs[cutoffs] = df_this_cutoff
#     return df_varied_cutoffs


# def get_best_NPMRS_varied_cutoffs(dfs_varied_cutoffs_random, dfs_varied_cutoffs_adaptive):
#     df_ = pd.DataFrame()
#     df_['category'] = ['ER', 'SW', 'PA', 'adaptive']
#     relevant_metrics_all = ['es','connectivity','pred_conf','node_deg_var', 'mask_feat_var','elbow_dist','curv_dist']
#     metrics_ = [m+' clean_val' for m in relevant_metrics_all]
#     for upper_bound in list(range(50,105,5)):
#         best_thresholds = []
#         lower_bound = 100-upper_bound if upper_bound!=95 else '05'
#         for category in df_['category']:
#             if category=='adaptive':
#                 df = dfs_varied_cutoffs_adaptive[f'{lower_bound}_{upper_bound}']
#             else:
#                 df_this_cutoff = dfs_varied_cutoffs_random[f'{lower_bound}_{upper_bound}']
#                 df = df_this_cutoff[df_this_cutoff['Graph Type']==category]
#             scores = []
#             for thresh in range(1,len(metrics_)+1):
#                 f1, _, _ = get_f1_tpr_fpr_from_binary_entries_and_threshold(df, metrics_, thresh)
#                 scores.append(np.round(f1,3))
#             best_thresh = int(np.argmax(scores))+1
#             best_thresholds.append(best_thresh)
#         df_[str(upper_bound)] = best_thresholds
#     return df_


# def get_metric_names():
#     return ['loss_max',     'loss_min',     'elbow',      'curv',     'es',     'unfaith',     'connectivity',     'pred_conf',     'node_deg_var',     'mask_feat_var',
#             'loss_max_dist','loss_min_dist','elbow_dist', 'curv_dist','es_dist','unfaith_dist','connectivity_dist','pred_conf_dist','node_deg_var_dist','mask_feat_var_dist']


# def explainer_metrics_plot(metric_value_dict, # or dataframe
#                         include_metrics =  ['elbow', 'curv', 'es', 'connectivity', 'pred_conf', 'node_deg_var', 'mask_feat_var', 'elbow_dist', 'curv_dist', 'es_dist', 'connectivity_dist', 'pred_conf_dist', 'node_deg_var_dist', 'mask_deg_var_dist'],
#                         raw_plot_path=None, 
#                         dist_plot_path=None, 
#                         plot=False,
#                         save_image = True,
#                         rows_cols = (2,5),
#                         figsize=(15,6)):
#     raw_metrics     = [metric for metric in include_metrics if 'dist' not in metric]
#     dist_metrics    = [metric for metric in include_metrics if 'dist' in metric]
#     handles_lists, labels_lists = [[],[]], [[],[]]
#     rows, cols = rows_cols
#     if isinstance(metric_value_dict, dict):
#         if plot==False:
#             matplotlib.use('Agg')  
#         else:
#             matplotlib.use('nbAgg')
#     for m, metrics in enumerate([raw_metrics, dist_metrics]):
#         fig,axs = plt.subplots(rows,cols,figsize=figsize) if 'dist' not in metrics[0] else plt.subplots(rows,cols,figsize=figsize)
#         for i, metric in enumerate(metrics):
#             ax = axs.flatten()[i]
#             if metric in include_metrics:
#                 if isinstance(metric_value_dict, dict):
#                     clean_validation_values     = metric_value_dict[metric]['clean_val']['values']
#                     clean_values                = metric_value_dict[metric]['clean']['values']
#                     backdoor_values             = metric_value_dict[metric]['backdoor']['values']
#                 else:
#                     clean_validation_values     = [list(metric_value_dict[metric])[i] for i in range(len(metric_value_dict)) if list(metric_value_dict['category'])[i]=='clean_val']
#                     clean_values                = [list(metric_value_dict[metric])[i] for i in range(len(metric_value_dict)) if list(metric_value_dict['category'])[i]=='clean']
#                     backdoor_values             = [list(metric_value_dict[metric])[i] for i in range(len(metric_value_dict)) if list(metric_value_dict['category'])[i]=='backdoor']
#                 inequality                  = metric_plot_info_dict[metric]['inequality']
#                 clean_validation_values = replace_none_nan_with_average(clean_validation_values)
#                 clean_values = replace_none_nan_with_average(clean_values)
#                 backdoor_values = replace_none_nan_with_average(backdoor_values)
#                 arrow, hists, lines = explainer_metrics_subplot(metric, clean_validation_values, clean_values, backdoor_values, inequality, ax)
#                 if arrow is not None:
#                     fig.patches.append(arrow)
#                 for (subplot, label) in zip(hists,['backdoor','clean', 'clean validation']):
#                     if subplot is not None and label not in labels_lists[m]:
#                         handles_lists[m].append(subplot[2][0])
#                         labels_lists[m].append(label)
#                 for (line, label) in zip(lines, ['Best ROC threshold','Clean Validation boundary','K-Means boundary','GMM boundary']):
#                     if line is not None and label not in labels_lists[m]:
#                         handles_lists[m].append(line)
#                         labels_lists[m].append(label)
#         plt.suptitle('Raw Values' if m == 0 else 'Distance from Clean Validation Distributions')
#         plt.subplots_adjust(bottom=0.14)
#         fig.legend(handles=handles_lists[m], labels=labels_lists[m], loc='lower center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=4)
#         plt.tight_layout()
#         if save_image==True:
#             plt.savefig(raw_plot_path if m == 0 else dist_plot_path, bbox_inches='tight')
#         if plot == True:
#             plt.show()
#         else:
#             plt.close()
#     return metric_plot_info_dict


# def explainer_metrics_subplot(metric, clean_validation_values, clean_values, backdoor_values, inequality, ax):
#     backdoor_color  = 'purple'  if 'dist' in metric else 'blue'
#     clean_color     = 'yellow'  if 'dist' in metric else 'orange'
#     if set(clean_validation_values) == set(clean_values) and set(clean_validation_values) == set(backdoor_values) and len(set(clean_validation_values))==1:
#         bins=None
#     else:
#         bins = universal_bins([clean_validation_values, clean_values, backdoor_values], num_bins=30)
#     inequality      = metric_plot_info_dict[metric]['inequality']
#     extrema_str     = 'max' if inequality=='more' else 'min'
#     backdoor_subplot    = ax.hist(backdoor_values,          bins=bins,label='backdoor',   alpha=0.6, color=backdoor_color)
#     clean_subplot       = ax.hist(clean_values,             bins=bins,label='clean',      alpha=0.6, color=clean_color)
#     clean_val_subplot   = ax.hist(clean_validation_values,  bins=bins,label='clean (validation)',  alpha=0.4, color='gray')
#     optimal_thresh      = get_boundary(clean_values, backdoor_values, clean_validation_values, inequality, thresh_type='optimal f1')
#     clean_val_thresh    = get_boundary(clean_values, backdoor_values, clean_validation_values, inequality, thresh_type='clean_val')
#     kmeans_thresh       = get_boundary(clean_values, backdoor_values, clean_validation_values, inequality, thresh_type='kmeans')
#     gmm_thresh          = get_boundary(clean_values, backdoor_values, clean_validation_values, inequality, thresh_type='gmm')
#     roc_line        = ax.axvline(optimal_thresh,    linestyle='solid',  label='best roc threshold',                 color='red')
#     clean_val_line  = ax.axvline(clean_val_thresh,  linestyle='--',     label=f'{extrema_str} clean validation',    color='blue')
#     kmeans_line     = ax.axvline(kmeans_thresh,     linestyle='-.',     label=f'kmeans boundary',                   color='blue')
#     gmm_line        = ax.axvline(gmm_thresh,        linestyle=':',      label=f'gmm boundary',                      color='blue')
#     clean_val_score = get_optimal_score('f1', clean_values, backdoor_values, clean_val_thresh, inequality)
#     title_str       = f'F1 using Clean Val Thresh: {np.round(clean_val_score,3)}'
#     arrow = None
#     if 'dist' not in metric:
#         trans = transforms.blended_transform_factory(ax.transAxes,ax.transAxes)
#         x_start, y_start    = 0, 1
#         dx                  = 0.05  if inequality=='more' else -0.05
#         arrow_color         = 'g'   if inequality=='more' else 'r'
#         arrow               = patches.FancyArrow(x_start, y_start, dx, 0, transform=trans,  color=arrow_color, width=0.02,  head_width=0.1, head_length=0.06)
#     ax.set_xlabel(metric_plot_info_dict[metric]['x_label'])
#     ax.set_ylabel('count')
#     ax.set_title(f'{metric_plot_info_dict[metric]["Title"]}\n{title_str}', fontsize=10)
#     hists = [backdoor_subplot, clean_subplot, clean_val_subplot]
#     lines = [roc_line, clean_val_line, kmeans_line, gmm_line]
#     return arrow, hists, lines


# def calculate_f1s(sub_df,values,requirement,var,relevant_cols):
#     f1s = []
#     for val in values:
#         if var not in ['Prob', 'Trigger Size']:
#             df_this_size = sub_df[(sub_df[var] >= val) & (sub_df[var] < val + (values[1] - values[0]))]
#         else:
#             df_this_size = sub_df[sub_df[var] == val]
#         this_df_backdoor = df_this_size[df_this_size['category'] == 'backdoor'][relevant_cols]
#         this_df_clean = df_this_size[df_this_size['category'] == 'clean'][relevant_cols]
#         tp = sum(this_df_backdoor.applymap(lambda x: str(x) == 'True Positive').sum(axis=1) >= requirement)
#         fn = len(this_df_backdoor) - tp
#         fp = sum(this_df_clean.applymap(lambda x: str(x) == 'False Positive').sum(axis=1) >= requirement)
#         tn = len(this_df_clean) - fp
#         if tp + 0.5 * (fp + fn) > 0:
#             f1 = tp / (tp + 0.5 * (fp + fn))
#             f1s.append(f1)
#         else:
#             f1s.append(np.nan)
#     f1s = interpolate_nans(np.array(values), np.array(f1s))
#     return f1s


# def get_f1_individual_metric(df,metric):
#     col = metric + ' clean_val'
#     tp = len(df[df[col]=='True Positive'])
#     fn = len(df[df[col]=='False Negative'])
#     fp = len(df[df[col]=='False Positive'])
#     if tp + 0.5 * (fp + fn) > 0:
#         f1 = tp / (tp + 0.5 * (fp + fn))
#     else:
#         f1 = np.nan
#     return f1


# def get_f1_tpr_fpr_from_binary_entries_and_threshold(df, metrics_, thresh):
#     df_backdoor = df[df['category']=='backdoor'][metrics_]
#     df_clean    = df[df['category']=='clean'][metrics_]
#     tp  = sum(df_backdoor.applymap(lambda x: str(x)=='True Positive').sum(axis=1) >= thresh)
#     fn = len(df_backdoor) - tp
#     fp  = sum(df_clean.applymap(lambda x: str(x)=='False Positive').sum(axis=1) >= thresh)
#     tn = len(df_clean) - fp
#     f1 = tp/(tp+0.5*(fn+fp))
#     tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
#     fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
#     fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
#     tnr = tn / (fp + tn) if (fp + tn) > 0 else 0
#     acc = (tp+tn)/(tp+tn+fp+fn)
#     return f1, tpr, fpr, fnr, tnr, acc


# def interpolate_nans(x, y):
#     nans = np.isnan(y)
#     non_nans = ~nans
#     interpolated_values = np.interp(x[nans], x[non_nans], y[non_nans])
#     y[nans] = interpolated_values
#     return y


# def count_saves_by_metric(individual_pos_neg_df, threshold_types=['roc','clean_val','kmeans','gmm'],relevant_columns=[],safe_types = ['True Positive', 'True Negative'],existing_dict_to_add_to = None):
#     if relevant_columns==[]:
#         relevant_columns = [c for c in individual_pos_neg_df.columns if all(cond not in c for cond in ['config','category', 'index']) and any(threshold_type in c for threshold_type in threshold_types)]
#     individual_pos_neg_df_ = individual_pos_neg_df[relevant_columns]
#     num_save_counts=None
#     if existing_dict_to_add_to==None:
#         num_save_counts = {c:0 for c in relevant_columns}
#     else:
#         num_save_counts = existing_dict_to_add_to
#     for i in range(len(individual_pos_neg_df_)):
#         row = list(individual_pos_neg_df_.iloc[i,:])
#         safe_index = [j for (j,val) in enumerate(row) if str(val) in safe_types]
#         if len(safe_index)==1:
#             passing_metric = individual_pos_neg_df_.columns[safe_index]
#             num_save_counts[str(passing_metric[0])] += 1
#     ranked_num_save_counts = dict(sorted(num_save_counts.items(), key=lambda item: item[1], reverse=True))
#     return ranked_num_save_counts


# def get_hyperparam_combos(graph_type_attack_KP_dict,graph_type,trigger_size, K_fractions=[0,0.5,1]):
#     graph_type_attack_KP_dict[graph_type][trigger_size] = []
#     if graph_type == 'ER':
#         probs = [1] if trigger_size == 2 else [0.5, 1]
#         for prob in probs:
#             graph_type_attack_KP_dict[graph_type][trigger_size].append([None, prob])
#     if graph_type == 'SW':
#         K_list = sorted(list(set([k for k in [2, trigger_size - 1] if k > 1 and k <= trigger_size])))
#         for K in K_list:
#             for prob in [0.01, 1]:
#                 graph_type_attack_KP_dict[graph_type][trigger_size].append([K, prob])
#     if graph_type == 'PA':
#         K_list = sorted(
#             list(set([k for k in set([1, int(0.5 * trigger_size), trigger_size - 1]) if k > 0 and k < trigger_size])))
#         K_list = []
#         for fraction in K_fractions:
#             if fraction==0:
#                 K_list.append(1)
#             elif fraction==1:
#                 K_list.append(trigger_size-1)
#             else:
#                 K_list.append(int(fraction*trigger_size))
#         K_list = sorted(list(set(K_list)))
#         for K in K_list:
#             graph_type_attack_KP_dict[graph_type][trigger_size].append([K, None])
#     return graph_type_attack_KP_dict