from   utils.data_utils import *
from   utils.general_utils import *
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from   matplotlib.lines import Line2D
from   matplotlib.patches import Patch
import networkx as nx


def curly_braces(xmin,cutoff,xmax,yy,category, inequality,ax):
    xspan = xmax - xmin
    if (category=='backdoor' and inequality=='less') or (category=='clean' and inequality=='more'):
        xmin, xmax = xmin, cutoff
    elif (category=='backdoor' and inequality=='more') or (category=='clean' and inequality=='less'):
        xmin, xmax = cutoff, xmax
    xmin_ = xmin + np.abs(0.05*xmin)
    xmax_ = xmax - np.abs(0.05*xmax)
    (xmin,xmax) = (xmin,xmax) if (xmin_<=xmax_) else (xmin,xmax)
    ax_xmin, ax_xmax = xmin, xmax
    xax_span = ax_xmax - ax_xmin
    ymin, ymax = 0, 5
    yspan = ymax - ymin
    if xax_span==0:
        resolution = 1
        beta = 300
    else:
        resolution = int(xspan/xax_span*100)*2+1
        beta = 300./xax_span
    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:int(resolution/2)+1]
    y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0]))) + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = yy + (.05*y - .01)*yspan

    text_x = (xmax+xmin)/2
    if (category=='backdoor' and inequality=='less'):# or (category=='clean' and inequality=='more'):
        ax.text(text_x, yy+.07*yspan, f'Predict {category.title()}', ha='center', va='bottom', fontsize=14)
    elif (category=='backdoor' and inequality=='more'):# or (category=='clean' and inequality=='less'):
        ax.text(text_x, yy+.07*yspan, f'Predict {category.title()}', ha='center', va='bottom', fontsize=14)
    ax.plot(x, y, color='black', lw=1)



def get_training_curve_image_path(dataset, classifier_hyperparams, attack_specs, model_hyp_set):
    model_name = get_model_name(classifier_hyperparams, attack_specs, model_hyp_set)
    image_path = f'{train_dir}/{dataset}/plots/{model_name}.png'
    return image_path

def train_test_subplot(scores, metric_name, epochs, ax=None, y_lims=[-0.05, 1.05], asrs=None,
                       clean_or_backdoor='backdoor'):
    if clean_or_backdoor == 'backdoor':
        subplot_labels = ['Train', 'Test (clean)', 'Test (backdoor)']
    elif clean_or_backdoor == 'clean':
        subplot_labels = ['Train', 'Test']
    if ax == None:
        use_ax = False
        ax = plt
    else:
        use_ax = True
        ax = ax
    ax.plot(range(epochs), scores[0], label=subplot_labels[0])
    ax.plot(range(epochs), scores[1], label=subplot_labels[1])
    if clean_or_backdoor == 'backdoor':
        ax.plot(range(epochs), scores[2], label=subplot_labels[2])
        if asrs is not None:
            ax.plot(range(epochs), asrs[0], label='Train ASR', linestyle='--')
            ax.plot(range(epochs), asrs[1], label='Test ASR', linestyle='--')

    if y_lims is None:
        pass
    elif isinstance(y_lims, list):
        if use_ax == True:
            ax.set_ylim(y_lims[0], y_lims[1])
        else:
            ax.ylim(y_lims[0], y_lims[1])
    if use_ax == True:
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Score')
        ax.set_title(metric_name)
    else:
        ax.xlabel('Epochs')
        ax.ylabel('Score')
        ax.title(metric_name)
    ax.legend()



def plot_training_results(dataset, 
                          plot,
                          accs,
                          losses, 
                          asrs,
                          classifier_hyperparams,
                          model_specs,
                          attack_specs=None):
    
    assert dataset is not None
    assert accs is not None
    assert losses is not None

    model_hyp_set, clean_or_backdoor = unpack_kwargs(model_specs, ['model_hyp_set','clean_or_backdoor'])
    model_type, balanced = unpack_kwargs(classifier_hyperparams,['model_type','balanced'])
    trigger_size, frac, prob, K, graph_type = unpack_kwargs(attack_specs,['trigger_size','frac','prob','K','graph_type'])

    epochs = len(accs[0])

    fig, axs = plt.subplots(nrows=1, ncols=3, sharey=False, figsize=(25, 5))
    if clean_or_backdoor == 'backdoor':
        title = f"Backdoor Attack - {dataset}, {graph_type}, trigger_size={trigger_size}, frac={frac}, prob={prob}, K={K}, set {model_hyp_set}"
        train_test_subplot(accs, 'Accuracy', epochs, ax=axs[0], asrs=asrs, clean_or_backdoor=clean_or_backdoor)
    else:
        title = f"Benign Model - {dataset}"
        train_test_subplot(accs, 'Accuracy', epochs, ax=axs[0], clean_or_backdoor=clean_or_backdoor)

    train_test_subplot(losses, 'Loss', epochs, ax=axs[1], y_lims=None, clean_or_backdoor=clean_or_backdoor)

    fig.suptitle(title)
    if clean_or_backdoor == 'backdoor':
        image_path = get_training_curve_image_path(dataset, classifier_hyperparams, attack_specs, model_hyp_set)
    else:
        plots_folder = f'{train_dir_cln}/{dataset}/plots'
        create_nested_folder(plots_folder)
        image_path = f'{plots_folder}/model_type_{model_type}_model_hyp_set_{model_hyp_set}_balanced_{balanced}.png'

    plt.savefig(image_path)
    if plot==True:
        plt.show()
    else:
        plt.close()


def node_features_from_mask(G, feat_mask,relative_vmin_vmax=False, color_vals=[], highlight_edges=[]):
    if len(color_vals)>0:
        node_color=[color_vals]*len(G.nodes())
        node_vmin, node_vmax = 0, 1
    else:
        if feat_mask is None:
            node_color = [1]*len(G.nodes())
            node_vmin, node_vmax  = 0, 1
        else:
            node_mask_to_node_mapping = {i:node for i, node in enumerate(G.nodes())}
            node_color = []
            for i,f in enumerate(feat_mask):
                node = node_mask_to_node_mapping[i]
                if node in G.nodes():
                    color_val = torch.max(f).item()
                    node_color.append(color_val)
            node_vmin, node_vmax = 0.2,1
            if relative_vmin_vmax==True:
                (node_vmin, node_vmax) = (torch.min(feat_mask).item(), torch.max(feat_mask).item())
    highlight_nodes = torch.tensor(list(highlight_edges)).unique().tolist()
    node_map = {node:i for (i,node) in enumerate(G.nodes())}
    highlight_nodes = [node_map[node] for node in highlight_nodes]
    node_vmin = node_vmin - 0.4*np.abs(np.max(node_color))
    return  node_color, node_vmin, node_vmax

def mask_to_dict_g(edge_mask, graph):
    edge_mask_dict = defaultdict(float)
    edge_index = None
    edge_min, edge_max = 100, 0
    if isinstance(graph, nx.Graph):
        edge_index = torch.tensor(list(graph.edges())).T
    elif isinstance(graph, GraphObject):
        edge_index = torch.tensor(graph.pyg_graph.edge_index)
    else:
        edge_index = torch.tensor(list(graph.g.edges())).T
    try:
        edge_indices = [(u.item(),v.item()) for [u,v] in edge_index.T.tolist()]
    except:
        edge_indices = [(u,v) for [u,v] in edge_index.T.tolist()]
    edge_mask_dict = {}
    for i,e in enumerate(edge_indices):
        val = edge_mask[i]
        edge_mask_dict[e] = val.item()
        if val < edge_min:
            edge_min = val
        if val > edge_max:
            edge_max = val
    return edge_mask_dict

def get_edge_features(G, triggered_edges=[], edge_mask_dict=None, relative_vmin_vmax=False,highlight_edges=[],color_vals=[], darkest_edge_width=2):
    edge_styles = []
    edge_colors = []
    for u, v in G.edges():
        if (u,v) in triggered_edges or (v,u) in triggered_edges or [u,v] in triggered_edges or [v,u] in triggered_edges:
            edge_styles.append('dotted')
        elif (u,v) in highlight_edges or (v,u) in highlight_edges:
            edge_styles.append('dashed')
        else:
            edge_styles.append('solid')
        if len(color_vals)>0:
            edge_colors=[color_vals]*len(G.edges())
        elif edge_mask_dict is None:
            edge_colors.append(1)
        else:
            uv_exists, vu_exists = False, False
            uv, vu = None, None
            try:
                uv = edge_mask_dict[(u, v)]
                uv_exists=True
            except:
                pass
            try:
                vu = edge_mask_dict[(v, u)]
                vu_exists=True
            except:
                pass
            if uv_exists and vu_exists:
                color_val = max(uv,vu)
                edge_colors.append(color_val)
            elif uv_exists:
                edge_colors.append(uv)
            elif vu_exists:
                edge_colors.append(vu)
    for i in range(len(edge_colors)):
        if edge_colors[i] == None:
            red_color_val = [c0 for (c0,c1,c2) in edge_colors if (c0,c1,c2) != None]
            min_color = min(red_color_val)
            edge_colors[i] = min_color
    (edge_vmin, edge_vmax) = (min(edge_colors), max(edge_colors)) if relative_vmin_vmax==True else (0.2, 1.0)
    edge_vmin -= 0.4*np.abs(np.max(edge_colors))
    highlighted_idxs = [i for i,style in enumerate(edge_styles) if style=='dashed']
    for i in highlighted_idxs:
        (u,v) = list(G.edges())[i]
        assert (u,v) in highlight_edges or (v,u) in highlight_edges or [u,v] in highlight_edges or [v,u] in highlight_edges
        edge_colors[i] = cm.Blues(122)[:3]
    edge_widths = [darkest_edge_width if color == edge_vmax else 0.75*darkest_edge_width for color in edge_colors]
    return edge_styles, edge_colors, edge_widths, edge_vmin, edge_vmax

def plot_mol(graph,             ax=None,            edge_mask=None,     feat_mask=None,             triggered_edges=[],     relative_vmin_vmax=True, 
             node_names=None,   plot_size=(6,6),    highlight_edges=[], color_map = plt.cm.Reds,    color_vals = [],cax=None, title_font_size=12,
             show=True,
             show_legend=True, lw=2):
    edge_mask_dict = None
    if edge_mask is not None:
        edge_mask_dict = mask_to_dict_g(edge_mask,graph)
    if ax is None:
        fig, ax = plt.subplots(dpi=120,figsize=plot_size)
    pos = nx.kamada_kawai_layout(graph)
    edge_styles, edge_colors, edge_widths, edge_vmin, edge_vmax = get_edge_features(graph, triggered_edges=triggered_edges, edge_mask_dict=edge_mask_dict,relative_vmin_vmax=relative_vmin_vmax, highlight_edges=highlight_edges, color_vals=color_vals, darkest_edge_width=lw)
    nx_edges = nx.draw_networkx_edges(graph, pos=pos, ax=ax, width=edge_widths, edge_color=edge_colors, style=edge_styles, edge_cmap=color_map, edge_vmin=edge_vmin, edge_vmax=edge_vmax)
    if node_names is None:
        node_names = {i:i for i, node in enumerate(graph.nodes())}
    node_colors, node_vmin, node_vmax = node_features_from_mask(graph, feat_mask, relative_vmin_vmax=relative_vmin_vmax, color_vals=color_vals, highlight_edges=highlight_edges)
    nx_nodes  = nx.draw_networkx_nodes(graph, pos=pos, node_size=150, node_color=node_colors, ax=ax, cmap=color_map, label=node_names, vmin=node_vmin, vmax=node_vmax)
    nx.draw_networkx_labels(graph, pos=pos, labels=node_names, ax=ax, font_color='grey')
    if len(set(edge_styles))>1 and show_legend==True:
        legend_patches = [Line2D([0], [0], linestyle=style, color=plt.cm.Reds(1.0)[:3], lw=lw) for style in ['solid', 'dashed']]
        legend_labels = ['Clean Edge', 'Trigger']
        ax.legend(legend_patches, legend_labels, loc='lower right')#,bbox_to_anchor=(0.5, -0.1))
    if cax is not None and feat_mask is not None and edge_mask_dict is not None:
        norm = matplotlib.colors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
        colorbar = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
        colorbar.set_array([])  
        cax.set_title('Mask Weights', fontsize=title_font_size)
        cb = plt.colorbar(colorbar,ax=ax,cax=cax)
        cb.ax.tick_params(length=0)
    if ax is None:
        fig.tight_layout()
        if show==True:
            plt.show()
    else:
        return nx_edges, nx_nodes 
    

def plot_explanation_results(explainer_hyperparams, backdoor_plot_inputs, clean_plot_inputs = [None,None,None], loss_config = [False, None], 
                             backdoor_elbow = None, backdoor_curvature = None, clean_elbow=None, clean_curvature=None, clean_connectivity=None, backdoor_connectivity=None,
                             title=None):
    
    [do_loss, loss_types] = loss_config
    ncols = 3 if do_loss==False else 4

    relative_vmin_vmax = explainer_hyperparams['relative_vmin_vmax']
    if clean_plot_inputs != [None, None, None, None] and clean_plot_inputs != None and clean_elbow != None and clean_curvature != None:
        do_clean = True
        nrows = 2 
    else:
        do_clean=False
        nrows = 1
    width_ratios = [2,2,1] if do_loss == False else [2,2,1,2]
    figsize = (18,10) if do_clean == True else (13,5)
    _, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, width_ratios=width_ratios)

    if do_clean==True:
        clean_axs    = [axs[0][0],axs[0][1],axs[0][2]] if do_loss==False else [axs[0][0],axs[0][1],axs[0][2],axs[0][3]]
        backdoor_axs = [axs[1][0],axs[1][1],axs[1][2]] if do_loss==False else [axs[1][0],axs[1][1],axs[1][2],axs[1][3]]
    else:
        do_clean = False
        backdoor_axs = [axs[0],axs[1],axs[2]] if do_loss==False else [axs[0],axs[1],axs[2],axs[3]]

    clean_explanation,clean_graph,loss_max_clean,loss_min_clean = None,None, 0, 100
    if do_clean==True:
        clean_explanation, clean_graph, _ = clean_plot_inputs
        plot_mol(clean_graph.nx_graph, ax=clean_axs[0], edge_mask=clean_explanation.edge_mask.clone().detach(),    feat_mask=clean_explanation.node_mask.clone().detach(), relative_vmin_vmax=relative_vmin_vmax)
        connectivity_str = '' if clean_connectivity is None else f'\nConnectivity = {clean_connectivity}'
        clean_axs[0].set_title(f'Clean Sample{connectivity_str}')

        clean_edge_mask_dict = mask_to_dict_g(clean_explanation.edge_mask.clone().detach(), clean_graph.nx_graph)
        clean_edge_mask      = square_edge_mask(clean_graph.pyg_graph.x, clean_edge_mask_dict)
        edge_mask_heat_map(clean_edge_mask, clean_graph.pyg_graph, ax = clean_axs[1], relative_vmin_vmax=relative_vmin_vmax)
        
        inverse_indices = [0]* len(clean_graph.nx_graph.nodes())
        for i,idx in enumerate(list(clean_graph.nx_graph.nodes())):
            inverse_indices[idx] = i
        node_mask_heat_map(clean_explanation.node_mask[inverse_indices,:], ax = clean_axs[2], relative_vmin_vmax=relative_vmin_vmax)
        
        loss_max_clean, loss_min_clean = max(clean_explanation['clf_loss_over_time']), min(clean_explanation['clf_loss_over_time'])
        if do_loss==True:
            plot_loss(clean_explanation,loss_types, ax=clean_axs[3], curvature=clean_curvature, elbow=clean_elbow)

    backdoor_explanation, backdoor_graph, _ = backdoor_plot_inputs

    plot_mol(backdoor_graph.nx_graph, ax=backdoor_axs[0], edge_mask=backdoor_explanation.edge_mask, feat_mask=backdoor_explanation.node_mask.clone().detach(), triggered_edges=backdoor_graph.pyg_graph.triggered_edges,relative_vmin_vmax=relative_vmin_vmax)
    connectivity_str = '' if backdoor_connectivity is None else f'\nConnectivity = {backdoor_connectivity}'
    backdoor_axs[0].set_title(f'Backdoor Sample{connectivity_str}')

    backdoor_edge_mask_dict = mask_to_dict_g(backdoor_explanation.edge_mask.clone().detach(), backdoor_graph)
    backdoor_edge_mask = square_edge_mask(backdoor_graph.pyg_graph.x, backdoor_edge_mask_dict)
    edge_mask_heat_map(backdoor_edge_mask, backdoor_graph.pyg_graph, ax = backdoor_axs[1], relative_vmin_vmax=relative_vmin_vmax)
    
    inverse_indices = [0]* len(backdoor_graph.nx_graph.nodes())
    for i,idx in enumerate(list(backdoor_graph.nx_graph.nodes())):
        inverse_indices[idx] = i
    node_mask_heat_map(backdoor_explanation.node_mask[inverse_indices,:], ax = backdoor_axs[2], relative_vmin_vmax=relative_vmin_vmax)
    
    loss_max_backdoor, loss_min_backdoor = max(backdoor_explanation['clf_loss_over_time']), min(backdoor_explanation['clf_loss_over_time'])
    if do_clean==True:
        y_min, y_max = min(loss_min_clean, loss_min_backdoor),  max(loss_max_clean, loss_max_backdoor)
    else:
        y_min, y_max = loss_min_backdoor, loss_max_backdoor
    if do_loss==True:
        assert loss_types is not None
        plot_loss(backdoor_explanation, loss_types, ax=backdoor_axs[3],curvature=backdoor_curvature, elbow=backdoor_elbow)
        backdoor_axs[3].set_ylim(y_min, y_max)
        if do_clean==True:
            clean_axs[3].set_ylim(y_min, y_max)

    if title is not None:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def edge_mask_heat_map(edge_mask_square, pyg_graph, ax=None, cax=None, relative_vmin_vmax=True,title_font_size=12):
    (vmin, vmax) = (torch.min(edge_mask_square).item(), torch.max(edge_mask_square).item()) if relative_vmin_vmax==True else (0.2,1)
    vmin = vmin-0.4*np.abs(vmax)
    if ax is None:
        heatmap = plt.imshow(edge_mask_square, cmap='Reds',vmin=vmin, vmax=vmax)
        plt.title("Edge Mask",fontsize=title_font_size)
        plt.xlabel("Node 1")
        plt.ylabel("Node 2")
        plt.xticks(range(len(pyg_graph.x)))
        plt.yticks(range(len(pyg_graph.x)))
        plt.tick_params(length=0)
        if cax:
            cax.set_title('Mask Weights')
            cb = plt.colorbar(heatmap,ax=ax,cax=cax)
            cb.ax.tick_params(length=0)
        for row_index, row in enumerate(edge_mask_square):
            if sum(row) != 0:
                col_index_max_value = torch.argmax(row)
                rect = plt.Rectangle((col_index_max_value-0.5, row_index-0.5), 1, 1, edgecolor='black', facecolor='none', linewidth=1)
                plt.gca().add_patch(rect)
        return  heatmap
    elif ax is not None:
        heatmap = ax.imshow(edge_mask_square, cmap='Reds',vmin=vmin, vmax=vmax)
        ax.set_title("Edge Mask",fontsize=title_font_size)
        ax.set_xlabel("Node 1")
        ax.set_ylabel("Node 2")
        ax.set_xticks(range(len(pyg_graph.x)))
        ax.set_yticks(range(len(pyg_graph.x)))
        ax.tick_params(length=0)
        for row_index, row in enumerate(edge_mask_square):
            if sum(row) != 0:
                col_index_max_value = torch.argmax(row)
                rect = plt.Rectangle((col_index_max_value-0.5, row_index-0.5), 1, 1, edgecolor='black', facecolor='none', linewidth=1)
                ax.add_patch(rect)
        if cax:
            cb = plt.colorbar(heatmap, ax=ax, cax=cax)
            cb.ax.tick_params(length=0)
        return heatmap
    

def node_mask_heat_map(feat_mask, ax=None, cax=None, relative_vmin_vmax=True,title_font_size=12):
    (vmin, vmax) = (torch.min(feat_mask).item(), torch.max(feat_mask).item()) if relative_vmin_vmax==True else (0.2,1)
    vmin = vmin-0.4*np.abs(vmax)
    if feat_mask.shape[1] == 1:
        feat_mask = feat_mask.reshape(-1,1)
    else:
        pass
    if ax is None:
        heatmap = plt.imshow(feat_mask, cmap='Reds',vmin=vmin, vmax=vmax)
        plt.title("Node Feature Mask",fontsize=title_font_size)
        plt.xticks([])
        plt.tick_params(length=0)
        if feat_mask.shape[1] == 1:
            plt.ylabel("Feature")
        plt.yticks(range(len(feat_mask)))
        plt.ylabel('Node')

        for row_index, row in enumerate(feat_mask):
            col_index_max_value = torch.argmax(row)
            rect = plt.Rectangle((col_index_max_value-0.5, row_index-0.5), 1, 1, edgecolor='black', facecolor='none', linewidth=0.5)
            plt.gca().add_patch(rect)
    
    elif ax is not None:
        try:
            heatmap = ax.imshow(feat_mask, cmap='Reds',vmin=vmin, vmax=vmax)
        except:
            feat_mask = feat_mask.cpu().numpy()
            heatmap = ax.imshow(feat_mask, cmap='Reds',vmin=vmin, vmax=vmax)

        ax.set_title("Node Feature Mask",fontsize=title_font_size)
        ax.set_xticks([])
        ax.set_xlabel('Feature (Node Degree)')
        ax.set_yticks(range(len(feat_mask)))
        ax.set_ylabel('Node')
        ax.tick_params(length=0)
        if feat_mask.shape[1] == 1:
            ax.set_ylabel("Feature")

        for row_index, row in enumerate(feat_mask):
            col_index_max_value = np.argmax(row)
            rect = plt.Rectangle((col_index_max_value-0.5, row_index-0.5), 1, 1, edgecolor='black', facecolor='none', linewidth=0.5)
            ax.add_patch(rect)
    if cax:
        cax.set_title('Mask Weights',fontsize=title_font_size)
        cb = plt.colorbar(heatmap, ax=cax)
        cb.ax.tick_params(length=0)
    return heatmap

def plot_loss(explanation, title=None, ax=None, curvature=None, elbow=None):
    epochs = range(len(explanation['clf_loss_over_time']))
    loss = explanation['clf_loss_over_time']
    
    min_loss_epoch=len(epochs)  if min(loss)==loss[-1]  else    np.argmin(loss)
    max_loss_epoch=0            if max(loss)==loss[0]   else    np.argmax(loss)
    min_loss, max_loss = min(loss), max(loss)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f'Min Loss: {np.round(min_loss,2)} (Epoch {min_loss_epoch})', markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label=f'Max Loss: {np.round(max_loss,2)} (Epoch {max_loss_epoch})', markerfacecolor='orange', markersize=10)
    ]

    if curvature is not None:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'Max Curvature: {np.round(curvature,2)}', markerfacecolor='red', markersize=10))

    if elbow is not None:
        if elbow != 0:
            elbow -= 1
        legend_elements.append(Line2D([0], [0], color='black', linestyle='--', label=f'Elbow Epoch: {elbow}'))
        
    if ax is None:
        plt.figure()
        ax = plt.gca()

    ax.set_xticks(range(len(epochs)+1)[::5])
    ax.set_xticklabels(range(len(epochs)+1)[::5])

    ax.plot(epochs, loss, label='Loss')
    ax.scatter([min_loss_epoch, max_loss_epoch], [min_loss, max_loss], color=['blue', 'orange'])
    if elbow is not None:
        ax.axvline(elbow, linestyle='--', color='black')
        ax.scatter([elbow], [loss[elbow]], color='red')

    ax.legend(handles=legend_elements, loc='upper right')
    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

data_shape_dict = get_info('data_shape_dict')
src_dir     = get_info('src_dir')
data_dir    = get_info('data_dir')
explanation_dir = get_info('explanation_dir')
explainer_dir = get_info('explanation_dir')
train_dir   = get_info('train_dir')
train_dir_cln = get_info('train_dir_cln')


def line_patch(linestyle='-',color='blue',lw=2,ax=None):
    if ax is not None:
        patch, = ax.plot([0], [0], linestyle=linestyle, lw=lw, color=color)
    else:
        patch, = plt.plot([0],[0], linestyle=linestyle, lw=lw, color=color)
    return patch


def get_7_centered_axes(fig):
    spec = gridspec.GridSpec(ncols=9, nrows=2, figure=fig)
    ax1 = fig.add_subplot(spec[0,1:3])
    ax2 = fig.add_subplot(spec[0,3:5])
    ax3 = fig.add_subplot(spec[0,5:7])
    ax4 = fig.add_subplot(spec[0,7:9])
    ax5 = fig.add_subplot(spec[1,2:4])
    ax6 = fig.add_subplot(spec[1,4:6])
    ax7 = fig.add_subplot(spec[1,6:8])
    return [ax1,ax2,ax3,ax4,ax5,ax6,ax7]


def legend_title_plot_procedure(fig, handles,labels,ncol,bbox=[0.555,-0.08], hspace=0,loc='lower center',title=None):
    if title is not None:
        x_ = bbox[0]
        fig.suptitle(title,fontsize=20,x=x_)
    fig.legend(handles, labels, loc=loc,fancybox=True,shadow=True,ncol=ncol,bbox_to_anchor=bbox,fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(hspace=hspace)
    plt.show()  

def box_subplot(clean_data, backdoor_data, clean_val_data, inequality, ax, metric=None, lower_q = None, upper_q = None, show_outliers=False, highlight_index=None, highlight_category=None,yy=None):
    bp_clean = ax.boxplot(clean_data, positions=[1], widths=0.35, patch_artist=True, vert=False,showfliers=show_outliers, medianprops=dict(color = "darkorange", linewidth = 1))
    plt.setp(bp_clean['boxes'], facecolor='orange', color='orange',alpha=0.5, label='clean')
    bp_clean_val = ax.boxplot(clean_val_data, positions=[2], widths=0.35, patch_artist=True, vert=False,showfliers=show_outliers,medianprops=dict(color = "#555555", linewidth = 1))
    plt.setp(bp_clean_val['boxes'], facecolor='gray', color='gray',alpha=0.5, label='clean val')
    bp_backdoor = ax.boxplot(backdoor_data, positions=[3], widths=0.35, patch_artist=True, vert=False,showfliers=show_outliers, medianprops=dict(color = "darkblue", linewidth = 1))
    plt.setp(bp_backdoor['boxes'], facecolor='blue', color='blue',alpha=0.5, label='backdoor')
    highlight_marker=None
    if highlight_index is not None:
        if highlight_category=='clean':
            highlighted_point_value = clean_data[highlight_index] # Value you want to highlight
            highlight_y = 2
            highlight_marker = ax.scatter([highlighted_point_value], [highlight_y], marker='X', s=50, color='red',alpha=1)
        elif highlight_category=='backdoor':
            highlighted_point_value = backdoor_data[highlight_index] # Value you want to highlight
            highlight_y = 3
            highlight_marker = ax.scatter([highlighted_point_value], [highlight_y], marker='X', s=50, color='red',alpha=1)
    if lower_q is not None and upper_q is not None:
        if inequality=='more':
            clean_val_cutoff = np.quantile(clean_val_data, upper_q)
        elif inequality=='less':
            clean_val_cutoff = np.quantile(clean_val_data, lower_q)
    elif lower_q is None or upper_q is None:
        if inequality == 'more':
            clean_val_cutoff = bp_clean_val['whiskers'][1].get_xdata()[0]
        elif inequality == 'less':
            clean_val_cutoff = bp_clean_val['whiskers'][0].get_xdata()[1]
    yy=4.1 if yy is None else yy
    clean_val_line, = ax.plot([clean_val_cutoff, clean_val_cutoff], [0, yy-0.1], linestyle='--', color='black')
    min_val = min(np.min(clean_data),np.min(backdoor_data),np.min(clean_val_data))
    max_val = max(np.max(clean_data),np.max(backdoor_data),np.max(clean_val_data))
    clean_min, clean_max = bp_clean['caps'][0].get_xdata()[0],bp_clean['caps'][1].get_xdata()[0]
    backdoor_min, backdoor_max = bp_backdoor['caps'][0].get_xdata()[0],bp_backdoor['caps'][1].get_xdata()[0]
    clean_val_min, clean_val_max = bp_clean_val['caps'][0].get_xdata()[0],bp_clean_val['caps'][1].get_xdata()[0]
    min_val_ = min(clean_min,backdoor_min,clean_val_min)
    max_val_ = max(clean_max,backdoor_max,clean_val_max)
    (min_val, max_val) = (min_val_, max_val_) if ((min_val_ <= clean_val_cutoff) and (clean_val_cutoff <= max_val_)) else (min_val, max_val)
    curly_braces(min_val, clean_val_cutoff, max_val, yy, 'backdoor', inequality,ax)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_yticks([])
    xlabel = metric_plot_info_dict[metric]['x_label']
    xlabel = 'Distance from Clean Validation' if xlabel == 't-score' else xlabel.title()
    xlabel = 'Raw Value' if 'Distance' not in xlabel else xlabel
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylim(0, 5)
    metric_title = metric_plot_info_dict[metric]['Title'] if 'dist' not in metric else metric_plot_info_dict[metric]['Title'] + ' Distance'
    ax.set_title(metric_title,fontsize=16)
    return clean_val_line, highlight_marker


def explainer_metrics_boxplot(df,metrics,plot=True,save_image=False, plot_path=None,figsize=None, lower_q=0.25, 
                              upper_q=0.75, show_outliers=False, highlight_index=None, highlight_category=None,
                              yy=None):
    figsize = (20,6) if figsize is None else figsize
    if len(metrics)==5 or len(metrics)==7:
        fig = plt.figure(figsize=figsize,tight_layout=True)
        if len(metrics) == 5:
            spec = gridspec.GridSpec(ncols=7, nrows=2, figure=fig)
            ax1 = fig.add_subplot(spec[0,1:3])
            ax2 = fig.add_subplot(spec[0,3:5])
            ax3 = fig.add_subplot(spec[0,5:7])
            ax4 = fig.add_subplot(spec[1,2:4])
            ax5 = fig.add_subplot(spec[1,4:6])
            axs = [ax1,ax2,ax3,ax4,ax5]
        elif len(metrics) == 7:
            spec = gridspec.GridSpec(ncols=9, nrows=2, figure=fig)
            ax1 = fig.add_subplot(spec[0,1:3])
            ax2 = fig.add_subplot(spec[0,3:5])
            ax3 = fig.add_subplot(spec[0,5:7])
            ax4 = fig.add_subplot(spec[0,7:9])
            ax5 = fig.add_subplot(spec[1,2:4])
            ax6 = fig.add_subplot(spec[1,4:6])
            ax7 = fig.add_subplot(spec[1,6:8])
            axs = [ax1,ax2,ax3,ax4,ax5,ax6,ax7]
    else:
        shape_dict = {6: (2,3), 8: (2,4), 9: (3,3), 10: (2,5), 12: (4,3), 15: (5,3), 16: (4,4), 18: (6,3), 20: (5,4)}
        if len(metrics) not in shape_dict.keys():
            nrows, ncols = 1, len(metrics)
        else:
            (nrows, ncols) = shape_dict[len(metrics)]
        if nrows is not None and ncols is not None:
            if figsize==None:
                figsize = (4*ncols, 3*nrows)
        fig,axs = plt.subplots(nrows,ncols,figsize=figsize)     
    handle_list, label_list = None, None
    for i in range(len(metrics)):
        metric = metrics[i]
        ax = axs[i]
        clean_data = list(df[df['category']=='clean'][metric])
        backdoor_data = list(df[df['category']=='backdoor'][metric])
        clean_val_data = list(df[df['category']=='clean_val'][metric])
        inequality = metric_plot_info_dict[metric]['inequality']
        clean_cutoff, highlight_marker = box_subplot(clean_data, backdoor_data, clean_val_data, inequality, ax, metric=metric, 
                                                    show_outliers=show_outliers, lower_q=lower_q, upper_q=upper_q, highlight_index=highlight_index, 
                                                    highlight_category=highlight_category,yy=yy)
        if handle_list is None:
            patch_1 = Patch(color='blue', label='Backdoor',alpha=0.5)
            patch_2 = Patch(color='orange', label='Clean',alpha=0.5)
            patch_3 = Patch(color='gray', label='Clean Validation',alpha=0.5)
            patch_4 = clean_cutoff
            handle_list = [patch_1, patch_2, patch_3, patch_4]
            label_list = ['Backdoor','Clean','Clean Validation','Threshold']
            if highlight_index is not None:
                handle_list.append(highlight_marker)
                label_list.append(f'{highlight_category.title()} Sample #{highlight_index}')
    fig.legend(handle_list, label_list,loc='lower center', bbox_to_anchor=(0.555, -0.08), fancybox=True, shadow=True, ncol=4,fontsize=16)
    fig.subplots_adjust(hspace=0.05)
    plt.tight_layout()
    if save_image==True:
        plt.savefig(plot_path)
    # if plot == True:
        # plt.show()
    # else:
    # plt.close()