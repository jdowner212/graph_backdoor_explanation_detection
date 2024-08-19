from   utils.config import *
from   utils.data_utils import *
from   utils.general_utils import *
from   torch.nn import Linear, Sequential, BatchNorm1d, ReLU,  ModuleList, Dropout
import torch.nn.functional as F
from   abc import abstractmethod
import lightning as L
import pytorch_lightning as pl
import torch
import torch.nn as nn
from   torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, GCNConv, GINConv, TopKPooling, SAGEConv, global_mean_pool, global_add_pool, GraphConv, GATConv
import torch.nn.functional as F
from   torch import Tensor
import torch_geometric
import torch_geometric.nn as geom_nn

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


'''much borrowed from: https://github.com/zaixizhang/graphbackdoor/blob/main/util.py'''

data_shape_dict = get_info('data_shape_dict')
src_dir     = get_info('src_dir')
data_dir    = get_info('data_dir')
explain_dir = get_info('explain_dir')
train_dir   = get_info('train_dir')
train_dir_cln = get_info('train_dir_cln')

gnn_layer_by_name = {
    "GCN": torch_geometric.nn.GCNConv,
    "GAT": torch_geometric.nn.GATConv,
    "GraphConv": torch_geometric.nn.GraphConv
}


class myReLU(torch.nn.Module):
    def forward(self, x, edge_index):
        return F.relu(x)

class myBatchNorm(torch.nn.Module):
    def __init__(self, num_features):
        super(myBatchNorm, self).__init__()
        self.bn = torch.nn.BatchNorm1d(num_features)

    def forward(self, x, edge_index):
        out = self.bn(x)
        return out

class GIN4(torch.nn.Module):
    def __init__(
            self, **kwargs):
        super(GIN4, self).__init__()

        num_node_features = kwargs.get('num_node_features')
        hidden_channels = kwargs.get('hidden_channels')
        num_layers = kwargs.get('num_layers')
        num_classes = kwargs.get('num_classes')
        dropout = kwargs.get('dropout')

        self.act = torch.nn.Tanh()

        convolution_layers   = torch.nn.ModuleList()
        batch_normalizations = torch.nn.ModuleList()

        __mlp_layers = [torch.nn.Linear(num_node_features, hidden_channels[0])]
        for _ in range(num_layers - 1):
            __mlp_layers.append(self.act)
            __mlp_layers.append(torch.nn.Linear(hidden_channels[0], hidden_channels[0]))
        convolution_layers.append(
            GINConv(torch.nn.Sequential(*__mlp_layers))
        )
        batch_normalizations.append(torch.nn.BatchNorm1d(hidden_channels[0]))

        num_layers: int = len(hidden_channels)
        for layer in range(num_layers - 1):
            __mlp_layers = [torch.nn.Linear(hidden_channels[layer], hidden_channels[layer + 1])]
            for _ in range(num_layers - 1):
                __mlp_layers.append(self.act)
                __mlp_layers.append(
                    torch.nn.Linear(hidden_channels[layer + 1], hidden_channels[layer + 1])
                )
            convolution_layers.append(
                GINConv(torch.nn.Sequential(*__mlp_layers))
            )
            batch_normalizations.append(
                torch.nn.BatchNorm1d(hidden_channels[layer + 1])
            )

        self.__convolution_layers: torch.nn.ModuleList = convolution_layers
        self.__batch_normalizations: torch.nn.ModuleList = batch_normalizations

        self.lin1 = Linear(hidden_channels[-1], hidden_channels[-1])
        self.drop = Dropout(p=dropout)
        self.lin2 = Linear(hidden_channels[-1], num_classes)
        self.pool = global_mean_pool

    def forward(self, x, edge_index, batch=None):
        num_layers = len(self.__convolution_layers)
        for layer in range(num_layers):
            x = self.__convolution_layers[layer](x, edge_index)
            x = self.act(x)
            x  = self.__batch_normalizations[layer](x)

        x = self.pool(x, batch)
        x = self.lin1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.lin2(x)
        return x


class _ClassificationModel(torch.nn.Module):
    def __init__(self):
        super(_ClassificationModel, self).__init__()

    def cls_encode(self, data) -> torch.Tensor:
        raise NotImplementedError

    def cls_decode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def cls_forward(self, data) -> torch.Tensor:
        return self.cls_decode(self.cls_encode(data))

class ClassificationSupportedSequentialModel(_ClassificationModel):
    def __init__(self):
        super(ClassificationSupportedSequentialModel, self).__init__()

    @property
    def sequential_encoding_layers(self) -> torch.nn.ModuleList:
        raise NotImplementedError

    def cls_encode(self, data) -> torch.Tensor:
        raise NotImplementedError

    def cls_decode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class _SAGELayer(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dropout: float,
    ):
        super(_SAGELayer, self).__init__()
        self._convolution = SAGEConv(input_channels, output_channels)
        self._dropout = Dropout(dropout)

    def forward(self, x, edge_index, batch=None) -> torch.Tensor:
        x = self._convolution(x, edge_index)
        x = F.relu(x)
        x = self._dropout(x)
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, **kwargs):
        super(GraphSAGE, self).__init__()

        num_node_features = kwargs['num_node_features']
        num_classes = kwargs['num_classes']
        num_layers = kwargs['num_layers']
        hidden_channels = kwargs['hidden_channels']
        self.dropout = kwargs['dropout']

        self.layers = ModuleList([SAGEConv(num_node_features, hidden_channels)])
        self.layers += [SAGEConv(hidden_channels,hidden_channels)]*(num_layers-2)
        self.layers += [SAGEConv(hidden_channels,num_classes)]


    def forward(self, x, edge_index, batch=None):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = torch.nn.Dropout(p=self.dropout)(x)
        return x

class Topkpool(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Topkpool, self).__init__()
        self.num_features = kwargs["num_node_features"]
        self.num_classes = kwargs["num_classes"]
        self.ratio = kwargs["ratio"]
        self.dropout = kwargs["dropout"]
        self.num_graph_features = 0#kwargs["num_graph_features"]

        self.conv1 = GraphConv(self.num_features, 128)
        self.pool1 = TopKPooling(128, ratio=self.ratio)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=self.ratio)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=self.ratio)

        self.lin1 = torch.nn.Linear(256 + self.num_graph_features, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, self.num_classes)

    def forward(self, x,edge_index,batch=None):
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)

        return x

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

device = torch.device('mps')

class DiffPool(torch.nn.Module):
    def __init__(self, **kwargs):
        super(DiffPool, self).__init__()

        num_node_features  = kwargs['num_node_features']
        hidden_channels = kwargs['hidden_channels']
        num_classes = kwargs['num_classes']

        self.dropout = kwargs['dropout']

        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        self.pool1 = TopKPooling(hidden_channels, ratio=0.8)
        self.pool2 = TopKPooling(hidden_channels, ratio=0.8)

        self.fc1 = torch.nn.Linear(hidden_channels*2, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gap(x, batch), gmp(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gap(x, batch), gmp(x, batch)], dim=1)

        x = x1 + x2

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        return x


class PlainGCN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(PlainGCN, self).__init__()
        self.num_features = kwargs.get('num_node_features')
        self.num_classes = kwargs.get('num_classes')
        self.hidden = kwargs.get('hidden_channels')
        self.num_layers = kwargs.get('num_layers')
        self.pool = global_mean_pool
        self.lin = Linear(self.hidden, self.num_classes)

        layers = []

        for layer_i in range(self.num_layers):
            in_dim = self.num_features  if layer_i == 0 else self.hidden
            layers.append(GCNConv(in_dim, self.hidden))
            layers.append(myReLU())
            layers.append(Dropout(p=0.5))

        self.layers = Sequential(*layers)
        self.pool = global_mean_pool
        self.final_linear = Linear(self.hidden, self.num_classes)

    def forward(self, x, edge_index, batch=None):
        for i, layer in enumerate(self.layers):
            if (i+1)%3 != 0:
                x = layer(x, edge_index)
            else:
                x = layer(x)

        x = self.pool(x, batch)
        x_out = self.lin(x)

        return x_out

class GCN3(torch.nn.Module):
    def __init__(self, **kwargs):
        super(GCN3, self).__init__()
        torch.manual_seed(12345)

        self.dropout = kwargs['dropout']

        hidden_channels     = kwargs['hidden_channels']
        num_node_features   = kwargs['num_node_features']
        num_conv_layers     = kwargs['num_conv_layers']
        batchnorm           = kwargs['batchnorm']
        num_classes         = kwargs['num_classes']

        layers = []

        for layer_i in range(num_conv_layers):
            in_dim = num_node_features if layer_i == 0 else hidden_channels
            layers.append(GCNConv(in_dim, hidden_channels))

            if layer_i != num_conv_layers - 1:
                layers.append(myReLU())
                if batchnorm[layer_i] == True:
                    layers.append(myBatchNorm(hidden_channels))

        self.layers = Sequential(*layers)
        self.pool = global_mean_pool
        self.final_dropout = F.dropout
        self.final_linear = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch=None):

        for layer in self.layers:
            x = layer(x, edge_index)

        x = self.pool(x, batch)
        x = self.final_dropout(x, p=self.dropout, training=self.training)
        x = self.final_linear(x)
        return x



''' "carate" https://github.com/cap-jmk/carate-pwc/blob/main/carate/models/base_model.py'''

class Model(torch.nn.Module):
    @abstractmethod
    def __init__(self, **kwargs):
        super(Model, self).__init__()
        self.num_classes = kwargs.get('num_classes')
        self.num_node_features = kwargs.get('num_node_features')
        self.hidden_channels = kwargs.get('hidden_channels')

    @abstractmethod
    def forward(
        self,
            x: int,
            edge_index: int,
            batch: int = None,
            edge_weight=None
    ) -> torch.Tensor:
        pass

class Net(Model):

    def __init__(self, **kwargs) -> None:
        super(Net, self).__init__(
            hidden_channels=kwargs.get('hidden_channels'), num_classes=kwargs.get('num_classes'), num_node_features=kwargs.get('num_node_features')
        )
        num_heads = kwargs.get('num_heads')
        dropout_forward = kwargs.get('dropout_forward')
        hidden_channels = kwargs.get('hidden_channels')
        num_classes = kwargs.get('num_classes')
        num_node_features = kwargs.get('num_node_features')

        self.dropout_gat = kwargs.get('dropout_gat')

        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels, dropout=dropout_forward, heads=num_heads)
        self.conv5 = GraphConv(hidden_channels * num_heads, hidden_channels)

        self.fc1 = Linear(hidden_channels, hidden_channels)
        self.fc2 = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout_gat, training=self.training)
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = F.relu(self.conv5(x, edge_index, edge_weight))

        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)

    def __str__(self):
        return "cgc_classification"
''''''

class NodeDropTransform(object):
    def __init__(self, drop_prob=0.2):
        self.drop_prob = drop_prob

    def __call__(self, data):
        mask = torch.rand(data.x.size(0)) > self.drop_prob
        data.x = data.x[mask]
        data.batch = data.batch[mask]

        removed_nodes = torch.where(mask == False)[0]
        mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)
        for node in removed_nodes:
            mask &= ~(data.edge_index == node).any(dim=0)
        data.edge_index = data.edge_index[:, mask]
        return data


class GIN(torch.nn.Module):
    """
    GIN
    https://mlabonne.github.io/blog/posts/2022-04-25-Graph_Isomorphism_Network.html
    """

    def __init__(self, **kwargs):
        super(GIN, self).__init__()
        num_node_features = kwargs.get('num_node_features')
        hidden_channels = kwargs.get('hidden_channels')
        num_classes = kwargs.get('num_classes')
        self.conv1 = GINConv(
            Sequential(Linear(num_node_features, hidden_channels),
                       BatchNorm1d(hidden_channels), ReLU(),
                       Linear(hidden_channels, hidden_channels), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(hidden_channels, hidden_channels), BatchNorm1d(hidden_channels), ReLU(),
                       Linear(hidden_channels, hidden_channels), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(hidden_channels, hidden_channels), BatchNorm1d(hidden_channels), ReLU(),
                       Linear(hidden_channels, hidden_channels), ReLU()))
        self.lin1 = Linear(hidden_channels * 3, hidden_channels * 3)
        self.relu = ReLU()
        self.lin2 = Linear(hidden_channels * 3, num_classes)
        self.pool1 = global_add_pool
        self.pool2 = global_add_pool
        self.pool3 = global_add_pool
        self.dropout = Dropout(kwargs.get('dropout'))

    def forward(self, x, edge_index, batch=None):
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        h1 = self.pool1(h1, batch)
        h2 = self.pool2(h2, batch)
        h3 = self.pool3(h3, batch)

        h = torch.cat((h1, h2, h3), dim=1)

        h = self.lin1(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.lin2(h)

        return h


class GIN2(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        num_node_features = kwargs.get('num_node_features')
        hidden_channels = kwargs.get('hidden_channels')
        num_layers = kwargs.get('num_layers')
        num_classes = kwargs.get('num_classes')
        self.dropout = kwargs.get('dropout')
        self.initialization = GINConv(
            Sequential(
                Linear(num_node_features, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                BatchNorm1d(hidden_channels),
            ),
	    eps = 0.,
	    train_eps=False)
        self.mp_layers = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.mp_layers.append(
                GINConv(
                    Sequential(
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        BatchNorm1d(hidden_channels),
                    ),
		    eps=0.,
		    train_eps=False)
	    )
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.relu = ReLU()
        self.drop = Dropout(p=self.dropout)
        self.lin2 = Linear(hidden_channels, num_classes)


    def forward(self, x, edge_index, batch=None):
        x = self.initialization(x, edge_index)
        for conv in self.mp_layers:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.lin2(x)
        return x



class GIN3(torch.nn.Module):
    def __init__(self, **kwargs):
        super(GIN3, self).__init__()
        self.dropout = kwargs.get('dropout')
        num_node_features = kwargs.get('num_node_features')
        hidden_channels = kwargs.get('hidden_channels')
        num_layers = kwargs.get('num_layers')
        num_classes = kwargs.get('num_classes')
        self.conv1 = GINConv(Sequential(
            Linear(num_node_features, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            BatchNorm1d(hidden_channels),
        ),
            train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(Sequential(
                    Linear(hidden_channels, hidden_channels),
                    ReLU(),
                    Linear(hidden_channels, hidden_channels),
                    ReLU(),
                    BatchNorm1d(hidden_channels),
                ),
                    train_eps=True))
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


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
#         # Drop edges
#         edge_mask = torch.rand(data.edge_index.size(1)) > self.edge_drop_prob
#         data.edge_index = data.edge_index[:, edge_mask]

#         # Add edges
#         num_nodes = data.num_nodes
#         num_edges_add = int(self.edge_add_prob * data.edge_index.size(1))
#         random_edges = torch.randint(0, num_nodes, (2, num_edges_add))
#         data.edge_index = torch.cat([data.edge_index, random_edges], dim=1)

#         return data

class GATLayer(nn.Module):
    def __init__(self, num_heads=1, concat_heads=True, alpha=0.2, **kwargs):
        """
        Args:
            c_in: Dimensionality of input features
            c_out: Dimensionality of output features
            num_heads: Number of heads, i.e. attention mechanisms to apply in parallel. The
                        output features are equally split up over the heads if concat_heads=True.
            concat_heads: If True, the output of the different heads is concatenated instead of averaged.
            alpha: Negative slope of the LeakyReLU activation.
        """
        super().__init__()
        num_node_features = kwargs.get('num_node_features')
        num_classes = kwargs.get('num_classes')


        self.num_heads = num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert num_classes % num_heads == 0, "Number of output features must be a multiple of the count of heads."
            num_classes = num_classes // num_heads

        # Sub-modules and parameters needed in the layer
        self.projection = nn.Linear(num_node_features, num_classes * num_heads)
        self.a = nn.Parameter(Tensor(num_heads, 2 * num_classes))  # One per head
        self.leakyrelu = nn.LeakyReLU(alpha)

        # Initialization from the original implementation
        nn.init.xavier_uniform_(self.projection.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, node_feats, adj_matrix, print_attn_probs=False):
        """Forward.

        Args:
            node_feats: Input features of the node. Shape: [batch_size, c_in]
            adj_matrix: Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
            print_attn_probs: If True, the attention weights are printed during the forward pass
                               (for debugging purposes)
        """
        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)

        node_feats = self.projection(node_feats)
        node_feats = node_feats.view(batch_size, num_nodes, self.num_heads, -1)
        edges = adj_matrix.nonzero(as_tuple=False)
        node_feats_flat = node_feats.view(batch_size * num_nodes, self.num_heads, -1)
        edge_indices_row = edges[:, 0] * num_nodes + edges[:, 1]
        edge_indices_col = edges[:, 0] * num_nodes + edges[:, 2]
        a_input = torch.cat(
            [
                torch.index_select(input=node_feats_flat, index=edge_indices_row, dim=0),
                torch.index_select(input=node_feats_flat, index=edge_indices_col, dim=0),
            ],
            dim=-1)

        attn_logits = torch.einsum("bhc,hc->bh", a_input, self.a)
        attn_logits = self.leakyrelu(attn_logits)

        attn_matrix = attn_logits.new_zeros(adj_matrix.shape + (self.num_heads,)).fill_(-9e15)
        attn_matrix[adj_matrix[..., None].repeat(1, 1, 1, self.num_heads) == 1] = attn_logits.reshape(-1)

        attn_probs = F.softmax(attn_matrix, dim=2)
        if print_attn_probs:
            print("Attention probs\n", attn_probs.permute(0, 3, 1, 2))
        node_feats = torch.einsum("bijh,bjhc->bihc", attn_probs, node_feats)

        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(dim=2)

        return node_feats
    
gnn_layer_by_name = {"GCN": geom_nn.GCNConv, "GAT": geom_nn.GATConv, "GraphConv": geom_nn.GraphConv}

class MLPModel(nn.Module):
    def __init__(self, num_layers=2, dp_rate=0.1, **kwargs):
        """MLPModel.

        Args:
            c_in: Dimension of input features
            c_hidden: Dimension of hidden features
            c_out: Dimension of the output features. Usually number of classes in classification
            num_layers: Number of hidden layers
            dp_rate: Dropout rate to apply throughout the network
        """
        super().__init__()
        num_node_features = kwargs.get('num_node_features')
        num_classes = kwargs.get('num_classes')
        hidden_channels = kwargs.get('hidden_channels')

        layers = []
        for l_idx in range(num_layers - 1):
            layers += [nn.Linear(num_node_features, hidden_channels), nn.ReLU(inplace=True), nn.Dropout(dp_rate)]
            num_node_features = hidden_channels
        layers += [nn.Linear(num_node_features, num_classes)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        return self.layers(x)


import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_max_pool


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(in_feats, out_feats)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return F.relu(x)


class GCN_(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[64, 32], dropout=0.2):
        super(GCN_, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(in_dim, hidden_dim[0]))
        for i in range(len(hidden_dim) - 1):
            self.layers.append(GCNLayer(hidden_dim[i], hidden_dim[i + 1]))
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim[-1], out_dim)
        )

    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = layer(x, edge_index)
        x = global_max_pool(x, batch)
        x = self.fc(x)
        return x


class GNNModel(nn.Module):
    def __init__(
        self,
        num_layers=2,
        dp_rate=0.1,
        edge_weight=False,
        **kwargs):

        super().__init__()
        num_node_features   = kwargs.get('num_node_features')
        hidden_channels     = kwargs.get('hidden_channels')
        self.edge_weight = edge_weight  # Store the choice
        layers = []
        for l_idx in range(num_layers - 1):
            layers += [
                geom_nn.GraphConv(
                                  in_channels=num_node_features,
                                  out_channels=hidden_channels,
                                  **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate),
            ]
            num_node_features = hidden_channels
        layers += [geom_nn.GraphConv(in_channels=hidden_channels,
                                     out_channels=hidden_channels,
                                     **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, batch=None, edge_weight=None):
        for layer in self.layers:
            if isinstance(layer, geom_nn.MessagePassing) and self.edge_weight==True:
                x = layer(x, edge_index, edge_weight=edge_weight)
            elif isinstance(layer, geom_nn.MessagePassing) and self.edge_weight==False:
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x
    

class GraphGNNModel(nn.Module):
    def __init__(self, dp_rate_linear=0.5, **kwargs):
        super().__init__()
        num_classes = kwargs.get('num_classes')
        hidden_channels = kwargs.get('hidden_channels')
        self.GNN = GNNModel(**kwargs)  # Not our prediction output yet!
        self.head = nn.Sequential(nn.Dropout(dp_rate_linear), nn.Linear(hidden_channels, num_classes))

    def forward(self, x, edge_index, batch_idx=None, edge_weight=None):
        if edge_weight is not None:
            x = self.GNN(x, edge_index, edge_weight=edge_weight)
        else:
            x = self.GNN(x, edge_index)
        x = geom_nn.global_mean_pool(x, batch_idx)
        x = self.head(x)

        return x
    

class GraphLevelGNN(L.LightningModule):
    def __init__(self, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = GraphGNNModel(**model_kwargs)
        self.loss_module = nn.BCEWithLogitsLoss() if self.hparams.c_out == 1 else nn.CrossEntropyLoss()
        self.train_losses = []
        self.test_losses = []
        self.train_accs = []
        self.test_accs = []

    def forward(self, x, edge_index, batch=None, mode='train'):
        x = self.model(x, edge_index, batch_idx=batch)
        x = x.squeeze(dim=-1)
        return x


class GNNModel_opt(torch.nn.Module):
    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GCN", dp_rate=0.1, **kwargs):
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers-1):
            layers += [
                gnn_layer(in_channels=in_channels,
                          out_channels=out_channels,
                          **kwargs),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(dp_rate)]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels,
                             out_channels=c_out,
                             **kwargs)]
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_weight=None, use_edge_weight=False):
        for l in self.layers:
            if isinstance(l, torch_geometric.nn.MessagePassing):
                if use_edge_weight == True:
                    x = l(x, edge_index, edge_weight=edge_weight)
                else:
                    x = l(x, edge_index)
            else:
                x = l(x)
        return x
    
class GraphGNNModel_opt(torch.nn.Module):
    def __init__(self, c_in, c_hidden, c_out, dp_rate_linear=0.5, **kwargs):
        super().__init__()

        self.GNN = GNNModel_opt(c_in=c_in,
                            c_hidden=c_hidden,
                            c_out=c_hidden,
                            **kwargs)
        self.head = torch.nn.Sequential(
            torch.nn.Dropout(dp_rate_linear),
            torch.nn.Linear(c_hidden, c_out))
        
    def forward(self, x, edge_index, batch_idx, edge_weight=None, use_edge_weight=True):
        if use_edge_weight == True:
            x = self.GNN(x, edge_index, edge_weight=edge_weight, use_edge_weight=True)
        else:
            x = self.GNN(x, edge_index)
        x = torch_geometric.nn.global_mean_pool(x, batch_idx)
        x = self.head(x)
        return x

class GraphLevelGNN_opt(pl.LightningModule):
    def __init__(self, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = GraphGNNModel_opt(**model_kwargs)
        self.loss_module = torch.nn.BCEWithLogitsLoss() if self.hparams.c_out == 1 else torch.nn.CrossEntropyLoss()
        self.lr = model_kwargs['lr']
        self.weight_decay = model_kwargs['weight_decay']

    def forward_not_pl(self, data, mode='train', use_edge_weight=False):
        x, edge_index, batch_idx, edge_weight = data.x, data.edge_index, data.batch, data.edge_weight
        if use_edge_weight == True:
            x = self.model(x, edge_index, batch_idx, edge_weight=edge_weight, use_edge_weight=True)
        else:
            x = self.model(x, edge_index, batch_idx)
        x = x.squeeze(dim=-1)
        return x
    
    def forward(self, data, mode="train"):
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch
        x = self.model(x, edge_index, batch_idx)
        x = x.squeeze(dim=-1)
        if self.hparams.c_out == 1:
            preds = (x > 0).float()
            data.y = data.y.float()
        else:
            preds = x.argmax(dim=-1)
        loss = self.loss_module(x, data.y.long())
        acc = (preds == data.y).sum().float() / preds.shape[0]
        return loss, acc

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay) # High lr because of small dataset and small model
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")

        self.log('train_loss', loss, batch_size=len(batch),prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, batch_size=len(batch),prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="val")
        self.log('val_loss', loss, batch_size=len(batch), prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, batch_size=len(batch), prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log('test_acc', acc, batch_size=len(batch), prog_bar=True, on_step=False, on_epoch=True)

    def test_step_2(self, batch, batch_idx=None, use_edge_weight=False):
        _, acc = self.forward_not_pl(batch, mode="test", use_edge_weight= use_edge_weight)
        self.log('test_acc', acc, batch_size=len(batch), prog_bar=True, on_step=False, on_epoch=True)


class generatorGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_possible_edges):
        super(generatorGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 32)
        self.edge_classifier = torch.nn.Linear(64, 1)

    def forward(self, x, edge_index, edge_weight=None):
        # Node feature extraction
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        # Edge feature construction
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        # Edge classification
        edge_pred = self.edge_classifier(edge_features)
        return edge_pred

class EdgeGenerator(torch.nn.Module):
    def __init__(self, node_features):
        super(EdgeGenerator, self).__init__()
        self.fc = torch.nn.Linear(in_features=node_features * 2, out_features=1)

    def forward(self, data, batch=None):
        x = data.x
        edge_index = data.edge_index
        num_nodes = x.size(0)
        possible_edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j and [i, j] not in edge_index.t().tolist() and [j, i] not in edge_index.t().tolist()]
        for edge in data.edge_index.t().tolist():
            assert (edge[0],edge[1]) not in possible_edges
        possible_edges = torch.tensor(possible_edges, dtype=torch.long).t().contiguous()
        edge_features = torch.cat((x[possible_edges[0]], x[possible_edges[1]]), dim=1)
        edge_features = edge_features + torch.randn_like(edge_features)
        print('\n\n\nedge_features:')
        print(edge_features)
        edge_scores = self.fc(edge_features).squeeze()
        print('edge_scores:')
        print(edge_scores)
        return edge_scores, possible_edges
    

class EdgeGeneratorHeavy(torch.nn.Module):
    def __init__(self, node_features, hidden_dim=64, depth=2, dropout_prob=0):
        super(EdgeGeneratorHeavy, self).__init__()
        self.fc_input = torch.nn.Linear(node_features * 2, hidden_dim)
        self.fc_hidden = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(depth - 1)])
        self.fc_output = torch.nn.Linear(hidden_dim, 1)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.batch_norms = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim) for _ in range(depth - 1)])

    def forward(self, data, batch=None):
        x = data.x
        possible_edges = get_all_possible_edges(data)
        possible_edges = torch.tensor(possible_edges, dtype=torch.long).t().contiguous()
        edge_features  = torch.cat((x[possible_edges[0]], x[possible_edges[1]]), dim=1)
        edge_features = self.fc_input(edge_features)
        edge_features = self.activation(edge_features)
        edge_features = self.dropout(edge_features)
        for fc_hidden, batch_norm in zip(self.fc_hidden, self.batch_norms):
            edge_features = fc_hidden(edge_features)
            try:
                edge_features = batch_norm(edge_features)
            except:
                pass
            edge_features = self.activation(edge_features)
            edge_features = self.dropout(edge_features)
        edge_scores = self.fc_output(edge_features).squeeze()
        return edge_scores, possible_edges


model_type_class_dict = {'gcn': GCN3, 'gin': GIN, 'gin2': GIN2,
                                     'gin3': GIN3, 'gin4': GIN4, 'gcn_plain': PlainGCN,
                                     'carate': Net, 'diffpool': DiffPool, 'topkpool': Topkpool,
                                     'sage': GraphSAGE, 'graphgnn': GraphGNNModel, 'graphlevelgnn': GraphLevelGNN}
def model_dict():
    return model_type_class_dict

