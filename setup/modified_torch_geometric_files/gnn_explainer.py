from math import sqrt
from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torchviz import make_dot
import networkx as nx
import torch.nn.functional as F
from torch_geometric.explain import ExplainerConfig, Explanation, ModelConfig
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel
from torch.autograd import gradcheck
import torch


class GNNExplainer(ExplainerAlgorithm):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and node features that play a crucial role in the predictions
    made by a GNN.

    .. note::

        For an example of using :class:`GNNExplainer`, see
        `examples/explain/gnn_explainer.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/explain/gnn_explainer.py>`_,
        `examples/explain/gnn_explainer_ba_shapes.py <https://github.com/
        pyg-team/pytorch_geometric/blob/master/examples/
        explain/gnn_explainer_ba_shapes.py>`_, and `examples/explain/
        gnn_explainer_link_pred.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/explain/gnn_explainer_link_pred.py>`_.

    Args:
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        **kwargs (optional): Additional hyper-parameters to override default
            settings in
            :attr:`~torch_geometric.explain.algorithm.GNNExplainer.coeffs`.
    """

    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
        'EPS': 1e-15,
    }

    def __init__(self,
                 epochs: int = 100, 
                 lr: float = 0.01, 
                 coeffs_dict: dict = None,
                 retain_graph_loss=False):
        
        super().__init__()
        self.epochs = epochs
        self.lr = lr

        if coeffs_dict is not None:
            self.coeffs.update(coeffs_dict)

        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None
        self.save_masks = False
        self.node_mask_over_time = None
        self.edge_mask_over_time = None
        self.clf_loss_over_time = []
        self.edge_size_loss_over_time = []
        self.edge_ent_loss_over_time = []
        self.node_size_loss_over_time = []
        self.node_ent_loss_over_time = []
        self.node_mask_grads_over_time = []
        self.edge_mask_grads_over_time = []
        self.predictions_over_time = []
        self.apply_sigmoid=False
        self.verbose=False
        self.epoch=0
        self.retain_graph_loss=retain_graph_loss

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")

        self._train(model, x, edge_index, target=target, index=index, **kwargs)

        node_mask = self._post_process_mask(
            self.node_mask,
            self.hard_node_mask,
            apply_sigmoid=self.apply_sigmoid,
        )

        try:
            edge_mask = self._post_process_mask(
                self.edge_mask,
                self.hard_edge_mask,
                apply_sigmoid=self.apply_sigmoid,
            )
        except:
            edge_mask = self.edge_mask


        self._clean_model(model)

        metrics_over_time = {'edge_mask_over_time':self.edge_mask_over_time,
                             'node_mask_over_time':self.node_mask_over_time,
                             'edge_grads_over_time':self.edge_mask_grads_over_time,
                             'node_grads_over_time':self.node_mask_grads_over_time,
                             'clf_loss_over_time':self.clf_loss_over_time,
                             'node_size_loss_over_time':self.node_size_loss_over_time,
                             'node_ent_loss_over_time':self.node_ent_loss_over_time,
                             'edge_size_loss_over_time':self.edge_size_loss_over_time,
                             'edge_ent_loss_over_time':self.edge_ent_loss_over_time,
                             'predictions_over_time': self.predictions_over_time,
                             'final_loss': self.final_loss
                             }
        explanation = Explanation(node_mask=node_mask, edge_mask=edge_mask, **metrics_over_time)

        return explanation

    def supports(self) -> bool:
        return True

    def _train(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):
        
        self._initialize_masks(x, edge_index)

        parameters = []

        node_mask_over_time = []
        edge_mask_over_time = []

        node_mask_grads_over_time = []
        edge_mask_grads_over_time = []

        if self.node_mask is not None:
            parameters.append(self.node_mask)
        if self.edge_mask is not None:
            set_masks(model, self.edge_mask, edge_index, apply_sigmoid=True)
            parameters.append(self.edge_mask)


        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        for i in range(self.epochs):
            optimizer.zero_grad()

            h = x if self.node_mask is None else x * self.node_mask.sigmoid()
            
            y_hat, y = model(h, edge_index, **kwargs), target

            self.predictions_over_time.append(y_hat)

            if index is not None:
                y_hat, y = y_hat[index], y[index]

            loss = self._loss(y_hat, y, edge_index)
            loss.backward(retain_graph=self.retain_graph_loss)
            optimizer.step()
            self.final_loss = loss

            '''
            In the first iteration, we collect the nodes and edges that are
            involved into making the prediction. These are all the nodes and
            edges with gradient != 0 (without regularization applied).
            '''
            if i == 0 and self.node_mask is not None:
                self.hard_node_mask = self.node_mask.grad != 0.0
            if i == 0 and self.edge_mask is not None:
                self.hard_edge_mask = self.edge_mask.grad != 0.0


            ''' 
            Appending current mask and grad values to lists 
            '''
            if self.node_mask is not None:
                current_node_mask = self.node_mask.clone().detach()
                current_node_mask = self._post_process_mask(current_node_mask, apply_sigmoid=self.apply_sigmoid)
                node_mask_over_time.append(current_node_mask)
                node_mask_grads_over_time.append(self.node_mask.grad.clone().detach())

            if self.edge_mask is not None:
                current_edge_mask = self.edge_mask.clone().detach()
                current_edge_mask = self._post_process_mask(current_edge_mask, apply_sigmoid=self.apply_sigmoid)
                edge_mask_over_time.append(current_edge_mask)
                edge_mask_grads_over_time.append(self.edge_mask.grad.clone().detach())

            self.epoch += 1

        '''
        Saving mask and grad values from all epochs
        '''
        if self.save_masks==True:
            self.node_mask_over_time = node_mask_over_time
            self.edge_mask_over_time = edge_mask_over_time
            self.node_mask_grads_over_time = node_mask_grads_over_time
            self.edge_mask_grads_over_time = edge_mask_grads_over_time

    def _initialize_masks(self, x: Tensor, edge_index: Tensor):
        node_mask_type = self.explainer_config.node_mask_type
        edge_mask_type = self.explainer_config.edge_mask_type

        device = x.device
        (N, F), E = x.size(), edge_index.size(1)
        std = 0.1

        if node_mask_type is None:
            self.node_mask = None
        elif node_mask_type == MaskType.object:
            self.node_mask = Parameter(torch.randn(N, 1, device=device) * std)
        elif node_mask_type == MaskType.attributes:
            self.node_mask = Parameter(torch.randn(N, F, device=device) * std)
        elif node_mask_type == MaskType.common_attributes:
            self.node_mask = Parameter(torch.randn(1, F, device=device) * std)
        else:
            assert False

        if edge_mask_type is None:
            self.edge_mask = None
        elif edge_mask_type == MaskType.object:
            std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
            self.edge_mask = Parameter(torch.randn(E, device=device) * std)

        else:
            assert False

    def _loss(self, y_hat: Tensor, y: Tensor, edge_index: Tensor) -> Tensor:

        if self.model_config.mode == ModelMode.binary_classification:
            clf_loss = self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            clf_loss = self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            clf_loss = self._loss_regression(y_hat, y)
        else:
            assert False
        self.clf_loss_over_time.append(clf_loss.clone().detach().item())

        edge_size_loss, edge_ent_loss, node_size_loss, node_ent_loss = 0, 0, 0, 0
        
        if self.hard_edge_mask is not None:
            assert self.edge_mask is not None

            m = self.edge_mask[self.hard_edge_mask].sigmoid()
            edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
            edge_size_loss = self.coeffs['edge_size'] * edge_reduce(m)
            self.edge_size_loss_over_time.append(edge_size_loss)

            ent = -m * torch.log(m + self.coeffs['EPS']) - (1 - m) * torch.log(1 - m + self.coeffs['EPS'])
            edge_ent_loss = self.coeffs['edge_ent'] * ent.mean()
            try:
                self.edge_ent_loss_over_time.append(edge_ent_loss.item())
            except:
                self.edge_ent_loss_over_time.append(edge_ent_loss)

        if self.hard_node_mask is not None:
            assert self.node_mask is not None
            m = self.node_mask[self.hard_node_mask].sigmoid()

            node_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
            try:
                node_size_loss = self.coeffs['node_feat_size'] * node_reduce(m)
            except:
                node_size_loss = self.coeffs['node_feat_size'] * node_reduce(m,dim=0)
            self.node_size_loss_over_time.append(node_size_loss)


            ent = -m * torch.log(m + self.coeffs['EPS']) - (1 - m) * torch.log(1 - m + self.coeffs['EPS'])
            node_ent_loss = self.coeffs['node_feat_ent'] * ent.mean()
            try:
                self.node_ent_loss_over_time.append(node_ent_loss.item())
            except:
                self.node_ent_loss_over_time.append(node_ent_loss)

        loss = clf_loss + edge_size_loss + edge_ent_loss + node_size_loss + node_ent_loss
        return loss

    def _clean_model(self, model):
        clear_masks(model)
        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None


class GNNExplainer_:
    r"""Deprecated version for :class:`GNNExplainer`."""

    coeffs = GNNExplainer.coeffs

    conversion_node_mask_type = {
        'feature': 'common_attributes',
        'individual_feature': 'attributes',
        'scalar': 'object',
    }

    conversion_return_type = {
        'log_prob': 'log_probs',
        'prob': 'probs',
        'raw': 'raw',
        'regression': 'raw',
    }

    def __init__(
        self,
        model: torch.nn.Module,
        epochs: int = 100,
        lr: float = 0.01,
        return_type: str = 'log_prob',
        feat_mask_type: str = 'feature',
        allow_edge_mask: bool = True,
        **kwargs,
    ):
        assert feat_mask_type in ['feature', 'individual_feature', 'scalar']

        explainer_config = ExplainerConfig(
            explanation_type='model',
            node_mask_type=self.conversion_node_mask_type[feat_mask_type],
            edge_mask_type=MaskType.object if allow_edge_mask else None,
        )
        model_config = ModelConfig(
            mode='regression'
            if return_type == 'regression' else 'multiclass_classification',
            task_level=ModelTaskLevel.node,
            return_type=self.conversion_return_type[return_type],
        )

        self.model = model
        self._explainer = GNNExplainer(epochs=epochs, lr=lr, **kwargs)
        self._explainer.connect(explainer_config, model_config)

    @torch.no_grad()
    def get_initial_prediction(self, *args, **kwargs) -> Tensor:

        training = self.model.training
        self.model.eval()

        out = self.model(*args, **kwargs)
        if (self._explainer.model_config.mode ==
                ModelMode.multiclass_classification):
            out = out.argmax(dim=-1)

        self.model.train(training)

        return out

    def explain_graph(
        self,
        x: Tensor,
        edge_index: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        self._explainer.model_config.task_level = ModelTaskLevel.graph

        explanation = self._explainer(
            self.model,
            x,
            edge_index,
            target=self.get_initial_prediction(x, edge_index, **kwargs),
            **kwargs,
        )
        return self._convert_output(explanation, edge_index)

    def explain_node(
        self,
        node_idx: int,
        x: Tensor,
        edge_index: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        self._explainer.model_config.task_level = ModelTaskLevel.node
        explanation = self._explainer(
            self.model,
            x,
            edge_index,
            target=self.get_initial_prediction(x, edge_index, **kwargs),
            index=node_idx,
            **kwargs,
        )
        return self._convert_output(explanation, edge_index, index=node_idx,
                                    x=x)

    def _convert_output(self, explanation, edge_index, index=None, x=None):
        node_mask = explanation.get('node_mask')
        edge_mask = explanation.get('edge_mask')

        if node_mask is not None:
            node_mask_type = self._explainer.explainer_config.node_mask_type
            if node_mask_type in {MaskType.object, MaskType.common_attributes}:
                node_mask = node_mask.view(-1)

        if edge_mask is None:
            if index is not None:
                _, edge_mask = self._explainer._get_hard_masks(
                    self.model, index, edge_index, num_nodes=x.size(0))
                edge_mask = edge_mask.to(x.dtype)
            else:
                edge_mask = torch.ones(edge_index.shape[1],
                                       device=edge_index.device)

        return node_mask, edge_mask
