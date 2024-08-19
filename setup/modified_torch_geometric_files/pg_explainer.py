import logging
from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import ReLU, Sequential

from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import (
    ExplanationType,
    ModelMode,
    ModelTaskLevel,
)
from torch_geometric.nn import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.utils import get_embeddings


class PGExplainer(ExplainerAlgorithm):
    r"""The PGExplainer model from the `"Parameterized Explainer for Graph
    Neural Network" <https://arxiv.org/abs/2011.04573>`_ paper.
    Internally, it utilizes a neural network to identify subgraph structures
    that play a crucial role in the predictions made by a GNN.
    Importantly, the :class:`PGExplainer` needs to be trained via
    :meth:`~PGExplainer.train` before being able to generate explanations:

    .. code-block:: python

        explainer = Explainer(
            model=model,
            algorithm=PGExplainer(epochs=30, lr=0.003),
            explanation_type='phenomenon',
            edge_mask_type='object',
            model_config=ModelConfig(...),
        )

        # Train against a variety of node-level or graph-level predictions:
        for epoch in range(30):
            for index in [...]:  # Indices to train against.
                loss = explainer.algorithm.train(epoch, model, x, edge_index,
                                                 target=target, index=index)

        # Get the final explanations:
        explanation = explainer(x, edge_index, target=target, index=0)

    Args:
        epochs (int): The number of epochs to train.
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.003`).
        **kwargs (optional): Additional hyper-parameters to override default
            settings in
            :attr:`~torch_geometric.explain.algorithm.PGExplainer.coeffs`.
    """

    coeffs = {
        'edge_size': 0.05,
        'edge_ent': 1.0,
        'temp': [5.0, 2.0],
        'bias': 0.0,
    }

    def __init__(self, epochs: int, lr: float = 0.003, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)

        self.mlp = Sequential(
            Linear(-1, 64),
            ReLU(),
            Linear(64, 1),
        )
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        self._curr_epoch = -1


        ''''''
        self.edge_mask = self.hard_edge_mask = None
        self.save_masks = False
        self.edge_mask_over_time = None
        self.clf_loss_over_time = []
        self.edge_size_loss_over_time = []
        self.edge_ent_loss_over_time = []
        self.edge_mask_grads_over_time = []
        self.predictions_over_time = []
        self.apply_sigmoid=False
        self.verbose=False
        self.epoch=0
        ''''''

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.mlp)



    def train(
        self,
        # epoch: int,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):
        r"""Trains the underlying explainer model.
        Needs to be called before being able to make predictions.

        Args:
            epoch (int): The current epoch of the training phase.
            model (torch.nn.Module): The model to explain.
            x (torch.Tensor): The input node features of a
                homogeneous graph.
            edge_index (torch.Tensor): The input edge indices of a homogeneous
                graph.
            target (torch.Tensor): The target of the model.
            index (int or torch.Tensor, optional): The index of the model
                output to explain. Needs to be a single index.
                (default: :obj:`None`)
            **kwargs (optional): Additional keyword arguments passed to
                :obj:`model`.
        """
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")

        if self.model_config.task_level == ModelTaskLevel.node:
            if index is None:
                raise ValueError(f"The 'index' argument needs to be provided "
                                 f"in '{self.__class__.__name__}' for "
                                 f"node-level explanations")
            if isinstance(index, Tensor) and index.numel() > 1:
                raise ValueError(f"Only scalars are supported for the 'index' "
                                 f"argument in '{self.__class__.__name__}'")



        edge_mask_over_time = []
        edge_mask_grads_over_time = []

        for epoch in range(self.epochs):            

            z = get_embeddings(model, x, edge_index, **kwargs)[-1]

            self.optimizer.zero_grad()
            temperature = self._get_temperature(epoch)

            inputs = self._get_inputs(z, edge_index, index)
            logits = self.mlp(inputs).view(-1)
            edge_mask = self._concrete_sample(logits, temperature)
            set_masks(model, edge_mask, edge_index, apply_sigmoid=True)

            if self.model_config.task_level == ModelTaskLevel.node:
                _, hard_edge_mask = self._get_hard_masks(model, index, edge_index,
                                                        num_nodes=x.size(0))
                edge_mask = edge_mask[hard_edge_mask]

            y_hat, y = model(x, edge_index, **kwargs), target

            if index is not None:
                y_hat, y = y_hat[index], y[index]

            loss = self._loss(y_hat, y, edge_mask)
            loss.backward()
            self.optimizer.step()

            if self.edge_mask is not None:
                current_edge_mask = self.edge_mask.clone().detach()
                current_edge_mask = self._post_process_mask(current_edge_mask, apply_sigmoid=self.apply_sigmoid)
                edge_mask_over_time.append(current_edge_mask)
                edge_mask_grads_over_time.append(self.edge_mask.grad.clone().detach())


            clear_masks(model)
            self._curr_epoch = epoch

        if self.save_masks==True:
            self.edge_mask_over_time = edge_mask_over_time
            self.edge_mask_grads_over_time = edge_mask_grads_over_time

        return float(loss)

    # def train(
    #     self,
    #     epoch: int,
    #     model: torch.nn.Module,
    #     x: Tensor,
    #     edge_index: Tensor,
    #     *,
    #     target: Tensor,
    #     index: Optional[Union[int, Tensor]] = None,
    #     **kwargs,
    # ):
    #     r"""Trains the underlying explainer model.
    #     Needs to be called before being able to make predictions.

    #     Args:
    #         epoch (int): The current epoch of the training phase.
    #         model (torch.nn.Module): The model to explain.
    #         x (torch.Tensor): The input node features of a
    #             homogeneous graph.
    #         edge_index (torch.Tensor): The input edge indices of a homogeneous
    #             graph.
    #         target (torch.Tensor): The target of the model.
    #         index (int or torch.Tensor, optional): The index of the model
    #             output to explain. Needs to be a single index.
    #             (default: :obj:`None`)
    #         **kwargs (optional): Additional keyword arguments passed to
    #             :obj:`model`.
    #     """
    #     if isinstance(x, dict) or isinstance(edge_index, dict):
    #         raise ValueError(f"Heterogeneous graphs not yet supported in "
    #                          f"'{self.__class__.__name__}'")

    #     if self.model_config.task_level == ModelTaskLevel.node:
    #         if index is None:
    #             raise ValueError(f"The 'index' argument needs to be provided "
    #                              f"in '{self.__class__.__name__}' for "
    #                              f"node-level explanations")
    #         if isinstance(index, Tensor) and index.numel() > 1:
    #             raise ValueError(f"Only scalars are supported for the 'index' "
    #                              f"argument in '{self.__class__.__name__}'")

    #     z = get_embeddings(model, x, edge_index, **kwargs)[-1]

    #     self.optimizer.zero_grad()
    #     temperature = self._get_temperature(epoch)

    #     inputs = self._get_inputs(z, edge_index, index)
    #     logits = self.mlp(inputs).view(-1)
    #     edge_mask = self._concrete_sample(logits, temperature)
    #     set_masks(model, edge_mask, edge_index, apply_sigmoid=True)

    #     if self.model_config.task_level == ModelTaskLevel.node:
    #         _, hard_edge_mask = self._get_hard_masks(model, index, edge_index,
    #                                                  num_nodes=x.size(0))
    #         edge_mask = edge_mask[hard_edge_mask]

    #     y_hat, y = model(x, edge_index, **kwargs), target

    #     if index is not None:
    #         y_hat, y = y_hat[index], y[index]

    #     loss = self._loss(y_hat, y, edge_mask)
    #     loss.backward()
    #     self.optimizer.step()

    #     clear_masks(model)
    #     self._curr_epoch = epoch

    #     return float(loss)

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

        self.train(model, x, edge_index, target=target, index=index, **kwargs)

        # if self._curr_epoch < self.epochs - 1:  # Safety check:
        #     raise ValueError(f"'{self.__class__.__name__}' is not yet fully "
        #                      f"trained (got {self._curr_epoch + 1} epochs "
        #                      f"from {self.epochs} epochs). Please first train "
        #                      f"the underlying explainer model by running "
        #                      f"`explainer.algorithm.train(...)`.")

        hard_edge_mask = None
        if self.model_config.task_level == ModelTaskLevel.node:
            if index is None:
                raise ValueError(f"The 'index' argument needs to be provided "
                                 f"in '{self.__class__.__name__}' for "
                                 f"node-level explanations")
            if isinstance(index, Tensor) and index.numel() > 1:
                raise ValueError(f"Only scalars are supported for the 'index' "
                                 f"argument in '{self.__class__.__name__}'")

            # We need to compute hard masks to properly clean up edges and
            # nodes attributions not involved during message passing:
            _, hard_edge_mask = self._get_hard_masks(model, index, edge_index,
                                                     num_nodes=x.size(0))

        z = get_embeddings(model, x, edge_index, **kwargs)[-1]

        inputs = self._get_inputs(z, edge_index, index)
        logits = self.mlp(inputs).view(-1)

        edge_mask = self._post_process_mask(logits, hard_edge_mask,
                                            apply_sigmoid=True)

        metrics_over_time = {'edge_mask_over_time':self.edge_mask_over_time,
                             'edge_grads_over_time':self.edge_mask_grads_over_time,
                             'clf_loss_over_time':self.clf_loss_over_time,
                             'edge_size_loss_over_time':self.edge_size_loss_over_time,
                             'edge_ent_loss_over_time':self.edge_ent_loss_over_time,
                             'predictions_over_time': self.predictions_over_time
                             }

        explanation = Explanation(edge_mask=edge_mask)

        return explanation

    def supports(self) -> bool:
        explanation_type = self.explainer_config.explanation_type
        if explanation_type != ExplanationType.phenomenon:
            logging.error(f"'{self.__class__.__name__}' only supports "
                          f"phenomenon explanations "
                          f"got (`explanation_type={explanation_type.value}`)")
            return False

        task_level = self.model_config.task_level
        if task_level not in {ModelTaskLevel.node, ModelTaskLevel.graph}:
            logging.error(f"'{self.__class__.__name__}' only supports "
                          f"node-level or graph-level explanations "
                          f"got (`task_level={task_level.value}`)")
            return False

        node_mask_type = self.explainer_config.node_mask_type
        if node_mask_type is not None:
            logging.error(f"'{self.__class__.__name__}' does not support "
                          f"explaining input node features "
                          f"got (`node_mask_type={node_mask_type.value}`)")
            return False

        return True

    ###########################################################################

    def _get_inputs(self, embedding: Tensor, edge_index: Tensor,
                    index: Optional[int] = None) -> Tensor:
        zs = [embedding[edge_index[0]], embedding[edge_index[1]]]
        if self.model_config.task_level == ModelTaskLevel.node:
            assert index is not None
            zs.append(embedding[index].view(1, -1).repeat(zs[0].size(0), 1))
        return torch.cat(zs, dim=-1)

    def _get_temperature(self, epoch: int) -> float:
        temp = self.coeffs['temp']
        return temp[0] * pow(temp[1] / temp[0], epoch / self.epochs)

    def _concrete_sample(self, logits: Tensor,
                         temperature: float = 1.0) -> Tensor:
        bias = self.coeffs['bias']
        eps = (1 - 2 * bias) * torch.rand_like(logits) + bias
        return (eps.log() - (1 - eps).log() + logits) / temperature

    def _loss(self, y_hat: Tensor, y: Tensor, edge_mask: Tensor) -> Tensor:
        if self.model_config.mode == ModelMode.binary_classification:
            clf_loss = self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            clf_loss = self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            clf_loss = self._loss_regression(y_hat, y)


        # Regularization loss:
        edge_mask = edge_mask.sigmoid()
        edge_size_loss = edge_mask.sum() * self.coeffs['edge_size']
        edge_mask = 0.99 * edge_mask + 0.005
        edge_ent = -edge_mask * edge_mask.log() - (1 - edge_mask) * (1 - edge_mask).log()
        edge_ent_loss = edge_ent.mean() * self.coeffs['edge_ent']

        self.clf_loss_over_time.append(clf_loss.clone().detach().item())
        self.edge_size_loss_over_time.append(edge_size_loss)
        try:
            self.edge_ent_loss_over_time.append(edge_ent_loss.item())
        except:
            self.edge_ent_loss_over_time.append(edge_ent_loss)
        
        loss = clf_loss + edge_size_loss + edge_ent_loss
        return loss
