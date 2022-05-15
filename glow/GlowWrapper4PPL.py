import os

import numpy as np
import torch

from torch_fidelity.generative_model_base import GenerativeModelBase
from torch_fidelity.helpers import vassert


class GenerativeModelWrap(GenerativeModelBase):
    def __init__(self, graph, level_id, z_type='uniform_0_1', num_classes=0):
        super().__init__()
        #vassert(type(z_size) is int and z_size > 0, 'z_size must be a positive integer')
        vassert(z_type in ('normal', 'unit', 'uniform_0_1'), f'z_type={z_type} not implemented')
        vassert(type(num_classes) is int and num_classes >= 0, 'num_classes must be a non-negative integer')
        output_shape = graph.flow.output_shapes[level_id]
        z_size = [output_shape[1], output_shape[2], output_shape[3]]
        self.level_id = level_id
        self._z_size = z_size
        self._z_type = z_type
        self._num_classes = num_classes
        self.graph = graph

    @property
    def z_size(self):
        return self._z_size

    @property
    def z_type(self):
        return self._z_type

    @property
    def num_classes(self):
        return self._num_classes

    @staticmethod
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def forward(self, z, eps_std=0.3):
        x = self.graph.reverse_flow(z=torch.tensor(z).cuda(),
                               y_onehot=None,
                               eps_std=eps_std,
                               dissec=dict(interrupt_z=self.level_id,
                                           interrupt_z_value=torch.tensor(z).cuda()))
        return x