import torch
import torch.nn.functional as F


class batch_norm(torch.nn.Module):
    def __init__(self, dim_hidden, type_norm, skip_connect=False, num_groups=1,
                 skip_weight=0.005):
        super(batch_norm, self).__init__()
        self.type_norm = type_norm
        self.skip_connect = skip_connect
        self.skip_weight = skip_weight
        self.dim_hidden = dim_hidden


    def forward(self, x):
        if self.type_norm == 'None':
            return x

        else:
            raise Exception(f'the normalization has not been implemented')
