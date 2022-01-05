import torch
import torch.nn as nn


class BaselineModel(nn.Module):
    def __init__(self, load_weights_path=None, lat_space_dim=18, in_feature_dim=18, cpu=False):
        super().__init__()
        self.latent_space_dim = lat_space_dim
        self.in_feature_dim = in_feature_dim

        self.l1 = nn.Linear(self.in_feature_dim, self.in_feature_dim)
        self.l2 = nn.Linear(self.in_feature_dim, self.latent_space_dim)

        self.l3 = nn.Linear(self.latent_space_dim, 20)
        self.l4 = nn.Linear(20, 1)

        if load_weights_path != None:
            self.load_state_dict(torch.load(load_weights_path, map_location = torch.device('cpu') if cpu else None))

    def forward(self, x, idx):
        x = self.l1(x)
        x = nn.functional.relu(x)
        x = self.l2(x)
        tmp = nn.functional.relu(x)

        # the last entry of idx is the index for the last entry
        # so +1 is equal to number of events
        x = torch.zeros(idx[-1] + 1, self.latent_space_dim).to(x.device)
        x.index_add_(0, idx, tmp)

        x = self.l3(x)
        x = nn.functional.relu(x)
        x = self.l4(x)
        # note we don't apply sigmoid here since it's inside the loss function
        # remember this when using the output in classification
        return x
