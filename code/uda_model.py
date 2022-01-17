import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Gradient reversal layer [ganin2015unsupervised]."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x
    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.alpha * grad_output.neg()
        return output, None


class UDAModel(nn.Module):
    def __init__(self, load_weights_path=None, lat_space_dim=18, in_feature_dim=18, cpu=False):
        super().__init__()
        self.latent_space_dim = lat_space_dim
        self.in_feature_dim = in_feature_dim

        # the usual feature extraction layers
        self.l1 = nn.Linear(self.in_feature_dim, self.in_feature_dim)
        self.l2 = nn.Linear(self.in_feature_dim, self.latent_space_dim)

        # the usual label prediction head
        self.l3 = nn.Linear(self.latent_space_dim, 20)
        self.l4 = nn.Linear(20, 1)

        # the additional domain classification head
        self.uda_l4 = nn.Linear(20, 1)

        if load_weights_path != None:
            self.load_state_dict(torch.load(load_weights_path, map_location = torch.device('cpu') if cpu else None))

    def forward(self, x, idx, alpha):
        x = self.l1(x)
        x = nn.functional.relu(x)
        x = self.l2(x)
        tmp = nn.functional.relu(x)

        # the last entry of idx is the index for the last entry
        # so +1 is equal to number of events
        x = torch.zeros(idx[-1] + 1, self.latent_space_dim).to(x.device)
        x.index_add_(0, idx, tmp)

        # the usual label prediction head (part 1)
        x = self.l3(x)
        x = nn.functional.relu(x)

        # the additional domain classification head
        x_uda = GradientReversalFunction.apply(x, alpha)
        x_uda = self.uda_l4(x_uda)

        # the usual label prediction head (part 2)
        x = self.l4(x)
        # note we don't apply sigmoid here since it's inside the loss function
        # remember this when using the output in classification

        return x, x_uda
