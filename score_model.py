import torch
import torch.nn as nn
import math
from score_blocks import CondRefineNetDilated


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Score_Model(nn.Module):
    def __init__(self, device, n_steps, sigma_min, sigma_max):
        '''
        Score Network.

        n_steps   : perturbation schedule steps (Langevin Dynamic step)
        sigma_min : sigma min of perturbation schedule
        sigma_min : sigma max of perturbation schedule

        '''
        super().__init__()
        self.device = device
        self.sigmas = torch.exp(torch.linspace(start=math.log(sigma_max), end=math.log(sigma_min), steps = n_steps)).to(device = device)
        self.conv_layer = CondRefineNetDilated(device, n_steps)
        self.to(device = device)

    # Loss Function
    def loss_fn(self, x, idx=None):
        '''
        This function performed when only training phase.

        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index. Else (inference phase), it is recommended that you specify.

        '''
        scores, target, sigma = self.forward(x, idx=idx, get_target=True)
        target = target.view(target.shape[0], -1)
        scores = scores.view(scores.shape[0], -1)        
        losses = torch.square(scores - target).mean(dim=-1) * sigma.squeeze() ** 2
        return losses.mean(dim=0)

    # S(theta, sigma)
    def forward(self, x, idx=None, get_target=False):
        '''
        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index. Else (inference phase), it is recommended that you specify.
        get_target : if True (training phase), target and sigma is returned with output (score prediction)

        '''

        if idx == None:
            idx = torch.randint(0, len(self.sigmas), (x.size(0), 1)).to(device = self.device)
            used_sigmas = self.sigmas[idx][:, :, None, None]
            noise = torch.randn_like(x)
            x_tilde = x + noise * used_sigmas
            idx = idx.squeeze()
        else:
            idx = torch.Tensor([idx for _ in range(x.size(0))]).to(device = self.device).long()
            x_tilde = x
            
        if get_target:
            target = - 1 / (used_sigmas ) * noise 

            
        output = self.conv_layer(x_tilde, idx)

        return (output, target, used_sigmas) if get_target else output