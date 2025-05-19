import torch
import torch.nn.functional as F
from torch import optim


class Backprop(object):
    def __init__(self, net, step_size=0.001, erank_step_size=0.001, loss='mse', opt='sgd', beta_1=0.9, beta_2=0.999, weight_decay=0.0,
                 to_perturb=False, perturb_scale=0.1, device='cpu', momentum=0):
        self.net = net
        self.to_perturb = to_perturb
        self.perturb_scale = perturb_scale
        self.device = device

        # define the optimizer
        if opt == 'sgd':
            self.opt = optim.SGD(self.net.parameters(), lr=step_size, weight_decay=weight_decay, momentum=momentum)
        elif opt == 'adam':
            self.opt = optim.Adam(self.net.parameters(), lr=step_size, betas=(beta_1, beta_2),
                                  weight_decay=weight_decay)
        elif opt == 'adamW':
            self.opt = optim.AdamW(self.net.parameters(), lr=step_size, betas=(beta_1, beta_2),
                                   weight_decay=weight_decay)
        
        self.opt_erank = optim.SGD(net.parameters(),
                                   lr=erank_step_size, weight_decay=weight_decay)

        # define the loss function
        self.loss = loss
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[self.loss]

        # Placeholder
        self.previous_features = None

    def effective_rank_loss(self, feature: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Compute the Shannon-entropy effective rank of the spectrum sv.
        
        Args:
        sv: 1D tensor of singular values (non-negative).
        eps: small constant to ensure numerical stability.
        Returns:
        scalar tensor: exp( - sum_i p_i * log(p_i) ).
        """
        # 1. Ensure non-negativity & normalize
        sv = torch.linalg.svdvals(feature.T).abs()
        total = sv.sum().clamp(min=eps)
        p = sv / total            # shape (r,)

        # 2. Compute entropy: sum p * log(p), but avoid log(0)
        entropy = -(p * torch.log(p + eps)).sum()

        # 3. Return exp(entropy)
        return torch.exp(entropy)

    def maximize_effective_rank(self, x, steps=1):
        """
        Run `steps` gradient-ascent updates to maximize the
        effective-rank of the last hidden activations.
        """
        x = x.to(self.device)
        for i in range(steps):
            self.opt_erank.zero_grad()
            _, features = self.net.predict(x)
            erank_losses = [self.effective_rank_loss(f) for f in features]
            loss_erank = - torch.stack(erank_losses).mean()
            loss_erank.backward()
            self.opt_erank.step()

        # return the final erank value (detached)
        return loss_erank.detach()

    def learn(self, x, target):
        """
        Learn using one step of gradient-descent
        :param x: input
        :param target: desired output
        :return: loss
        """
        self.opt.zero_grad()
        output, features = self.net.predict(x=x)
        loss = self.loss_func(output, target)
        self.previous_features = features

        loss.backward()
        self.opt.step()
        if self.to_perturb:
            self.perturb()
        if self.loss == 'nll':
            return loss.detach(), output.detach()
        return loss.detach()

    def perturb(self):
        with torch.no_grad():
            for i in range(int(len(self.net.layers)/2)+1):
                self.net.layers[i * 2].bias +=\
                    torch.empty(self.net.layers[i * 2].bias.shape, device=self.device).normal_(mean=0, std=self.perturb_scale)
                self.net.layers[i * 2].weight +=\
                    torch.empty(self.net.layers[i * 2].weight.shape, device=self.device).normal_(mean=0, std=self.perturb_scale)
