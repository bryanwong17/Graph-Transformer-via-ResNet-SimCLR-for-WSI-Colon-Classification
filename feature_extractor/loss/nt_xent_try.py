import torch
import numpy as np
from zmq import device


class NTXentLoss_try(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss_try, self).__init__()
        self.batch_size = batch_size #128
        self.temperature = temperature # 0.5
        self.device = device # cuda if available
        self.similarity_function = self._get_similarity_function(use_cosine_similarity) # use cosine similarity if true

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        # zis and zjs are two representations that come from same image
        representations = torch.cat([zjs, zis], dim=0)
        n_samples = len(representations) # 2 * batch size

        # Full similarity matrix
        cov = torch.mm(representations, representations.t().contiguous())
        sim = torch.exp(cov / self.temperature)

        # Negative similarity
        mask = ~torch.eye(n_samples, device=self.device).bool()
        # remove the diagonal (positive)
        neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

        # Positive similarity
        pos = torch.exp(torch.sum(zis * zjs, dim=-1) / self.temperature)
        pos = torch.cat([pos,pos], dim=0)

        loss = -torch.log(pos / neg).mean()
        return loss
