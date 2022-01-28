import sys
import os
import torch
import random
import numpy as np

from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .ViT import *
from .gcn import GCNBlock

from torch_geometric.nn import GCNConv, DenseGraphConv, dense_mincut_pool
from torch.nn import Linear

# def _rank3_trace(x):
#     return torch.einsum('ijj->i', x)


# def _rank3_diag(x):
#     eye = torch.eye(x.size(1)).type_as(x)
#     out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1))
#     return out

# EPS = 1e-15

# def dense_mincut_pool(x, adj, s, mask=None):
#     r"""MinCUt pooling operator from the `"Mincut Pooling in Graph Neural
#     Networks" <https://arxiv.org/abs/1907.00481>`_ paper

#     .. math::
#         \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
#         \mathbf{X}

#         \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
#         \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

#     based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
#     \times N \times C}`.
#     Returns pooled node feature matrix, coarsened symmetrically normalized
#     adjacency matrix and two auxiliary objectives: (1) The minCUT loss

#     .. math::
#         \mathcal{L}_c = - \frac{\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{A}
#         \mathbf{S})} {\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{D}
#         \mathbf{S})}

#     where :math:`\mathbf{D}` is the degree matrix, and (2) the orthogonality
#     loss

#     .. math::
#         \mathcal{L}_o = {\left\| \frac{\mathbf{S}^{\top} \mathbf{S}}
#         {{\|\mathbf{S}^{\top} \mathbf{S}\|}_F} -\frac{\mathbf{I}_C}{\sqrt{C}}
#         \right\|}_F.

#     Args:
#         x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
#             \times N \times F}` with batch-size :math:`B`, (maximum)
#             number of nodes :math:`N` for each graph, and feature dimension
#             :math:`F`.
#         adj (Tensor): Symmetrically normalized adjacency tensor
#             :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
#         s (Tensor): Assignment tensor :math:`\mathbf{S} \in \mathbb{R}^{B
#             \times N \times C}` with number of clusters :math:`C`. The softmax
#             does not have to be applied beforehand, since it is executed
#             within this method.
#         mask (BoolTensor, optional): Mask matrix
#             :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
#             the valid nodes for each graph. (default: :obj:`None`)

#     :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
#         :class:`Tensor`)
#     """

#     x = x.unsqueeze(0) if x.dim() == 2 else x
#     adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
#     s = s.unsqueeze(0) if s.dim() == 2 else s

#     (batch_size, num_nodes, _), k = x.size(), s.size(-1)

#     s = torch.softmax(s, dim=-1)

#     if mask is not None:
#         mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
#         x, s = x * mask, s * mask

#     out = torch.matmul(s.transpose(1, 2), x)
#     out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

#     # MinCUT regularization.
#     mincut_num = _rank3_trace(out_adj)
#     d_flat = torch.einsum('ijk->ij', adj)
#     d = _rank3_diag(d_flat)
#     mincut_den = _rank3_trace(
#         torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
#     mincut_loss = -(mincut_num / mincut_den)
#     mincut_loss = torch.mean(mincut_loss)

#     # Orthogonality regularization.
#     ss = torch.matmul(s.transpose(1, 2), s)
#     i_s = torch.eye(k).type_as(ss)
#     ortho_loss = torch.norm(
#         ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
#         i_s / torch.norm(i_s), dim=(-1, -2))
#     ortho_loss = torch.mean(ortho_loss)

#     # Fix and normalize coarsened adjacency matrix.
#     ind = torch.arange(k, device=out_adj.device)
#     out_adj[:, ind, ind] = 0
#     d = torch.einsum('ijk->ij', out_adj)
#     d = torch.sqrt(d)[:, None] + EPS
#     out_adj = (out_adj / d) / d.transpose(1, 2)

#     return out, out_adj, mincut_loss, ortho_loss

class Classifier(nn.Module):
    def __init__(self, n_class):
        super(Classifier, self).__init__()

        self.embed_dim = 64
        self.num_layers = 3
        self.node_cluster_num = 100

        self.transformer = VisionTransformer(num_classes=n_class, embed_dim=self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.criterion = nn.CrossEntropyLoss()

        self.bn = 1
        self.add_self = 1
        self.normalize_embedding = 1
        # input_dim:512, output_dim: 64, use torch.nn.BatchNorm1d(output_dim), add self (y += x), F.normalize
        self.conv1 = GCNBlock(512,self.embed_dim,self.bn,self.add_self,self.normalize_embedding,0.,0)       # 64->128
        self.pool1 = Linear(self.embed_dim, self.node_cluster_num)                                          # 100-> 20
        # self.relu1 = nn.ReLU()

    def forward(self,node_feat,labels,adj,mask,is_print=False, graphcam_flag=False):
        # node_feat, labels = self.PrepareFeatureLabel(batch_graph)
        # cls_loss=node_feat.new_zeros(self.num_layers)
        # rank_loss=node_feat.new_zeros(self.num_layers-1)
        X=node_feat
        # p_t=[]
        # pred_logits=0
        # visualize_tools=[]
        # visualize_tools1=[labels.cpu()]
        # embeds=0
        # concats=[]
        
        # layer_acc=[]
        
        # return tensor with a dimension of size one
        # 1 was inserted in the shape of the array at axis 2
        X=mask.unsqueeze(2)*X
        X = self.conv1(X, adj, mask)
        s = self.pool1(X)
        # s = self.relu1(s)
        
        if graphcam_flag:
            s_matrix = torch.argmax(s[0], dim=1)
            from os import path
            torch.save(s_matrix, 'graphcam/s_matrix.pt')
            torch.save(s[0], 'graphcam/s_matrix_ori.pt')
            
            if path.exists('graphcam/att_1.pt'):
                os.remove('graphcam/att_1.pt')
                os.remove('graphcam/att_2.pt')
                os.remove('graphcam/att_3.pt')
    
        X, adj, mc1, o1 = dense_mincut_pool(X, adj, s, mask)
        b, _, _ = X.shape
        cls_token = self.cls_token.repeat(b, 1, 1)
        X = torch.cat([cls_token, X], dim=1)

        out = self.transformer(X)

        # loss
        loss = self.criterion(out, labels)
        loss = loss + mc1 + o1
        # pred
        pred = out.data.max(1)[1]

        if graphcam_flag:
            print('GraphCAM enabled')
            p = F.softmax(out, dim=1)
            torch.save(p, 'graphcam/prob.pt')
            index = np.argmax(out.cpu().data.numpy(), axis=-1)
            print("indexxx")
            print(index)

            for index_ in range(2):
                one_hot = np.zeros((1, out.size()[-1]), dtype=np.float32)
                one_hot[0, index_] = out[0][index_]
                one_hot_vector = one_hot
                one_hot = torch.from_numpy(one_hot).requires_grad_(True)
                one_hot = torch.sum(one_hot.cuda() * out)       #!!!!!!!!!!!!!!!!!!!!out-->p
                self.transformer.zero_grad()
                one_hot.backward(retain_graph=True)

                kwargs = {"alpha": 1}
                cam = self.transformer.relprop(torch.tensor(one_hot_vector).to(X.device), method="transformer_attribution", is_ablation=False, 
                                            start_layer=0, **kwargs)

                torch.save(cam, 'graphcam/cam_{}.pt'.format(index_))

        return pred,labels,loss
