import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import repeat

class get_model(nn.Module):
    def __init__(self, num_classes=3):
        super(get_model, self).__init__()
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.pt_layer1 = PointTransformerLayer(dim = 64, pos_mlp_hidden_dim=64, attn_mlp_hidden_mult=4, num_neighbors=20)
        self.pt_layer2 = PointTransformerLayer(dim = 64, pos_mlp_hidden_dim=64, attn_mlp_hidden_mult=4, num_neighbors=20)
        self.pt_layer3 = PointTransformerLayer(dim = 64, pos_mlp_hidden_dim=64, attn_mlp_hidden_mult=4, num_neighbors=20)
        self.pt_layer4 = PointTransformerLayer(dim = 64, pos_mlp_hidden_dim=64, attn_mlp_hidden_mult=4, num_neighbors=20)
        self.fc1 = nn.Linear(64, 32)
        # self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        
        out = points.permute(0, 2, 1) # (B, 3, N)
        out = F.relu(self.conv1(out)) # (B, 64, N)
        out = out.permute(0, 2, 1)    # (B, N, 64)

        out = self.pt_layer1(out, pos)
        out = self.pt_layer2(out, pos)

        out = out.permute(0, 2, 1)
        out = F.max_pool1d(out, out.size(2)).squeeze(2)
        out = F.relu(self.fc1(out))
        
        x = self.fc3(out)
        return x
    

def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

class PointTransformerLayer(nn.Module):
    def __init__(self, dim, pos_mlp_hidden_dim=64, attn_mlp_hidden_mult=4, num_neighbors=20):
        super(PointTransformerLayer, self).__init__()
        self.num_neighbors = num_neighbors
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, dim)
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(dim*2, dim*attn_mlp_hidden_mult),
            nn.ReLU(),
            nn.Linear(dim*attn_mlp_hidden_mult, 1)
        )
        self.to_qkv = nn.Linear(dim, dim*3, bias=False)

    def forward(self, x, pos, mask=None):
        """
        x: (B, N, C)
        pos: (B, N, 3)
        mask: (B, N)
        output: (B, N, C)
        """
        n, num_neighbors = x.shape[1], self.num_neighbors
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        
        knn = torch.cdist(pos, pos)
        dist, indices = knn.topk(num_neighbors, largest = False)
        
        p1 =  pos[:, :, None, :]
        p2 = batched_index_select(pos, indices, dim = 1)
        rel_pos = p1 - p2

        rel_pos_emb = self.pos_mlp(rel_pos)
        qk_rel = q[:, :, None, :] - k[:, None, :, :]
        
        # expand values
        v = repeat(v, 'b j d -> b i j d', i = n)
        
        # determine k nearest neighbors for each point, if specified
        if num_neighbors < n:
            knn = torch.cdist(pos, pos)
            dist, indices = knn.topk(num_neighbors, largest = False)
            indices = indices.to(v.device)

            v = batched_index_select(v, indices, dim = 2)
            qk_rel = batched_index_select(qk_rel, indices, dim = 2)

        v = v + rel_pos_emb
        inp = torch.cat((qk_rel, rel_pos_emb), dim = -1)
        sim = self.attn_mlp(inp)
        # attention
        attn = sim.softmax(dim = -2)
        return einsum('b i j d, b i j d -> b i d', attn, v)