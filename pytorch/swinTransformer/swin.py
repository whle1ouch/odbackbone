import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchsummary import summary 


class PatchEmbedding(nn.Module):
    
    def __init__(self, model_dim, patch_size) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, model_dim, patch_size, stride=patch_size, bias=False)
        
    def forward(self, x):
        x = self.conv(x)
        bs, patch_depth, h, w = x.shape
        out = x.reshape((bs, patch_depth, h * w)).transpose((-1, -2)) # bs * patch_num * patch_depth, patch_depth = model_dim
        return out 
        
        
class MultiHeadAttention(nn.Module):
    
    def __init__(self, model_dim, num_head) -> None:
        super().__init__()
        assert model_dim % num_head == 0
        self.num_head = num_head
        self.attn_proj = nn.Linear(model_dim, 3*model_dim)
        self.linear = nn.Linear(model_dim, model_dim)
        
    def forward(self, x, mask = None):
        bs, sql_len, model_dim = x.shape
        x_proj = self.attn_proj(x)
        q, k, v = x_proj.chunk(-1)
        
        num_head = self.num_head
        head_dim = model_dim // self.num_head
        
        
        q = q.reshape((bs, sql_len, num_head, head_dim)).transpose(1, 2)
        q = q.reshape((bs * num_head, sql_len, head_dim))
        
        k = k.reshape((bs, sql_len, num_head, head_dim)).transpose(1, 2)
        k = k.reshape((bs * num_head, sql_len, head_dim))
        
        v = v.reshape((bs, sql_len, num_head, head_dim)).transpose(1, 2)
        v = v.reshape((bs * num_head, sql_len, head_dim))
        
        if mask is None:
            attn_prob = F.softmax(torch.bmm(q, k.transpose(-1, -2)) / math.sqrt(head_dim), dim=-1) # sql_len * model_dim
        else:
            additive_mask = mask.tile((bs * num_head, 1, 1))
            attn_prob = F.softmax(torch.bmm(q, k.transpose(-1, -2)) / math.sqrt(head_dim) + additive_mask, dim=-1)
        
        out = torch.bmm(attn_prob, v)
        out = out.reshae((bs, num_head, sql_len, head_dim)).transpose(1, 2)
        out = out.reshape((bs, sql_len, model_dim))
        out = self.linear(out)
        return attn_prob, out
    
   
class PatchMerging(nn.Nodule):
    
    def __init__(self, model_dim, merge_size, output_depth_scale=0.5) -> None:
        super().__init__()
        self.merge_size = merge_size
        self.proj = nn.Linear(model_dim * merge_size * merge_size, int(model_dim * merge_size * merge_size * output_depth_scale))
        
    def forward(self, x):
        bs, patch_num, model_dim = x.shape
        merge_size = self.merge_size
        patch_size = int(math.sqrt(patch_num))
        x = x.transpose(-2, -1).reshape((bs, model_dim, patch_size, patch_size))
        x = F.unfold(x, (merge_size, merge_size), stride=(merge_size, merge_size)).transopose(-1, -2) # bs,  (patch_num / (merge_size**2)),  model_dim *(merge_size**2)
        x = self.proj(x) # bs,  (patch_num / (merge_size**2)), model_dim * merge_size * merge_size * output_depth_scale 
        return x 
    
    
class SwinTransfomerBlock(nn.Module):
    
    def __init__(self, model_dim, num_head,  window_size, shift=False) -> None:
        super().__init__()
        
        self.window_size = window_size
        self.shift = shift
        
        self.ln1 = nn.LayerNorm(model_dim)
        self.msa = MultiHeadAttention(model_dim, num_head)
        
        self.ln2 = nn.LayerNorm(model_dim)
        self.linear = nn.Linear(model_dim, model_dim)
        
    
    def forward(self, x):
        window_size = self.window_size
        bs, patch_num, model_dim = x.shape
        
        
    def reshape_with_window(self, x):
        window_size = self.window_size
        bs, patch_num, model_dim = x.shape
        patch_size = int(math.sqrt(patch_num))
        x = x.transpose(1, 2).reshape((bs, model_dim, patch_size, patch_size))
        x = F.unfold(x, (window_size, window_size), stride=(window_size, window_size))
        
        
        