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
        out = x.reshape((bs, patch_depth, h * w)).transpose(-1, -2) # bs * patch_num * patch_depth, patch_depth = model_dim
        return out 
        
        
class WindowAttention(nn.Module):
    
    def __init__(self, model_dim, num_head, window_size, qkv_bias=True) -> None:
        super().__init__()
        assert model_dim % num_head == 0
        self.num_head = num_head
        self.attn_proj = nn.Linear(model_dim, 3*model_dim, bias=qkv_bias)
        self.linear = nn.Linear(model_dim, model_dim)
        self.window_size = window_size 
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_head))
        nn.init.xavier_normal_(self.relative_position_bias_table)
        
        
        coord_size = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coord_size, coord_size], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1 
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        
        
        
    def forward(self, x, mask = None):
        bs, sql_len, model_dim = x.shape
        x_proj = self.attn_proj(x)
        q, k, v = x_proj.chunk(3, dim=-1)
        num_head = self.num_head
        head_dim = model_dim // self.num_head
        
        
        q = q.reshape((bs, sql_len, num_head, head_dim)).transpose(1, 2)
        q = q.reshape((bs * num_head, sql_len, head_dim))
        
        k = k.reshape((bs, sql_len, num_head, head_dim)).transpose(1, 2)
        k = k.reshape((bs * num_head, sql_len, head_dim))
        
        v = v.reshape((bs, sql_len, num_head, head_dim)).transpose(1, 2)
        v = v.reshape((bs * num_head, sql_len, head_dim))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().tile(bs, 1,  1)
        if mask is None:
            attn_prob = F.softmax(torch.bmm(q, k.transpose(-1, -2)) / math.sqrt(head_dim) + relative_position_bias, dim=-1) # sql_len * model_dim
        else:
            additive_mask = mask.tile((num_head, 1, 1))
        
            attn_prob = F.softmax(torch.bmm(q, k.transpose(-1, -2)) / math.sqrt(head_dim) + additive_mask + relative_position_bias, dim=-1)
        
        out = torch.bmm(attn_prob, v)
        out = out.reshape((bs, num_head, sql_len, head_dim)).transpose(1, 2)
        out = out.reshape((bs, sql_len, model_dim))
        out = self.linear(out)
        return attn_prob, out
    
   
class PatchMerging(nn.Module):
    
    def __init__(self, model_dim, merge_size, output_depth_scale=0.5) -> None:
        super().__init__()
        self.merge_size = merge_size
        self.proj = nn.Linear(model_dim * merge_size * merge_size, int(model_dim * merge_size * merge_size * output_depth_scale))
        
    def forward(self, x):
        bs, patch_num, model_dim = x.shape
        merge_size = self.merge_size
        patch_size = int(math.sqrt(patch_num))
        x = x.transpose(-2, -1).reshape((bs, model_dim, patch_size, patch_size))
        x = F.unfold(x, (merge_size, merge_size), stride=(merge_size, merge_size)).transpose(-1, -2) # bs,  (patch_num / (merge_size**2)),  model_dim * (merge_size**2)
        x = self.proj(x) # bs,  (patch_num / (merge_size**2)), model_dim * merge_size * merge_size * output_depth_scale 
        return x 
    
    
class SuccessiveSwinTransfomerBlock(nn.Module):
    
    def __init__(self, model_dim, num_head,  window_size) -> None:
        super().__init__()
        
        self.window_size = window_size
        
        self.ln1 = nn.LayerNorm(model_dim)
        self.wmsa = WindowAttention(model_dim, num_head, window_size)
        self.ln2 = nn.LayerNorm(model_dim)
        self.mlp1 = nn.Linear(model_dim, model_dim)
        
        self.ln3 = nn.LayerNorm(model_dim)
        self.swmsa = WindowAttention(model_dim, num_head, window_size)
        self.ln4 = nn.LayerNorm(model_dim)
        self.mlp2 = nn.Linear(model_dim, model_dim)
        
        
        
    
    def forward(self, x):
        bs, patch_num, model_dim = x.shape
        window_size = self.window_size
        patch_size = int(math.sqrt(patch_num))
        window_num = self.get_window_num(patch_size)
        
        y = self.ln1(x)
        y = self.sequence2image(y, window_size, window_num)
        _, y = self.wmsa(y)
        y = self.image2sequence(y, bs, window_num, window_size, model_dim)
        y = x + y
        
        x = y 
        y = self.ln2(x)
        y = self.mlp1(y)
        y = x + y
        
        x = y
        y = self.ln3(x)
        y = self.sequence2shift(y, window_size, window_num, patch_size)
        mask = self.get_shift_mask(bs, window_size, window_num, patch_size)
        _, y = self.swmsa(y, mask)
        y = self.shift2sequence(y, bs, window_num, window_size, model_dim)
        y = x + y
        
        x = y
        y = self.ln4(x)
        y = self.mlp2(y)
        y + x + y
        return y
        
        
    def get_window_num(self, patch_size):
        return patch_size // self.window_size
    
    @staticmethod
    def image2sequence(x, batch_size, window_num, window_size, model_dim):
        return x.reshape((batch_size, window_num * window_num * window_size * window_size, model_dim))

    @staticmethod
    def shift2sequence(x, batch_size, window_num, window_size, model_dim):
        x = x.reshape((batch_size, window_num, window_num, window_size, window_size, model_dim)).transpose(2, 3)
        patch_size = window_num * window_size
        shift_size = window_size // 2
        x = x.reshape((batch_size, patch_size, patch_size, model_dim))
        x = torch.roll(x, (shift_size, shift_size), dims=(1, 2))
        return x.reshape(batch_size, patch_size * patch_size, model_dim)
        
    @staticmethod
    def sequence2image(x, window_size, window_num):
        bs, _, model_dim = x.shape
        x = x.reshape((bs, window_num, window_size, window_num, window_size, model_dim)).transpose(2, 3)
        x = x.reshape((bs * window_num * window_num, window_size * window_size, model_dim))
        return x
    
    @staticmethod
    def sequence2shift(x, window_size, window_num, patch_size):
        bs, _, model_dim = x.shape
        shift_size = window_size // 2
        num_windows = window_num * window_num
        num_patch_in_window = window_size * window_size
        x = x.reshape((bs, patch_size, patch_size, model_dim))
        x = torch.roll(x, (-shift_size, -shift_size), dims=(1, 2))
        x = x.reshape((bs, window_num, window_size, window_num, window_size, model_dim)).transpose(2, 3)
        x = x.reshape((bs * num_windows, num_patch_in_window, model_dim))
        return x
    
    @staticmethod
    def get_shift_mask(batch_size, window_size, window_num, patch_size):
        num_windows = window_num * window_num
        num_patch_in_window = window_size * window_size
        shift_size = window_size // 2
        index_mat = torch.zeros(patch_size, patch_size)
        for i in range(patch_size):
            for j in range(patch_size):
                row_times = (i + window_size // 2) // window_size
                col_times = (j + window_size // 2) // window_size
                index_mat[i][j] = row_times * (patch_size // window_size) + col_times + 1
        index_mat = torch.roll(index_mat, (-shift_size, -shift_size), dims=(0, 1))
        index_mat = index_mat.unsqueeze(0)
        index_mat = index_mat.reshape((1, window_num, window_size, window_num, window_size)).transpose(2, 3)
        
        c1 = index_mat.reshape((1, num_windows, num_patch_in_window)).unsqueeze(-1)
        c1 = c1.tile(batch_size, 1, 1)
        c2 = (c1 - c1.transpose(-1, -2))  == 0
        c2 = c2.to(torch.float32)
        mask = (1 - c2) * (1e-9)
        mask = mask.reshape((batch_size * num_windows, num_patch_in_window, num_patch_in_window))
        return mask
    

class SwinTransformer(nn.Module):
    
    def __init__(self, patch_size, C, layer_numbers, num_head, window_size) -> None:
        super().__init__()
        assert len(layer_numbers) == 4
        model_dim = C
        
        self.add_module("patch_embedding", PatchEmbedding(C, patch_size))
        for i, num in enumerate(layer_numbers):
            assert num % 2 == 0
            successive_num = num // 2
            ml = []
            if i > 0:
                ml.append(PatchMerging(model_dim, 2, 0.5))
                model_dim *= 2
            ml += [SuccessiveSwinTransfomerBlock(model_dim, num_head, window_size) for _ in range(successive_num)]
            self.add_module(f"starge{i+1}", nn.Sequential(*ml))
            
    
    def forward(self, x):
        for name, m in self.named_children():
            x = m(x)
        return x
    
def swin_t(patch_size, num_head, window_size):
    return SwinTransformer(patch_size, 96, [2, 2, 6, 2], num_head, window_size)

def swin_s(patch_size, num_head, window_size):
    return SwinTransformer(patch_size, 96, [2, 2, 18, 2], num_head, window_size)

def swin_b(patch_size, num_head, window_size):
    return SwinTransformer(patch_size, 128, [2, 2, 18, 2], num_head, window_size)

def swin_l(patch_size, num_head, window_size):
    return SwinTransformer(patch_size, 192, [2, 2, 18, 2], num_head, window_size)


if __name__ == "__main__":
    model = swin_t(4, 2, 7)
    summary(model, (3, 224, 224))