import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np 
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
    
    def __init__(self, model_dim, num_head, window_size, qkv_bias=True, pretrained_window_size=[0, 0]) -> None:
        super().__init__()
        assert model_dim % num_head == 0
        self.num_head = num_head
        self.attn_proj = nn.Linear(model_dim, 3*model_dim, bias=qkv_bias)
        self.linear = nn.Linear(model_dim, model_dim)
        self.window_size = window_size
        
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_head, 1, 1))), requires_grad=True)
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True), 
                                     nn.ReLU(inplace=True), 
                                     nn.Linear(512, num_head, bias=True))
        
        self.pretrained_window_size = pretrained_window_size
        relative_coords_h = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w], indexing="ij")).permute(1, 2, 0).contiguous().unsqueeze(0) 
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)
        
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
        self.softmax = nn.Softmax(dim=-1)

    
        
    def forward(self, x, mask=None):
        bs, sql_len, model_dim = x.shape
        x_proj = self.attn_proj(x)
        q, k, v = x_proj.chunk(3, dim=-1)
        num_head = self.num_head
        head_dim = model_dim // self.num_head
        
        
        q = q.reshape((bs, sql_len, num_head, head_dim)).transpose(1, 2)
        k = k.reshape((bs, sql_len, num_head, head_dim)).transpose(1, 2)
        v = v.reshape((bs, sql_len, num_head, head_dim)).transpose(1, 2)
    

        attn = F.normalize(q, dim=-1)  @ F.normalize(k,dim=-1).transpose(-1, -2)
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        attn = attn * logit_scale
        
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_head)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1) 
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0) #bs, num_head, sql_len, sql_len
        
        
        if mask is None:
            attn_prob = self.softmax(attn)
        else:
            num_windows = mask.shape[0]  # num_windows, sql_len, sql_len
            attn = attn.reshape(bs // num_windows, num_windows, num_head, sql_len, sql_len) +  mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape(bs, num_head, sql_len, sql_len)
            attn_prob = self.softmax(attn) 
           
        out = (attn_prob @ v).transpose(1, 2).reshape(bs, sql_len, model_dim) 
        out = self.linear(out)
        return out
    
    
class PatchMerging(nn.Module):
    
    def __init__(self, input_shape, model_dim, merge_size, output_depth_scale=0.5) -> None:
        super().__init__()
        self.merge_size = merge_size
        self.input_shape = input_shape
        self.proj = nn.Linear(model_dim * merge_size * merge_size, int(model_dim * merge_size * merge_size * output_depth_scale))
        
    def forward(self, x):
        bs, length, model_dim = x.shape
        height, width = self.input_shape
        assert length == height * width, "erorr"
        merge_size = self.merge_size
        x = x.transpose(-2, -1).reshape((bs, model_dim, height, width))
        x = F.unfold(x, (merge_size, merge_size), stride=(merge_size, merge_size)).transpose(-1, -2) # bs,  (patch_num / (merge_size**2)),  model_dim * (merge_size**2)
        x = self.proj(x) # bs,  (patch_num / (merge_size**2)), model_dim * merge_size * merge_size * output_depth_scale 
        return x 
    
    
class TransformerBlockV2(nn.Module):
    
    def __init__(self, input_shape, model_dim, num_head, window_size, shift_size, mlp_ratio, dropout_rate) -> None:
        super().__init__()
        assert len(input_shape) == 2, "input image shape error"
        assert 0 <= shift_size < window_size ,"shift size mush less than window size"
        self.window_size = window_size
        self.shift_size = shift_size
        self.input_shape = input_shape
        self.ln1 = nn.LayerNorm(model_dim)
        self.swmsa = WindowAttention(model_dim, num_head, window_size)
        self.ln2 = nn.LayerNorm(model_dim)
    
        mlp_hidden_dim = mlp_ratio * model_dim
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_hidden_dim), 
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, model_dim),
            nn.Dropout(dropout_rate)
            )
        
        
        if self.shift_size > 0:
            height, width = self.input_shape
            image_mask = torch.zeros(1, height, width, 1)
            h_slices = [
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)
            ]
            w_slices = [
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)
            ]
            cl = 0
            for hs in h_slices:
                for ws in w_slices:
                    image_mask[:, hs, ws, :] = cl
                    cl += 1
            mask_windows = self.window_partition(image_mask)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        
        self.register_buffer("attn_mask", attn_mask)
            
        
    
    def forward(self, x):
        bs, length, model_dim = x.shape
        height, width = self.input_shape
        assert length == height * width, "error length != height * width"

        shortcut = x 
        x = x.reshape(bs, height, width, model_dim)
        if self.shift_size > 0:
            x = torch.roll(x, (-self.shift_size, -self.shift_size), dims=(1, 2))
            x = self.window_partition(x)
        else:
            x = self.window_partition(x)
        x = x.reshape(-1, self.window_size * self.window_size, model_dim)
        x = self.swmsa(x, self.attn_mask)
        x = self.window_reverse(x)
        
        if self.shift_size > 0:
            x = torch.roll(x, (self.shift_size, self.shift_size), dims=(1, 2))
        x = x.reshape(bs, length, model_dim)
        x = self.ln1(x)
        x = x + shortcut
        shortcut = x
        
        x = self.mlp(x)
        x = self.ln2(x)
        x = x + shortcut
        return x
        
    
    def window_partition(self, x):
        batch, height, width, channel = x.shape
        x = x.reshape((batch, height // self.window_size, self.window_size, width // self.window_size, self.window_size, channel)).transpose(2, 3)
        x = x.contiguous().view(-1, self.window_size, self.window_size, channel)
        return x
    
    def window_reverse(self, x):
        height, width = self.input_shape
        batch_size = int(x.shape[0] /  (height * width / self.window_size / self.window_size))
        x = x.reshape(batch_size, height // self.window_size, width // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, height, width, -1)
        return x

    
    @staticmethod
    def get_attn_mask(window_size, window_num, patch_size):
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
        c2 = (c1 - c1.transpose(-1, -2))  == 0
        c2 = c2.to(torch.float32)
        mask = (1 - c2) * (1e-9)
        mask = mask.reshape((num_windows, num_patch_in_window, num_patch_in_window))
        return mask
    

class StageLayer(nn.Module):
    
    def __init__(self, depth, input_shape, model_dim, num_head, window_size, mlp_ratio, dropout_rate, downsample):
        super().__init__() 
        self.blocks = nn.ModuleList(
            [TransformerBlockV2(
                input_shape,
                model_dim,
                num_head,
                window_size,
                shift_size= 0 if i % 2 == 0 else window_size // 2,
                mlp_ratio= mlp_ratio, 
                dropout_rate=dropout_rate
                ) for i in range(depth)]
        )
        
        if downsample:
            self.merge = PatchMerging(input_shape, model_dim, 2)
        else:
            self.merge = nn.Identity()
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.merge(x)
        return x
        

class SwinTransformer(nn.Module):
    
    def __init__(self, image_shape, patch_size, C, depths, num_head, window_size, mlp_ratio=4, dropout_rate=0, num_classes=0) -> None:
        super().__init__()
        assert len(depths) == 4
        model_dim = C
        
        self.add_module("patch_embedding", PatchEmbedding(C, patch_size))
        input_shape = (image_shape[0] // patch_size, image_shape[1] // patch_size)
        
        for i, d in enumerate(depths):
            self.add_module(f"starge{i+1}", StageLayer(
                depth=d,
                input_shape=input_shape,
                model_dim=model_dim,
                num_head=num_head,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout_rate=dropout_rate, 
                downsample=True if i < len(depths)-1 else False
                ))
            if i < len(depths) -1:
                model_dim *= 2
                input_shape = (input_shape[0] // 2, input_shape[1] // 2)
                
        
        if num_classes > 0:
            head = nn.Sequential(nn.LayerNorm(model_dim),
                                nn.AdaptiveAvgPool1d(1),
                                nn.Linear(model_dim, num_classes))
        else:
            head = nn.Identity()
        self.add_module("head", head)

    
    def forward(self, x):
        
        for name, m in self.named_children():
            x = m(x)
        return x
    

def swinv2_l(input_size, patch_size, num_head, window_size):
    return SwinTransformer(input_size, patch_size, 192, [2, 2, 18, 2], num_head, window_size)    
    

if __name__ == "__main__":
    model = swinv2_l((224, 224), 4, 2, 7)
    summary(model, (3, 224, 224))