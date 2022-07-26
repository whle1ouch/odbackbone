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
        self.softmax = nn.Softmax(dim=-1)

        
        
        
    def forward(self, x, mask=None):
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
        
        attn = torch.bmm(q, k.transpose(-1, -2)) / math.sqrt(head_dim) + relative_position_bias 
        if mask is None:
            attn_prob = self.softmax(attn)
        else:
            num_windows = mask.shape[0]
            b_ = bs // num_windows
            mask = mask.tile(b_*num_head, 1, 1)
            attn = attn + mask
            attn_prob = self.softmax(attn) 
           
        out = torch.bmm(attn_prob, v)
        out = out.reshape((bs, num_head, sql_len, head_dim)).transpose(1, 2)
        out = out.reshape((bs, sql_len, model_dim))
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
    
    
class SwinMLPBlock(nn.Module):
    
    def __init__(self, input_shape, model_dim, num_head, window_size, shift_size, mlp_ratio, dropout_rate) -> None:
        super().__init__()
        assert len(input_shape) == 2, "input image shape error"
        assert 0 <= shift_size < window_size ,"shift size mush less than window size"
        self.window_size = window_size
        self.shift_size = shift_size
        self.input_shape = input_shape
        self.num_head = num_head
        
        self.ln1 = nn.LayerNorm(model_dim)
        self.padding = [self.window_size - self.shift_size, self.shift_size, self.window_size-self.shift_size, self.shift_size]
        self.spatial_mlp = nn.Conv1d(self.num_head * self.window_size ** 2, self.num_head * self.window_size ** 2, kernel_size=1, groups=self.num_head)
        
        self.ln2 = nn.LayerNorm(model_dim)
        mlp_hidden_dim = mlp_ratio * model_dim
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_hidden_dim), 
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, model_dim),
            nn.Dropout(dropout_rate)
            )
        
        
        
            
        
    
    def forward(self, x):
        bs, length, model_dim = x.shape
        height, width = self.input_shape
        assert length == height * width, "error length != height * width"

        shortcut = x 
        x = self.ln1(x)
        x = x.reshape(bs, height, width, model_dim)
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            x = F.pad(x, [0, 0, P_l, P_r, P_t, P_b], "constant", 0)
        
        _, _h, _w, _ = x.shape
        x = self.window_partition(x)
        x = x.view(-1, self.window_size*self.window_size, model_dim)
        x = x.view(-1, self.window_size*self.window_size, self.num_head, model_dim // self.num_head).transpose(1, 2)
        x = x.reshape(-1, self.num_head * self.window_size * self.window_size, model_dim // self.num_head)
        
        x = self.spatial_mlp(x)
        x = x.view(-1, self.num_head, self.window_size*self.window_size, model_dim // self.num_head).transpose(1, 2)
        x = x.reshape(-1, self.window_size*self.window_size, model_dim)
        x = x.reshape(-1, self.window_size, self.window_size, model_dim)
        
        x = self.window_reverse(x)
        
        if self.shift_size > 0:
            x = x[:, P_t: -P_b, P_l:-P_r, :].contiguous()
        x = x.reshape(bs, length, model_dim)
        
        x = x + shortcut
        shortcut = x
        
        x = self.ln2(x)
        x = self.mlp(x)
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

    

class StageLayer(nn.Module):
    
    def __init__(self, depth, input_shape, model_dim, num_head, window_size, mlp_ratio, dropout_rate, downsample):
        super().__init__() 
        self.blocks = nn.ModuleList(
            [SwinMLPBlock(
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
    
def swin_t(input_size, patch_size, num_head, window_size):
    return SwinTransformer(input_size, patch_size, 96, [2, 2, 6, 2], num_head, window_size)

def swin_s(input_size, patch_size, num_head, window_size):
    return SwinTransformer(input_size, patch_size, 96, [2, 2, 18, 2], num_head, window_size)

def swin_b(input_size, patch_size, num_head, window_size):
    return SwinTransformer(input_size, patch_size, 128, [2, 2, 18, 2], num_head, window_size)

def swin_l(input_size, patch_size, num_head, window_size):
    return SwinTransformer(input_size, patch_size, 192, [2, 2, 18, 2], num_head, window_size)


if __name__ == "__main__":
    model = swin_t((224, 224), 4, 2, 7)
    summary(model, (3, 224, 224))