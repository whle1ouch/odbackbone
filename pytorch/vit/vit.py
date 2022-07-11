import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


def image2emb_naive(image: torch.Tensor, path_size: int, weight: torch.Tensor):
    """ 'image to embedding' through blockmeal method 

    Args:
        image (torch.Tensor): input image tensor, must be 4-D (batch, in_channels, height, weight)
        path_size (int): the size of each patch block.
        weight (torch.Tensor): output weight matrix, must be 2-D (patch_depth, output_dim), and patch_depth = patch_size * patch_size * in_channels 

    Returns:
        torch.Tensor: image embedding tensor 
    """
    # input shape: b * c * h * w, patch_size: p, weigth_shape: patch_pixel_num * output_dim
    # unfold: b * (c * p * p) * ( h // p * w // p ) = b * patch_depth * patch_num
    patch = F.unfold(image, kernel_size=path_size, stride=path_size).transpose(-1, -2) # b * patch_num * patch_depth
    patch_embedding = patch @ weight  # b * patch_num * output_dim
    return patch_embedding

def image2emb_conv(image, kernel, stride):
    """'image to embedding' through conv method 

    Args:
        mage (torch.Tensor): input image tensor, must be 4-D (batch, in_channels, height, weight)
        kernel (torch.Tensor): 2-D convolution kernal, size is (output_dim, in_channels. patch_size, patch_size)
        stride (int): 2-D convolution stride

    Returns:
        torch.Tensor: _description_
    """
    # 
    conv_output = F.conv2d(image, kernel, stride=stride) # b * oc * p * p 
    batch, output_channel, output_height, output_weight = conv_output.shape
    patch_embedding = conv_output.reshape((batch, output_channel, output_height * output_weight)).transpose(-1, -2)
    return patch_embedding


class ImageEmbeding(nn.Module):
    
    def __init__(self, patch_size: int, model_dim: int, max_token:int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, out_channels=model_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token_embedding = nn.Parameter(torch.randn(1, 1, model_dim))
        self.position_embedding_table = nn.Parameter(torch.randn(max_token, model_dim))
    
    def forward(self, x):
        patch_embedding = self.conv(x)
        bs, oc, h, w = patch_embedding.shape
        patch_embedding = torch.reshape(patch_embedding, (bs, oc, h*w))
        patch_embedding = torch.transpose(patch_embedding, -1, -2)
        
        cls_token_embeding = torch.tile(self.cls_token_embedding, [patch_embedding.shape[0], 1, 1])
        token_embedding = torch.cat([cls_token_embeding, patch_embedding], dim=1)
        seq_len = token_embedding.shape[1]
        position_embedding = torch.tile(self.position_embedding_table[:seq_len], [token_embedding.shape[0], 1, 1])
        token_embedding += position_embedding
        return token_embedding
        

class Vit(nn.Module):
    
    def __init__(self, patch_size: int, model_dim: int, num_class: int, encoder_head: int = 8, num_layers: int = 6, max_token: int = 16) -> None:
        super().__init__()
        self.image2embed = ImageEmbeding(patch_size, model_dim, max_token)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=encoder_head)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifer = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, num_class)
            )
        
        
    def forward(self, x):
        x = self.image2embed(x)
        encoder_output = self.transformer_encoder(x)
        cls_token_output = encoder_output[:, 0]
        x = self.classifer(cls_token_output)
        nn.BatchNorm1d
        return x 
        
    
if __name__ == "__main__":
    model = Vit(4, 16, 10)
    summary(model, (3, 8, 8))