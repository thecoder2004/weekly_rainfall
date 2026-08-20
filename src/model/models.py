import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Combined_Spatial, TemporalExactor, PredictionHead, SpatialExactor2, TemporalExactorSTrans, PredictionHead2
from .stransformer import PatchEmbedding, PositionEmbedding, MHABlock, WindowMultiHeadAttention, UpsampleWithTransposedConv,SEResNet, PatchEmbedding2, PositionEmbedding2
import timm
from timm import create_model
from torchvision import transforms
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math

import torch.nn as nn
import torch.nn.functional as F

class SimpleUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(SimpleUpsample, self).__init__()
        self.scale_factor = scale_factor
        # Sử dụng 'bilinear' cho dữ liệu mượt, 'nearest' cho kết quả sắc nét hơn
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        # Conv 1x1 để điều chỉnh số kênh và tinh chỉnh đặc trưng sau khi upsample
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x có shape [B, H, W, C] từ các bước trước
        # Cần permute về [B, C, H, W] cho các lớp Conv2d và Upsample
        x = x.permute(0, 3, 1, 2) 
        x = self.upsample(x)
        x = self.conv(x)
        # Permute lại về định dạng [B, H, W, C] để phù hợp với phần còn lại của mô hình
        x = x.permute(0, 2, 3, 1)
        return x
from peft import LoraConfig, get_peft_model, TaskType


from src.model.gsmap_vit import VITGSMAP


class VIFOS_SPATIAL_TEMPORAL_EXTRACTOR(nn.Module):
    def __init__(self, config):
        super(VIFOS_SPATIAL_TEMPORAL_EXTRACTOR, self).__init__()
        self.config = config
        self.patch_size = config.MODEL.PATCH_SIZE
        
        self.embed_dim = 192
        self.hidden_dim = config.MODEL.TEMPORAL.HIDDEN_DIM
        self.num_layers = config.MODEL.SWIN_TRANSFORMER.NUM_LAYERS
        self.dropout = config.MODEL.DROPOUT
        
        
        self.patch_embed = PatchEmbedding(self.patch_size, config.MODEL.IN_CHANNEL, self.embed_dim)
        self.window_attention = WindowMultiHeadAttention(self.embed_dim, config.MODEL.SWIN_TRANSFORMER.WINDOW_SIZE, 
                                                         config.MODEL.SWIN_TRANSFORMER.NUM_HEADS,
                                                         self.num_layers, config.MODEL.SWIN_TRANSFORMER.FF_DIM, self.dropout)
        self.temporal_exactor = TemporalExactorSTrans(self.embed_dim, self.hidden_dim, self.num_layers)
        num_patches = self.cal_num_patches([self.config.DATA.HEIGHT, self.config.DATA.WIDTH])
        
        self.pos_embed = PositionEmbedding(num_patches, self.embed_dim)
        self.upsample = UpsampleWithTransposedConv(self.hidden_dim, self.embed_dim, scale_factor=self.patch_size)  # Upsample with transposed convolution

        self.esp_temporal = nn.ModuleList(
            VITGSMAP(config = self.config, out_ch=self.embed_dim)
            for _ in range(1)
        )

        # Set pretrained = False to not use pretrained weight for image-net
        vit = timm.create_model("vit_tiny_patch16_224", pretrained=True, drop_path_rate=self.dropout)
        
        vit.blocks = vit.blocks[:self.config.TRAIN.NUM_VITBLOCKS]
        
        lora_config = LoraConfig(
            r=self.config.MODEL.R,
            lora_alpha=self.config.MODEL.R,
            target_modules=["qkv", "fc1", "fc2", "proj"],
            lora_dropout=self.dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        vit.blocks = vit.blocks[:self.config.TRAIN.NUM_VITBLOCKS]
        
        peft_vit = get_peft_model(vit, lora_config)

        
        self.spatial_encoder = peft_vit.blocks 
        print("Tích hợp LoRA hoàn tất.")
        

        middle = self.embed_dim//2
        
        self.proj_x = nn.Sequential(
            nn.Linear(self.embed_dim, middle),
            nn.LayerNorm(middle),
            nn.Linear(middle, self.embed_dim)
        )
        self.proj_h = nn.Sequential(
            nn.Linear(self.embed_dim, middle),
            nn.LayerNorm(middle),
            nn.Linear(middle, self.embed_dim)
        )
        self.prompt_type = config.MODEL.PROMPT_TYPE
        self.add_type = config.MODEL.TEMPORAL.ADDING_TYPE
        # self.h_after = nn.Parameter(torch.zeros(self.config.TRAIN.BATCH_SIZE, self.config.DATA.HEIGHT, self.config.DATA.WIDTH, self.embed_dim))
        if self.prompt_type == 0:
            max_delta_t = config.MODEL.TEMPORAL.MAX_DELTA_T
            embed_dim = self.embed_dim
            pos_encoding = torch.zeros(max_delta_t, embed_dim)
            position = torch.arange(0, max_delta_t, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
            pos_encoding[:, 0::2] = torch.sin(position * div_term)
            if embed_dim % 2 == 1:
                pos_encoding[:, 1::2] = torch.cos(position * div_term)[:, :-1]
            else:
                pos_encoding[:, 1::2] = torch.cos(position * div_term)
            self.delta_t = nn.Parameter(pos_encoding, requires_grad=True)
        else:
            raise("Wrong prompt_type")
        
        self.prediction_head = PredictionHead2(self.embed_dim,
                                              use_layer_norm=config.MODEL.USE_LAYER_NORM,
                                              dropout=self.dropout)

    def cal_num_patches(self, img_size):
        h, w = img_size[0], img_size[1]
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        padded_h, padded_w = h + pad_h, w + pad_w
        num_patches = (padded_h // self.patch_size) * (padded_w // self.patch_size)
        return num_patches
    
    def add_prompt_vecs(self, temporal_embedding, lead_time):
        list_prompt = []
        if self.prompt_type == 0:
            if self.add_type == 0:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    # print(lt)
                    assert lt < len(self.delta_t), f"lead_time {lt} out of range"
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return temporal_embedding + add_prompt
            

            elif self.add_type == 1:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return torch.concat([temporal_embedding, add_prompt], -1)
            else:
                raise("Wrong adding type value")
            
        else:
            raise("Wrong prompt type value")

    def forward(self, x):
        esp = x[2]
        lead_time = x[1]
        x = x[0]
        batch_size, n_ts, n_ft, h, w = x.shape
        x = x.view(batch_size * n_ts, n_ft, h, w)  
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h)) 
        padded_h, padded_w = h + pad_h, w + pad_w
        
        x = self.patch_embed(x)  
        x = self.pos_embed(x)  
        h_patch = padded_h // self.patch_size
        w_patch = padded_w // self.patch_size
        x = self.spatial_encoder(x)
        
        x = x.reshape(batch_size, n_ts, h_patch, w_patch, -1) 
        x = self.temporal_exactor(x) 
        x = self.upsample(x)  
        x = x[:, :h, :w, :] 
        h_after = self.esp_temporal[0](esp)
        
        x = self.add_prompt_vecs(x, lead_time) # Uncomment/comment to use/not use LT-embedding
                
                
        x = x + h_after # Uncomment/comment to use/not use GsMAP
        self.res = x
        return x 
  

class VIFOS_CONV3D(nn.Module):
    def __init__(self, config):
        super(VIFOS_CONV3D, self).__init__()
        self.config = config
        self.patch_size = config.MODEL.PATCH_SIZE
        self.embed_dim = 192 
        self.dropout = config.MODEL.DROPOUT
       
        
        
        self.patch_embed = PatchEmbedding2(self.patch_size, config.MODEL.IN_CHANNEL, self.embed_dim)
        
        
        self.scale_time_factor, num_patches = self.cal_num_patches([self.config.MODEL.ECMWF_TIME_STEP, self.config.DATA.HEIGHT, self.config.DATA.WIDTH])

        self.pos_embed = PositionEmbedding2(embed_dim=self.embed_dim)
        
        self.upsample = UpsampleWithTransposedConv(self.embed_dim * self.scale_time_factor * (config.MODEL.TEMPORAL.ADDING_TYPE + 1), self.embed_dim, scale_factor=self.patch_size)
        self.esp_temporal = nn.ModuleList(
            VITGSMAP(config = self.config, out_ch=self.embed_dim)
            for _ in range(1)
        )
        print("Tích hợp LoRA vào khối spatial_encoder...")

        # Set pretrained = False to not use pretrained weight for image-net
        vit = timm.create_model("vit_tiny_patch16_224", pretrained=True, drop_path_rate=self.dropout)

        
        
        vit.blocks = vit.blocks[:self.config.TRAIN.NUM_VITBLOCKS]
        
        lora_config = LoraConfig(
            r=self.config.MODEL.R,
            lora_alpha=self.config.MODEL.R,
            target_modules=["qkv", "fc1", "fc2", "proj"],
            lora_dropout=self.dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        vit.blocks = vit.blocks[:self.config.TRAIN.NUM_VITBLOCKS]
        
        peft_vit = get_peft_model(vit, lora_config)

        
        self.spatial_encoder = peft_vit.blocks 
        print("Tích hợp LoRA hoàn tất.")
        

        middle = self.embed_dim//2
        
        self.proj_x = nn.Sequential(
            nn.Linear(self.embed_dim, middle),
            nn.LayerNorm(middle),
            nn.Linear(middle, self.embed_dim)
        )
        self.proj_h = nn.Sequential(
            nn.Linear(self.embed_dim, middle),
            nn.LayerNorm(middle),
            nn.Linear(middle, self.embed_dim)
        )
        self.prompt_type = config.MODEL.PROMPT_TYPE
        self.add_type = config.MODEL.TEMPORAL.ADDING_TYPE
        # self.h_after = nn.Parameter(torch.zeros(self.config.TRAIN.BATCH_SIZE, self.config.DATA.HEIGHT, self.config.DATA.WIDTH, self.embed_dim))
        if self.prompt_type == 0:
            max_delta_t = config.MODEL.TEMPORAL.MAX_DELTA_T
            embed_dim = self.embed_dim
            pos_encoding = torch.zeros(max_delta_t, embed_dim)
            position = torch.arange(0, max_delta_t, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
            pos_encoding[:, 0::2] = torch.sin(position * div_term)
            if embed_dim % 2 == 1:
                pos_encoding[:, 1::2] = torch.cos(position * div_term)[:, :-1]
            else:
                pos_encoding[:, 1::2] = torch.cos(position * div_term)
            self.delta_t = nn.Parameter(pos_encoding, requires_grad=True)
        else:
            raise("Wrong prompt_type")
        
        self.prediction_head = PredictionHead2(self.embed_dim,
                                                use_layer_norm=config.MODEL.USE_LAYER_NORM,
                                                dropout=self.dropout)

    def cal_num_patches(self, img_size):
        
        t, h, w = img_size[0], img_size[1], img_size[2]
        pad_t = (self.patch_size - t % self.patch_size) % self.patch_size
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        padded_t, padded_h, padded_w = t + pad_t, h + pad_h, w + pad_w
        num_patches = (padded_h // self.patch_size) * (padded_w // self.patch_size) * (padded_t // self.patch_size)
        return (padded_t // self.patch_size), num_patches
    
    def add_prompt_vecs(self, temporal_embedding, lead_time):
        
        list_prompt = []
        if self.prompt_type == 0:
            if self.add_type == 0:
                for lt in lead_time:
                    lt = int(lt)
                    lt -= 1
                    assert lt < len(self.delta_t), f"lead_time {lt} out of range"
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt, 0)
                repetition_factors = (1, 1, 1, temporal_embedding.shape[3]//add_prompt.shape[3])
                # print(temporal_embedding.shape, add_prompt.shape)
                add_prompt = add_prompt.repeat(repetition_factors)
                
                # print(temporal_embedding.shape, add_prompt.shape)
                return temporal_embedding + add_prompt
            elif self.add_type == 1:
                for lt in lead_time:
                    lt -= 7
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt, 0)
                repetition_factors = (1, 1, 1, self.scale_time_factor)
                add_prompt = add_prompt.repeat(repetition_factors)
                return torch.concat([temporal_embedding, add_prompt], -1)
            else:
                raise("Wrong adding type value")
        else:
            raise("Wrong prompt type value")

    def forward(self, x):
        
        esp = x[2]
        
        lead_time = x[1]
        x_begin = x[0]
        batch_size, n_ts, n_ft, h, w = x_begin.shape
        
        
        x = x_begin.permute(0, 2, 1, 3, 4)
        pad_t = (self.patch_size - n_ts % self.patch_size) % self.patch_size
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))
        padded_t, padded_h, padded_w = n_ts + pad_t, h + pad_h, w + pad_w
        x = x.view(batch_size, n_ft, padded_t, padded_h, padded_w)

        
        x_sequence, x_grid = self.patch_embed(x)
        pos_embedding = self.pos_embed(x_grid)
        x = x_sequence + pos_embedding
        
        x = self.spatial_encoder(x)
        
        
        h_patch = padded_h // self.patch_size
        w_patch = padded_w // self.patch_size
        x = x.reshape(batch_size, h_patch, w_patch, -1) 
        
        x = self.upsample(x)
        x = x[:, :h, :w, :]

        

        h_after = self.esp_temporal[0](esp)
        
        x = self.add_prompt_vecs(x, lead_time) # Uncomment/comment to use/not use LT-embedding
        
        
        x = x + h_after # Uncomment/comment to use/not use GsMAP
        x = self.prediction_head(x)
        self.res = x
        
        x = x 
        return x 
#ver_5: not pretrain
#ver_4: num  vit
#ver_3: patch embedding

### round 1
#ver_6: full
#ver_5: not pretrain
#ver_4: not GsMAP
#ver_4b: not Lt embedding
#ver_3: spatial-temporal