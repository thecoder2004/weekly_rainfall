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
from .MSGNet import ScaleGraphModel2D
 
class SwinTransformer_Ver3(nn.Module):
    def __init__(self, config):
        super(SwinTransformer_Ver3, self).__init__()
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
        
        vit = timm.create_model("vit_tiny_patch16_224", pretrained=True, drop_rate = self.dropout, attn_drop_rate=self.dropout/2, drop_path_rate=self.dropout/2)
        self.spatial_encoder = vit.blocks[:config.TRAIN.NUM_VITBLOCKS]
     

        self.prompt_type = config.MODEL.PROMPT_TYPE
        self.add_type = config.MODEL.TEMPORAL.ADDING_TYPE
        if self.prompt_type == 0:    
            self.delta_t = nn.Parameter(torch.randn(config.MODEL.TEMPORAL.MAX_DELTA_T, self.hidden_dim))
        else:
            raise("Wrong prompt_type")
        
        self.prediction_head = PredictionHead(self.embed_dim,
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
                    
                    lt -= 7
                    
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
        lead_time = x[1]
        x = x[0]
        batch_size, n_ts, n_ft, h, w = x.shape
        
        x = x.view(batch_size * n_ts, n_ft, h, w)  # (batch_size * n_ts, n_ft, h, w)

        # Step 0: Pad the input to make h and w divisible by patch_size
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # Pad (left, right, top, bottom)
        padded_h, padded_w = h + pad_h, w + pad_w
        
        # Step 1: Patch embedding
        x = self.patch_embed(x)  # (batch_size * n_ts, num_patches, embed_dim) ==> B, Patch, embeddim, use Con3D

        # Step 2: Position embedding
        x = self.pos_embed(x)  # (batch_size * n_ts, num_patches, embed_dim)

        # Step 3: Reshape for window-based attention
        h_patch = padded_h // self.patch_size
        w_patch = padded_w // self.patch_size
        x = self.spatial_encoder(x)
        # Step 4: Apply window-based multi-head attention
        # x = self.window_attention(x)  # (batch_size * n_ts, h_patch, w_patch, embed_dim)
        

        
        ## Step 4.1 To-Do temporal-exactor 
        x = x.reshape(batch_size, n_ts, h_patch, w_patch, -1) # (batch_size, n_ts, h_patch, w_patch, embed_dim)
        x = self.temporal_exactor(x) # (batch_size, h_patch, w_patch, embed_dim) ==> Khong can
        
        ## Step 4.2 To-do adding delta_t the expected output shape is : batch, h_patch, w_patch, embed_dim
        x = self.add_prompt_vecs(x, lead_time) # (batch_size, h_patch, w_patch, embed_dim)
        
        # Step 5: Upsample to original resolution
        x = self.upsample(x)  # (batch_size, h, w, embed_dim)
        x = x[:, :h, :w, :] # (batch_size, h, w, embed_dim)

        # Step 6: To-Do add prediction head on it
        x = self.prediction_head(x) # (batch_size, h, w)
        self.res = x
        return x 
    
class SwinTransformer_Ver4(nn.Module):
    def __init__(self, config):
        super(SwinTransformer_Ver4, self).__init__()
        self.config = config
        self.patch_size = config.MODEL.PATCH_SIZE
        self.embed_dim = 192 
        self.dropout = config.MODEL.DROPOUT
        self.patch_embed = PatchEmbedding2(self.patch_size, config.MODEL.IN_CHANNEL, self.embed_dim)
        self.scale_time_factor, num_patches = self.cal_num_patches([self.config.MODEL.ECMWF_TIME_STEP, self.config.DATA.HEIGHT, self.config.DATA.WIDTH])
        self.pos_embed = PositionEmbedding2(self.embed_dim)
        self.upsample = UpsampleWithTransposedConv(self.embed_dim * self.scale_time_factor * (config.MODEL.TEMPORAL.ADDING_TYPE + 1), self.embed_dim, scale_factor=self.patch_size)
        self.esp_temporal = nn.ModuleList(
            VITGSMAP(config = self.config, out_ch=self.embed_dim)
            for _ in range(1)
        )
        print("LoRA config...")
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
        print("LoRA config ok")
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
                
                add_prompt = add_prompt.repeat(repetition_factors)
                
                
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
        if len(x) >= 3:
            esp = x[2]
        else: esp = None
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
        # h_after = self.proj_h(h_after)
        x = self.add_prompt_vecs(x, lead_time)
        
        
        x = x # + h_after
        x = self.prediction_head(x)
        self.res = x
        x = x 
        return x 
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math

import torch.nn as nn
import torch.nn.functional as F

from peft import LoraConfig, get_peft_model, TaskType
class SwinTransformer_Ver5(nn.Module):
    def __init__(self, config):
        super(SwinTransformer_Ver5, self).__init__()
        self.config = config
        self.patch_size = config.MODEL.PATCH_SIZE
        self.embed_dim = 192 
        self.dropout = config.MODEL.DROPOUT
        self.patch_embed = PatchEmbedding2(self.patch_size, config.MODEL.IN_CHANNEL, self.embed_dim)
        self.scale_time_factor, num_patches = self.cal_num_patches([self.config.MODEL.ECMWF_TIME_STEP, self.config.DATA.HEIGHT, self.config.DATA.WIDTH])
        self.pos_embed = PositionEmbedding2(self.embed_dim)
        self.upsample = UpsampleWithTransposedConv(self.embed_dim * self.scale_time_factor * (config.MODEL.TEMPORAL.ADDING_TYPE + 1), self.embed_dim, scale_factor=self.patch_size)
        self.esp_temporal = nn.ModuleList(
            VITGSMAP(config = self.config, out_ch=self.embed_dim)
            for _ in range(1)
        )
        print("LoRA config...")
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
        print("LoRA config ok")
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
                
                add_prompt = add_prompt.repeat(repetition_factors)
                
                
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
        if len(x) >= 3:
            esp = x[2]
        else: esp = None
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
        h_after = self.proj_h(h_after)
        # x = self.add_prompt_vecs(x, lead_time)
        
        
        x = x + h_after
        x = self.prediction_head(x)
        self.res = x
        x = x 
        return x 
######################################################################################


from src.model.gsmap_vit import VITGSMAP
class SwinTransformer_Ver6(nn.Module):
    def __init__(self, config):
        super(SwinTransformer_Ver6, self).__init__()
        self.config = config
        self.patch_size = config.MODEL.PATCH_SIZE
        self.embed_dim = 192 
        self.dropout = config.MODEL.DROPOUT
        self.patch_embed = PatchEmbedding2(self.patch_size, config.MODEL.IN_CHANNEL, self.embed_dim)
        self.scale_time_factor, num_patches = self.cal_num_patches([self.config.MODEL.ECMWF_TIME_STEP, self.config.DATA.HEIGHT, self.config.DATA.WIDTH])
        self.pos_embed = PositionEmbedding2(self.embed_dim)
        self.upsample = UpsampleWithTransposedConv(self.embed_dim * self.scale_time_factor * (config.MODEL.TEMPORAL.ADDING_TYPE + 1), self.embed_dim, scale_factor=self.patch_size)
        self.esp_temporal = nn.ModuleList(
            VITGSMAP(config = self.config, out_ch=self.embed_dim)
            for _ in range(1)
        )
        print("LoRA config...")
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
        print("LoRA config ok")
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
                
                add_prompt = add_prompt.repeat(repetition_factors)
                
                
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
        if len(x) >= 3:
            esp = x[2]
        else: esp = None
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
        h_after = self.proj_h(h_after)
        x = self.add_prompt_vecs(x, lead_time)
        
        
        x = x + h_after
        x = self.prediction_head(x)
        self.res = x
        x = x 
        return x 
