import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import CLIPImageProcessor
import timm

from transformers.modeling_utils import get_parameter_device, get_parameter_dtype

@dataclass
class CoCaVisionCfg:
    image_size: int = 448
    hidden_size: int = 768 
    context_size: int = 768


def get_norm_constants(which_img_norm: str = 'imagenet'):
    constants_zoo = {'imagenet': {'mean': (0.485, 0.456, 0.406),
                                  'std': (0.229, 0.224, 0.225)},
                     'uniform': {'mean': (0.5, 0.5, 0.5),
                                 'std': (0.5, 0.5, 0.5)}}
    constants = constants_zoo[which_img_norm]
    return constants.get('mean'), constants.get('std')

class CoCaVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.cfg_only = CoCaVisionCfg() # TODO: add more config options

        self.vision_tower_name = vision_tower

        if vision_tower == 'coca_vit-l':
            self.cfg_only.context_size = 1024

        if args.mm_vision_select_layer == -1:
            self.select_layer = args.mm_vision_select_layer
        else:
            raise NotImplementedError('select_layer not implemented for CoCaVisionTower')
        
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()

    def load_model(self):
        vision_cfg = self.config
        
        mean, std = get_norm_constants('imagenet')
        self.image_processor = CLIPImageProcessor(size={"shortest_edge": vision_cfg.image_size},
                                                  crop_size=vision_cfg.image_size,
                                                  do_resize=True,
                                                  do_center_crop=True,
                                                  do_rescale=True,
                                                  do_normalize=True,
                                                  do_convert_rgb=True,
                                                  rescale_factor=1 / 255,
                                                  image_mean=mean,
                                                  image_std=std)

        if self.vision_tower_name == 'coca_vit-l':
            self.vision_tower = timm.create_model("vit_large_patch16_224", 
                                                  img_size=vision_cfg.image_size, 
                                                  patch_size=16, 
                                                  init_values=1., 
                                                  num_classes=0, 
                                                  dynamic_img_size=True)
            self.vision_tower.forward = self.vision_tower.forward_features
        else:
            raise ValueError(f'Unknown vision tower: {self.vision_tower_name}')
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        if self.select_feature == 'patch':
            image_features = image_forward_outs
        elif self.select_feature == 'cls_patch':
            raise NotImplementedError('cls_patch not implemented for CoCaVisionTower')
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return get_parameter_device(self.vision_tower)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self.vision_tower)

    @property
    def config(self):
        return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size
    
    @property
    def context_size(self):
        return self.config.context_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
    

