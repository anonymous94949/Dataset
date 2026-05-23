import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from gcn_lib import Grapher, act_layer

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from torch_scatter import scatter_mean
from segment_edge import pixel_to_patch_segment, segment_based_edge_index, pixel_to_patch_segment_overlapping


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'gnn_patch16_224': _cfg(
        crop_pct=0.9, input_size=(3, 224, 224),
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class Stem(nn.Module):
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//8, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//8),
            act_layer(act),
            nn.Conv2d(out_dim//8, out_dim//4, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//4),
            act_layer(act),
            nn.Conv2d(out_dim//4, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class DeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        self.n_blocks = opt.n_blocks
        drop_path = opt.drop_path
        self.use_segmentation_edge = opt.use_segmentation_edge

        self.stem = Stem(out_dim=channels, act=act)
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]
        print('dpr', dpr)
        num_knn = [int(x.item()) for x in torch.linspace(k, 2*k, self.n_blocks)]
        print('num_knn', num_knn)
        max_dilation = 196 // max(num_knn)

        self.pos_embed = nn.Parameter(torch.zeros(1, channels, 14, 14))

        self.backbone = nn.ModuleList([nn.Sequential(
                Grapher(channels, num_knn[i], min(i // 4 + 1, max_dilation), conv, act, norm,
                        bias, stochastic, epsilon, 1, drop_path=dpr[i],
                        use_segmentation_edge=self.use_segmentation_edge),
                FFN(channels, channels * 4, act=act, drop_path=dpr[i])
            ) for i in range(self.n_blocks)])

        self.prediction = Seq(nn.Conv2d(channels, 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer(act),
                              nn.Dropout(opt.dropout),
                              nn.Conv2d(1024, opt.n_classes, 1, bias=True))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def get_feature(self):
        return self.feature

    def forward(self, inputs, train_mech='vig', b_size=32, n_image=8, seg_result=None):
        x = self.stem(inputs) + self.pos_embed
        B, C, H, W = x.shape
        batch_num = B // n_image

        for i in range(self.n_blocks):
            if self.use_segmentation_edge == 'seg_only':
                seg_edge_index = segment_based_edge_index(x, segment_labels=seg_result)
                x = self.backbone[i][0](x, seg_label=seg_edge_index)
                x = self.backbone[i][1](x)
            elif self.use_segmentation_edge in ['seg_knn', 'knn_in_seg']:
                x = self.backbone[i][0](x, seg_label=seg_result)
                x = self.backbone[i][1](x)
            else:
                x = self.backbone[i][0](x)
                x = self.backbone[i][1](x)

        x = F.adaptive_avg_pool2d(x, 1)

        if train_mech == 'vig':
            self.feature = x.squeeze()

        if train_mech == 'vig_pano_feat':
            index = torch.arange(batch_num).repeat_interleave(n_image).to(inputs.device)
            x = scatter_mean(x, index, dim=0)
            self.feature = x.squeeze()

        output = self.prediction(x).squeeze(-1).squeeze(-1)

        if train_mech == 'vig_pano_pred':
            index = torch.arange(batch_num).repeat_interleave(n_image).to(inputs.device)
            output = scatter_mean(output, index, dim=0)
            self.feature = output

        return output


@register_model
def vig_ti_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0,
                     num_knn=9, use_segmentation_edge=False, **kwargs):
            self.k = num_knn
            self.conv = 'mr'
            self.act = 'gelu'
            self.norm = 'batch'
            self.bias = True
            self.n_blocks = 12
            self.n_filters = 192
            self.n_classes = num_classes
            self.dropout = drop_rate
            self.use_dilation = True
            self.epsilon = 0.2
            self.use_stochastic = False
            self.drop_path = drop_path_rate
            self.use_segmentation_edge = use_segmentation_edge

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model


@register_model
def vig_s_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0,
                     num_knn=9, use_segmentation_edge=False, **kwargs):
            self.k = num_knn
            self.conv = 'mr'
            self.act = 'gelu'
            self.norm = 'batch'
            self.bias = True
            self.n_blocks = 16
            self.n_filters = 320
            self.n_classes = num_classes
            self.dropout = drop_rate
            self.use_dilation = True
            self.epsilon = 0.2
            self.use_stochastic = False
            self.drop_path = drop_path_rate
            self.use_segmentation_edge = use_segmentation_edge

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model


@register_model
def vig_b_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0,
                     num_knn=9, use_segmentation_edge=False, **kwargs):
            self.k = num_knn
            self.conv = 'mr'
            self.act = 'gelu'
            self.norm = 'batch'
            self.bias = True
            self.n_blocks = 16
            self.n_filters = 640
            self.n_classes = num_classes
            self.dropout = drop_rate
            self.use_dilation = True
            self.epsilon = 0.2
            self.use_stochastic = False
            self.drop_path = drop_path_rate
            self.use_segmentation_edge = use_segmentation_edge

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model
