import os, sys
import os.path as osp

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import math

from models.backbone import build_backbone
from models.transformer import build_transformer
from utils.misc import clean_state_dict
from models.fore-mod import FAB

class Net(nn.Module):
    def __init__(self, backbone, transfomer, num_class):
        super().__init__()
        self.backbone = backbone
        self.transformer = transfomer
        self.num_class = num_class
        hidden_dim = transfomer.d_model
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_class, hidden_dim)
        self.fc = nn.linear(num_class, hidden_dim, bias=True)
        self.fa = FAB()
        self.conv = nn.Conv1d(200, 100, kernel_size=1, padding=0)

    def forward(self, input):
        src, pos = self.backbone(input)
        src, pos = src[-1], pos[-1]
        src1 = self.fa(src).flatten(2)
        src2 = torch.cat((src.flatten(2), src1), 2).permute(0, 2, 1)
        src = self.conv(src2).permute(0, 2, 1).reshape(10, 2048, 10, 10)
        query_input = self.query_embed.weight
        hs, sim = self.transformer(self.input_proj(src), query_input, pos)[0]  # B,K,d
        out = self.fc(hs[-1])

        return out

    def finetune_paras(self):
        from itertools import chain
        return chain(self.transformer.parameters(), self.fc.parameters(), self.input_proj.parameters(),
                     self.query_embed.parameters())

    def load_backbone(self, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=torch.device(dist.get_rank()))
        # import ipdb; ipdb.set_trace()
        self.backbone[0].body.load_state_dict(clean_state_dict(checkpoint['state_dict']), strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(path, checkpoint['epoch']))


def build(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = Net(
        backbone=backbone,
        transfomer=transformer,
        num_class=args.num_class
    )

    if not args.keep_input_proj:
        model.input_proj = nn.Identity()
        print("set model.input_proj to Indentify!")

    return model


