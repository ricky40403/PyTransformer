import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import copy
from transformers.torchTransformer import TorchTransformer
# from quantize import QConv2d
model = models.__dict__["inception_v3"]()
model.cuda()
model = model.eval()

transofrmer = TorchTransformer()

net = transofrmer.summary(model)
# transofrmer.visualize(model, save_name= "example", graph_size = 80)