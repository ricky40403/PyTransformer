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
input_tensor = torch.randn([1, 3, 224, 224])		
input_tensor = input_tensor.cuda()		
net = transofrmer.summary(model, input_tensor=input_tensor)
# transofrmer.visualize(model, save_name= "example", graph_size = 80)