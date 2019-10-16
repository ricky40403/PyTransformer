import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import copy
from transformers.torchTransformer import TorchTransformer
# from quantize import QConv2d
model = models.__dict__["googlenet"]()
model.eval()
# print(model)
#print(len(model._modules))
# sys.exit()



#print(model._modules)
#print(model)
#sys.exit()

# print("----------------------------")
transofrmer = TorchTransformer()
#transofrmer.register(nn.Conv2d, QConv2d)

#transofrmer.trans_layers(model)

#net = transofrmer._build_graph(model)
#print(model)
# net = transofrmer.summary(model)
net = transofrmer.summary(model)
transofrmer.visualize(model, save_name= "example", graph_size = 50)
# print("QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ")
# transofrmer.visualize(model, save_name= "example2")
#print(net)