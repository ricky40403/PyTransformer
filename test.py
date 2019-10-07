import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

from torchTransformer import TorchTransformer
from quantize import QConv2d
model = models.__dict__["resnet18"]()
#print(len(model._modules))
#sys.exit()



#print(model._modules)
#print(model)
#sys.exit()

print("----------------------------")
transofrmer = TorchTransformer()
#transofrmer.register(nn.Conv2d, QConv2d)

#transofrmer.trans_layers(model)

#net = transofrmer._build_graph(model)
#print(model)
#net = transofrmer.summary(model)
transofrmer.visualize(model)
#print(net)