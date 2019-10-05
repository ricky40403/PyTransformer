import torch.nn as nn
import torchvision
import torchvision.models as models

from torchTransformer import TorchTransformer

model = models.__dict__["resnet101"]()
#print(len(model._modules))
#sys.exit()



#print(model._modules)
#print(model)
#sys.exit()

print("----------------------------")
transofrmer = TorchTransformer()
#print(model)
net = transofrmer.summary(model)
#print(net)