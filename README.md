# PyTranformer



## summary
This repository implement the summary function similar to keras summary()  

```
model = models.__dict__["resnet18"]()
model.eval()
transofrmer = TorchTransformer()
net = transofrmer.summary(model)
```  

the example is in [example.ipynb](/examples/example.ipynb)

## visualize
visualize using [graphviz](https://graphviz.readthedocs.io/en/stable/) and [pydot](https://pypi.org/project/pydot/)  
it will show the architecture.  
Such as alexnet in torchvision:
```
model = models.__dict__["alexnet"]()
model.eval()
transofrmer = TorchTransformer()
transofrmer.visualize(model, save_name= "example", graph_size = 80)
# graph_size can modify to change the size of the output graph
# graphviz does not auto fit the model's layers, which mean if the model is too deep, it will become too small to see.
# so change the graph size to enlarge the layer 
```  
<img src=/examples/alexnet.png  height =500  width=100> 

other example is in [examples](/examples)

## Note
Suggest that the layers input should not be too many because the graphviz may generate image slow.(eg: densenet161 in torchvision 0.4.0 may stuck when generating png)

## TODO
- [ ] support registration for custom layer
- [x] activation size calculation for supported layers
- [x] network summary output as in keras
- [x] model graph visualization
