FlopCo
=====

FlopCo is a Python library that aims at making FLOPs and MACs counting simple and accessible for Pytorch neural networks.
Moreover FlopCo allows to collect oter useful model statistics, such as number of parameters, shapes of layer inputs/outputs, etc. 

Requirements
-----
- numpy
- pytorch

Installation
-----
```pip install flopco-pytorch ```

Quick start
-----
```python
from flopco import FlopCo
from torchvision.models import resnet50

device = 'cuda'
model = resnet50().to(device)

# Estimate model statistics by making one forward pass througth the model, 
# for the input image of size 3 x 224 x 224

stats = FlopCo(model, img_size = (1, 3, 224, 224), device = device)

print(stats.total_macs, stats.relative_flops)
```

List of estimated statistics includes:
- total number of FLOPs/MACs/parameters
- number of FLOPs/MACs/parameters for each layer
- relative number of FLOPs/MACs/parameters for each layer
- input/output shapes for each layer

By default for statistics counting nn.Conv2d and nn.Linear layers  are used. 
To include more layer types in computation, pass ```instances``` to the constructor

```python
stats = FlopCo(model,
               img_size = (1, 3, 224, 224),
               device = device,
               instances = [nn.Conv2d, nn.Linear,\
                            nn.BatchNorm2d, nn.ReLU,\
                            nn.MaxPool2d, nn.AvgPool2d,\
                            nn.Softmax]
               )
 ```

License
-----

Project is distributed under [MIT License](https://github.com/juliagusak/flopco-pytorch/blob/master/LICENSE.txt)
