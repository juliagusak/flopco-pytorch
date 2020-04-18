FlopCo
=====

FlopCo is a Python library that aims to make FLOPs and MACs counting simple and accessible for Pytorch neural networks.
Moreover FlopCo allows to collect other useful model statistics, such as number of parameters, shapes of layer inputs/outputs, etc.

Requirements
-----
- numpy
- tensorflow>=2.0

Installation
-----
```pip install flopco-keras ```

Quick start
-----
```python
from flopco import FlopCo
import tensorflow as tf

model = tf.keras.applications.ResNet101()
stats = FlopCo(model)

print(f"FLOPs: {stats.total_flops}")
print(f"MACs: {stats.total_macs}")
print(f"Relative FLOPs: {stats.relative_flops}")
```

List of estimated statistics includes:
- total number of FLOPs/MACs
- number of FLOPs/MACs for each layer
- relative number of FLOPs/MACs for each layer

Make sure your tf.keras model is builded properly

MACS for:
- ResNet50: 3879147569 (3.8B)
- ResNet101: 7601604657 (7.6B)
- ResNet152: 11326470193 (11.3B)

Same as [eq here](https://neurohive.io/ru/vidy-nejrosetej/resnet-34-50-101/)

License
-----

Project is distributed under [MIT License](https://github.com/juliagusak/flopco-pytorch/blob/master/LICENSE.txt)
