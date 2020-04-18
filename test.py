import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""
import tensorflow as tf
import glob
import sys 
import numpy as np
import time
import logging
import json
from flopco import FlopCo

FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT,level=logging.DEBUG)
log = logging.getLogger("basic")

m_hash = "cd0d753f898b816928ff01b2a05bde18"
filename = f"../nasbench_latency/data/config128/{m_hash}.h5"
# keylist = [f for f in glob.glob(f"{H5_PATH}*.h5", recursive=True)]
# filename = keylist[5678]
log.info(f"filename: {filename}")
tf.debugging.set_log_device_placement(True)
with tf.device(f'CPU'):
    model = tf.keras.models.load_model(filename)

device = 'cpu'
# Estimate model statistics by making one forward pass througth the model, 
# for the input image of size 3 x 92 x 92

stats = FlopCo(model)

print(stats.total_flops, stats.total_macs, stats.relative_flops)


resnet50 = tf.keras.applications.ResNet101()

stats = FlopCo(resnet50)

print(stats.total_flops, stats.total_macs, stats.relative_flops)


1453484047
1469841408

3879147569
726663168
7601604657
11326470193