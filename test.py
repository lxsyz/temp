# -*- coding: utf-8 -*-

import tensorflow as tf
import torch
import numpy as np

i = np.loadtxt('tf_i.txt', dtype=np.float32)
w = np.loadtxt('tf_w.txt', dtype=np.float32)

torch_input = torch.tensor(i).reshape(27,1,512)
torch_w = torch.tensor(w)
print(torch_input.matmul(torch_w).numpy())

print(np.matmul(i,w))

print(tf.matmul(tf.convert_to_tensor(i), tf.convert_to_tensor(w)))

