import torch as trc
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"]= "10.3.0"
gpu = trc.device("cuda" if trc.cuda.is_available() else "cpu")

ex1 = trc.zeros(1, 1, 5, 5)
ex1[0, 0, :, 2] = 1

conv1 = trc.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
res1 = conv1(ex1)
print(res1)  #shows result

ex2 = ex1.to(gpu)
conv2 = conv1.to(gpu)

res2 = conv2(ex2) #error here
print(res2)