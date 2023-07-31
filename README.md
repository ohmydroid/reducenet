# ReduceNet:Shrink, Distill
This repo presents a compact paradigm of knowledge distillation, where student network is a subnetwork of teacher network by removing nonlinear bottleneck branch, thereby sharing similar network architecture. Teacher network can obtain higher model performance by expanding width of nonlinear bottleneck branch while student network architecture keeps unchanged and inherits partial structure and weights from its teacher, which can effectively reduce computational cost for knowledge distillation. Finally, the goal of such pattern is to transfer ability from teacher network to student network combing combing existing and mature knowledge distillation techniques, offering an elegant distillation framework to bridge neural architecture search domain. For simplicity, we choose residual ConvBlock as our basic building block. Our design pattern can extend to any other building blocks such as inverted residual block in MobileNetV2 and DenseNet block.

There are two branches before the final Conv3 of the Basic Block. One is a non-linear branch with bottleneck structure (Conv3BnRelu and Conv1BNRelu in series, and the width of the bottleneck is determined by the expansion parameter). The other is a single-layer non-linear convolutional branch (Conv3BNRelu).

The training is divided into two stages: 

*  Both branches are trained at the same time to train a larger teacher network

*  The non-linear bottleneck branch is discarded, and the large network is degraded into a smaller student network. The student network reuses weights of teacher network during distillation process.



#### Code is based on repo [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)

#### We use [torchsummaryX](https://github.com/nmhkahn/torchsummaryX) to count parameters and MAdds of model

more detail [知乎文章]（https://zhuanlan.zhihu.com/p/634198940?）

# Result on Cifar10
| Model  Type  | Params | MAdds  | Acc (%) |
|--------------|--------|--------|---------|
| ResNet20-1-T | 412.1K | 62.0M  | 91.97   |
| ResNet20-1-S | 272.5K | 40.8M  | 92.72   |
| ResNet20-2-T | 1.6M   | 247.3M | 93.27   |
| ResNet20-2-S | 1.1M   | 162.4M | 94.40   |
| ResNet20-4-T | 6.5M   | 987.4M | 93.74   |
| ResNet20-4-S | 4.3M   | 647.7M | 95.08   |
| ResNet56-1-T | 1.3M   | 194.2M | 92.84   |
| ResNet56-1-S | 855.8K | 125.7M | 93.79   |
| ResNet56-2-T | 5.3M   | 775.8M | 94.11   |
| ResNet56-2-S | 3.4M   | 502.1M | 95.27   |

Symbols "T" and "S" in model type column denote teacher and student models respectively. Integers 1, 2 and 4 represent width factor of network.


# just run
```python
python main.py -m reduce20
```
```python
python main.py -m reduce56
```
```
sh run.sh
```



# To do
- [x] Reuse weights during second training.
- [x] Add vanilla knowledge distillation and other distillation techniques.
- [ ] Remove redundant attributes for student network instance








