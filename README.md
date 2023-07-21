# ReduceNet:Shrink, Distill
This repo presents a compact paradigm of knowledge distillation, where student network is a subnetwork of teacher network by removing nonlinear bottleneck branch, thereby sharing similar network architecture. Teacher network can obtain higher model performance by expanding width of nonlinear bottleneck branch while student network architecture keeps unchanged and inherits partial structure and weights from its teacher, which can effectively reduce computational cost for knowledge distillation. Finally, the goal of such pattern is to transfer ability from teacher network to student network combing combing existing and mature knowledge distillation techniques, offering an elegant distillation framework to bridge neural architecture search domain. For simplicity, we choose residual ConvBlock as our basic building block. Our design pattern can extend to any other building blocks such as inverted residual block in MobileNetV2 and DenseNet block.

There are two branches before the final Conv3 of the Basic Block. One is a non-linear branch with bottleneck structure (Conv3BnRelu and Conv1BNRelu in series, and the width of the bottleneck is determined by the expansion parameter). The other is a single-layer non-linear convolutional branch (Conv3BNRelu).

The training is divided into two stages: 

*  Both branches are trained at the same time to train a larger teacher network

*  The non-linear bottleneck branch is discarded, and the large network is degraded into a smaller student network. The student network reuses weights of teacher network during distillation process.



#### Code is based on repo [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)

#### We use [torchsummaryX](https://github.com/nmhkahn/torchsummaryX) to count parameters and MAdds of model

more detail [知乎文章]（https://zhuanlan.zhihu.com/p/634198940?）




# just run
```python
python main.py -m reduce20
```
```python
python main.py -m reduce56
```
```shell
sh run.sh
```



# To do
- [x] Reuse weights during second training.
- [x] Add vanilla knowledge distillation and other distillation techniques.
- [ ] Remove redundant attributes for student network instance








