# ReduceNet
ReduceNet不再采用VanillaNet那样在训练阶段让LambdaReLU逐渐由非线性转为线性，而直接转向对DepthShrinker的精简。与Depth Shrinker不同，完全抛弃学习mask甄别激活重要性的过程和原始的蒸馏方案。

训练分为两个阶段；
* 第一阶段scaler为1.0,LambdaReLU为纯粹的ReLU，利用expansion增加模型规模以得到一个大模型的精度。
* 第二阶段scaler为0.,冻结分类层的权重（将来会继续冻结basic block中conv3,bn2,bn3的权重，或者直接采用蒸馏的方式）LambdaReLU为纯粹的identity mapping。这个时候，该网络相当于是第一阶段的大网络退化而来的小网络。我们通过冻结大网络的分类层权重,希望能够引导小网络得到较高的模型性能。第二次训练的lr scheduler估计需要进一步人为调整,目前代码功能还不完善,有待进一步测试.



#### Code is based on repo [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)

#### We use [torchsummaryX](https://github.com/nmhkahn/torchsummaryX) to count parameters and MAdds of model




# just run
```python
python main.py -m reduce20
```
```python
python main.py -m reduce56
```



# To do
* 第二次训练时，继续复用basic block中conv3，bn2, bn3的权重
* 由于目前ReduceNet采用复用分类层，算是“知识引导”，可以看作是一种隐晦的蒸馏方式，不需要两个模型参与蒸馏。目前的整个pipeline是非常简洁的，如果效果不好，就直接采用最朴素的蒸馏方式。
* 增加融合算子的操作






