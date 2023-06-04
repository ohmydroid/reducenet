# ReduceNet
ReduceNet不再像VanillaNet那样在训练阶段让LambdaReLU逐渐由非线性转为线性，也不再是DepthShrinker的精简版本。

训练分为两个训练阶段；
* 在第一个阶段，Basic Block最后的Conv3之前存在两个分支，一个是bottleneck结构的非线性分支（Conv3BnRelu和Conv1BNRelu串联，bottleneck中间的宽度由参数expansion决定），一个是单层非线性卷积的分支（Conv3BNRelu），两个分支同时参与训练。
* 第二阶段，丢弃bottleneck结构的非线性分支，大网络退化成小网络。小网络复用大网络的分类层，部分卷积层和BN层，引入LORA线性分支增加小网络学习能力，事后融合.根据情况，可以在LORA的中间插入VanillaNet的LambdaReLU，在训练过程中引入非线性，最终转为线性。

第二次训练的lr scheduler估计需要进一步人为调整,目前代码功能还没完全实现,效果也有待进一步调式.



#### Code is based on repo [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)

#### We use [torchsummaryX](https://github.com/nmhkahn/torchsummaryX) to count parameters and MAdds of model

具体见[知乎文章]（https://zhuanlan.zhihu.com/p/634198940?）




# just run
```python
python main.py -m reduce20
```
```python
python main.py -m reduce56
```



# To do

* 第二次训练时，继续复用一些层的参数
* 引入LORA，利用VanillaNet中的LambdaReLU(deep training strategy) 改进LORA
* 由于目前ReduceNet采用复用分类层，算是“知识引导”，可以看作是一种隐晦的蒸馏方式，不需要两个模型参与蒸馏。目前的整个pipeline是非常简洁的，如果效果不好，就直接采用最朴素的蒸馏方式,soft labels 和特征对齐。







