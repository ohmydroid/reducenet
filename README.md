# ReduceNet
####  ReduceNet can reduce network depth following [deep training startegy of VanillaNet](https://arxiv.org/abs/2305.12972). The difference is as follows:
1. ReduceNet is totally compatible with current model design
2. we introduce expansion factor to create bottleneck structure （bottleneck condensation） for basic block (conv3 and conv1), making it possible to utilize wider (unlimited width) network during training but will not increase extra computational overhead during inference.


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

1. Add code to fuse operators within basic block
2. Explore more elegant way to reduce network depth





