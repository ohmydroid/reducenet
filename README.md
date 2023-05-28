# ReduceNet
####  ReduceNet can reduce network depth following [deep training startegi of VanillaNet](https://arxiv.org/abs/2305.12972) while adopting more radical strategy to shrink net. The difference is as follows:
1. ReduceNet is totally comppatible with current model design
2. There is no any non-linear activation in the network during inference
3. Residual connection is allowed to improve model peformance during training and be fused into conv operator during inference
4. we introduce expansion factor to make bottleneck structure for basic block (conv3 and conv1), making it possible to utilize wider (unlimited width) network during training
5. Theoretically, the whole network of ReduceNet can be fused into a single linear layer during inferencce while utilizing deep non-linear layers during training

#### Code is based on repo [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)

#### We use [torchsummaryX](https://github.com/nmhkahn/torchsummaryX) to count parameters and MAdds




# just run
```python
python main.py -m reduce20
```
```python
python main.py -m reduce56
```





# To do
1. combine  structural re-parameterization techniques such as RepVGG
2. fuse operators within basic block and turn the whole network into a single layer
3. explore more elegant way to reduce network depth
4. decrease the number of KxK conv operator and computations of overlapped area following the sprit of "all you need is weighted sum of patches"
5. introduce residual connection into down-sampling layer during training phase
