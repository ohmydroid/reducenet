# ReduceNet
####  ReduceNet can reduce network depth following [deep training startegi of VanillaNet](https://arxiv.org/abs/2305.12972) while adopting more radical strategy to shrink net. Theoretically, ReduceNet can be fused into a single layer for the whole network during inferencce while utilize deep non-linear layers during training.

#### Code is based on repo [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)

#### We use [torchsummaryX](https://github.com/nmhkahn/torchsummaryX) to count parameters and MAdds




# just run
python main.py -m reduce20
# python main.py -m reduce56






# To do
1. combine  structural re-parameterization techniques such as RepVGG
2. fuse operators within basic block and turn the whole network into a single layer
3. explore more elegant way to reduce network depth
4. decrease the number of KxK conv operator and computations of overlapped area following the sprit of "all you need is weighted sum of patches"
5. introduce residual connection into down-sampling layer during training phase
