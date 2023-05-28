# ReduceNet
####  ReduceNet can reduce network depth following the deep training startegi of VanillaNet(https://arxiv.org/abs/2305.12972).




# just run
python main.py -m reduce20






To do
1. combing reparameterized techniques such as RepVGG
2. fuse operators within basic block and turn the whole network into a single layer
3. explore more elegant way to reduce network depth
4. decrease the number of KxK conv operator and computations of overlapped area following the sprit of "all you need is weighted sum of patches".
5. introduce residual connection into down-sampling layer during training phase
