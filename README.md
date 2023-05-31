# ReduceNet
####  ReduceNet can reduce network depth following [deep training startegy of VanillaNet](https://arxiv.org/abs/2305.12972) while adopting more radical strategy to shrink net. The difference is as follows:
1. ReduceNet is totally compatible with current model design
#removed Residual connection is allowed to improve model peformance during training and be regarded as a special convolution kernel and fused into conv operator during inference
2. we introduce expansion factor to create bottleneck structure （bottleneck condensation） for basic block (conv3 and conv1), making it possible to utilize wider (unlimited width) network during training but will not increase extra computational overhead during inference. Essentially, bottleneck condensation is a superset of RepVGG with only a single branch 



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

1. Add code to fuse operators within basic block and turn the whole network into a single layer by fuse all blocks (fused conv)
2. Explore more elegant way to reduce network depth, I am not sure...
3. Decrease the number of KxK conv operator and computations of overlapped area following the sprit of "weighted sum of patches is all you need"

# Cite

```latex
@misc{onedroid,
  author = {onedroid, P.W.D.},
  title = {ReduceNet},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},


```

