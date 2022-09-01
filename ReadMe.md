# Others

1. Corpus Preprocessing Example.ipynb : Corpus Preprocessing Example 
2. Plot Confusion Matrix Example.ipynb : Confusion Matrix Code example
2. cosine_annealing_with_warmup.ipynb : cosine_annealing_with_warmup Code example. I edited min values when `start point` `smaller than base min lr`. If anyone who find error, Please Tell me.


# Batch Normalizae in Pytorch
* pytorch에서는 layer를 freezing 해도 Batch normalization이 학습되는 문제가 있음
* https://yjs-program.tistory.com/212?category=804886 참조
```python
# 학습에서 Batch Normalization freeze 하는 방법
for module in model.modules():
    if isinstance(module, nn.BatchNorm2d):
        if hasattr(module, 'weight'):
            module.weight.requires_grad_(False)
        if hasattr(module, 'weight'):
            module.bias.requires_grad_(False)
        module.track_running_stats = False
        module.affine = False
```
