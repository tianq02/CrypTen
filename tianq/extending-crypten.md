from tutorials.Tutorial_4_Classification_with_Encrypted_Neural_Networks import private_model

# 在crypten中加入自己的协议

虽然说crypten几乎没有关于扩展协议的文档，但没有人阻止你瞎折腾。我们计划用crypten的框架实现Cenia的方案，目前还是没头苍蝇。

## Crypten的代码结构：

MPCTensor -> binary.py,arithmetic.py 对应布尔秘密共享和算数秘密共享

> 注意：测试发现全连接计算实际使用的是GEMM而不是Linear，我不理解，

修改全连接协议：

`crypten.nn.module.py->class Linear` 特别简单的线性全连接计算，看起来是使用x的多态（Tensor/MPCTensor）性质实现的

到下一层：

## snippets

打印crypten模型结构

```python
import crypten.nn

private_model: crypten.nn.Module = crypten.nn.Module() # 自己的模型|从文件加载的模型

# Examine the structure of the encrypted CrypTen network
for name, curr_module in private_model._modules.items():
    print("\n===============================================================")
    print("Name:", name, "\tModule:", curr_module)
    for param_name, param in curr_module.named_parameters():
        print("\nParam name:", param_name)
        print(param)
    for var_name, var_value in vars(curr_module).items():
        if var_name[0] == "_":  # Only print non-private variables
            continue
        print("Variable name:", var_name, "\tvalue:", var_value)
```

Module的调试输出  
作用：在模块初始化时打印参数和变量，帮助检查模型结构

```python
# None:no debug, "verbose": print module init params, else: print module init only
debug: Optional[str] = "verbose"
```

edit: 叔叔懒了，现在只有True/False

## 关于算数秘密共享

CrypTen中默认的实现中场景为：A持有秘密值，A想在ABC三方之间建立秘密共享，最终ABC各自持有一个份额

具体而言，ABC三方执行PRZS协议，ABC各自持有0的份额，A再往0份额加上秘密值，作为自己的秘密共享份额，BC直接使用0份额

注意看arithmetic.py中的`PRZS`方法，这里将各个参与方排列成一个环。每一方自己生成随机数并传给下一方，持有份额=自己的随机数-收到的随机数。

如此一来，各方份额必然相加为0，满足PRZS要求。且3方以上时，各参与方互相不知道本地份额。

## 关于`torch.Tensor`的精度问题

把秘密份额转换成torch的张量时，好像会由于数据结构的原因爆精度，导致份额恢复不出来

```python
print("Sum & Scale:", sum(s for r, s in shares) / 65536)
print("Sum2", torch.Tensor([s for r, s in shares]).sum(dim=0) / 65536)
```

结果：

```plaintext
Rank: 0, Share: [ 3336106720203657081 -2393457681650778829   626208987372982748]
Rank: 1, Share: [-3839082320315716272 -1587141446561267151  -131080352916936437]
Rank: 2, Share: [ 502975600112124727 3980599128212177052 -495128634455849703]
Sum & Scale: [1. 2. 3.]
Sum2 tensor([1.7180e+11, 0.0000e+00, 0.0000e+00])
```

解决方法1：像上面片段的原始实现一样，直接用numpy的sum  
解决方法2：\(没病别用\)
```python
print("Sum2", torch.from_numpy(numpy.array([s for r, s in shares])).sum(dim=0) / 65536)
```

## Cenia的方案到底怎么做？

注意到了吗？Cenia方案虽然是4PC，但C和P实际上只在初始化/结束时露个面，其余时间都是S1S2在秘密份额上推理，秘密份额在两方之间共享

**Cenia其实是2PC外包！！！**