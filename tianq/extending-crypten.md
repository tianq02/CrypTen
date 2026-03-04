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

