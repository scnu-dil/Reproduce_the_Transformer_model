通过阅读论文*Attention is all you need*来复现**Transformer**模型

# 已完成
- [x] 输入数据处理部分
- [x] transformer模型的训练部分
- [x] transformer模型的验证部分
- [x] transformer模型的推理部分
- [x] 输出数据生成部分

# 待完成
- [ ] 将当前代码拆分为各个模块
- [ ] 添加对模型训练部分，测试部分困惑度PPL和准确率ACC的图
- [ ] 优化模型代码，添加更多注释
- [ ] 构造输入参数约束函数
- [ ] 添加命令行参数模式
- [x] 添加参考论文和代码的链接
- [ ] 训练模型结束保存模型

# 使用方法
+ 当前就直接运行'all.py'文件即可；
+ 'CUDA_VISIBLE_DEVICES=0 python all.py'，指定GPU显卡来运行模型，'all.py'文件包括模型训练，验证和推理三个功能;
+ 可以在'parameters.py'文件中修改模型的全部参数。

# 注意
+ 该复现的Tranformer模型主要是参考论文 [*Attention is all you need*](https://arxiv.org/abs/1706.03762)
+ 代码主要参考了该博主完整的transformer代码 [*jadore801120*](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
+ 代码参考了该博主的transformer结构代码 [*luozhouyang*](https://luozhouyang.github.io/transformer/)
