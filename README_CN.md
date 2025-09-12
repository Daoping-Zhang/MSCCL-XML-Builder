# MSCCL-XML-Builder

一个用于生成Microsoft集合通信库(MSCCL) XML文件的Python库，支持自定义集合通信算法的细粒度数据流控制。

## 概述

Microsoft集合通信库(MSCCL)是一个在多个加速器上执行自定义集合通信算法的平台。MSCCL-XML-Builder为生成MSCCL XML配置文件提供了比原始MSCCLang工具更灵活和细粒度的替代方案。

### 相比MSCCLang的关键优势

1. **细粒度API控制**：虽然MSCCLang只提供隐式转换为特定步骤操作(s, r, copy, rcs)的复制语义，但MSCCL-XML-Builder在提供基于块的抽象的同时，还提供对步骤操作的直接访问。

2. **灵活的依赖管理**：显式指定同一GPU上任意两个步骤之间的依赖关系，而不是仅依靠基于数据关系的自动依赖推理。

3. **高级线程块(TB)管理**：直接TB创建和绑定，实现TB和通信通道的灵活管理。

## 功能特性

- **基于块的操作**：保持MSCCLang直观的块抽象
- **直接步骤操作**：对send、recv、copy和recv-copy-send(rcs)操作的细粒度控制
- **显式依赖**：手动指定步骤依赖以实现精确的控制流
- **线程块管理**：灵活的TB创建和通道分配
- **RCS优化**：自动将recv+send操作合并为优化的rcs操作
- **XML生成**：生成与MSCCL运行时兼容的干净、格式化XML输出

## 安装
### GitHub直接安装​

```bash
pip3 install git+https://github.com/Daoping-Zhang/MSCCL-XML-Builder.git

```



### 本地开发安装

```bash
git clone https://github.com/Daoping-Zhang/MSCCL-XML-Builder.git
cd MSCCL-XML-Builder

# 以开发模式安装
pip install -e .


```

## 故障排除



跨GPU依赖错误：依赖只能在同一GPU上的步骤之间添加。

通道冲突错误：GPU上的每个通道只能有一个TB发送到特定rank，一个TB从特定rank接收。

步骤索引不匹配：确保send和recv步骤具有匹配的索引，否则会导致收发逻辑混乱。

安装时导入错误：尝试通过将项目路径添加到sys.path的替代安装方法，或使用python setup.py develop而不是pip install -e .

## Limitations

- 没有内置算法验证（将在未来版本中提供）

## License
本项目采用MIT许可证。


## Acknowledgments
- Microsoft Research的MSCCL框架
- 启发此工作的MSCCLang研究
- MSCCL生态系统的贡献者