# 集合通信算法框架 - 代码架构文档

## 1. 架构概述

本框架是一个用于生成集合通信算法XML配置的Python框架，采用层次化面向对象设计。框架通过类的继承关系管理复杂的集合通信算法，并能将配置导出为标准XML格式。

## 2. 核心类层次结构

### 2.1 类继承关系
Algo (算法类)
└── GPU (GPU类)
    └── TB (Thread Block类)
        └── Step (步骤类)

### 2.2 辅助类
- **Chunk类**: 数据块抽象，提供高级API操作

## 3. 各层类的详细设计

### 3.1 Step类 (操作步骤)
**职责**: 表示单个原子操作，是框架的最小执行单元

**核心字段**:
- `s`: Step在TB中的索引位置
- `type`: 操作类型 ("cpy"/"s"/"r"/"rcs"/"nop")
- 缓冲区字段: `srcbuf`, `dstbuf`, `srcoff`, `dstoff`, `cnt`
- 依赖字段: `depid`, `deps`, `hasdep`

**新增字段**:
- `send_index/recv_index`: 发送/接收索引
- `peer_step`: 对端Step引用 (用于跨GPU通信)
- `send_peer/recv_peer`: RCS操作的双重peer关系
- `dep_list`: 依赖Step列表 (新的依赖管理)
- `depended_by_list`: 被依赖Step列表
- `position_fixed`: 位置是否已固定
- `original_index`: 保存原始插入顺序

**关键方法**:
- `add_dep(dep_step)`: 添加依赖关系
- `remove_dep(dep_step)`: 移除依赖关系
- `to_xml()`: 导出为XML元素

### 3.2 TB类 (Thread Block执行单元)
**职责**: 管理一组相关的Step操作，通常对应一个通信通道

**核心字段**:
- `id`: TB标识
- `send/recv`: 发送/接收目标rank (-1表示无操作)
- `chan`: 绑定的通道ID
- `steps`: Step队列
- `send_index/recv_index`: 索引计数器

**关键方法**:
- `add_step(step)`: 添加Step并维护连续性
- `sort_steps_by_index()`: 按index排序Step (防止死锁)
- `insert_nop_step(position, dep_steps)`: 插入nop Step处理多依赖

**排序规则（仅针对P2P 全双工TB）**:
- 发送操作: (send_index * 2, 0)
- 接收操作: (recv_index * 2 + 1, 1)
- RCS操作: (recv_index * 2, 2)

### 3.3 GPU类 (GPU资源管理)
**职责**: 管理单个GPU上的所有TB和通道资源

**核心字段**:
- `id`: GPU的rank标识
- `tbs`: TB队列
- `channel_usage`: 通道使用情况追踪

**通道冲突检查**:
```python
channel_usage[channel_id] = {
    'send': [target_ranks],  # 发送目标列表
    'recv': [source_ranks]   # 接收源列表
}
```

**关键方法**:

- add_tb(tb): 添加TB并检查通道冲突
- find_tb(send, recv, chan): 查找匹配条件的TB
- build_dependencies(merge_rcs): 构建依赖关系
- _can_merge_rcs(): 检查是否可以合并recv+send为rcs
- _merge_recv_send_to_rcs(): 执行RCS合并

### 3.4 Algo类 (顶层算法管理)

**职责**: 管理整个算法的全局配置和GPU资源

**核心字段**:

- 算法元数据: name, proto, coll
- 配置参数: nchannels, nchunksperloop, ngpus
- 数据范围: minBytes, maxBytes
- 操作模式: inplace, outofplace
**关键方法**:

- build_all_dependencies(merge_rcs, sort): 构建所有依赖关系
- save_xml(filename): 导出为XML文件

## 4. Chunk数据抽象层
### 4.1 设计理念
Chunk类提供高级数据操作抽象，隐藏底层TB和Step的复杂性。

### 4.2 核心属性

- gpu_id: int          # 所属GPU
- chunk_type: str      # 类型 ("input"/"output"/"scratch")  
- index: int           # 起始索引
- size: int            # 数据块大小
- algo: Algo           # 所属算法实例
### 4.3 操作映射
- copy() → 创建"cpy"类型Step
- send() → 创建"s"类型Step
- recv() → 创建"r"类型Step
- rcs() → 创建"rcs"类型Step
- copy_diff() → 创建send+recv Step对
## 5. 依赖管理系统
### 5.1 两阶段处理机制

依赖收集阶段: 调用API时将依赖保存到dep_list
依赖构建阶段: 统一调用build_all_dependencies()处理

### 5.2 依赖合法性检查

``` python
# 合法依赖：同GPU不同TB
step_a._gpu_id == step_b._gpu_id  # ✓
step_a._tb != step_b._tb           # ✓

# 非法依赖：跨GPU
step_a._gpu_id != step_b._gpu_id  # ✗

# 需手动管理：同TB内
step_a._tb == step_b._tb          # 需要通过添加顺序控制
```

### 5.3 死锁避免策略
- TB内Step按index排序确保send在recv之前
- 多轮遍历确保所有依赖都能解析
- RCS合并优化减少通信步骤

## 6. RCS合并机制
### 6.1 合并条件

```python
def _can_merge_rcs(tb, send_step_idx):
    recv_step = tb.steps[send_step_idx - 1]
    send_step = tb.steps[send_step_idx]
    
    # 基本条件
    if not (send_step.type == "s" and recv_step.type == "r"):
        return False
    
    # 数据一致性
    if not (send_step.cnt == recv_step.cnt and 
            send_step.srcbuf == recv_step.dstbuf and 
            send_step.srcoff == recv_step.dstoff):
        return False
    
    # 依赖条件 (三种情况之一)
    return (len(send_step.dep_list) == 0 or
            (len(send_step.dep_list) == 1 and send_step.dep_list[0] == recv_step) or
            (send_step.depid == tb.id and send_step.deps == recv_step.s))
```

### 6.2 合并收益
- 减少通信步骤数量
- 优化执行效率
- 保持语义等价性
## 7. XML导出系统
### 7.1 层次化导出
每个类都实现to_xml()方法，构建对应的XML元素：

- Step → <step> 元素
- TB → <tb> 元素包含多个<step>
- GPU → <gpu> 元素包含多个<tb>
- Algo → <algo> 元素包含多个<gpu>

### 7.2 自动计算字段

``` python
# GPU元素自动计算buffer深度
gpu_elem.set("i_chunks", str(max_i_chunks))
gpu_elem.set("o_chunks", str(max_o_chunks))  
gpu_elem.set("s_chunks", str(max_s_chunks))
```

## 8. 架构优势
### 8.1 分层抽象
- 细粒度控制 (Step/TB/GPU级别)
- 粗粒度易用性 (Chunk级别)
- 清晰的职责分离
### 8.2 灵活性设计
- 支持动态TB创建
- 自动ID分配机制
- 可选的RCS优化
## 8.3 一致性保证
- 严格的通道冲突检查
- peer_step机制确保通信匹配
- 位置固定机制避免依赖混乱