## 1. API层次结构

框架提供两套完整的API体系：
- **细粒度API**: 直接操作Step/TB/GPU/Algo类
- **粗粒度API**: 基于Chunk类的高级操作

## 2. 核心类API

### 2.1 Algo类

#### 构造函数
```python
Algo(name: str, proto: str = "Simple", nchannels: int = 1, 
     nchunksperloop: int = 2, ngpus: int = 2, coll: str = "allgather",
     inplace: int = 1, outofplace: int = 1, minBytes: int = 0, maxBytes: int = 0)
```
**参数说明**:

- name: 算法名称
- proto: 协议类型，默认"Simple"
- nchannels: 通道数量
- nchunksperloop: 每轮处理的chunk数
- ngpus: GPU数量
- coll: 集合通信类型 ("allgather", "allreduce"等)
- inplace/outofplace: 是否支持原地/异地操作 (0/1)
- minBytes/maxBytes: 数据大小范围
**核心方法**

``` python
def get_gpu(self, gpu_id: int) -> GPU
```

获取指定ID的GPU实例。

``` python
def build_all_dependencies(self, merge_rcs: bool = False, sort: bool = True) -> None
```

构建所有GPU的依赖关系。

- merge_rcs: 是否启用RCS合并优化
- sort: 是否先对P2P Step进行排序
``` python
def save_xml(self, filename: str) -> None
```
将算法配置导出为XML文件。

### 2.2 GPU类
**基本方法**
``` python
def get_next_tb_id(self) -> int
```

返回下一个可用的TB ID。

``` python
def add_tb(self, tb: TB) -> None
```
添加TB到GPU，自动进行连续性和通道冲突检查。

```python
def find_tb(self, send: int = None, recv: int = None, chan: int = None) -> Optional[TB]
```
查找匹配条件的TB实例。

**高级方法**
``` python
def check_channel_conflict(self, tb: TB) -> None
```

检查TB的通道使用是否与现有TB冲突。

**冲突规则**:

- 同一channel不能有多个TB发送到相同rank
- 同一channel不能有多个TB从相同rank接收

### 2.3 TB类

构造函数
``` python
TB(id: Optional[int] = None, send: int = -1, recv: int = -1, chan: int = 0)
```

**参数说明**:

- id: TB标识，None时自动分配
- send/recv: 发送/接收目标rank，-1表示无操作
- chan: 绑定的通道ID
**核心方法**
``` python
def add_step(self, step: Step) -> None
#添加Step到TB，自动设置索引和TB引用。

def sort_steps_by_index(self) -> None
#按send/recv index对steps排序，避免死锁。

def insert_nop_step(self, position: int, dep_steps: List[Step]) -> Step
#在指定位置插入nop step处理多重依赖。
```

### 2.4 Step类
**构造函数**
``` python
Step(s: Optional[int] = None, type: str = "nop", srcbuf: str = "i", 
     srcoff: int = -1, dstbuf: str = "o", dstoff: int = -1, cnt: int = 0,
     depid: int = -1, deps: int = -1, hasdep: int = 0)
```

操作类型:

- "cpy": 同GPU内拷贝
- "s": 发送操作
- "r": 接收操作
- "rcs": Reduce-Copy-Send复合操作
- "nop": 无操作 (仅用于依赖管理)
**核心方法**
``` python
依赖管理
def add_dep(self, dep_step: Step) -> None
#添加依赖关系，自动进行合法性检查。

def remove_dep(self, dep_step: Step) -> None
#移除依赖关系，自动维护被依赖列表。
```

## 3. Chunk高级API
### 3.1 构造函数
``` python
Chunk(gpu_id: int, chunk_type: str, index: int, size: int, algo: Algo)
```

**chunk_type映射**:

- "input" → "i" (输入缓冲区)
- "output" → "o" (输出缓冲区)
- "scratch" → "s" (临时缓冲区)

### 3.2 同GPU操作
**copy - 同GPU内拷贝**

``` python

def copy(self, dest_chunk: Chunk, channel_id: int, tb: TB = None, 
         dep_steps: List[Step] = None) -> Step
```

功能: 在同一GPU内执行数据拷贝
检查项: 同GPU、大小一致
返回: 创建的copy类型Step

处理逻辑:

查找或创建TB (send=-1, recv=-1, chan=channel_id)
创建"cpy"类型Step
添加依赖关系到dep_list

### 3.3 跨GPU操作
**copy_diff - 跨GPU拷贝**
``` python
def copy_diff(self, dest_chunk: Chunk, channel_id: int, 
              dep_steps: List[Step] = None) -> Tuple[Step, Step]
``` 
功能: 自动拆解为send+recv操作对
返回: (send_step, recv_step)

自动处理:

- 设置peer_step关系
- 检查send_index和recv_index一致性
- 同步srcbuf/dstbuf信息

**send - 发送操作**
``` python
def send(self, dest_rank: int, channel_id: int, dep_steps: List[Step] = None, 
         bidirectional: bool = True) -> Step
```

参数:

- dest_rank: 目标GPU rank
- bidirectional: 控制TB创建模式
- bidirectional影响:

- - True: 创建send=recv=dest_rank的TB (双向通信)
- - False: 创建send=dest_rank, recv=-1的TB (单向发送)

**recv - 接收操作**
``` python
def recv(self, src_rank: int, channel_id: int, dep_steps: List[Step] = None, 
         bidirectional: bool = True) -> Step
```
与send操作对称，处理数据接收。

### 3.4 复合操作
```python
rcs - Reduce-Copy-Send
def rcs(self, dest_chunk: Chunk, intermediate_rank: int, channel_id: int, 
        dep_steps: List[Step] = None) -> Step
功能: 在指定GPU上执行reduce-copy-send操作
约束: 三个rank (src, dest, intermediate) 必须互不相同

处理逻辑:

检查rank唯一性和chunk大小一致性
在intermediate_rank上查找或创建TB
创建"rcs"类型Step
```
4. 系统级API
4.1 依赖构建
algo.build_all_dependencies(merge_rcs=False, sort=True)
执行流程:

排序阶段 (如果sort=True): 对所有TB的Step按index排序
构建阶段: 多轮遍历处理所有依赖关系
merge_rcs优化:

自动检测相邻的recv+send操作
满足条件时合并为单个rcs操作
减少通信步骤，提高效率
4.2 XML导出
algo.save_xml("output.xml")
生成标准格式的XML配置文件。

5. 使用模式和最佳实践
5.1 推荐工作流程
# 1. 创建算法实例
algo = Algo(name="my_algorithm", ngpus=4, nchannels=2)

# 2. 定义数据块
chunks = [Chunk(gpu_id=i, chunk_type="output", index=0, size=1, algo=algo) 
          for i in range(4)]

# 3. 定义通信操作
steps = []
for i in range(4):
    next_rank = (i + 1) % 4
    step = chunks[i].send(next_rank, channel_id=0)
    steps.append(step)

# 4. 添加依赖关系 (可选)
# step2.add_dep(step1)

# 5. 构建依赖并导出
algo.build_all_dependencies(merge_rcs=True)
algo.save_xml("output.xml")
5.2 错误预防技巧
避免通道冲突
# ✓ 正确：不同操作使用不同通道
chunk_a.send(1, channel_id=0)
chunk_b.send(2, channel_id=1)

# ✗ 错误：同通道多重绑定
chunk_a.send(1, channel_id=0)
chunk_c.send(1, channel_id=0)  # 冲突！
合理管理依赖
# ✓ 正确：同GPU不同TB的依赖
step_gpu0_tb0 = chunk_a.copy(chunk_b, channel_id=0)
step_gpu0_tb1 = chunk_c.send(1, channel_id=1, dep_steps=[step_gpu0_tb0])

# ✗ 错误：跨GPU依赖
step_gpu0 = chunk_gpu0.send(1, channel_id=0)
step_gpu1 = chunk_gpu1.send(2, channel_id=1, dep_steps=[step_gpu0])  # 非法！
5.3 调试技巧
# 检查通道使用情况
for gpu in algo.gpus:
    print(f"GPU {gpu.id} channels: {gpu.channel_usage}")

# 检查依赖关系
for tb in gpu.tbs:
    for step in tb.steps:
        if step.dep_list:
            print(f"Step {step.s} depends on: {[dep.s for dep in step.dep_list]}")
6. 高级特性
6.1 自动TB管理
框架自动处理TB的创建和ID分配，用户通常不需要手动创建TB实例。

6.2 RCS优化
# 启用RCS合并可以自动优化这种模式：
recv_step = chunk_a.recv(src_rank=0, channel_id=0)
send_step = chunk_a.send(dest_rank=2, channel_id=0, dep_steps=[recv_step])

# 构建时自动合并为：
# rcs_step (type="rcs")
6.3 灵活的依赖管理
# 可以在任何时候添加依赖
step1 = chunk_a.send(1, channel_id=0)
step2 = chunk_b.send(2, channel_id=1) 
step3 = chunk_c.copy(chunk_d, channel_id=2)

# 后续添加依赖
step3.add_dep(step1)
step3.add_dep(step2)
