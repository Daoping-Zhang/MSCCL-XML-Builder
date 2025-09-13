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
```

添加Step到TB，自动设置索引和TB引用。

``` python

def sort_steps_by_index(self) -> None
```
按send/recv index对steps排序，避免死锁。
``` python

def insert_nop_step(self, position: int, dep_steps: List[Step]) -> Step
```
在指定位置插入nop step处理多重依赖。


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
**依赖管理**
``` python

def add_dep(self, dep_step: Step) -> None
```
添加依赖关系，自动进行合法性检查。
``` python

def remove_dep(self, dep_step: Step) -> None
```

移除依赖关系，自动维护被依赖列表。

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
  - True: 创建send=recv=dest_rank的TB (双向通信)
  - False: 创建send=dest_rank, recv=-1的TB (单向发送)

**recv - 接收操作**
``` python
def recv(self, src_rank: int, channel_id: int, dep_steps: List[Step] = None, 
         bidirectional: bool = True) -> Step
```
与send操作对称，处理数据接收。

### 3.4 复合操作
**rcs - Reduce-Copy-Send**
``` python
def rcs(self, dest_chunk: Chunk, intermediate_rank: int, channel_id: int, 
        dep_steps: List[Step] = None) -> Step
```
功能: 在指定GPU上执行copy-send操作


## 4. 系统级API
### 4.1 依赖构建
``` python
algo.build_all_dependencies(merge_rcs=False, sort=True)
```
执行流程:
- 排序阶段 (如果sort=True): 对所有TB的Step按index排序
- 构建阶段: 多轮遍历处理所有依赖关系

merge_rcs优化:
- 自动检测相邻的recv+send操作
- 满足条件时合并为单个rcs操作
- 减少通信步骤，提高效率
### 4.2 XML导出
``` python
algo.save_xml("output.xml")
```
生成标准格式的XML配置文件。

## 5. 使用模式和最佳实践
### 5.1 推荐工作流程
``` python
# 使用示例
from msccl_xml_builder import Algo, Chunk,TB

if __name__ == "__main__":
    # 创建算法

    ngpus=2
    chunk=ngpus
    algo = Algo(name="ag_test", proto = "Simple", nchannels = 1, 
                 nchunksperloop = chunk, ngpus = ngpus, coll = "allgather",
                 inplace = 1, outofplace = 1, minBytes = 0, maxBytes = 0)
    
    #显示的创建本地copytb
    local_tbs = {}      # {rank: tb} - 本地操作TB
    for rank in range(ngpus):

        local_tb = TB(send=-1, recv=-1, chan=0)
        gpu = algo.get_gpu(rank)

        gpu.add_tb(local_tb)
        local_tbs[rank] = local_tb
        
    # 本地copy
    copy_steps = []
    for rank in range(ngpus):
        src_chunk = Chunk(gpu_id=rank, chunk_type="i", index=0, size=1, algo=algo)
        dest_chunk = Chunk(gpu_id=rank, chunk_type="o", index=rank, size=1, algo=algo)
        copy_step = src_chunk.copy(dest_chunk, channel_id=0)
        copy_steps.append(copy_step)

    #创建P2P tb
    rank = 0
    
    peer_rank=1

    gpu = algo.get_gpu(rank)
    
    peer_gpu = algo.get_gpu(peer_rank)

    tb = TB(send=peer_rank, recv=peer_rank, chan=0)
    
    peer_tb = TB(send=rank, recv=rank, chan=0)

    gpu.add_tb(tb)

    peer_gpu.add_tb(peer_tb)

    #数据发送，并设置依赖

    src_chunk = Chunk(gpu_id=rank, chunk_type="o", index=rank, size=1, algo=algo)
    dest_chunk = Chunk(gpu_id=peer_rank, chunk_type="o", index=peer_rank, size=1, algo=algo)

    send_step = src_chunk.send(dest_rank=peer_rank, channel_id= 0, dep_steps=[copy_steps[rank]])
    recv_step = dest_chunk.recv(src_rank=rank, channel_id= 0)

    #需要绑定send recv
    send_step.peer_step = recv_step
    recv_step.peer_step = send_step

    

    recv_step.add_dep(copy_steps[peer_rank]) #支持后续显示的添加依赖

    #可以通过copy_diff 实现send recv

    src_chunk = Chunk(gpu_id=peer_rank, chunk_type="o", index=peer_rank, size=1, algo=algo)
    dest_chunk = Chunk(gpu_id=rank, chunk_type="o", index=rank, size=1, algo=algo)

    peer_send_step, peer_recv_step = src_chunk.copy_diff(dest_chunk, channel_id=0, dep_steps=[copy_steps[peer_rank]])

    algo.build_all_dependencies() #需要显示的调用依赖构建
    
    # 保存XML
    algo.save_xml("output.xml")
    print("XML文件已生成: output.xml")
``` 


