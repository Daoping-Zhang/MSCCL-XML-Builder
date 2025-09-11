from typing import List, Tuple
from .step import Step
from .tb import TB

class Chunk:
    def __init__(self, gpu_id: int, chunk_type: str, index: int, size: int, algo):
        self.gpu_id = gpu_id
        self.chunk_type = chunk_type  # "input", "output", "scratch"
        self.index = index
        self.size = size
        self.algo = algo
    
    def _get_buf_name(self) -> str:
        type_map = {"input": "i", "output": "o", "scratch": "s"}
        return type_map.get(self.chunk_type, self.chunk_type)
    
    def copy(self, dest_chunk: 'Chunk', channel_id: int, tb: 'TB' = None, dep_steps: List[Step] = None) -> Step:
        if dep_steps is None:
            dep_steps = []
            
        # Check same GPU and size
        if self.gpu_id != dest_chunk.gpu_id:
            raise ValueError("Copy operation requires chunks on the same GPU")
        if self.size != dest_chunk.size:
            raise ValueError("Copy operation requires chunks of the same size")
        
        gpu = self.algo.get_gpu(self.gpu_id)
        
        # Find or create TB with send=-1, recv=-1, channel=channel_id
        if tb is None:
            tb = gpu.find_tb(send=-1, recv=-1, chan=channel_id)
            
        if tb is None:
            tb = TB(send=-1, recv=-1, chan=channel_id)
            gpu.add_tb(tb)
        
        # Create copy step
        step = Step(type="cpy", srcbuf=self._get_buf_name(), srcoff=self.index,
                   dstbuf=dest_chunk._get_buf_name(), dstoff=dest_chunk.index,
                   cnt=self.size, depid=-1, deps=-1, hasdep=0)
        
        # 添加依赖到dep_list
        for dep_step in dep_steps:
            step.add_dep(dep_step)
        
        tb.add_step(step)
        return step
    
    def copy_diff(self, dest_chunk: 'Chunk', channel_id: int, dep_steps: List[Step] = None, bidirectional: bool = True) -> Tuple[Step, Step]:
        """跨rank copy操作，拆解为send+recv"""
        if dep_steps is None:
            dep_steps = []
            
        if self.gpu_id == dest_chunk.gpu_id:
            raise ValueError("copy_diff requires chunks on different GPUs")
        if self.size != dest_chunk.size:
            raise ValueError("copy_diff requires chunks of the same size")
        
        # 创建send step
        send_step = self.send(dest_chunk.gpu_id, channel_id, dep_steps, bidirectional=bidirectional)
        
        # 创建recv step  
        recv_step = dest_chunk.recv(self.gpu_id, channel_id, [], bidirectional=bidirectional)
        
        # 设置peer关系
        send_step.peer_step = recv_step
        recv_step.peer_step = send_step

        send_step.dstbuf = recv_step.dstbuf
        send_step.dstoff = recv_step.dstoff

        recv_step.srcbuf = send_step.srcbuf
        recv_step.srcoff = send_step.srcoff
        
        # 检查index一致性
        if send_step.send_index != recv_step.recv_index:
            raise ValueError(f"Index mismatch: send_index={send_step.send_index}, recv_index={recv_step.recv_index}")
        
        return send_step, recv_step
    
    def send(self, dest_rank: int, channel_id: int, dep_steps: List[Step] = None, 
             bidirectional: bool = True) -> Step:
        if dep_steps is None:
            dep_steps = []
            
        gpu = self.algo.get_gpu(self.gpu_id)
        
        # Find or create appropriate TB
        tb = gpu.find_tb(send=dest_rank, chan=channel_id)
        if tb is None:
            if bidirectional:
                # Check if recv exists for the same channel
                if gpu.find_tb(recv=dest_rank, chan=channel_id):
                    raise ValueError(f"Channel {channel_id} already has recv from rank {dest_rank}")
                tb = TB(send=dest_rank, recv=dest_rank, chan=channel_id)
            else:
                tb = TB(send=dest_rank, recv=-1, chan=channel_id)
            gpu.add_tb(tb)
        
        # Create send step
        step = Step(type="s", srcbuf=self._get_buf_name(), srcoff=self.index,
                   dstbuf="o", dstoff=-1, cnt=self.size, depid=-1, deps=-1, hasdep=0)
        
        # 添加依赖到dep_list
        for dep_step in dep_steps:
            step.add_dep(dep_step)
        
        tb.add_step(step)
        return step
    
    def recv(self, src_rank: int, channel_id: int, dep_steps: List[Step] = None, 
             bidirectional: bool = True) -> Step:
        if dep_steps is None:
            dep_steps = []
            
        gpu = self.algo.get_gpu(self.gpu_id)
        
        # Find or create appropriate TB
        tb = gpu.find_tb(recv=src_rank, chan=channel_id)
        if tb is None:
            if bidirectional:
                # Check if send exists for the same channel
                if gpu.find_tb(send=src_rank, chan=channel_id):
                    raise ValueError(f"Channel {channel_id} already has send to rank {src_rank}")
                tb = TB(send=src_rank, recv=src_rank, chan=channel_id)
            else:
                tb = TB(send=-1, recv=src_rank, chan=channel_id)
            gpu.add_tb(tb)
        
        # Create recv step
        step = Step(type="r", srcbuf="i", srcoff=-1,
                   dstbuf=self._get_buf_name(), dstoff=self.index,
                   cnt=self.size, depid=-1, deps=-1, hasdep=0)
        
        # 添加依赖到dep_list
        for dep_step in dep_steps:
            step.add_dep(dep_step)
        
        tb.add_step(step)
        return step
    
    def rcs(self, dest_chunk: 'Chunk', intermediate_rank: int, channel_id: int, 
            dep_steps: List[Step] = None) -> Step:
        if dep_steps is None:
            dep_steps = []
            
        # Check constraints
        ranks = {self.gpu_id, dest_chunk.gpu_id, intermediate_rank}
        if len(ranks) != 3:
            raise ValueError("RCS operation requires three different GPU ranks")
        if self.size != dest_chunk.size:
            raise ValueError("RCS operation requires chunks of the same size")
        
        gpu = self.algo.get_gpu(intermediate_rank)
        
        # Find or create TB
        tb = gpu.find_tb(send=dest_chunk.gpu_id, recv=self.gpu_id, chan=channel_id)
        if tb is None:
            # Check for conflicts
            if gpu.find_tb(send=dest_chunk.gpu_id, chan=channel_id):
                raise ValueError(f"Channel {channel_id} already has send to rank {dest_chunk.gpu_id}")
            if gpu.find_tb(recv=self.gpu_id, chan=channel_id):
                raise ValueError(f"Channel {channel_id} already has recv from rank {self.gpu_id}")
            
            tb = TB(send=dest_chunk.gpu_id, recv=self.gpu_id, chan=channel_id)
            gpu.add_tb(tb)
        
        # Create RCS step
        step = Step(type="rcs", srcbuf=self._get_buf_name(), srcoff=self.index,
                   dstbuf=dest_chunk._get_buf_name(), dstoff=dest_chunk.index,
                   cnt=self.size, depid=-1, deps=-1, hasdep=0)
        
        # 添加依赖到dep_list
        for dep_step in dep_steps:
            step.add_dep(dep_step)
        
        tb.add_step(step)
        return step