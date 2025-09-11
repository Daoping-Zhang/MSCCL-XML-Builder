import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple
from .step import Step

class TB:
    def __init__(self, id: Optional[int] = None, send: int = -1, recv: int = -1, chan: int = 0):
        self.id = id
        self.send = send
        self.recv = recv
        self.chan = chan
        self.steps: List[Step] = []
        
        # 索引维护
        self.send_index: int = 0
        self.recv_index: int = 0
    
    def get_next_step_id(self) -> int:
        return len(self.steps)
    
    def add_step(self, step: Step) -> None:
        if step.s is None:
            step.s = self.get_next_step_id()
        elif step.s != len(self.steps):
            raise ValueError(f"Step s={step.s} is not continuous. Expected {len(self.steps)}")
        
        # 设置step的TB引用
        step._tb = self
        step.original_index = len(self.steps)
        
        # 根据step类型设置index
        if step.type == "s":  # send
            step.send_index = self.send_index
            self.send_index += 1
        elif step.type == "r":  # recv
            step.recv_index = self.recv_index
            self.recv_index += 1
        elif step.type == "rcs":  # recv-copy-send
            step.recv_index = self.recv_index
            step.send_index = self.send_index
            self.recv_index += 1
            self.send_index += 1
        
        self.steps.append(step)
    
    def sort_steps_by_index(self) -> None:
        """根据send/recv index对steps进行排序"""
        def get_sort_key(step: Step) -> Tuple[int, int]:
            # 先按照index排序，send在前recv在后
            if step.type == "s":
                return (step.send_index * 2, 0)  # send: (index*2, 0)
            elif step.type == "r": 
                return (step.recv_index * 2 + 1, 1)  # recv: (index*2+1, 1)
            elif step.type == "rcs":
                return (step.recv_index * 2, 2)  # rcs按recv_index排序
            else:
                # nop和cpy保持原有位置
                return (step.original_index * 1000, 3)
        
        self.steps.sort(key=get_sort_key)
        
        # 重新设置s字段
        for i, step in enumerate(self.steps):
            step.s = i
    
    def insert_nop_step(self, position: int, dep_steps: List[Step]) -> Step:
        """在指定位置插入nop step(s)处理多个依赖"""
        if len(dep_steps) == 0:
            return None
        elif len(dep_steps) == 1:
            # 单个依赖，插入一个nop step
            nop_step = Step(type="nop", srcbuf="i", srcoff=-1, dstbuf="o", dstoff=-1, cnt=0)
            nop_step._tb = self
            nop_step.position_fixed = True
            nop_step.depid = dep_steps[0]._tb.id
            nop_step.deps = dep_steps[0].s
            
            self.steps.insert(position, nop_step)
            
            # 重新设置后续steps的s字段
            for i in range(position, len(self.steps)):
                self.steps[i].s = i
                
            return nop_step
        else:
            # 多个依赖，插入(len(dep_steps)-1)个nop steps
            # 最后一个依赖不插入nop，直接作为原始step的依赖
            num_nops_to_insert = len(dep_steps) - 1
            
            # 插入前(len(dep_steps)-1)个nop steps
            for i in range(num_nops_to_insert):
                dep_step = dep_steps[i]
                nop_step = Step(type="nop", srcbuf="i", srcoff=-1, dstbuf="o", dstoff=-1, cnt=0)
                nop_step._tb = self
                nop_step.position_fixed = True
                
                # 设置依赖关系
                nop_step.depid = dep_step._tb.id
                nop_step.deps = dep_step.s
                
                # 插入到正确位置
                insert_pos = position + i
                self.steps.insert(insert_pos, nop_step)
            
            # 重新设置所有steps的s字段
            for i in range(len(self.steps)):
                self.steps[i].s = i
            
            return None
        
    def to_xml(self) -> ET.Element:
        tb_elem = ET.Element("tb")
        tb_elem.set("id", str(self.id))
        tb_elem.set("send", str(self.send))
        tb_elem.set("recv", str(self.recv))
        tb_elem.set("chan", str(self.chan))
        
        for step in self.steps:
            tb_elem.append(step.to_xml())
        
        return tb_elem