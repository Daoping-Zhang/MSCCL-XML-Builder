import xml.etree.ElementTree as ET
from typing import List, Optional

class Step:
    def __init__(self, s: Optional[int] = None, type: str = "nop", srcbuf: str = "i", 
                 srcoff: int = -1, dstbuf: str = "o", dstoff: int = -1, cnt: int = 0,
                 depid: int = -1, deps: int = -1, hasdep: int = 0):
        self.s = s
        self.type = type
        self.srcbuf = srcbuf
        self.srcoff = srcoff
        self.dstbuf = dstbuf
        self.dstoff = dstoff
        self.cnt = cnt
        self.depid = depid
        self.deps = deps
        self.hasdep = hasdep
        
        # 索引和peer关系
        self.send_index: Optional[int] = None
        self.recv_index: Optional[int] = None
        self.peer_step: Optional['Step'] = None
        self.send_peer: Optional['Step'] = None  # for rcs
        self.recv_peer: Optional['Step'] = None  # for rcs
        
        # 新增依赖管理字段
        self.dep_list: List['Step'] = []  # 依赖的step列表
        self.depended_by_list: List['Step'] = []  # 被依赖的step列表（新添加）

        self.position_fixed: bool = False  # 位置是否固定
        self.original_index: Optional[int] = None  # 保存原始插入顺序
    
    def add_dep(self, dep_step: 'Step') -> None:
        """添加依赖关系"""
        # 检查是否为合法依赖
        if self._get_gpu_id() != dep_step._get_gpu_id():
            raise ValueError("Cross-GPU dependency is not allowed")
        
        if self._get_tb() == dep_step._get_tb():
            raise ValueError("Dependencies within the same TB should be managed by controlling step order manually")
        
        if dep_step not in self.dep_list:
            self.dep_list.append(dep_step)
            dep_step.hasdep = 1
            # 同时维护被依赖关系
            if self not in dep_step.depended_by_list:
                dep_step.depended_by_list.append(self)

    def remove_dep(self, dep_step: 'Step') -> None:
        """移除依赖关系"""
        if dep_step in self.dep_list:
            self.dep_list.remove(dep_step)
            # 同时移除被依赖关系
            if self in dep_step.depended_by_list:
                dep_step.depended_by_list.remove(self)
            
            # 如果dep_step不再被任何step依赖，设置hasdep=0
            if len(dep_step.depended_by_list) == 0:
                dep_step.hasdep = 0
    
    def _get_gpu_id(self) -> int:
        """获取step所属的GPU ID"""
        return getattr(self, '_gpu_id', -1)
    
    def _get_tb(self):
        """获取step所属的TB"""
        return getattr(self, '_tb', None)
    
    def to_xml(self) -> ET.Element:
        step_elem = ET.Element("step")
        step_elem.set("s", str(self.s))
        step_elem.set("type", self.type)
        step_elem.set("srcbuf", self.srcbuf)
        step_elem.set("srcoff", str(self.srcoff))
        step_elem.set("dstbuf", self.dstbuf)
        step_elem.set("dstoff", str(self.dstoff))
        step_elem.set("cnt", str(self.cnt))
        step_elem.set("depid", str(self.depid))
        step_elem.set("deps", str(self.deps))
        step_elem.set("hasdep", str(self.hasdep))
        return step_elem