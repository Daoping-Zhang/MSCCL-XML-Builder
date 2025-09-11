import xml.etree.ElementTree as ET
from typing import List, Optional, Dict
from .tb import TB
from .step import Step

class GPU:
    def __init__(self, id: int):
        self.id = id
        self.tbs: List[TB] = []
        # 修改数据结构：channel_usage[channel_id] = {'send': [target_ranks], 'recv': [source_ranks]}
        self.channel_usage: Dict[int, Dict[str, List[int]]] = {}
    
    def get_next_tb_id(self) -> int:
        return len(self.tbs)
    
    def check_channel_conflict(self, tb: TB) -> None:
        """
        检查TB的channel使用是否冲突
        规则：在同一个channel中，不能有两个TB具有相同的send目标或recv源
        """
        chan = tb.chan
        if chan in self.channel_usage:
            existing = self.channel_usage[chan]
            
            # 检查send冲突：如果当前TB要send到某个rank，检查是否已有其他TB也send到同一个rank
            if tb.send != -1 and 'send' in existing and tb.send in existing['send']:
                raise ValueError(f"GPU {self.id} Channel {chan} already has a TB sending to rank {tb.send}")
            
            # 检查recv冲突：如果当前TB要从某个rank recv，检查是否已有其他TB也从同一个rank recv
            if tb.recv != -1 and 'recv' in existing and tb.recv in existing['recv']:
                raise ValueError(f"GPU {self.id} Channel {chan} already has a TB receiving from rank {tb.recv}")
    
    def add_tb(self, tb: TB) -> None:
        if tb.id is None:
            tb.id = self.get_next_tb_id()
        elif tb.id != len(self.tbs):
            raise ValueError(f"TB id={tb.id} is not continuous. Expected {len(self.tbs)}")
        
        self.check_channel_conflict(tb)
        
        # Update channel usage
        chan = tb.chan
        if chan not in self.channel_usage:
            self.channel_usage[chan] = {'send': [], 'recv': []}
        
        if tb.send != -1:
            self.channel_usage[chan]['send'].append(tb.send)
        if tb.recv != -1:
            self.channel_usage[chan]['recv'].append(tb.recv)
            
        self.tbs.append(tb)
    
    def find_tb(self, send: int = None, recv: int = None, chan: int = None) -> Optional[TB]:
        """
        查找匹配条件的TB
        """
        for tb in self.tbs:
            if ((send is None or tb.send == send) and 
                (recv is None or tb.recv == recv) and
                (chan is None or tb.chan == chan)):
                return tb
        return None
    
    def sort_all_tb_steps(self) -> None:
        """对所有P2PTB中的steps进行排序"""
        for tb in self.tbs:
            if tb.send == tb.recv:
                tb.sort_steps_by_index()
    
    def build_dependencies(self, merge_rcs: bool = False) -> None:
        """构建该GPU下所有step的依赖关系"""
        # 设置所有step的GPU ID引用
        for tb in self.tbs:
            for step in tb.steps:
                step._gpu_id = self.id
        
        max_iterations = 100 # 防止无限循环
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            all_fixed = True
            
            for tb in self.tbs:
                i = 0  # 使用while循环，因为合并操作可能改变steps数量
                while i < len(tb.steps):
                    step = tb.steps[i]
                    
                    if step.position_fixed:
                        i += 1
                        continue
                    
                    # 检查依赖
                    if not step.dep_list:
                        # 无依赖，可以固定位置
                        can_fix = True
                    else:
                        # 检查所有依赖是否已固定位置
                        can_fix = all(dep.position_fixed for dep in step.dep_list)
                    
                    if can_fix:
                        # 构建依赖关系
                        if step.dep_list:
                            if len(step.dep_list) == 1:
                                # 单个依赖
                                dep_step = step.dep_list[0]
                                step.depid = dep_step._tb.id
                                step.deps = dep_step.s
                            else:
                                # 多个依赖，需要插入nop step
                                tb.insert_nop_step(i, step.dep_list)
                                # 当前step依赖最后一个原始依赖
                                dep_step = step.dep_list[-1]
                                step.depid = dep_step._tb.id
                                step.deps = dep_step.s
                        
                        # 在固定位置之前，检查是否可以执行rcs合并
                        merged = False
                        if merge_rcs and self._can_merge_rcs(tb, i):
                            merged = self._merge_recv_send_to_rcs(tb, i)
                            if merged:
                                # 合并后不增加i，因为当前位置的step已经变化
                                continue
                        
                        # 如果没有合并或合并失败，正常固定位置
                        if not merged:
                            step.position_fixed = True
                            i += 1
                    else:
                        all_fixed = False
                        i += 1
            
            if all_fixed:
                break
        
        if iteration >= max_iterations:
            raise RuntimeError("Failed to resolve all dependencies within maximum iterations")

    def _can_merge_rcs(self, tb: TB, send_step_idx: int) -> bool:
        """
        检查是否可以将recv+send合并为rcs
        """
        if tb.send == tb.recv:
            return False
        
        if send_step_idx <= 0 or send_step_idx >= len(tb.steps):
            return False
        
        send_step = tb.steps[send_step_idx]
        recv_step = tb.steps[send_step_idx - 1]
        
        # 检查基本条件：当前是send，上一个是recv
        if not (send_step.type == "s" and recv_step.type == "r" and 
                send_step.cnt == recv_step.cnt and 
                send_step.srcbuf == recv_step.dstbuf and 
                send_step.srcoff == recv_step.dstoff):
            return False
        
        # 检查依赖条件
        if len(send_step.dep_list) == 0:
            return True
        elif (len(send_step.dep_list) == 1 and 
              send_step.dep_list[0] == recv_step):
            return True
        elif (len(send_step.dep_list) == 1 and 
              send_step.depid == tb.id and 
              send_step.deps == recv_step.s):
            return True
        
        return False

    def _merge_recv_send_to_rcs(self, tb: TB, send_step_idx: int) -> bool:
        """
        将recv+send合并为rcs操作
        """
        try:
            recv_step = tb.steps[send_step_idx - 1]
            send_step = tb.steps[send_step_idx]
            
            # 检查recv操作是否仅被该send操作依赖
            recv_only_depended_by_send = (
                len(recv_step.depended_by_list) == 1 and 
                recv_step.depended_by_list[0] == send_step
            )
            
            # 创建新的rcs step
            rcs_step = Step(
                s=recv_step.s,  # 使用recv step的位置
                type="rcs",
                srcbuf=send_step.srcbuf,  # rcs的源buffer来自send
                srcoff=send_step.srcoff,  # rcs的源offset来自send
                dstbuf=recv_step.dstbuf,  # rcs的目标buffer来自send
                dstoff=recv_step.dstoff,  # rcs的目标offset来自send
                cnt=recv_step.cnt,  # 数据量通常相同
                depid=recv_step.depid,   # 继承recv的依赖
                deps=recv_step.deps,
                hasdep=0 if recv_only_depended_by_send else recv_step.hasdep
            )
            
            # 设置TB引用和索引
            rcs_step._tb = tb
            rcs_step.send_index = send_step.send_index
            rcs_step.recv_index = recv_step.recv_index
            rcs_step.position_fixed = True  # 合并后直接固定位置
            
            # 设置peer关系
            rcs_step.recv_peer = recv_step.peer_step
            rcs_step.send_peer = send_step.peer_step
            
            # 处理依赖关系转移
            # 1. 继承recv的依赖关系
            rcs_step.dep_list = recv_step.dep_list.copy()
            
            # 2. 处理send的被依赖关系（其他step依赖send的情况）
            for dependent_step in send_step.depended_by_list:
                # 将依赖send的step改为依赖rcs
                if send_step in dependent_step.dep_list:
                    dependent_step.dep_list.remove(send_step)
                    dependent_step.dep_list.append(rcs_step)
                rcs_step.depended_by_list.append(dependent_step)
            
            # 3. 处理recv的被依赖关系（除了send之外的其他依赖）
            for dependent_step in recv_step.depended_by_list:
                if dependent_step != send_step:  # 跳过send，因为它已经被合并了
                    # 将依赖recv的step改为依赖rcs
                    if recv_step in dependent_step.dep_list:
                        dependent_step.dep_list.remove(recv_step)
                        dependent_step.dep_list.append(rcs_step)
                    rcs_step.depended_by_list.append(dependent_step)
            
            # 4. 清理被合并steps的依赖关系
            for dep_step in recv_step.dep_list:
                if recv_step in dep_step.depended_by_list:
                    dep_step.depended_by_list.remove(recv_step)
                    dep_step.depended_by_list.append(rcs_step)
            
            # 清理send的依赖关系
            for dep_step in send_step.dep_list:
                if send_step in dep_step.depended_by_list:
                    dep_step.depended_by_list.remove(send_step)
            
            # 替换steps: 移除recv和send，插入rcs
            tb.steps.pop(send_step_idx)  # 先移除send (索引较大的)
            tb.steps.pop(send_step_idx - 1)  # 再移除recv
            tb.steps.insert(send_step_idx - 1, rcs_step)  # 插入rcs到recv的原位置
            
            # 更新后续steps的s字段
            for j in range(send_step_idx - 1, len(tb.steps)):
                tb.steps[j].s = j
            
            return True
            
        except Exception as e:
            return False
    
    def to_xml(self) -> ET.Element:
        gpu_elem = ET.Element("gpu")
        gpu_elem.set("id", str(self.id))
        
        # 计算各种buffer的最大深度
        max_i_chunks = 0
        max_o_chunks = 0  
        max_s_chunks = 0
        
        for tb in self.tbs:
            for step in tb.steps:
                # 检查srcbuf
                if step.srcbuf == "i" and step.srcoff >= 0:
                    max_i_chunks = max(max_i_chunks, step.srcoff + step.cnt)
                elif step.srcbuf == "o" and step.srcoff >= 0:
                    max_o_chunks = max(max_o_chunks, step.srcoff + step.cnt)
                elif step.srcbuf == "s" and step.srcoff >= 0:
                    max_s_chunks = max(max_s_chunks, step.srcoff + step.cnt)
                
                # 检查dstbuf
                if step.dstbuf == "i" and step.dstoff >= 0:
                    max_i_chunks = max(max_i_chunks, step.dstoff + step.cnt)
                elif step.dstbuf == "o" and step.dstoff >= 0:
                    max_o_chunks = max(max_o_chunks, step.dstoff + step.cnt)
                elif step.dstbuf == "s" and step.dstoff >= 0:
                    max_s_chunks = max(max_s_chunks, step.dstoff + step.cnt)
        
        gpu_elem.set("i_chunks", str(max_i_chunks))
        gpu_elem.set("o_chunks", str(max_o_chunks))
        gpu_elem.set("s_chunks", str(max_s_chunks))
        
        for tb in self.tbs:
            gpu_elem.append(tb.to_xml())
        
        return gpu_elem