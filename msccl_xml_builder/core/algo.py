import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List

class Algo:
    def __init__(self, name: str, proto: str = "Simple", nchannels: int = 1, 
                 nchunksperloop: int = 2, ngpus: int = 2, coll: str = "allgather",
                 inplace: int = 1, outofplace: int = 1, minBytes: int = 0, maxBytes: int = 0):
        self.name = name
        self.proto = proto
        self.nchannels = nchannels
        self.nchunksperloop = nchunksperloop
        self.ngpus = ngpus
        self.coll = coll
        self.inplace = inplace
        self.outofplace = outofplace
        self.minBytes = minBytes
        self.maxBytes = maxBytes
        
        # Initialize GPUs - 延迟导入避免循环导入
        from .gpu import GPU
        self.gpus: List[GPU] = []
        for i in range(ngpus):
            self.gpus.append(GPU(i))
    
    def get_gpu(self, gpu_id: int):
        if 0 <= gpu_id < len(self.gpus):
            return self.gpus[gpu_id]
        raise ValueError(f"GPU {gpu_id} not found")
    
    def build_all_dependencies(self, merge_rcs: bool = False, sort: bool = True) -> None:
        """构建所有GPU的依赖关系"""
        # 第一步：对所有TB中的steps进行排序
        if sort:
            for gpu in self.gpus:
                gpu.sort_all_tb_steps()
        
        # 第二步：构建依赖关系
        for gpu in self.gpus:
            gpu.build_dependencies(merge_rcs)
    
    def to_xml(self) -> ET.Element:
        algo_elem = ET.Element("algo")
        algo_elem.set("name", self.name)
        algo_elem.set("proto", self.proto)
        algo_elem.set("nchannels", str(self.nchannels))
        algo_elem.set("nchunksperloop", str(self.nchunksperloop))
        algo_elem.set("ngpus", str(self.ngpus))
        algo_elem.set("coll", self.coll)
        algo_elem.set("inplace", str(self.inplace))
        algo_elem.set("outofplace", str(self.outofplace))
        algo_elem.set("minBytes", str(self.minBytes))
        algo_elem.set("maxBytes", str(self.maxBytes))
        
        for gpu in self.gpus:
            algo_elem.append(gpu.to_xml())
        
        return algo_elem
    
    def save_xml(self, filename: str) -> None:
        xml_elem = self.to_xml()
        rough_string = ET.tostring(xml_elem, 'unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_string = reparsed.toprettyxml(indent="  ")
        
        # Remove empty lines and XML declaration for cleaner output
        lines = [line for line in pretty_string.split('\n') if line.strip() and not line.strip().startswith('<?xml')]
        pretty_string = '\n'.join(lines)
        
        with open(filename, 'w') as f:
            f.write(pretty_string)