#!/usr/bin/env python3

try:
    from msccl_xml_builder import Algo, Chunk
    print("Installation successful!")
    
    # 创建一个简单的算法测试
    algo = Algo(name="test_algo", ngpus=2, coll="allgather")
    print(f"Created algorithm: {algo.name} with {algo.ngpus} GPUs")
    
    # 创建一些chunk并进行基本操作
    chunk1 = Chunk(0, "i", 0, 1, algo)
    chunk2 = Chunk(1, "o", 0, 1, algo)
    
    # 测试跨GPU通信
    send_step, recv_step = chunk1.copy_diff(chunk2, channel_id=0)
    print(f"Created communication: {send_step.type} -> {recv_step.type}")
    
    # 构建依赖并生成XML
    algo.build_all_dependencies()
    algo.save_xml("test_output.xml")
    print("XML file generated successfully: test_output.xml")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Please check if the package is installed correctly")
except Exception as e:
    print(f"Runtime error: {e}")
    import traceback
    traceback.print_exc()