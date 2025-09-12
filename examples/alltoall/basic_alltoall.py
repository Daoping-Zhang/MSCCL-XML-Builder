from xml_generator_tools import Algo, Chunk

def generate_alltoall_xml(node_nus: int = 8, gpus_pernode:int =8, instances: int = 1, filename: str = None):
    """
    生成AllToAll算法的XML文件
    
    算法逻辑：
    - 每个rank i 将 input[j] 发送到 rank j 的 output[i]
    - 如果 i == j，则是本地copy操作
    - 所有rank之间都进行数据交互
    
    Args:
        ngpus: GPU数量
        filename: 输出文件名
    """


    ngpus=node_nus*gpus_pernode
    nchunksperloop = ngpus*instances
    nchannles = instances
    if filename is None:
        filename = f"alltoall_{ngpus}gpus.xml"

    # 创建算法实例
    algo = Algo(
        name="alltoall",
        proto="Simple", 
        nchannels=nchannles,  # 使用单个channel
        nchunksperloop=nchunksperloop,
        ngpus=ngpus,
        coll="allreduce", 
        inplace=0,
        outofplace=1,
        minBytes=0,
        maxBytes=0
    )
    
    print(f"生成AllToAll算法: {ngpus}个GPU")
    
    # 为每个rank创建操作
    for instance in range(instances):
        for src_rank in range(ngpus):
            print(f"  处理rank {src_rank}")
            
            for dest_rank in range(ngpus):
                if src_rank == dest_rank:
                    # 本地copy: input[src_rank] -> output[src_rank]
                    src_chunk = Chunk(gpu_id=src_rank, chunk_type="i", index=src_rank*instances+instance, size=1, algo=algo)
                    dest_chunk = Chunk(gpu_id=src_rank, chunk_type="o", index=src_rank*instances+instance, size=1, algo=algo)
                    copy_step = src_chunk.copy(dest_chunk, channel_id=instance)
                    print(f"    instance {instance} channel {instance} rank {src_rank} -> rank {dest_rank} (本地copy)")
                else:
                    # 跨rank通信: rank src_rank 的 input[dest_rank] -> rank dest_rank 的 output[src_rank]
                    src_chunk = Chunk(gpu_id=src_rank, chunk_type="i", index=dest_rank*instances +instance, size=1, algo=algo)
                    dest_chunk = Chunk(gpu_id=dest_rank, chunk_type="o", index=src_rank*instances +instance, size=1, algo=algo)
                    
                    # 使用copy_diff方法进行跨rank传输
                    send_step, recv_step = src_chunk.copy_diff(dest_chunk, channel_id=instance)
                    print(f"    instance {instance} channel {instance} rank {src_rank} -> rank {dest_rank} (跨rank传输)")
    
    # 构建依赖关系
    print("构建依赖关系...")
    algo.build_all_dependencies()
    
    # 保存XML文件
    filename=f"basic_alltoall_128gpus_{instances}instances.xml"
    algo.save_xml(filename)
    print(f"XML文件已生成: {filename}")
    
    return algo


# 主函数
if __name__ == "__main__":
    # 生成基础版本
    print("=== 生成基础AllToAll算法 ===")


    generate_alltoall_xml(node_nus=16,gpus_pernode=8, instances=4)
    
 