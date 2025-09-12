from msccl_xml_builder import Algo, Chunk

def generate_alltoall_2step_xml(node_nums: int = 16, gpus_pernode: int = 8, instances: int = 1, p2pchannels: int = 1, filename: str = None):
    """
    生成两阶段AllToAll算法的XML文件
    
    添加依赖关系：
    - 第二阶段的转发操作必须等待第一阶段的接收操作完成
    - 确保scratch buffer中有数据后才能转发
    """
    
    ngpus = node_nums * gpus_pernode
    nchunksperloop = ngpus * instances

    nchannels = instances *p2pchannels
    
    if filename is None:
        filename = f"alltoall_2step_{node_nums}nodes_{gpus_pernode}gpus_{instances}instances_{p2pchannels}p2pchannels.xml"
    
    # 创建算法实例
    algo = Algo(
        name="alltoall_2step",
        proto="Simple",
        nchannels=nchannels,
        nchunksperloop=nchunksperloop,
        ngpus=ngpus,
        coll="allreduce",
        inplace=0,
        outofplace=1,
        minBytes=0,
        maxBytes=0
    )
    
    print(f"生成两阶段AllToAll算法: {node_nums}个节点, {gpus_pernode}个GPU/节点, {instances}个实例")
    
    # 用于存储第一阶段的接收操作，以便第二阶段建立依赖关系
    # 结构：recv_steps[intermediary_rank][scratch_index] = recv_step
    
    
    for instance in range(instances):
        print(f"\n=== 处理实例 {instance} ===")
        
        # ===== 第一阶段：数据发送和中转 =====
        print("第一阶段：数据发送和中转")
        
        recv_steps = {}
        send_map = {}
        recv_map = {}
        # 初始化一个5x5的二维数组，所有元素为0
        rows, cols = ngpus, ngpus
        channel_map = [[0] * cols for _ in range(rows)]

        for node_id in range(node_nums):
            print(f"\n  处理节点 {node_id}")
            
            for local_rank in range(gpus_pernode):
                global_rank = node_id * gpus_pernode + local_rank
                print(f"    处理rank {global_rank} (节点{node_id}, local_rank{local_rank})")
                
                # 该rank要发送给所有其他rank的数据
                for dest_global_rank in range(ngpus):
                    dest_node = dest_global_rank // gpus_pernode
                    dest_local_rank = dest_global_rank % gpus_pernode


                    
                    # 源数据：input buffer中索引为dest_global_rank的数据
                    src_chunk = Chunk(
                        gpu_id=global_rank,
                        chunk_type="i",
                        index=dest_global_rank * instances + instance,
                        size=1,
                        algo=algo
                    )
                    
                    if dest_node == node_id:
                        # 本节点内传输
                        if global_rank == dest_global_rank:
                            # 发送给自己：copy操作
                            dest_chunk = Chunk(
                                gpu_id=dest_global_rank,
                                chunk_type="o",
                                index=global_rank * instances + instance,
                                size=1,
                                algo=algo
                            )

                            channel = instance*p2pchannels + channel_map[global_rank][dest_global_rank]%p2pchannels
                            channel_map[global_rank][dest_global_rank] = channel_map[global_rank][dest_global_rank] + 1
                            copy_step = src_chunk.copy(dest_chunk, channel_id = channel )
                            
                            print(f"      自拷贝: rank{global_rank} input[{dest_global_rank * instances + instance}] -> output[{global_rank * instances + instance}]")
                        else:
                            # 发送给本节点其他rank
                            dest_chunk = Chunk(
                                gpu_id=dest_global_rank,
                                chunk_type="o",
                                index=global_rank * instances + instance,
                                size=1,
                                algo=algo
                            )

                            channel = instance*p2pchannels + channel_map[global_rank][dest_global_rank]%p2pchannels
                            channel_map[global_rank][dest_global_rank] = channel_map[global_rank][dest_global_rank] + 1

                            send_step, recv_step = src_chunk.copy_diff(dest_chunk, channel_id=channel)
                            print(f"      节点内传输: rank{global_rank} -> rank{dest_global_rank}")
                    
                    elif local_rank == dest_local_rank:
                        # 跨节点，同local_rank：直接发送
                        dest_chunk = Chunk(
                            gpu_id=dest_global_rank,
                            chunk_type="o",
                            index=global_rank * instances + instance,
                            size=1,
                            algo=algo
                        )

                        channel = instance*p2pchannels + channel_map[global_rank][dest_global_rank]%p2pchannels
                        channel_map[global_rank][dest_global_rank] = channel_map[global_rank][dest_global_rank] + 1

                        send_step, recv_step = src_chunk.copy_diff(dest_chunk, channel_id=channel)
                        print(f"      跨节点同local_rank: rank{global_rank} -> rank{dest_global_rank}")
                    
                    else:
                        # 跨节点，不同local_rank：需要中转
                        # 发送给本节点对应的local_rank进行中转
                        intermediary_rank = node_id * gpus_pernode + dest_local_rank
                        
                        # scratch buffer索引：按目标节点距离组织
                        if dest_node > node_id:
                            node_distance = dest_node - node_id - 1
                        else:
                            node_distance = (node_nums - node_id) + dest_node - 1
                        
                        scratch_index = instances*node_distance * gpus_pernode + local_rank*instances + instance
                        
                        dest_chunk = Chunk(
                            gpu_id=intermediary_rank,
                            chunk_type="s",
                            index=scratch_index,
                            size=1,
                            algo=algo
                        )

                        channel = instance*p2pchannels + channel_map[global_rank][intermediary_rank]%p2pchannels
                        channel_map[global_rank][intermediary_rank] = channel_map[global_rank][intermediary_rank] + 1
                        
                        send_step, recv_step = src_chunk.copy_diff(dest_chunk, channel_id=channel)
                        
                        # 记录接收操作，用于后续建立依赖关系
                        if intermediary_rank not in recv_steps:
                            recv_steps[intermediary_rank] = {}
                        recv_steps[intermediary_rank][scratch_index] = recv_step
                        
                        print(f"      中转发送: rank{global_rank} -> rank{intermediary_rank} scratch[{scratch_index}] (目标node{dest_node}, 距离{node_distance})")
        
        # ===== 第二阶段：中转数据转发 =====
        print(f"\n第二阶段：中转数据转发")
        
        for node_id in range(node_nums):
            for local_rank in range(gpus_pernode):
                global_rank = node_id * gpus_pernode + local_rank
                print(f"\n  处理中转rank {global_rank} (节点{node_id}, local_rank{local_rank})")
                
                # 该rank需要转发给其他节点同local_rank的数据
                for dest_node in range(node_nums):
                    if dest_node == node_id:
                        continue  # 跳过本节点
                    
                    dest_global_rank = dest_node * gpus_pernode + local_rank
                    
                    # 转发来自本节点其他local_rank的数据
                    for src_local_rank in range(gpus_pernode):
                        if src_local_rank == local_rank:
                            continue  # 跳过自己的数据（已经在第一阶段直接发送了）
                        
                        src_global_rank = node_id * gpus_pernode + src_local_rank
                        
                        # 计算在scratch buffer中的索引
                        if dest_node > node_id:
                            node_distance = dest_node - node_id - 1
                        else:
                            node_distance = (node_nums - node_id) + dest_node - 1
                        
                        scratch_index = instances*node_distance * gpus_pernode + src_local_rank*instances + instance
                        
                        src_chunk = Chunk(
                            gpu_id=global_rank,
                            chunk_type="s",
                            index=scratch_index,
                            size=1,
                            algo=algo
                        )
                        
                        dest_chunk = Chunk(
                            gpu_id=dest_global_rank,
                            chunk_type="o",
                            index=src_global_rank * instances + instance,
                            size=1,
                            algo=algo
                        )
                        
                        # 查找对应的接收操作，建立依赖关系
                        dep_steps = []
                        if (global_rank in recv_steps and 
                            scratch_index in recv_steps[global_rank]):
                            dep_recv_step = recv_steps[global_rank][scratch_index]
                            dep_steps.append(dep_recv_step)
                            print(f"      建立依赖: 转发操作依赖于scratch[{scratch_index}]的接收完成")

                            channel = instance*p2pchannels + channel_map[global_rank][dest_global_rank]%p2pchannels
                            channel_map[global_rank][dest_global_rank] = channel_map[global_rank][dest_global_rank] + 1

                            send_step, recv_step = src_chunk.copy_diff(dest_chunk, channel_id=channel, dep_steps=dep_steps)
                            print(f"    转发: rank{global_rank} scratch[{scratch_index}] -> rank{dest_global_rank} output[{src_global_rank * instances + instance}]")
                            # ===== 存储第二阶段的发送和接收操作 =====
                            # 初始化嵌套字典结构
                            if node_id not in send_map:
                                send_map[node_id] = {}
                                recv_map[node_id] = {}
                            if dest_node not in send_map[node_id]:
                                send_map[node_id][dest_node] = {}
                                recv_map[node_id][dest_node] = {}
                            
                            # 存储发送和接收操作
                            send_map[node_id][dest_node][local_rank] = send_step
                            recv_map[node_id][dest_node][local_rank] = recv_step
                            
                            print(f"    转发: rank{global_rank} scratch[{scratch_index}] -> rank{dest_global_rank} output[{src_global_rank * instances + instance}]")
                            print(f"      存储到send_map[{node_id}][{dest_node}][{local_rank}]和recv_map[{node_id}][{dest_node}][{local_rank}]")


    print("\n构建依赖关系...")
    algo.build_all_dependencies()
    
    print(f"保存XML文件: {filename}")
    algo.save_xml(filename)
    print(f"两阶段AllToAll XML文件已生成: {filename}")
    
    return algo

# 主函数
if __name__ == "__main__":
    print("=== 生成两阶段AllToAll算法 ===")
    


    generate_alltoall_2step_xml(node_nums=16, gpus_pernode=8, instances=1, p2pchannels=2 )
