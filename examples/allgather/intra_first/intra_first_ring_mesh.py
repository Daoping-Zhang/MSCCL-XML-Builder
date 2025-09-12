from xml_generator_tools import Algo, Chunk
from xml_generator_tools import TB


def generate_intra_first_ring_mesh_allgather_xml(node_num: int = 4, gpus_per_node: int = 4, instances:int = 1, ring_channels:int = 1,p2p_channels:int = 1, proto: str = "LL128",filename: str = None):
    """
    正确实现双结构Ring AllGather算法 - 修正机内环数据传播逻辑
    """
    ngpus = node_num * gpus_per_node


    channels_per_instance = max(ring_channels,p2p_channels)

    nchannls = instances*channels_per_instance

    nchunksperloop=ngpus*instances
    
    if filename is None:
        filename = f"intra_first_ring_mesh_{node_num}nodes_{gpus_per_node}gpus.xml"
    
    # 创建算法实例
    algo = Algo(
        name="intra_first_ring_mesh_allgather",
        proto=proto, 
        nchannels= nchannls,
        nchunksperloop=nchunksperloop,
        ngpus=ngpus,
        coll="allgather",
        inplace=1,
        outofplace=1,
        minBytes=0,
        maxBytes=0
    )
    
    def get_node_id(rank):
        return rank // gpus_per_node
    
    def get_local_rank(rank):
        return rank % gpus_per_node
    
    def get_global_rank(node_id, local_rank):
        return node_id * gpus_per_node + local_rank
    
    print(f"生成正确的双ring算法: {node_num}节点 × {gpus_per_node}GPU = {ngpus}总GPU")
    


    data_index_map = [[] for _ in range(ngpus)]
    recv_step_map = [[] for _ in range(ngpus)]
            # 初始化两个列表，每个 GPU 对应一个空列表
    for rank in range(ngpus):
        gpu = algo.get_gpu(rank)
        node_id = get_node_id(rank)
        local_rank = get_local_rank(rank)
        

        
        #创建ring TB
        # 第二类：机间环TB
        # 计算机间环的send和recv目标
        inter_send_rank = get_global_rank((node_id + 1) % node_num, local_rank)  # 下一个节点的相同local_rank
        inter_recv_rank = get_global_rank((node_id - 1) % node_num, local_rank)  # 上一个节点的相同local_rank
        
        channel_id = 0

        inter_tb = TB(send=inter_send_rank, recv=inter_recv_rank, chan=channel_id)
        gpu.add_tb(inter_tb)

    # 第一步：本地copy
    copy_steps = []
    for rank in range(ngpus):
        src_chunk = Chunk(gpu_id=rank, chunk_type="i", index=0, size=1, algo=algo)
        dest_chunk = Chunk(gpu_id=rank, chunk_type="o", index=rank, size=1, algo=algo)
        copy_step = src_chunk.copy(dest_chunk, channel_id=0)
        copy_steps.append(copy_step)
        data_index_map[rank].append(rank)
        recv_step_map[rank].append(copy_step)

    #第二步骤机内数据聚合mesh算法

    for node_id in range(node_num):
        print(f"  处理节点{node_id}的机内数据分发")
        node_ranks = [get_global_rank(node_id, local_rank) for local_rank in range(gpus_per_node)]
            
  
                
        # 为每个有数据要分发的rank创建分发操作
        for sender_rank in node_ranks:

            for receiver_rank in node_ranks:

                if receiver_rank == sender_rank:
                    continue

                src_chunk = Chunk(gpu_id=sender_rank, chunk_type="i", index=0, size=1, algo=algo)
                dest_chunk = Chunk(gpu_id=receiver_rank, chunk_type="o", index = sender_rank, size=1, algo=algo)

                send_step, recv_step = src_chunk.copy_diff(dest_chunk, channel_id=0) 
                                                            
                data_index_map[receiver_rank].append(sender_rank)
                recv_step_map[receiver_rank].append(recv_step)


        

            
                



    
    #第三步骤机间ring算法数据同步

    inter_node_steps = {}
    inter_node_recv_data = {}
    
    for local_rank_id in range(gpus_per_node):
        print(f"  处理机间环 {local_rank_id}")
        inter_node_steps[local_rank_id] = {}
        inter_node_recv_data[local_rank_id] = {}
        
        # 参与这个环的所有rank
        ring_ranks = [get_global_rank(node_id, local_rank_id) for node_id in range(node_num)]
        
        # 执行(node_num-1)步传输
        prev_recv_steps = {rank: copy_steps[rank] for rank in ring_ranks}
        
        
        for step in range(node_num - 1):
            print(f"    机间环 {local_rank_id} 第{step+1}步")
            current_recv_steps = {}
            current_recv_data = {}
            
            for i, rank in enumerate(ring_ranks):
                next_rank = ring_ranks[(i + 1) % len(ring_ranks)]
                prev_rank = ring_ranks[(i - 1) % len(ring_ranks)]
                
                # 计算要发送的数据的原始owner
                data_owner_index = (i - step) % len(ring_ranks)
                data_owner_rank = ring_ranks[data_owner_index]

                owner_rank_step = (i - step) % len(ring_ranks)
                send_channel= 0
                
                # 计算要接收的数据的原始owner
                recv_data_owner_index = ((i - 1) - step) % len(ring_ranks)
                recv_data_owner_rank = ring_ranks[recv_data_owner_index]


                recv_rank_step = ((i - 1) - step) % len(ring_ranks)
                recv_channel = 0
                
                # Send操作
                if step == 0:


                    src_chunk = Chunk(gpu_id=rank, chunk_type="o", index=data_owner_rank//gpus_per_node *gpus_per_node, size=gpus_per_node, algo=algo)
                    send_step = src_chunk.send(dest_rank=next_rank, channel_id= send_channel,
                                            dep_steps=recv_step_map[rank], bidirectional=False)
                    
                    # Recv操作
                    dest_chunk = Chunk(gpu_id=rank, chunk_type="o", index = recv_data_owner_rank//gpus_per_node *gpus_per_node, size=gpus_per_node, algo=algo)
                    recv_step = dest_chunk.recv(src_rank=prev_rank, channel_id = recv_channel , bidirectional=False)
                    
                    current_recv_steps[rank] = recv_step
                    current_recv_data[rank] = recv_data_owner_rank  # 记录接收到的数据的offset

                else:

                    src_chunk = Chunk(gpu_id=rank, chunk_type="o", index=data_owner_rank//gpus_per_node *gpus_per_node, size=gpus_per_node, algo=algo)
                    send_step = src_chunk.send(dest_rank=next_rank, channel_id= send_channel,
                                            dep_steps=[prev_recv_steps[rank]], bidirectional=False)
                    
                    # Recv操作
                    dest_chunk = Chunk(gpu_id=rank, chunk_type="o", index = recv_data_owner_rank//gpus_per_node *gpus_per_node, size=gpus_per_node, algo=algo)
                    recv_step = dest_chunk.recv(src_rank=prev_rank, channel_id = recv_channel , bidirectional=False)
                    
                    current_recv_steps[rank] = recv_step
                    current_recv_data[rank] = recv_data_owner_rank  # 记录接收到的数据的offset
                
                    # 设置peer关系
                    send_step.peer_step = recv_step
                    recv_step.peer_step = send_step
            
            # 保存这一步的信息
            inter_node_steps[local_rank_id][step] = current_recv_steps.copy()
            inter_node_recv_data[local_rank_id][step] = current_recv_data.copy()
            prev_recv_steps = current_recv_steps
                                                                

        

                


                        
                        

 
    # 构建依赖关系
    print("构建依赖关系...")
    algo.build_all_dependencies(True)
    
    filename = f"intra_first_ring_mesh_{node_num}nodes_{gpus_per_node}gpus_{instances}instances_{ring_channels}ringChannels_{p2p_channels}p2p_channels.xml"
    # 保存XML文件
    algo.save_xml(filename)
    print(f"修正后的XML文件已生成: {filename}")
    
    return algo

# 测试
if __name__ == "__main__":

    generate_intra_first_ring_mesh_allgather_xml(node_num=16, gpus_per_node=8, instances=1, ring_channels=1, p2p_channels = 1)
