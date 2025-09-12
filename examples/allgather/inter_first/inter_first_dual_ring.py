from msccl_xml_builder import Algo, Chunk
from msccl_xml_builder import TB

def generate_dual_ring_allgather_xml(node_num: int = 4, gpus_per_node: int = 4, instances:int = 1, data_steps:int=1, inter_ring_channels:int = 1,intra_ring_instances:int = 1, intra_ring_channels:int = 1, proto: str = "Simple",filename: str = None):

    ngpus = node_num * gpus_per_node

    channels_per_instance = max(inter_ring_channels, intra_ring_channels*intra_ring_instances)

    nchannls = instances* channels_per_instance
    
    nchunksperloop=ngpus*instances*data_steps

    total_data_steps = instances*data_steps

    
    if filename is None:
        filename = f"inter_first_dual_ring_{node_num}nodes_{gpus_per_node}gpus.xml"
    
    # 创建算法实例
    algo = Algo(
        name="inter_first_dual_ring_allgather",
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
    


    for instance in range(instances):
    
        local_tbs = {}      # {rank: tb} - 本地操作TB
        inter_ring_tbs = {} # {rank: tb} - 机间环TB  
        intra_ring_tbs = {} # {rank: tb} - 机内环TB
        
        #显示的创建tb
        for rank in range(ngpus):
            gpu = algo.get_gpu(rank)
            node_id = get_node_id(rank)
            local_rank = get_local_rank(rank)
            
            # 第一类：本地操作TB (send=-1, recv=-1)
            local_tb = TB(send=-1, recv=-1, chan=instance)
            
            gpu.add_tb(local_tb)
            local_tbs[rank] = local_tb
            
            #创建ring TB
            for channel in range(inter_ring_channels):
                # 第二类：机间环TB
                # 计算机间环的send和recv目标
                inter_send_rank = get_global_rank((node_id + 1) % node_num, local_rank)  # 下一个节点的相同local_rank
                inter_recv_rank = get_global_rank((node_id - 1) % node_num, local_rank)  # 上一个节点的相同local_rank
                
                channel_id = instance*channels_per_instance + channel

                inter_tb = TB(send=inter_send_rank, recv=inter_recv_rank, chan=channel_id)
                
                gpu.add_tb(inter_tb)
                inter_ring_tbs[rank] = inter_tb

            for channel in range(intra_ring_instances*intra_ring_channels):
                # 第三类：机内环TB
                # 计算机内环的send和recv目标
                
                channel_id = instance*channels_per_instance + channel

                intra_send_rank = get_global_rank(node_id, (local_rank + 1) % gpus_per_node)  # 同节点下一个local_rank
                intra_recv_rank = get_global_rank(node_id, (local_rank - 1) % gpus_per_node)  # 同节点上一个local_rank
                
                intra_tb = TB(send=intra_send_rank, recv=intra_recv_rank, chan=channel_id)
                gpu.add_tb(intra_tb)
                intra_ring_tbs[rank] = intra_tb
                
                #print(f"  rank {rank}: 本地TB, 机间TB(send={inter_send_rank}, recv={inter_recv_rank}), 机内TB(send={intra_send_rank}, recv={intra_recv_rank})")
    for data_step in range(total_data_steps):

        instance = data_step//data_steps
        # 第一步：本地copy
        copy_steps = []
        for rank in range(ngpus):
            src_chunk = Chunk(gpu_id=rank, chunk_type="i", index=data_step, size=1, algo=algo)
            dest_chunk = Chunk(gpu_id=rank, chunk_type="o", index=rank*total_data_steps+data_step, size=1, algo=algo)
            copy_step = src_chunk.copy(dest_chunk, channel_id=instance)
            copy_steps.append(copy_step)
        
        # 第二步：机间环传输
        # inter_node_recv_data[local_rank][step] = {rank: 接收到的数据的offset}
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
                    send_channel= channels_per_instance*instance + (owner_rank_step%inter_ring_channels)
                    
                    # 计算要接收的数据的原始owner
                    recv_data_owner_index = ((i - 1) - step) % len(ring_ranks)
                    recv_data_owner_rank = ring_ranks[recv_data_owner_index]


                    recv_rank_step = ((i - 1) - step) % len(ring_ranks)
                    recv_channel = channels_per_instance*instance + (recv_rank_step%inter_ring_channels)
                    
                    # Send操作
                    src_chunk = Chunk(gpu_id=rank, chunk_type="o", index=data_owner_rank*total_data_steps+data_step, size=1, algo=algo)
                    send_step = src_chunk.send(dest_rank=next_rank, channel_id= send_channel,
                                            dep_steps=[prev_recv_steps[rank]], bidirectional=False)
                    
                    # Recv操作
                    dest_chunk = Chunk(gpu_id=rank, chunk_type="o", index = recv_data_owner_rank*total_data_steps+data_step, size=1, algo=algo)
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
        
        # 第三步：机内环传输
        
        for node_id in range(node_num):
            
            print(f"  处理节点{node_id}的机内环")
            node_ranks = [get_global_rank(node_id, local_rank) for local_rank in range(gpus_per_node)]
            
            # 执行node_num个周期
            for cycle in range(node_num):

                intra_ring_instance = cycle%intra_ring_instances
                print(f"    节点{node_id} 机内环 第{cycle+1}个周期")
                
                # 确定这个周期的数据依赖和要传播的数据
                if cycle == 0:
                    # 第1个周期：依赖本地copy，传播本地数据
                    cycle_deps = {rank: copy_steps[rank] for rank in node_ranks}
                    # 要传播的数据就是每个rank自己的数据
                    cycle_data_map = {rank: rank for rank in node_ranks}  # rank -> 该rank拥有的数据offset
                    print(f"      第1周期依赖本地copy，传播本地数据")
                else:
                    # 第N个周期：依赖机间环第(N-1)步，传播机间环收集的数据
                    step_idx = cycle - 1
                    cycle_deps = {}
                    cycle_data_map = {}
                    
                    for rank in node_ranks:
                        local_rank = get_local_rank(rank)
                        if (local_rank in inter_node_steps and 
                            step_idx in inter_node_steps[local_rank] and
                            rank in inter_node_steps[local_rank][step_idx]):
                            
                            cycle_deps[rank] = inter_node_steps[local_rank][step_idx][rank]
                            # 要传播的数据是机间环这一步接收到的数据
                            cycle_data_map[rank] = inter_node_recv_data[local_rank][step_idx][rank]
                            print(f"      rank {rank} 依赖机间环{local_rank}第{step_idx+1}步，传播数据offset={cycle_data_map[rank]}")
                        else:
                            cycle_deps[rank] = copy_steps[rank]
                            cycle_data_map[rank] = rank

                
                # 执行这个周期的(gpus_per_node-1)步机内环
                prev_recv_steps = cycle_deps.copy()
                
                for step in range(gpus_per_node - 1):
                    print(f"      节点{node_id} 周期{cycle+1} 步骤{step+1}")
                    current_recv_steps = {}
                    
                    for i, rank in enumerate(node_ranks):
                        next_rank = node_ranks[(i + 1) % len(node_ranks)]
                        prev_rank = node_ranks[(i - 1) % len(node_ranks)]


                        # 计算要发送的数据：这一步应该由哪个rank发送数据
                        sender_index = (i - step) % len(node_ranks)
                        sender_rank = node_ranks[sender_index]

                        sender_rank_step = sender_rank%gpus_per_node
                        send_channel = instance*channels_per_instance + intra_ring_instance*intra_ring_channels + (sender_rank_step%intra_ring_channels)
                        
                        # 发送的数据是sender_rank在这个周期拥有的数据
                        data_offset = cycle_data_map[sender_rank]
                        
                        # 计算要接收的数据：从prev_rank接收数据
                        recv_sender_index = ((i - 1) - step) % len(node_ranks)
                        recv_sender_rank = node_ranks[recv_sender_index]

                        recv_sender_rank_step = recv_sender_rank%gpus_per_node
                        recv_channel = instance*channels_per_instance + intra_ring_instance*intra_ring_channels + (recv_sender_rank_step%intra_ring_channels)

                        recv_data_offset = cycle_data_map[recv_sender_rank]
                        
                        # Send操作
                        src_chunk = Chunk(gpu_id=rank, chunk_type="o", index=data_offset*total_data_steps+data_step, size=1, algo=algo)
                        send_step = src_chunk.send(dest_rank=next_rank, channel_id=send_channel,
                                                dep_steps=[prev_recv_steps[rank]], bidirectional=False)
                        
                        # Recv操作
                        dest_chunk = Chunk(gpu_id=rank, chunk_type="o", index=recv_data_offset*total_data_steps+data_step, size=1, algo=algo)
                        recv_step = dest_chunk.recv(src_rank=prev_rank, channel_id=recv_channel, bidirectional=False)
                        
                        current_recv_steps[rank] = recv_step
                        
                        # 设置peer关系
                        send_step.peer_step = recv_step
                        recv_step.peer_step = send_step
                        
                        print(f"        rank {rank} 发送数据offset={data_offset} 到 rank {next_rank}")
                        print(f"        rank {rank} 接收数据offset={recv_data_offset} 从 rank {prev_rank}")
        

                    prev_recv_steps = current_recv_steps
    
    # 构建依赖关系
    print("构建依赖关系...")
    algo.build_all_dependencies(True)
    
    filename = f"inter_first__dual_ring_{data_steps}s_{instances}instances_{inter_ring_channels}interRingChannels_{intra_ring_instances}intraRingInstances_{intra_ring_channels}intraRingChannels.xml"
    # 保存XML文件
    algo.save_xml(filename)
    print(f"修正后的XML文件已生成: {filename}")
    
    return algo

# 测试
if __name__ == "__main__":

    generate_dual_ring_allgather_xml(node_num=16, gpus_per_node=8, instances=1, data_steps=1, inter_ring_channels=16 , intra_ring_instances=1, intra_ring_channels = 1)


