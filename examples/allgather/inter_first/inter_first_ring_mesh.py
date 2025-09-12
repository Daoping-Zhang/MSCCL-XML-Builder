from msccl_xml_builder import Algo, Chunk
from msccl_xml_builder import TB

def generate_inter_first_ring_mesh_allgather_xml(node_num: int = 4, gpus_per_node: int = 4, instances:int = 1, ring_channels:int = 1,p2p_channels:int = 1, proto: str = "LL128",filename: str = None):

    ngpus = node_num * gpus_per_node


    channels_per_instance = max(ring_channels,p2p_channels)

    nchannls = instances*channels_per_instance

    nchunksperloop=ngpus*instances
    
    if filename is None:
        filename = f"inter_first_ring_mesh_{node_num}nodes_{gpus_per_node}gpus.xml"
    
    # 创建算法实例
    algo = Algo(
        name="inter_first_ring_mesh__allgather",
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
            for channel in range(ring_channels):
                # 第二类：机间环TB
                # 计算机间环的send和recv目标
                if instance%2 == 0:
                    inter_send_rank = get_global_rank((node_id + 1) % node_num, local_rank)  # 下一个节点的相同local_rank
                    inter_recv_rank = get_global_rank((node_id - 1) % node_num, local_rank)  # 上一个节点的相同local_rank
                    
                    channel_id = instance*ring_channels+channel

                    inter_tb = TB(send=inter_send_rank, recv=inter_recv_rank, chan=channel_id)
                    gpu.add_tb(inter_tb)
                    inter_ring_tbs[rank] = inter_tb
                else:

                    inter_send_rank = get_global_rank((node_id - 1) % node_num, local_rank)  # 下一个节点的相同local_rank
                    inter_recv_rank = get_global_rank((node_id + 1) % node_num, local_rank)  # 上一个节点的相同local_rank
                    
                    channel_id = instance*ring_channels+channel

                    inter_tb = TB(send=inter_send_rank, recv=inter_recv_rank, chan=channel_id)
                    gpu.add_tb(inter_tb)
                    inter_ring_tbs[rank] = inter_tb


        # 第一步：本地copy
        copy_steps = []
        for rank in range(ngpus):
            src_chunk = Chunk(gpu_id=rank, chunk_type="i", index=instance, size=1, algo=algo)
            dest_chunk = Chunk(gpu_id=rank, chunk_type="o", index=rank*instances+instance, size=1, algo=algo)
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
                    if instance%2 == 0:
                        next_rank = ring_ranks[(i + 1) % len(ring_ranks)]
                        prev_rank = ring_ranks[(i - 1) % len(ring_ranks)]
                        
                        # 计算要发送的数据的原始owner
                        data_owner_index = (i - step) % len(ring_ranks)
                        data_owner_rank = ring_ranks[data_owner_index]

                        owner_rank_step = (i - step) % len(ring_ranks)
                        send_channel= ring_channels*instance + (owner_rank_step%ring_channels)
                        
                        # 计算要接收的数据的原始owner
                        recv_data_owner_index = ((i - 1) - step) % len(ring_ranks)
                        recv_data_owner_rank = ring_ranks[recv_data_owner_index]


                        recv_rank_step = ((i - 1) - step) % len(ring_ranks)
                        recv_channel = ring_channels*instance + (recv_rank_step%ring_channels)
                        
                        # Send操作
                        src_chunk = Chunk(gpu_id=rank, chunk_type="o", index=data_owner_rank*instances+instance, size=1, algo=algo)
                        send_step = src_chunk.send(dest_rank=next_rank, channel_id= send_channel,
                                                dep_steps=[prev_recv_steps[rank]], bidirectional=False)
                        
                        # Recv操作
                        dest_chunk = Chunk(gpu_id=rank, chunk_type="o", index = recv_data_owner_rank*instances+instance, size=1, algo=algo)
                        recv_step = dest_chunk.recv(src_rank=prev_rank, channel_id = recv_channel , bidirectional=False)
                        
                        current_recv_steps[rank] = recv_step
                        current_recv_data[rank] = recv_data_owner_rank  # 记录接收到的数据的offset
                        
                        # 设置peer关系
                        send_step.peer_step = recv_step
                        recv_step.peer_step = send_step
                    else:

                        next_rank = ring_ranks[(i - 1) % len(ring_ranks)]
                        prev_rank = ring_ranks[(i + 1) % len(ring_ranks)]
                        
                        # 计算要发送的数据的原始owner
                        data_owner_index = (i - step) % len(ring_ranks)
                        data_owner_rank = ring_ranks[data_owner_index]

                        owner_rank_step = (i - step) % len(ring_ranks)
                        send_channel= ring_channels*instance + (owner_rank_step%ring_channels)
                        
                        # 计算要接收的数据的原始owner
                        recv_data_owner_index = ((i - 1) - step) % len(ring_ranks)
                        recv_data_owner_rank = ring_ranks[recv_data_owner_index]


                        recv_rank_step = ((i - 1) - step) % len(ring_ranks)
                        recv_channel = ring_channels*instance + (recv_rank_step%ring_channels)
                        
                        # Send操作
                        src_chunk = Chunk(gpu_id=rank, chunk_type="o", index=data_owner_rank*instances+instance, size=1, algo=algo)
                        send_step = src_chunk.send(dest_rank=next_rank, channel_id= send_channel,
                                                dep_steps=[prev_recv_steps[rank]], bidirectional=False)
                        
                        # Recv操作
                        dest_chunk = Chunk(gpu_id=rank, chunk_type="o", index = recv_data_owner_rank*instances+instance, size=1, algo=algo)
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
                
        # 第三步：机内数据分发
        for node_id in range(node_num):
            print(f"  处理节点{node_id}的机内数据分发")
            node_ranks = [get_global_rank(node_id, local_rank) for local_rank in range(gpus_per_node)]
            
            # 执行node_num个周期的数据分发
            for cycle in range(node_num):
                print(f"    节点{node_id} 机内分发 第{cycle+1}个周期")
                
                # 确定这个周期每个rank要分发的数据和依赖
                cycle_deps = {}
                cycle_data_map = {}
                
                for rank in node_ranks:
                    if cycle == 0:
                        # 第1个周期：依赖本地copy，分发本地数据
                        cycle_deps[rank] = copy_steps[rank]
                        cycle_data_map[rank] = rank  # 分发自己的数据
                        print(f"      rank {rank} 第1周期：依赖本地copy，分发数据offset={rank}")
                    else:
                        # 第N个周期：依赖机间环第(N-1)步，分发机间环收到的数据
                        step_idx = cycle - 1
                        local_rank = get_local_rank(rank)
                        
                        if (local_rank in inter_node_steps and 
                            step_idx in inter_node_steps[local_rank] and
                            rank in inter_node_steps[local_rank][step_idx]):
                            
                            # 依赖机间环的接收操作
                            cycle_deps[rank] = inter_node_steps[local_rank][step_idx][rank]
                            # 分发机间环这一步接收到的数据
                            cycle_data_map[rank] = inter_node_recv_data[local_rank][step_idx][rank]
                            print(f"      rank {rank} 第{cycle+1}周期：依赖机间环{local_rank}第{step_idx+1}步，分发数据offset={cycle_data_map[rank]}")
                        else:
                            # 如果没有对应的机间环数据，跳过这个rank的分发
                            print(f"      rank {rank} 第{cycle+1}周期：无对应机间环数据，跳过分发")
                            continue

                # 执行这个周期的机内数据分发
                print(f"      执行第{cycle+1}周期的机内数据分发...")
                
                # 为每个有数据要分发的rank创建分发操作
                for sender_rank in node_ranks:
                    if sender_rank not in cycle_data_map:
                        continue  # 跳过没有数据要分发的rank
                        
                    data_offset = cycle_data_map[sender_rank]
                    sender_local_rank = get_local_rank(sender_rank)
                    

                    
                    print(f"        rank {sender_rank} 分发数据offset={data_offset} (周期{cycle+1})")
                    
                    # 向节点内所有其他rank发送数据
                    for receiver_rank in node_ranks:
                        if receiver_rank == sender_rank:
                            continue  # 不向自己发送
                        
                        # 使用copy_diff API实现跨rank数据传输
                        src_chunk = Chunk(gpu_id=sender_rank, chunk_type="o", 
                                        index=data_offset*instances+instance, size=1, algo=algo)
                        dest_chunk = Chunk(gpu_id=receiver_rank, chunk_type="o", 
                                        index=data_offset*instances+instance, size=1, algo=algo)
                        
                        # copy_diff会自动创建send和recv操作，并设置peer关系
                        send_step, recv_step = src_chunk.copy_diff(dest_chunk, channel_id=cycle%p2p_channels, 
                                                                dep_steps=[cycle_deps[sender_rank]])
                        
                        
                        print(f"          {sender_rank} -> {receiver_rank}: 数据offset={data_offset}, channel={channel_id}, 依赖step={cycle_deps[sender_rank].s if hasattr(cycle_deps[sender_rank], 's') else 'N/A'}")

 
    # 构建依赖关系
    print("构建依赖关系...")
    algo.build_all_dependencies(merge_rcs=False)
    
    filename = f"ring_mesh_allgather_{node_num}nodes_{gpus_per_node}gpus_{instances}instances_{ring_channels}ringChannels_{p2p_channels}p2p_channels.xml"
    # 保存XML文件
    algo.save_xml(filename)
    print(f"修正后的XML文件已生成: {filename}")
    
    return algo

# 测试
if __name__ == "__main__":

    generate_inter_first_ring_mesh_allgather_xml(node_num=4, gpus_per_node=1, instances=1, ring_channels=1, p2p_channels = 1)
