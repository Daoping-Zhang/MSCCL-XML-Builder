from xml_generator_tools import Algo, Chunk
import math

def generate_communication_phases(node_nums: int):
    """
    生成避免incast的通信调度方案
    """
    if node_nums & (node_nums - 1) != 0:
        raise ValueError("节点数必须是2的整数次方")
    
    if node_nums < 2:
        return []
    
    schedule = []
    
    for phase in range(node_nums - 1):
        pairs = []
        used = set()
        
        for node in range(node_nums):
            if node in used:
                continue
                
            partner = (node + phase + 1) % node_nums
            
            if partner not in used:
                pairs.append((min(node, partner), max(node, partner)))
                used.add(node)
                used.add(partner)
        
        schedule.append(pairs)
        print(f"  阶段 {phase + 1}: {pairs}")
    
    return schedule



def generate_alltoall_2step_xml(node_nums: int = 16, gpus_pernode: int = 8, instances: int = 1, filename: str = None):
    """
    生成两阶段AllToAll算法的XML文件
    
    添加依赖关系：
    - 第二阶段的转发操作必须等待第一阶段的接收操作完成
    - 确保scratch buffer中有数据后才能转发
    """
    
    ngpus = node_nums * gpus_pernode
    nchunksperloop = ngpus * instances
    nchannels = instances
    
    if filename is None:
        filename = f"alltoall_2step_dep_{node_nums}nodes_{gpus_pernode}gpus_{instances}instances.xml"
    
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
                            copy_step = src_chunk.copy(dest_chunk, channel_id=instance)
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
                            send_step, recv_step = src_chunk.copy_diff(dest_chunk, channel_id=instance)
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
                        send_step, recv_step = src_chunk.copy_diff(dest_chunk, channel_id=instance)
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
                        
                        send_step, recv_step = src_chunk.copy_diff(dest_chunk, channel_id=instance)
                        
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
                        
                            send_step, recv_step = src_chunk.copy_diff(dest_chunk, channel_id=instance, dep_steps=dep_steps)
                            print(f"    转发: rank{global_rank} scratch[{scratch_index}] -> rank{dest_global_rank} output[{src_global_rank * instances + instance}]")
                            
                            # ===== 存储第二阶段的发送和接收操作 =====
                            # 初始化嵌套字典结构
                            if node_id not in send_map:
                                send_map[node_id] = {}
                            if dest_node not in send_map[node_id]:
                                send_map[node_id][dest_node] = {}
                            if local_rank not in send_map[node_id][dest_node]:
                                send_map[node_id][dest_node][local_rank] = []

                            if dest_node not in recv_map:
                                recv_map[dest_node] = {}

                            if node_id not in recv_map[dest_node]:
                                recv_map[dest_node][node_id] = {}

                            if local_rank not in recv_map[dest_node][node_id]:
                                recv_map[dest_node][node_id][local_rank] = []
                            
                            # 存储发送和接收操作（按顺序添加到列表中）
                            send_map[node_id][dest_node][local_rank].append(send_step)

                            recv_map[dest_node][node_id][local_rank].append(recv_step)
                            
                            print(f"      存储到send_map[{node_id}][{dest_node}][{local_rank}][{len(send_map[node_id][dest_node][local_rank])-1}]")


        # ===== 第三阶段：设置分阶段通信的依赖关系 =====
        print(f"\n第三阶段：设置分阶段通信的依赖关系")    
        # 生成通信阶段分组

        comm_phases = generate_communication_phases(node_nums)

# ===== 第三阶段：设置分阶段通信的依赖关系 =====
        print(f"\n第三阶段：设置分阶段通信的依赖关系")
        
        # 按通信阶段设置依赖关系
        for phase_idx, phase_pairs in enumerate(comm_phases):
            print(f"\n  === 处理通信阶段 {phase_idx} ===")
            print(f"  通信对: {phase_pairs}")
            
            if phase_idx == 0:
                print("    第一阶段无需额外依赖")
                continue
            
            # 获取上一阶段的通信对
            prev_phase_pairs = comm_phases[phase_idx - 1]
            print(f"  上一阶段通信对: {prev_phase_pairs}")
            
            # 为当前阶段的每个通信对设置依赖关系
            for src_node, dest_node in phase_pairs:
                print(f"\n    设置通信对 ({src_node}, {dest_node}) 的依赖关系")
                
                # 处理双向通信的依赖
                for direction in [(src_node, dest_node),(dest_node,src_node)]:
                    sender_node, receiver_node = direction
                    print(f"      处理方向: node{sender_node} -> node{receiver_node}")
                    
                    # 找到上一阶段中该sender_node参与的通信对
                    prev_receiver_node = None
                    for prev_src, prev_dest in prev_phase_pairs:
                        if prev_src == sender_node:
                            prev_receiver_node = prev_dest
                            break
                        elif prev_dest == sender_node:
                            prev_receiver_node = prev_src
                            break
                    
                    if prev_receiver_node is None:
                        print(f"        未找到node{sender_node}在上一阶段的通信对")
                        continue
                    
                    print(f"        上一阶段: node{sender_node} -> node{prev_receiver_node}")
                    
                    for local_rank in range(gpus_pernode):
                        # 当前阶段的第一个转发step
                        if (sender_node in send_map and 
                            receiver_node in send_map[sender_node] and 
                            local_rank in send_map[sender_node][receiver_node] and
                            len(send_map[sender_node][receiver_node][local_rank]) > 0):
                            
                            current_first_send = send_map[sender_node][receiver_node][local_rank][0]
                            current_first_recv = recv_map[sender_node][receiver_node][local_rank][0]
                            
                            # 上一阶段的最后一个转发step
                            if (sender_node in send_map and 
                                prev_receiver_node in send_map[sender_node] and 
                                local_rank in send_map[sender_node][prev_receiver_node] and
                                len(send_map[sender_node][prev_receiver_node][local_rank]) > 0):
                                
                                prev_last_send = send_map[sender_node][prev_receiver_node][local_rank][-1]
                                
                                prev_last_recv = recv_map[sender_node][prev_receiver_node][local_rank][-1]
                                
                                # 添加发送依赖
                                if current_first_send._get_tb() != prev_last_send._get_tb():
                                    current_first_send.add_dep(prev_last_send)
                                    print(f"          ✓ local_rank{local_rank}_添加发送依赖: TB{current_first_send._get_tb().id}-step{current_first_send.s} 依赖 TB{prev_last_send._get_tb().id}-step{prev_last_send.s}")
                                
                                # 添加接收依赖
                                if current_first_recv._get_tb() != prev_last_recv._get_tb():
                                    current_first_recv.add_dep(prev_last_recv)
                                    print(f"          ✓ local_rank{local_rank}_添加接收依赖: TB{current_first_recv._get_tb().id}-step{current_first_recv.s} 依赖 TB{prev_last_recv._get_tb().id}-step{prev_last_recv.s}")
                            else:
                                print(f"        未找到上一阶段的转发step")
                        else:
                            print(f"        未找到当前阶段的转发step")
    print("\n构建依赖关系...")
    algo.build_all_dependencies()
    
    print(f"保存XML文件: {filename}")
    algo.save_xml(filename)
    print(f"两阶段AllToAll XML文件已生成: {filename}")
    
    return algo


# 主函数
if __name__ == "__main__":
    print("=== 生成两阶段AllToAll算法 ===")
    

    generate_alltoall_2step_xml(node_nums=16, gpus_pernode=8, instances=8)