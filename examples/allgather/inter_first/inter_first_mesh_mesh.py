from xml_generator_tools import Algo, Chunk
from xml_generator_tools import TB

def generate_inter_first_mesh_mesh_allgather_xml(node_num: int = 4, gpus_per_node: int = 4, instances:int = 1, ring_channels:int = 1,p2p_channels:int = 1, proto: str = "LL128",filename: str = None):
    """
    正确实现双结构Ring AllGather算法 - 修正机内环数据传播逻辑
    """
    ngpus = node_num * gpus_per_node


    channels_per_instance = max(ring_channels,p2p_channels)

    nchannls = instances*channels_per_instance

    nchunksperloop=ngpus*instances
    
    if filename is None:
        filename = f"inter_first_mesh_mesh_{node_num}nodes_{gpus_per_node}gpus.xml"
    
    # 创建算法实例
    algo = Algo(
        name="inter_first_mesh_mesh_allgather",
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
    


    data_index_map = {}
    recv_step_map = {}
            # 初始化两个列表，每个 GPU 对应一个空列表
    for instance in range(instances):
        data_index_map[instance] = [[] for _ in range(ngpus)]
        recv_step_map[instance] = [[] for _ in range(ngpus)]

    for instance in range(instances):
    
        local_tbs = {}      # {rank: tb} - 本地操作TB
        inter_ring_tbs = {} # {rank: tb} - 机间环TB  
        intra_ring_tbs = {} # {rank: tb} - 机内环TB
        

            
                


        # 第一步：本地copy
        copy_steps = []
        for rank in range(ngpus):
            src_chunk = Chunk(gpu_id=rank, chunk_type="i", index=instance, size=1, algo=algo)
            dest_chunk = Chunk(gpu_id=rank, chunk_type="o", index=rank*instances+instance, size=1, algo=algo)
            copy_step = src_chunk.copy(dest_chunk, channel_id=instance)
            copy_steps.append(copy_step)
            data_index_map[instance][rank].append(rank*instances+instance)
            recv_step_map[instance][rank].append(copy_step)
        
        # 第二步：机间mesh传输


        for src_node in range(node_num):
            for peer_node in range(node_num):

                if  src_node == peer_node:
                    continue

                for local_rank in range(gpus_per_node):

                    src_rank = src_node*gpus_per_node + local_rank

                    peer_rank = peer_node*gpus_per_node + local_rank

                    src_chunk = Chunk(gpu_id=src_rank, chunk_type="o", index=src_rank*instances+instance, size=1, algo=algo)
                    dest_chunk = Chunk(gpu_id=peer_rank, chunk_type="o", index = src_rank*instances+instance, size=1, algo=algo)

                    send_step, recv_step = src_chunk.copy_diff(dest_chunk, channel_id=0,dep_steps=[copy_steps[src_rank]]) 
                                                                
                    data_index_map[instance][peer_rank].append(src_rank*instances+instance)
                    recv_step_map[instance][peer_rank].append(recv_step)
        

                
        # 第三步：机内数据分发

    for node_id in range(node_num):
        print(f"  处理节点{node_id}的机内数据分发")
        node_ranks = [get_global_rank(node_id, local_rank) for local_rank in range(gpus_per_node)]
            
  
                
                # 为每个有数据要分发的rank创建分发操作
        for sender_rank in node_ranks:
            for receiver_rank in node_ranks:
                if receiver_rank == sender_rank:
                    continue

                for i in range(instances*node_num):
                        
                        print(f"i:{i} instance:{i%instances} index:{i//instances}")
                        
                        data_offset = data_index_map[i%instances][sender_rank][i//instances]

                        dep_recv_step = recv_step_map[i%instances][sender_rank][i//instances]
                    

                        
                        # 使用copy_diff API实现跨rank数据传输
                        src_chunk = Chunk(gpu_id=sender_rank, chunk_type="o", 
                                        index=data_offset, size=1, algo=algo)
                        dest_chunk = Chunk(gpu_id=receiver_rank, chunk_type="o", 
                                        index=data_offset, size=1, algo=algo)
                        
                        # copy_diff会自动创建send和recv操作，并设置peer关系
                        send_step, recv_step = src_chunk.copy_diff(dest_chunk, channel_id=0, 
                                                                dep_steps=[dep_recv_step])
                        
                        

 
    # 构建依赖关系
    print("构建依赖关系...")
    algo.build_all_dependencies(True)
    
    filename = f"mesh_mesh_allgather_{node_num}nodes_{gpus_per_node}gpus_{instances}instances_{ring_channels}ringChannels_{p2p_channels}p2p_channels.xml"
    # 保存XML文件
    algo.save_xml(filename)
    print(f"修正后的XML文件已生成: {filename}")
    
    return algo

# 测试
if __name__ == "__main__":
    # 生成小规模测试版本
    print("生成修正版本...")


    generate_inter_first_mesh_mesh_allgather_xml(node_num=16, gpus_per_node=8, instances=1, ring_channels=1, p2p_channels = 1)
