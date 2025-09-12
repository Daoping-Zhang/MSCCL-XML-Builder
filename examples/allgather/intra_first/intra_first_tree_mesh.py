from xml_generator_tools import Algo, Chunk
from xml_generator_tools import TB

def generate_intra_first_tree_mesh_allgather_xml(node_num: int = 4, gpus_per_node: int = 4, instances:int = 1, ring_channels:int = 1,p2p_channels:int = 1, proto: str = "LL128",filename: str = None):
    """
    正确实现双结构Ring AllGather算法 - 修正机内环数据传播逻辑
    """
    ngpus = node_num * gpus_per_node


    channels_per_instance = max(ring_channels,p2p_channels)

    nchannls = instances*channels_per_instance

    nchunksperloop=ngpus*instances
    
    if filename is None:
        filename = f"intra_first_tree_mesh_{node_num}nodes_{gpus_per_node}gpus.xml"
    
    # 创建算法实例
    algo = Algo(
        name="intra_first_tree_mesh_allgather",
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

    # 第一步：本地copy
    copy_steps = []
    for rank in range(ngpus):
        src_chunk = Chunk(gpu_id=rank, chunk_type="i", index=0, size=1, algo=algo)
        dest_chunk = Chunk(gpu_id=rank, chunk_type="o", index=rank, size=1, algo=algo)
        copy_step = src_chunk.copy(dest_chunk, channel_id=0)
        copy_steps.append(copy_step)
        data_index_map[rank].append(rank)
        recv_step_map[rank].append(copy_step)

    # 第二步：机内Mesh算法数据聚合
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


        

            
                



    
    # 第二步：机间Tree算法数据聚合

    count = 1
    pre_tree_recv_steps = {}

    while count < node_num:

        current_recv_steps = {}



        for src_node in range(node_num):

            
            peer_node = src_node^count

            for local_rank in range(gpus_per_node):

                src_rank = src_node*gpus_per_node + local_rank

                peer_rank = peer_node*gpus_per_node + local_rank

                send_index = (src_node // count)*gpus_per_node * count

                if count == 1:

                    src_chunk = Chunk(gpu_id=src_rank, chunk_type="o", index=send_index, size=count*gpus_per_node, algo=algo)

                    dst_chunk = Chunk(gpu_id=peer_rank, chunk_type="o", index=send_index, size=count*gpus_per_node, algo=algo)

                    send_step, recv_step = src_chunk.copy_diff(dst_chunk, channel_id=0, dep_steps=recv_step_map[src_rank])
                    
                    current_recv_steps[peer_rank] = recv_step
                else:

                    src_chunk = Chunk(gpu_id=src_rank, chunk_type="o", index=send_index, size=count*gpus_per_node, algo=algo)

                    dst_chunk = Chunk(gpu_id=peer_rank, chunk_type="o", index=send_index, size=count*gpus_per_node, algo=algo)

                    send_step, recv_step = src_chunk.copy_diff(dst_chunk, channel_id=0, dep_steps=[pre_tree_recv_steps[src_rank]])
                    current_recv_steps[peer_rank] = recv_step                    




                print(f"cunt:{count} src_node:{src_node} local_rank:{local_rank} sendindex:{send_index}")
        
        count  *= 2

        for rank in range(ngpus):

            pre_tree_recv_steps[rank] = current_recv_steps[rank]
                                                                

        

                



                        
                        

 
    # 构建依赖关系
    print("构建依赖关系...")
    algo.build_all_dependencies(True)
    
    filename = f"intra_first_tree_mesh_{node_num}nodes_{gpus_per_node}gpus_{instances}instances_{ring_channels}ringChannels_{p2p_channels}p2p_channels.xml"
    # 保存XML文件
    algo.save_xml(filename)
    print(f"修正后的XML文件已生成: {filename}")
    
    return algo

# 测试
if __name__ == "__main__":

    generate_intra_first_tree_mesh_allgather_xml(node_num=16, gpus_per_node=8, instances=1, ring_channels=1, p2p_channels = 1)
