from xml_generator_tools import Algo, Chunk

def generate_inter_first_tree_mesh_allgather_xml(node_num: int = 4, gpus_per_node: int = 4, instances:int = 1, ring_channels:int = 1,p2pchannels:int =1, proto: str = "LL128",filename: str = None):
    """
    正确实现双结构Ring AllGather算法 - 修正机内环数据传播逻辑
    """
    ngpus = node_num * gpus_per_node


    nchannls = instances*max(ring_channels,p2pchannels)

    #nchannls = 2

    #instances = 1

    nchunksperloop=ngpus*instances
    
    if filename is None:
        filename = f"inter_first_tree_mesh_{node_num}nodes_{gpus_per_node}gpus.xml"
    
    # 创建算法实例
    algo = Algo(
        name="inter_first_tree_mesh_allgather",
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
    
    from xml_generator_tools import TB

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
            
            
        # 初始化两个列表，每个 GPU 对应一个空列表
        data_index_map = [[] for _ in range(ngpus)]
        recv_step_map = [[] for _ in range(ngpus)]
                

        # 初始化两个列表，每个 GPU 对应一个空列表
        data_index_map = [[] for _ in range(ngpus)]
        recv_step_map = [[] for _ in range(ngpus)]

        # 第一步：本地copy
        copy_steps = []
        for rank in range(ngpus):
            src_chunk = Chunk(gpu_id=rank, chunk_type="i", index=instance, size=1, algo=algo)
            dest_chunk = Chunk(gpu_id=rank, chunk_type="o", index=rank*instances+instance, size=1, algo=algo)

            copy_step = src_chunk.copy(dest_chunk, channel_id=instance)

            data_index_map[rank].append(rank*instances+instance)
            recv_step_map[rank].append(copy_step)

            copy_steps.append(copy_step)
        
        # 第二步：机间Tree传输


        count = 1

        while count < node_num:



            tmp_data_index_map = [[] for _ in range(ngpus)]
            tmp_recv_step_map = [[] for _ in range(ngpus)]

            for src_node in range(node_num):

                
                peer_node = src_node^count

                for local_rank in range(gpus_per_node):

                    src_rank = src_node*gpus_per_node + local_rank

                    peer_rank = peer_node*gpus_per_node + local_rank

                    for i in range(len(data_index_map[src_rank])):

                        send_index = data_index_map[src_rank][i]
                        src_chunk = Chunk(gpu_id=src_rank, chunk_type="o", index=send_index, size=1, algo=algo)

                        dst_chunk = Chunk(gpu_id=peer_rank, chunk_type="o", index=send_index, size=1, algo=algo)

                        send_step, recv_step = src_chunk.copy_diff(dst_chunk, channel_id=instance, dep_steps=[recv_step_map[src_rank][i]])

                        tmp_data_index_map[peer_rank].append(send_index)
                        tmp_recv_step_map[peer_rank].append(recv_step)

                        print(f"cunt:{count} src_node:{src_node} local_rank:{local_rank} sendindex:{send_index}")
            
            count  *= 2
            for rank in range(ngpus):
                data_index_map[rank].extend(tmp_data_index_map[rank])
                recv_step_map[rank].extend(tmp_recv_step_map[rank])
        
        # 第三步：机内数据分发
        for node_id in range(node_num):
            print(f"  处理节点{node_id}的机内数据分发")
            node_ranks = [get_global_rank(node_id, local_rank) for local_rank in range(gpus_per_node)]

            for sender_rank in node_ranks:
                for receiver_rank in node_ranks:
                    if receiver_rank == sender_rank:
                        continue  # 不向自己发送

                    for i in range(len(data_index_map[sender_rank])):
                        
                        data_index = data_index_map[sender_rank][i]
                        dep_recv_step = recv_step_map[sender_rank][i]
                        src_chunk = Chunk(gpu_id=sender_rank, chunk_type="o", 
                                                index=data_index, size=1, algo=algo)
                        dest_chunk = Chunk(gpu_id=receiver_rank, chunk_type="o", 
                                                index=data_index, size=1, algo=algo)
                        
                        send_step, recv_step = src_chunk.copy_diff(dest_chunk, channel_id=instance*p2pchannels + i%p2pchannels, 
                                                                    dep_steps=[dep_recv_step])


 
    # 构建依赖关系
    print("构建依赖关系...")
    algo.build_all_dependencies(merge_rcs=True,sort=True)
    
    #filename = f"ring_mesh_allgather_diifchannel_{node_num}nodes_{gpus_per_node}gpus_{instances}instances_{ring_channels}ringChannels.xml"

    filename = f"inter_first_tree_mesh_{ngpus}gpus_{instances}instances_{p2pchannels}p2pchannels.xml"

    # 保存XML文件
    algo.save_xml(filename)
    print(f"修正后的XML文件已生成: {filename}")
    
    return algo

# 测试
if __name__ == "__main__":
    # 生成小规模测试版本
    print("生成修正版本...")



    generate_inter_first_tree_mesh_allgather_xml(node_num=8, gpus_per_node=4, instances=1, ring_channels=1,p2pchannels=2)




