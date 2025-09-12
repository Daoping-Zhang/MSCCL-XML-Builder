from xml_generator_tools import Algo, Chunk

def generate_allgather_ring_xml(ngpus: int = 128, instances: int = 1, ring_channels: int = 1, filename: str = None):
    """
    生成AllGather Ring算法的XML文件
    
    Args:
        ngpus: GPU数量
        nchunksperloop: 每轮循环的chunk数量
        filename: 输出文件名，如果为None则自动生成
    """
    if filename is None:
        filename = f"allgather_ring_oop_{ngpus}gpus.xml"
    
    channels = ring_channels*instances
    nchunksperloop = instances*ngpus

    # 创建算法实例
    algo = Algo(
        name="allgather_ring_oop",
        proto="Simple", 
        nchannels=channels,
        nchunksperloop=nchunksperloop,
        ngpus=ngpus,
        coll="allgather",
        inplace=1,
        outofplace=1,
        minBytes=0,
        maxBytes=0
    )
    for instance in range(instances):
        # 第一步：为每个GPU创建本地copy操作
        copy_steps = []
        for rank in range(ngpus):
            src_chunk = Chunk(gpu_id=rank, chunk_type="i", index=instance, size=1, algo=algo)
            dest_chunk = Chunk(gpu_id=rank, chunk_type="o", index=rank*instances+instance, size=1, algo=algo)
            copy_step = src_chunk.copy(dest_chunk, channel_id=instance)
            copy_steps.append(copy_step)
        
        # 第二步：为每个GPU创建tb 1（ring连接）
        for rank in range(ngpus):
            from xml_generator_tools import TB
            gpu = algo.get_gpu(rank)
            next_rank = (rank + 1) % ngpus
            prev_rank = (rank - 1) % ngpus
            
            for channel in range(ring_channels):
                # 创建tb 1
                tb1 = TB(send=next_rank, recv=prev_rank, chan=ring_channels*instance + channel)
                #print(f"creat tb send {next_rank} recv {prev_rank} channel {ring_channels*instance + channel}")
                gpu.add_tb(tb1)
        
        # 第三步：执行(ngpus-1)轮ring传输
        # 记录每个rank在每轮的recv step，用于下一轮的依赖
        recv_steps_by_rank = [None] * ngpus
        
        for round_num in range(ngpus - 1):
            current_send_steps = []
            current_recv_steps = []
            
            for rank in range(ngpus):
                next_rank = (rank + 1) % ngpus
                prev_rank = (rank - 1) % ngpus
                
                # 计算当前轮次要传输的数据的原始owner
                data_owner = (rank - round_num) % ngpus
                
                # 创建send操作
                src_chunk = Chunk(gpu_id=rank, chunk_type="o", index=data_owner*instances+instance, size=1, algo=algo)
                
                dep_steps = []
                if round_num == 0:
                    # 第一轮依赖本地copy
                    dep_steps = [copy_steps[rank]]
                else:
                    # 后续轮次依赖上一轮该rank的recv操作
                    if recv_steps_by_rank[rank] is not None:
                        dep_steps = [recv_steps_by_rank[rank]]
                
                send_step = src_chunk.send(dest_rank=next_rank, channel_id= ring_channels*instance + (data_owner%ring_channels), dep_steps=dep_steps, bidirectional=False)
                current_send_steps.append(send_step)
                
                # 创建recv操作
                # 当前rank接收的数据owner是prev_rank在当前轮次发送的数据owner
                recv_data_owner = (prev_rank - round_num) % ngpus
                dest_chunk = Chunk(gpu_id=rank, chunk_type="o", index=recv_data_owner*instances+instance, size=1, algo=algo)

                recv_step = dest_chunk.recv(src_rank=prev_rank, channel_id=instance*ring_channels+(recv_data_owner%ring_channels), bidirectional=False)
                current_recv_steps.append(recv_step)
                
                # 设置peer关系
                send_step.peer_step = recv_step
                recv_step.peer_step = send_step
            
            # 更新recv_steps_by_rank为下一轮做准备
            recv_steps_by_rank = current_recv_steps.copy()
    
    # 构建所有依赖关系
    algo.build_all_dependencies(True)
    filename = f"ring_instances{instances}_ringchannels{ring_channels}_gpus{ngpus}.xml"
    # 保存XML文件
    algo.save_xml(filename)
    print(f"XML文件已生成: {filename}")
    
    return algo

# 主函数
if __name__ == "__main__":

    

    generate_allgather_ring_xml(ngpus=128, instances=1, ring_channels=2)

    
