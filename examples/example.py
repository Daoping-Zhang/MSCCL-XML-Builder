# 使用示例
from msccl_xml_builder import Algo, Chunk,TB

if __name__ == "__main__":
    # 创建算法

    ngpus=2
    chunk=ngpus
    algo = Algo(name="ag_test", proto = "Simple", nchannels = 1, 
                 nchunksperloop = chunk, ngpus = ngpus, coll = "allgather",
                 inplace = 1, outofplace = 1, minBytes = 0, maxBytes = 0)
    
    #显示的创建本地copytb
    local_tbs = {}      # {rank: tb} - 本地操作TB
    for rank in range(ngpus):

        local_tb = TB(send=-1, recv=-1, chan=0)
        gpu = algo.get_gpu(rank)

        gpu.add_tb(local_tb)
        local_tbs[rank] = local_tb
        
    # 本地copy
    copy_steps = []
    for rank in range(ngpus):
        src_chunk = Chunk(gpu_id=rank, chunk_type="i", index=0, size=1, algo=algo)
        dest_chunk = Chunk(gpu_id=rank, chunk_type="o", index=rank, size=1, algo=algo)
        copy_step = src_chunk.copy(dest_chunk, channel_id=0)
        copy_steps.append(copy_step)

    #创建P2P tb
    rank = 0
    
    peer_rank=1

    gpu = algo.get_gpu(rank)
    
    peer_gpu = algo.get_gpu(peer_rank)

    tb = TB(send=peer_rank, recv=peer_rank, chan=0)
    
    peer_tb = TB(send=rank, recv=rank, chan=0)

    gpu.add_tb(tb)

    peer_gpu.add_tb(peer_tb)

    #数据发送，并设置依赖

    src_chunk = Chunk(gpu_id=rank, chunk_type="o", index=rank, size=1, algo=algo)
    dest_chunk = Chunk(gpu_id=peer_rank, chunk_type="o", index=peer_rank, size=1, algo=algo)

    send_step = src_chunk.send(dest_rank=peer_rank, channel_id= 0, dep_steps=[copy_steps[rank]])
    recv_step = dest_chunk.recv(src_rank=rank, channel_id= 0)

    #需要绑定send recv
    send_step.peer_step = recv_step
    recv_step.peer_step = send_step

    

    recv_step.add_dep(copy_steps[peer_rank]) #支持后续显示的添加依赖

    #可以通过copy_diff 实现send recv

    src_chunk = Chunk(gpu_id=peer_rank, chunk_type="o", index=peer_rank, size=1, algo=algo)
    dest_chunk = Chunk(gpu_id=rank, chunk_type="o", index=rank, size=1, algo=algo)

    peer_send_step, peer_recv_step = src_chunk.copy_diff(dest_chunk, channel_id=0, dep_steps=[copy_steps[peer_rank]])

    algo.build_all_dependencies() #需要显示的调用依赖构建
    
    # 保存XML
    algo.save_xml("output.xml")
    print("XML文件已生成: output.xml")