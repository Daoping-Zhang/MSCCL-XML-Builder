# msccl-xml-generator
# MSCCL-XML-Builder

A Python library for generating Microsoft Collective Communication Library (MSCCL) XML files that define custom collective communication algorithms with fine-grained control over data flow operations.

## Overview

Microsoft Collective Communication Library (MSCCL) is a platform to execute custom collective communication algorithms for multiple accelerators. MSCCL-XML-Builder provides a more flexible and fine-grained alternative to the original MSCCLang tool for generating MSCCL XML configuration files.

### Key Advantages over MSCCLang

1. **Fine-grained API Control**: While MSCCLang only provides copy semantics that are implicitly translated into specific step operations (s, r, copy, rcs), MSCCL-XML-Builder provides direct access to step operations alongside chunk-based abstractions.

2. **Flexible Dependency Management**: Explicit specification of dependencies between any two steps on the same GPU, rather than relying solely on automatic dependency inference based on data relationships.

3. **Advanced ThreadBlock (TB) Management**: Direct TB creation and binding, enabling flexible management of TBs and communication channels.

## Features

- **Chunk-based Operations**: Maintain MSCCLang's intuitive chunk abstraction
- **Direct Step Operations**: Fine-grained control over send, recv, copy, and recv-copy-send (rcs) operations  
- **Explicit Dependencies**: Manual specification of step dependencies for precise control flow
- **ThreadBlock Management**: Flexible TB creation and channel assignment
- **RCS Optimization**: Automatic merging of recv+send operations into optimized rcs operations
- **XML Generation**: Clean, formatted XML output compatible with MSCCL runtime

## Installation

### Local Development Installation

```bash
# Clone or download the project to your local machine
cd /path/to/msccl-xml-builder

# Install in development mode
pip install -e .

# Or if you encounter issues, use:
python setup.py develop
Alternative Setup (if pip install fails)
Add the project directory to your Python path:

import sys
import os
sys.path.insert(0, '/path/to/msccl-xml-builder')

from msccl_xml_builder import Algo, Chunk
Verify Installation
python test_simple.py
Quick Start
from msccl_xml_builder import Algo, Chunk

# Create algorithm with 4 GPUs
algo = Algo(name="ring_allgather", ngpus=4, coll="allgather")

# Define input chunks for each GPU
chunks = []
for gpu_id in range(4):
    chunk = Chunk(gpu_id, "input", gpu_id, 1, algo)
    chunks.append(chunk)

# Define communication pattern (ring topology)
steps = []
for gpu_id in range(4):
    next_gpu = (gpu_id + 1) % 4
    src_chunk = chunks[gpu_id]
    
    # Create output chunk on next GPU
    dst_chunk = Chunk(next_gpu, "output", gpu_id, 1, algo)
    
    send_step, recv_step = src_chunk.copy_diff(dst_chunk, channel_id=0)
    steps.extend([send_step, recv_step])

# Build dependencies and generate XML
algo.build_all_dependencies(merge_rcs=True, sort=True)
algo.save_xml("ring_allgather.xml")
Core Concepts
Chunk
Represents a data chunk with operations like:

copy(): Intra-GPU copy operation
copy_diff(): Inter-GPU copy (generates send+recv pair)
send(): Send operation to another GPU
recv(): Receive operation from another GPU
rcs(): Optimized recv-copy-send operation
Step
Low-level operation unit with types:

s: Send operation
r: Receive operation
cpy: Copy operation
rcs: Recv-copy-send operation
nop: No operation (for dependency management)
ThreadBlock (TB)
Groups steps that execute on the same CUDA threadblock:

Manages send/recv operations for specific GPU pairs
Controls communication channel assignment
Handles step ordering and indexing
Dependencies
Explicit dependency management:

step.add_dep(other_step): Add dependency relationship
Cross-TB dependencies only (intra-TB handled by step ordering)
Same-GPU dependencies only
API Reference
Algo Class
algo = Algo(name="my_algo", ngpus=4, coll="allgather")
algo.build_all_dependencies(merge_rcs=True, sort=True)
algo.save_xml("output.xml")
Chunk Class
chunk = Chunk(gpu_id=0, chunk_type="input", index=0, size=1, algo=algo)

# Operations
copy_step = chunk.copy(dest_chunk, channel_id=0)
send_step, recv_step = chunk.copy_diff(dest_chunk, channel_id=0)
send_step = chunk.send(dest_rank=1, channel_id=0)
recv_step = chunk.recv(src_rank=1, channel_id=0)
rcs_step = chunk.rcs(dest_chunk, intermediate_rank=1, channel_id=0)
Step Class
step = Step(type="s", srcbuf="i", srcoff=0, dstbuf="o", dstoff=0, cnt=1)
step.add_dep(other_step)  # Add dependency
step.remove_dep(other_step)  # Remove dependency
Examples
Basic AllGather (Ring)
from msccl_xml_builder import Algo, Chunk

def create_ring_allgather(ngpus=4):
    algo = Algo(name="ring_allgather", ngpus=ngpus, coll="allgather")
    
    # Each GPU starts with its own chunk
    input_chunks = []
    for gpu_id in range(ngpus):
        chunk = Chunk(gpu_id, "input", gpu_id, 1, algo)
        input_chunks.append(chunk)
    
    # Ring communication: each GPU sends to next GPU
    for step in range(ngpus - 1):
        for gpu_id in range(ngpus):
            src_gpu = (gpu_id - step) % ngpus
            next_gpu = (gpu_id + 1) % ngpus
            
            # Source chunk location
            src_chunk = Chunk(gpu_id, "input" if step == 0 else "output", 
                            src_gpu, 1, algo)
            
            # Destination chunk location  
            dst_chunk = Chunk(next_gpu, "output", src_gpu, 1, algo)
            
            send_step, recv_step = src_chunk.copy_diff(dst_chunk, channel_id=0)
    
    algo.build_all_dependencies(merge_rcs=True)
    return algo

# Generate XML
algo = create_ring_allgather(4)
algo.save_xml("ring_allgather_4gpu.xml")
AllReduce with Dependencies
from msccl_xml_builder import Algo, Chunk

def create_allreduce_with_deps(ngpus=4):
    algo = Algo(name="tree_allreduce", ngpus=ngpus, coll="allreduce")
    
    # Reduce phase: gather to GPU 0
    reduce_steps = []
    for gpu_id in range(1, ngpus):
        src_chunk = Chunk(gpu_id, "input", 0, 1, algo)
        dst_chunk = Chunk(0, "scratch", gpu_id, 1, algo)
        send_step, recv_step = src_chunk.copy_diff(dst_chunk, channel_id=0)
        reduce_steps.extend([send_step, recv_step])
    
    # Broadcast phase: distribute from GPU 0
    broadcast_steps = []
    for gpu_id in range(1, ngpus):
        src_chunk = Chunk(0, "output", 0, 1, algo) 
        dst_chunk = Chunk(gpu_id, "output", 0, 1, algo)
        send_step, recv_step = src_chunk.copy_diff(dst_chunk, channel_id=1)
        
        # Add dependency: broadcast must wait for reduce
        for reduce_step in reduce_steps:
            if reduce_step.type == "r":  # Only depend on recv steps
                send_step.add_dep(reduce_step)
        
        broadcast_steps.extend([send_step, recv_step])
    
    algo.build_all_dependencies(merge_rcs=True)
    return algo

# Generate XML
algo = create_allreduce_with_deps(4)
algo.save_xml("tree_allreduce_4gpu.xml")
Project Structure
msccl-xml-builder/
├── README.md
├── setup.py
├── requirements.txt
├── test_simple.py           # Basic import and functionality test
├── msccl_xml_builder/
│   ├── __init__.py
│   └── core/
│       ├── __init__.py
│       ├── step.py          # Step class definition
│       ├── tb.py            # ThreadBlock class
│       ├── gpu.py           # GPU class
│       ├── algo.py          # Algorithm class
│       └── chunk.py         # Chunk class with operations
Development
Running Tests
# Basic functionality test
python test_simple.py

# Create your own test
python test_installation.py
Creating Custom Algorithms
# Example: Custom algorithm with manual TB management
from msccl_xml_builder.core import TB, Step

# Create custom TB
tb = TB(send=1, recv=2, chan=0)
gpu.add_tb(tb)

# Add steps manually
step1 = Step(type="r", srcbuf="i", srcoff=-1, dstbuf="s", dstoff=0, cnt=1)
step2 = Step(type="s", srcbuf="s", srcoff=0, dstbuf="o", dstoff=-1, cnt=1)

tb.add_step(step1)
tb.add_step(step2)

# Add cross-TB dependency
step2.add_dep(other_tb_step)
Troubleshooting
Common Issues
Q: Getting "Cross-GPU dependency is not allowed" error
A: Dependencies can only be added between steps on the same GPU. Use proper data flow through send/recv operations for cross-GPU coordination.

Q: Channel conflict errors
A: Each channel on a GPU can only have one TB sending to a specific rank and one TB receiving from a specific rank.

Q: Step index mismatches
A: Ensure send and recv steps have matching indices. Use bidirectional=True (default) for automatic pairing.

Q: Import errors during installation
A: Try the alternative setup method by adding the project path to sys.path, or use python setup.py develop instead of pip install -e .

Debug Mode
# Enable detailed logging for dependency resolution
algo.build_all_dependencies(merge_rcs=True, sort=True)
Limitations
Currently supports Python 3.7+
Dependencies are limited to same-GPU steps
XML output format is fixed (MSCCL-compatible)
No built-in algorithm validation (coming in future versions)
License
This project is licensed under the MIT License.

Acknowledgments
Microsoft Research for the MSCCL framework
The MSCCLang research that inspired this work
Contributors to the MSCCL ecosystem