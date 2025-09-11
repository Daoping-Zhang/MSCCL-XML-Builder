import os
import sys

# 定义要创建的目录和文件列表（去掉外层的 msccl-xml-builder）
directories = [
    "examples",
    "msccl_xml_builder/core",
    "msccl_xml_builder/utils",
    "tests",
    "docs"
]

files = [
    "README.md",
    "README_CN.md",
    "setup.py",
    "requirements.txt",
    "LICENSE",
    ".gitignore",
    "examples/__init__.py",
    "examples/basic_allgather.py",
    "examples/advanced_allreduce.py",
    "msccl_xml_builder/__init__.py",
    "msccl_xml_builder/core/__init__.py",
    "msccl_xml_builder/core/step.py",
    "msccl_xml_builder/core/tb.py",
    "msccl_xml_builder/core/gpu.py",
    "msccl_xml_builder/core/algo.py",
    "msccl_xml_builder/core/chunk.py",
    "msccl_xml_builder/utils/__init__.py",
    "msccl_xml_builder/utils/xml_formatter.py",
    "tests/__init__.py",
    "tests/test_step.py",
    "tests/test_tb.py",
    "tests/test_gpu.py",
    "tests/test_algo.py",
    "tests/test_chunk.py",
    "docs/api_reference.md",
    "docs/user_guide.md",
    "docs/examples.md"
]

def create_directories():
    """创建所有需要的目录"""
    for dir_path in directories:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"创建目录: {dir_path}")
        else:
            print(f"目录已存在: {dir_path}")

def create_files():
    """创建所有需要的文件"""
    for file_path in files:
        if not os.path.exists(file_path):
            # 确保父目录存在
            parent_dir = os.path.dirname(file_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            
            # 创建空文件
            with open(file_path, 'w') as f:
                pass
            print(f"创建文件: {file_path}")
        else:
            print(f"文件已存在: {file_path}")

def main():
    print("开始创建项目结构...")
    print(f"当前工作目录: {os.getcwd()}")
    create_directories()
    create_files()
    print("\n✅ 项目结构创建完成！")

if __name__ == "__main__":
    main()