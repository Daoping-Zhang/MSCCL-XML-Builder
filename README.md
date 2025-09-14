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


## Installation

### Direct Installation from GitHub

To install the latest version directly from the repository:

```bash
pip3 install git+https://github.com/Daoping-Zhang/MSCCL-XML-Builder.git

```


### Local Development Installation

For contributing to development or modifying the code:


```bash
# Clone the repository
git clone https://github.com/Daoping-Zhang/MSCCL-XML-Builder.git
cd msccl-xml-builder

# Install in editable/development mode
pip3 install -e .

```

## Troubleshooting
Q: Getting "Cross-GPU dependency is not allowed" error
A: Dependencies can only be added between steps on the same GPU. Use proper data flow through send/recv operations for cross-GPU coordination.

Q: Channel conflict errors
A: Each channel on a GPU can only have one TB sending to a specific rank and one TB receiving from a specific rank.

Q: Step index mismatches
A: Ensure send and recv steps have matching indices. Use bidirectional=True (default) for automatic pairing.

Q: Import errors during installation
A: Try the alternative setup method by adding the project path to sys.path, or use python setup.py develop instead of pip install -e .


## Limitations

- No built-in algorithm validation (coming in future versions)
## License
This project is licensed under the MIT License.

## Acknowledgments
- Microsoft Research for the MSCCL framework
- The MSCCLang research that inspired this work
- Contributors to the MSCCL ecosystem