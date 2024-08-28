
# MLIR for Dummies

### What is MLIR?
MLIR (Multi-Level Intermediate Representation) is a framework in the LLVM project that helps in creating and optimizing compilers. It enables the use of several intermediate representations at various levels, making it easier to implement domain specific optimizations and hence create Domain Specific Compilers. 

It can be used to create compiler frontends as well as backends.

```mermaid
flowchart LR
    subgraph Applications/Compilers
        HLS_Chisel["HLS/Chisel"]
        ONNX["ONNX"]
        PyTorch["PyTorch"]
        TensorFlow["TensorFlow"]
    end

    subgraph MLIR
        subgraph Dialects
            affine["affine"]
            arith["arith"]
            scf["scf"]
        end
        Dialect["Dialect"]
        Shared_Optimizations["Shared Optimizations"]
    end

    subgraph Backends
        LLVM_IR["LLVM IR"]
        CIRCT["CIRCT (FIRRTL)"]
        SPIRV["SPIR-V for GPU"]
        TPU_IR["TPU IR"]
    end

    subgraph Hardware_Devices
        CPU["CPU"]
        GPU["GPU"]
        FPGA["FPGA"]
        TPU["TPU"]
        XPU["XPU"]
    end

    HLS_Chisel --> MLIR
    ONNX --> MLIR
    PyTorch --> MLIR
    TensorFlow --> MLIR
    MLIR --> Backends
    Backends --> Hardware_Devices
```

