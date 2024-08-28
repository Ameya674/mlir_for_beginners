
# MLIR for Beginners

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
        direction LR
        LLVM_IR["LLVM IR"]
        CIRCT["CIRCT (FIRRTL)"]
        SPIRV["SPIR-V for GPU"]
        TPU_IR["TPU IR"]
    end

    subgraph Hardware_Devices
        direction LR
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

### What makes MLIR special?

#### Multilevel IR
Most compilers convert high-level programming languages into an Intermediate Representation (IR), like LLVM IR. However, this process often misses out on domain-specific optimizations and can make custom optimizations challenging. MLIR (Multi-Level Intermediate Representation) addresses this by offering multiple levels of IR. It gradually translates high-level code down through these levels, allowing for domain-specific and high-level optimizations at each stage. This results in a more optimized and efficient IR tailored to specific needs.

But this concept isn't new and can be implemented with any tool or language and not just MLIR. 

#### Same syntax for defining all dialects
MLIR lower languages in the form of dialects(which is basically another language or IR) which consists of operations where you can define your own operations. All these dialects have the same syntax, which makes writing multiple dialects super easy. This is what makes MLIR really powerful.

##### Tensorflow
```bash
%x = "tf.Conv2d"(%input, %filter)
   {strides: [1,1,2,1], padding: "SAME", dilations: [2,1,1,1]}
   : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
```
##### XLA HLO
```bash
%m = "xla.AllToAll"(%z)
   {split_dimension: 1, concat_dimension: 0, split_count: 2}
   : (memref<300x200x32xf32>) -> memref<600x100x32xf32>
```
##### LLVM IR
```bash
%f = llvm.add %a, %b
   : !llvm.float
```
### Operation Structure

All dialects, which are just a collection of operations are written in the following syntax. 

```bash
%res:2 = "mydialect.morph"(%input#3) { some_attribute = true, other_attribute = 1.5 } \
       : (!mydialect<"custom_type">) -> (!mydialect<"other_type">, !mydialect<"other_type">) \
       loc(callsite("foo" at "mysource.cc":10:8))

# %res:2                  : Name of the result (2 values returned)
# "mydialect.morph"       : Op Id (mydialect is dialect prefix, morph is operation name)
# (%input#3)              : Argument to the operation
# { ... }                 : List of attributes (constant named arguments)
# (!mydialect<"custom_type">) : Input type specification
# -> (... , ...)          : Output type specification (two output types)
# loc(...)                : Location information (source file and position)
```
### IR Structure
```mermaid
graph TD
    subgraph Op1[Operation]
        subgraph Reg1[Region]
            subgraph Block1[Block]
                subgraph Op2[Operation]
                    subgraph Reg2[Region]
                        subgraph Block2[Block]
                        end
                        subgraph Block3[Block]
                        end
                        subgraph Block4[Block]
                        end
                    end
                    subgraph Reg3[Region]
                        subgraph Block5[Block]
                        end
                    end
                end
            end
        end
    end

    style Op1 fill:#FF6666,stroke:#333,stroke-width:2px
    style Reg1 fill:#FFCC66,stroke:#333,stroke-width:2px
    style Block1 fill:#66CC66,stroke:#333,stroke-width:2px
    style Op2 fill:#FF6666,stroke:#333,stroke-width:2px
    style Reg2 fill:#FFCC66,stroke:#333,stroke-width:2px
    style Reg3 fill:#FFCC66,stroke:#333,stroke-width:2px
    style Block2 fill:#66CC66,stroke:#333,stroke-width:2px
    style Block3 fill:#66CC66,stroke:#333,stroke-width:2px
    style Block4 fill:#66CC66,stroke:#333,stroke-width:2px
    style Block5 fill:#66CC66,stroke:#333,stroke-width:2px
```
