# Building MLIR with LLVM

LLVM is a framework used to develop a frontend for any langauge and hence develop a new language and develop a backend for any architecture. MLIR stands for Multi Level Intermediate Representation and is a software framework for developing multi level IR compiler frameworks.

## Building from source

### Requirements
We require build-essential cmake ninja-build python3 git and clang
```bash
sudo apt update
sudo apt install build-essential cmake ninja-build python3 git clang
```

### Clone the LLVM project and build it
```bash
git clone https://github.com/llvm/llvm-project.git
mkdir llvm-project/build && cd llvm-project/build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON
# Using clang and lld speeds up the build, we recommend adding:
#  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON
# CCache can drastically speed up further rebuilds, try adding:
#  -DLLVM_CCACHE_BUILD=ON
# Optionally, using ASAN/UBSAN can find bugs early in development, enable with:
# -DLLVM_USE_SANITIZER="Address;Undefined"
# Optionally, enabling integration tests as well
# -DMLIR_INCLUDE_INTEGRATION_TESTS=ON
cmake --build . --target check-mlir
```

This successfully builds llvm which is ready for use.
