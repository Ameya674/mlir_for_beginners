# Chapter 1 - Toy language to its AST

The Toy language is a very simple language with a few mathematical functionalities and a syntax which is a mixture of python and c++.

## File Structure of Ch1 compiler

```mermaid
graph TD
  ch1_compiler["llvm_project/mlir/examples/toy/Ch1"]

  toyc.cpp["toyc.cpp"]
  include["include"]
  parser["parser"]
  cmakelists.txt["CMakeLists.txt"]

  ch1_compiler --> toyc.cpp
  ch1_compiler --> include
  ch1_compiler --> parser
  ch1_compiler --> cmakelists.txt

  AST.cpp["AST.cpp"]

  parser --> AST.cpp

  toy["toy"]

  include --> toy

  AST.h["AST.h"]
  Lexer.h["Lexer.h"]
  Parser.h["Parser.h"]

  toy --> AST.h
  toy --> Lexer.h
  toy --> Parser.h
```

**toyc.cpp -** The entry point for the compiler. Takes the file as input and calls the function to dump the AST present in the AST.h file. 
**AST.h -** Calls the Lexer.h and Parser.h files to convert the source code into the AST.
**AST.cpp -** Contains the code for the AST.h file.
**Lexer.h -** Converts the source code into tokens.
**Parser.h -** Uses the Recursive Descent Parser to create the AST.

### Code instance of the toy language 

```bash
```
