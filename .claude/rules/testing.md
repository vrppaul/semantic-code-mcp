---
paths:
  - "tests/**/*.py"
---

# Testing Rules

## Framework
- pytest with pytest-asyncio
- Test files mirror source structure

## TDD / Spec-Driven Development
- Write tests BEFORE implementation
- Tests define the expected behavior/requirements
- Tests should verify requirements and expectations, NOT implementation details

## Test Philosophy
- Tests are specifications of behavior
- Test WHAT the code should do, not HOW it does it
- A test should fail if requirements are broken, not if implementation changes
- Example: test "chunker extracts function with its docstring" not "chunker calls _parse_node 3 times"

## Coverage Focus
- Chunker correctness (AST extraction)
- Search relevance (semantic matching)
- Storage operations (LanceDB CRUD)
- Edge cases (empty files, syntax errors, large files)
