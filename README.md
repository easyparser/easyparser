## Installation

```bash
pip install ...
```

## Features

- Correct.
    - Correctly parsing. Maintain sectional layout structure. Sections are first class citizen. Note that, due to the complexity and variety of how structures can be represented in different file types, there can be errors. `easyparser` treats this as best effort. Refer xxx for cases where determining can be difficult.
- Fast.
    - At least 100MB/s parsing
    - At least 100MB/s splitting
- Extensible.
    - Easy to add new strategy
    - Easy to change configure of the current strategy
- Traceable: trace from chunk to source.
- Developer-friendly
    - Evaluation
    - Benchmark
    - Config selector
    - Docker
    - CLI to chunk
- Complete.
    - All common file types
    - All chunking strategies

## Cookbook

TBD.

## Examples

TBD. Code snippet of prominent features:

- Drop-in replacement for file parsing.
- Show case of maintaining sectional layout structure. Show case the `Chunk` interface.
- Show case of a notable chunking strategy.
- Use as tool for agent.

## Contributing

Ensure that you have `git` and `git-lfs` installed. `git` will be used for version control and `git-lfs` will be used for test data.

```bash
# Clone the repository
git clone git@github.com:easyparser/easyparser.git
cd easyparser

# Fetch the test data
git submodule update --init --recursive

# Install development dependnecy
pip install -e ".[dev]"

# Initialize pre-commit hooks
pre-commit install
```

## License

Apache 2.0.
