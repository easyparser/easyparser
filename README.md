## Installation

```bash
pip install ...
```

## Features

Library responsibility:

Graph:
Files ----> Parse ----> Chunk ----> Index ----> Search ----> Represent retrieval context -----> LLM
We are responsible for:
- Parse, Chunk
- Represent retrieval context

**How `easyparser` is different?** We chunk the document structure rather than the text. This is to take the document structure into account to create more sensible chunks.

- Maintain sectional layout structure during parsing and chunking (*).
- Supported formats (refer ... for suitable parsers for each format):
    - Text: html, md, txt, epub, latex, org, rtf, rst
    - Office documents: pdf, docx, pptx, xlsx
    - Images: jpg, png
    - Audio: wav, mp3
    - Video: mp4
    - Code: ipynb, (coming soon: py, js)
    - Data interchange: csv, json yaml, toml
- Content linking across files.
- Suppport LLM integration for correct content parsing.
- Task description for agent-oriented RAG strategies.
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

(*) Due to the complexity and variety of how structures can be represented in different file types, there can be errors. `easyparser` treats this as best effort. Refer xxx for difficult cases. File an issue if you encounter a problem.

## Usage

### Add LLM support

By default, `easyparser` uses the `llm` ([repo](https://github.com/simonw/llm)) with
alias `easyparser-llm` to interact with LLM. Please setup the desired LLM
provider according to their docs, and set the alias `easyparser-llm` to that
model. Example, using Gemini model (as of April 2025):

```bash
# Install the LLM gemini
$ llm install llm-gemini

# Set the Gemini API key
$ llm keys set gemini

# Alias LLM to 'easyparser-llm' (you can see other model ids by running `llm models`)
$ llm aliases set easyparser-llm gemini-2.5-flash-preview-04-17

# Check the LLM is working correctly
$ llm -m easyparser-llm "Explain quantum mechanics in 100 words"
```

## Cookbook

TBD.

- Application in agent-oriented RAG strategies.

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
