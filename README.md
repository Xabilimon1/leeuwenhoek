# Leeuwenhoek

Taking apart the tiny networks of ARC's [AlgZoo](https://github.com/alignment-research-center/alg-zoo), one specimen at a time.

ARC published AlgZoo in January 2026: RNNs and transformers with 8 to 1,408 parameters, trained on four algorithmic tasks. Their post analyses the *2nd argmax* family. This project works on the families nobody has explained yet (*argmedian*, *median*, *longest cycle*), climbing each family from its smallest model upwards, and studies how AI coding agents do on the same task.

Status: starting. Not affiliated with ARC.

## Setup

```bash
uv sync
uv run pytest
```
