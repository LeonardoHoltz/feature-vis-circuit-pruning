# InceptionV1 circuits

Feature visualization experiments on Lucid's InceptionV1 (the `inception5h` weights, same channel indices as
Lucid and the OpenAI Microscope), using [Lucent](https://github.com/greentfrapp/lucent), Lucid's PyTorch port.

- **Dropout during optimization**: visualize a channel while randomly switching off neurons in earlier layers.
- **Attribution**: rank the channels of the previous layer by activation × gradient to a target channel.
- **Top-down circuit pruning**: a local web app that builds a tree from a root channel down to earlier layers,
  keeps only the channels you select, and renders the root with everything else switched off.

## Setup

Requires [uv](https://docs.astral.sh/uv/). A CUDA GPU is recommended; everything also runs on CPU, slowly.

```bash
uv sync
```

## Usage

```bash
# Feature visualization; one row per dropout rate
uv run featviz_dropout.py --layer mixed4e --channel 55 --rates 0 0.1 0.3 0.5
uv run featviz_dropout.py --layer mixed4d --channel 493 --neuron       # center neuron instead of the whole channel
uv run featviz_dropout.py --layer mixed4e --channel 55 --rates 0 --keep mixed4d:493,460   # zero the other 4d channels

# Top-k channels of the previous layer by attribution, as a labeled grid
uv run top_channels.py --layer mixed4e --channel 55 --k 30

# Top-down circuit pruning app -> http://127.0.0.1:8000
uv run circuit_app.py --layer mixed4e --channel 55
```

Layers: `conv2d0`–`conv2d2`, `mixed3a`–`mixed5b`. Targets of `featviz_dropout.py` and the app's root must be `mixed*`.
Images are written to the working directory and are git-ignored.

## Tests

```bash
uv run pytest
```
