# Attention Visualizer

<video src="docs/demo.mp4" controls width="100%"></video>


## Description

A small [Gradio](https://gradio.app) web UI for inspecting masked language model attention weights through [bertviz](https://github.com/jessevig/bertviz).

The input should be pretokenized with `[MASK]` tokens, the tool will output the model's predictions plus an interactive grid of attention heatmaps — one per layer × head.



## Installation
### Prerequisites
The `undertale` package providing the model and tokenizer is assumed to be importable already.


### Installing
Install the Python package with:

```bash
pip install -e .
```

This installs `gradio`, `bertviz`, and `torch`.

## Configuration

The server reads two environment variables at startup:

| Variable | Purpose |
| --- | --- |
| `UNDERTALE_TOKENIZER_PATH` | path to the trained tokenizer |
| `UNDERTALE_MASKEDLM_CHECKPOINT` | path to the masked LM model checkpoint (`.ckpt`) |

If either is missing the process exits immediately with a clear error.

## Usage
Make sure that the required the environment variables from the previous step are set; if not, set them in the current shell session or source them from e.g. an `.env` file.

```bash
export UNDERTALE_TOKENIZER_PATH=/path/to/tokenizer
export UNDERTALE_MASKEDLM_CHECKPOINT=/path/to/checkpoint.ckpt
```

Next, start the server using:
```bash
python server.py
```

Now you can open <http://127.0.0.1:8888>, paste pretokenized input including at least one `[MASK]` token and click *Submit*.


## What it shows

For each request the server runs a single forward pass that returns both the filled output tokens and the per-layer attention weights, then hands the cropped weights and decoded tokens to `bertviz.model_view` to visualize attentions.

## Project layout

```
.
├── pyproject.toml      project metadata and dependencies
├── README.md           this file
└── server.py           Gradio app
```

The viz server is intentionally a single file. The inference logic is one function (`visualize` in `server.py`).


## Troubleshooting

**`RuntimeError: Missing required environment variable(s): ...`** — you need to export both env vars listed in the `Configureation` section before launching. Put them in a `.env` and `source` it if you launch the server often.

**`ModuleNotFoundError: No module named 'undertale'`** — the `undertale` package isn't installed. `pip install -e /path/to/undertale` from the project root.

**The visualizer renders but every cell is uniform** — your input doesn't contain a `[MASK]` token, so the model is just doing reconstruction with no masked positions to fill. Attention is still real and visualizable, but the predicted output equals the input.

**The visualizer renders but tokens look wrong** — make sure your input is *pretokenized* (whitespace-separated subword tokens in the form the model expects).