# Undertale: Function Namer

A Binary Ninja plugin that renames a function by sending its disassembly to
an [Undertale Inference Server](../inference-server) and applying the
predicted name.

## Workflow

1. Select a function in Binary Ninja and run **Undertale > Infer and Rename
   Function** (right-click in the disassembly/IL view, or the Tools menu).
2. The plugin reads the function's disassembly and pretokenizes it into the
   form the model was trained on.
3. The pretokenized disassembly is sent to the inference server for analysis.
4. The plugin polls the inference server for the function name prediction.
5. The predicted name is applied to the function as a single undoable
   action.

Naming runs in the background, so Binary Ninja stays responsive while
waiting on the inference server.

## Installation

The installable plugin lives in
[`undertale_function_namer/`](undertale_function_namer). Symlink (or copy)
that directory into Binary Ninja's user plugin folder:

```bash
# In Linux
ln -s "/path/to/undertale_function_namer" ~/.binaryninja/plugins/undertale_function_namer

# In macOS
ln -s "/path/to/undertale_function_namer" ~/Library/Application\ Support/Binary\ Ninja/plugins/undertale_function_namer  # macOS
```

Then restart Binary Ninja to register the plugin.

## Configuration

On first use, the plugin prompts you to configure the inference server connection. The connection can be specified as either:

* a TCP address (`host:port`), or
* a Unix domain socket path

This configuration is saved in Binary Ninja's user settings and persists across restarts. To update it, select **Undertale > Reconfigure Inference Server Connection**. This clears the saved connection and immediately prompts you to configure a new one.


## Requirements

- Binary Ninja build 4000 or newer, with Python 3 scripting enabled.
- Network or filesystem access to a running Undertale Inference Server (see
  [`extras/inference-server`](../inference-server)).

## Development

> [!NOTE]
> To keep the plugin independent of Undertale's much heavier dependency set, the disassembly code is duplicated here. `undertale_function_namer/_disassembly.py` is a byte-for-byte copy of [`undertale/pipeline/disassembly.py`](https://github.com/undertale-re/undertale/blob/binaryninja-plugin/undertale/pipeline/disassembly.py).
>
> If either copy changes, update the other as well. `tests/unit.py` (`TestPipelineDisassemblyCodeConsistency`) verifies that the two remain identical.

