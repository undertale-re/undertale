import html
import os
from contextlib import contextmanager
from functools import partial
from math import sqrt

import gradio as gr
import torch
from bertviz import model_view

from undertale.models import tokenizer
from undertale.models.maskedlm import InstructionTraceTransformerEncoderForMaskedLM
from undertale.models.tokenizer import TOKEN_PAD


def check_env() -> None:
    required = ("UNDERTALE_TOKENIZER_PATH", "UNDERTALE_MASKEDLM_CHECKPOINT")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): " + ", ".join(missing)
        )


@contextmanager
def capture_attention(model):
    collected = [[] for _ in model.encoder.layers]
    handles = []
    for layer_idx, layer in enumerate(model.encoder.layers):
        for head in layer.attention.heads:

            def hook(module, inp, _out, i=layer_idx):
                state = inp[0]
                mask = inp[1] if len(inp) > 1 else None
                q, k = module.q(state), module.k(state)
                scores = torch.bmm(q, k.transpose(-2, -1)) / sqrt(q.size(-1))
                if mask is not None:
                    attn_mask = mask.unsqueeze(-2).bool()
                    scores = scores.masked_fill(~attn_mask, float("-inf"))
                collected[i].append(torch.softmax(scores, dim=-1))

            handles.append(head.register_forward_hook(hook))
    yield collected
    for h in handles:
        h.remove()


def visualize(text: str, tok, model):
    encoded = tok.encode(text)
    tokens = torch.tensor(encoded.ids).unsqueeze(0).to(model.device)
    mask = torch.tensor(encoded.attention_mask).unsqueeze(0).to(model.device)

    with capture_attention(model) as layer_attentions:
        with torch.no_grad():
            filled = model.infer(tokens, mask)

    attention = (
        torch.stack(
            [torch.stack(layer, dim=0) for layer in layer_attentions],
            dim=0,
        )
        .squeeze(2)
        .cpu()
    )
    mask_bool = mask.squeeze(0).bool().cpu()
    layers = [
        layer[:, mask_bool, :][:, :, mask_bool].unsqueeze(0) for layer in attention
    ]

    predicted = (
        tok.decode(filled.tolist(), skip_special_tokens=False)
        .replace(TOKEN_PAD, "")
        .strip()
    )

    bertviz_html = model_view(layers, predicted.split(" "), html_action="return").data
    document = (
        "<!doctype html><html><head><style>"
        "body { margin: 0; padding: 1rem; display: flex; justify-content: center; }"
        f"</style></head><body>{bertviz_html}</body></html>"
    )
    iframe = (
        f'<iframe srcdoc="{html.escape(document, quote=True)}" '
        'width="100%" height="800" frameborder="0" '
        'style="border: none;"></iframe>'
    )

    return predicted, iframe


def main() -> None:
    check_env()

    tok = tokenizer.load(os.environ["UNDERTALE_TOKENIZER_PATH"])
    model = InstructionTraceTransformerEncoderForMaskedLM.load_from_checkpoint(
        os.environ["UNDERTALE_MASKEDLM_CHECKPOINT"]
    )
    model.eval()

    fn = partial(visualize, tok=tok, model=model)
    css = ".resizable textarea { resize: vertical; }"

    with gr.Blocks(title="Attention Visualizer", css=css) as demo:
        gr.Markdown("# Attention Visualizer")
        gr.Markdown("Paste pretokenized input including at least one `[MASK]` token.")

        text = gr.Textbox(
            lines=4,
            label="Input",
            placeholder="mov eax [MASK]",
            elem_classes="resizable",
        )
        button = gr.Button("Submit", variant="primary")
        prediction = gr.Textbox(
            label="Prediction",
            interactive=False,
            lines=2,
            elem_classes="resizable",
        )

        output = gr.HTML(label="Attention")

        button.click(fn=fn, inputs=text, outputs=[prediction, output])

    demo.launch(server_name="127.0.0.1", server_port=8888)


if __name__ == "__main__":
    main()
