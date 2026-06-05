import html
import os
from functools import partial

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


def visualize(text: str, tok, model):
    encoded = tok.encode(text)
    tokens = torch.tensor(encoded.ids).unsqueeze(0).to(model.device)
    mask = torch.tensor(encoded.attention_mask).unsqueeze(0).to(model.device)

    with torch.no_grad():
        filled, attention = model.infer(tokens, mask, attn_weights=True)

    attention = attention.cpu()
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
    css = "#input-box textarea { resize: vertical; }"

    with gr.Blocks(title="Attention Visualizer", css=css) as demo:
        gr.Markdown("# Attention Visualizer")
        gr.Markdown("Paste pretokenized input including at least one `[MASK]` token.")

        text = gr.Textbox(
            lines=4,
            label="Input",
            placeholder="mov eax [MASK]",
            elem_id="input-box",
        )
        button = gr.Button("Submit", variant="primary")
        prediction = gr.Textbox(
            label="Prediction",
            interactive=False,
            lines=2,
            elem_id="input-box",
        )

        output = gr.HTML(label="Attention")

        button.click(fn=fn, inputs=text, outputs=[prediction, output])

    demo.launch(server_name="127.0.0.1", server_port=8888)


if __name__ == "__main__":
    main()
