import random

import torch

from undertale.models.item import tokenizer
from undertale.models.item.model import TransformerEncoderForMaskedLM
from undertale.models.item.tokenizer import pretokenize


def load_model(path):
    return TransformerEncoderForMaskedLM.load_from_checkpoint(path)


def load_tokenizer(path):
    return tokenizer.load(path)


def _mask(pretokenized):
    pretokenized_list = pretokenized.split(" ")
    random_index = random.randint(0, len(pretokenized_list) - 1)
    pretokenized_list[random_index] = "[MASK]"
    return " ".join(pretokenized_list)


def _tokenize(inputs, tok, pretokenized=False, masked=False):
    if not pretokenized:
        inputs = pretokenize(inputs)
    if not masked:
        inputs = _mask(inputs)

    encoded = tok.encode(inputs)
    tokens, mask = torch.tensor(encoded.ids), torch.tensor(encoded.attention_mask)

    return inputs, encoded, tokens, mask


def predict(inputs, tok, model, pretokenized=False, masked=False):
    inputs, _, tokens, mask = _tokenize(inputs, tok, pretokenized, masked)
    output, attns = model(tokens.unsqueeze(0), mask.unsqueeze(0), attn_weights=True)
    output = output.squeeze()

    masked_tokens = torch.where(tokens == tok.token_to_id(tokenizer.TOKEN_MASK))[0]
    logits = output[masked_tokens]
    probs = torch.softmax(logits, dim=-1)
    topk_probs, topk_indices = torch.topk(probs, k=5, dim=-1)

    topk = {}
    for token_position, probabilities, columns in zip(
        masked_tokens, topk_probs, topk_indices
    ):
        topk[token_position.item()] = [
            (tok.decode([column]), round(probability.item(), 4))
            for column, probability in zip(columns, probabilities)
        ]

    filled = torch.where(
        tokens == tok.token_to_id(tokenizer.TOKEN_MASK),
        torch.argmax(output, dim=-1),
        tokens,
    )

    predicted = tok.decode(filled.tolist(), skip_special_tokens=False)
    predicted = predicted.replace(tokenizer.TOKEN_PAD, "").strip()

    print("Masked Inputs:".ljust(17) + f"{inputs}")
    print("Predicted:".ljust(17) + f"{predicted}")

    return attns, mask, predicted.split(" "), topk


def remove_padded_tokens(mask, attns):
    mask = torch.tensor(mask, dtype=torch.bool)

    unpadded = []
    for head in attns:
        attns = []
        for attn in head:
            attns.append(attn[:, mask, :][:, :, mask].squeeze(0))
        unpadded.append(torch.stack(attns, dim=0).unsqueeze(0))

    return unpadded
