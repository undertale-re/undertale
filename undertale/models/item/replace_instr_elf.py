#
# Sample use:
#
# python -m undertale.models.item.replace_instr_elf  -t ~/undertale_shared/models/item/item.tokenizer.nixpkgs-disassembled-rizin.json -c ~/undertale_shared/models/item/nixpkgs-full-pytorch/checkpoints/epoch\=38-train_loss\=0.16-valid_f1\=0.96.ckpt  ~/trident-obfuscated sym.ls
# python -m undertale.models.item.replace_instr_elf  -t ~/undertale_shared/models/item/item.tokenizer.nixpkgs-disassembled-rizin.json -c ~/undertale_shared/models/item/nixpkgs-full-pytorch/checkpoints/epoch\=38-train_loss\=0.16-valid_f1\=0.96.ckpt  ~/trident-obfuscated ls
#
# -t is the tokenizer to use
# -c is the masked language model to use
# 2nd to last arg is a binary elf file
# last arg is function name (assume not stripped)
#
# We use r2 to find that instruction, pull out all of its bytes, and
# then hand those back to r2 for disassmbly.  All of this is to
# simulate what happens in our pipelines, which hand bytes for a
# function to rizin (or r2) for disassembly.
#
# Then we pretokenize using the item code, and peform the following
# experiment.
#
# Starting with 0th instruction, we collect instructions until we hit
# token limit (512) but only keep full instructions.  This is the
# window of code that anomaly_map operates upon.  This function gets
# handed a window that starts with the 1st instruction, then one that
# starts with the 20th instruction, then one that starts with the 40th
# instruction, etc. These different windows overlap.  Could have done
# a window starting at 0, at 1, 2, etc.  But that seemed overkill.
#
# anomaly_map gets a string that is pretokenized and includes a
# sequence of instructions from the input function. It masks all the
# tokens in the 1st instruction and has the model fill in those
# blanks. For just the tokens that represent the instruction that was
# masked, we add up the log probabilities for the new tokens predicted
# by the model, and also for the original tokens that were masked.  We
# subract these two, and get a result that will be 0 if the predicted
# instruction is the same and will be positive if a more likely
# alternative was imagined by the model.
#
# Returned by anomaly map is a list of triples which are the original
# and predicted instructions for masking and the log probability for
# the predicted vs the original.
#
# This list, for each of the window starts (token 0, token 20, token 40, ... etc) is output in a file
# window-0-nt
# window-20-nt
# etc, where 'nt' is the number of tokens in this window
#


import argparse
import json
import math
import re

import binaryninja
import r2pipe
from binaryninja.architecture import Architecture
from binaryninja.enums import InstructionTextTokenType
from torch import argmax, softmax, tensor, where

from . import tokenizer
from .model import TransformerEncoderForMaskedLM
from .tokenizer import pretokenize


def get_insn_inds_pretokens(pretokens, insn_num_desired):
    inds = []
    insn_num = 0
    for i in range(len(pretokens)):
        if pretokens[i] == next_token:
            insn_num += 1
        if insn_num > insn_num_desired:
            break
        if insn_num == insn_num_desired and pretokens[i] != next_token:
            inds.append(i)
    return inds


def get_insn_inds_tokens(tokens, insn_num_desired):
    inds = []
    insn_num = 0
    for i in range(len(tokens)):
        if tokens[i] == pad_token_id:
            continue
        if int(tokens[i]) == next_token_id:
            insn_num += 1
        if insn_num == insn_num_desired and int(tokens[i]) != next_token_id:
            inds.append(i)
    return inds


# compute something to indicate how much better or worse the
# tokens in new_tokens are compared with those in old_tokens.
# Attention is restricted to just the tokens at inds which
# were the ones that were masked.  We compute likelihood ratio
# of new/old and then take log and sum that.  So we are
# returning the avg log likelihood ratio for the masked
# tokens, new vs old.
# we also print out the lr for each new/old pair
def pp(inds, old_tokens, new_tokens, probs):
    # for these inds
    # compute probs and multiply them
    lpo = 0.0
    lpn = 0.0
    for ind in inds:
        oid = old_tokens[ind]
        nid = new_tokens[ind]
        ot = tok.id_to_token(oid)
        nt = tok.id_to_token(nid)
        op = probs[ind][oid]
        lpo += math.log(op)
        np = probs[ind][nid]
        lpn += math.log(np)
        # likelihood ratio for new token vs old one
        lr = np / op
        print(f"({ot},{nt},{lr:.2f}) ", end="")
        print("")
        # in log domain, this corresponds to p(new token seq) / p(old token seq)
    return lpn - lpo


# Use MLM to fill in all masked tokens with highest prob token.
# Also return probs for (all?) tok in the sequence.
# Note: Probably only the probs for originally masked tokens will be meaningful
def fill_in_masked(tokens_masked, attn):
    # and compute output
    output = model(tokens_masked.unsqueeze(0), attn.unsqueeze(0)).squeeze()
    # now get predicted tokens
    filled = where(
        tokens_masked == tok.token_to_id(tokenizer.TOKEN_MASK),
        argmax(output, dim=-1),
        tokens_masked,
    )
    probs = softmax(output, dim=-1)
    return (filled, probs)


# Use the MLM to replace an entire instruction.  Note that we
# have already figured out which token inds correspond to each
# insn_num.
def replace_toks(tokens, attn, inds):
    # in a copy of tokens, change all the tokens in instruction i to [MASK]
    tokens_masked = tokens.clone().detach()
    for ind in inds:
        tokens_masked[ind] = mask_token_id
    # breakpoint()
    return fill_in_masked(tokens_masked, attn)


# Use the MLM to replace token @ ind. Mask it, use MLM to
# predict most likely replacement, and replace.
def replace_tok(tokens, attn, ind):
    tokens_masked = tokens.clone().detach()
    tokens_masked[ind] = mask_token_id
    return fill_in_masked(tokens_masked, attn)


# just translate tokens seq into string
def decode(tokens):
    dec = tok.decode(tokens.tolist(), skip_special_tokens=False)
    return dec.replace(tokenizer.TOKEN_PAD, "").strip()


def anomaly_map(pretok_disas_str):
    encoded = tok.encode(pretok_disas_str)
    tokens, attn = tensor(encoded.ids), tensor(encoded.attention_mask)

    num_insns = 1
    for x in tokens:
        if x == next_token_id:
            num_insns += 1
    instr_to_inds = {}
    for i in range(num_insns):
        instr_to_inds[i] = get_insn_inds_tokens(tokens, i)
        for ind in instr_to_inds[i]:
            assert ind < 512

    # iterate over instructions in the input
    lpri = []
    for instr in range(num_insns):
        print(f"\ninstruction {instr}: ", end="")
        inds = instr_to_inds[instr]

        def instr_dec(the_tokens, the_inds):
            instr_txt = ""
            for ind in the_inds:
                dec = tok.decode([the_tokens[ind]], skip_special_tokens=False)
                instr_txt += dec.replace(tokenizer.TOKEN_PAD, "").strip()
                instr_txt += " "
            return instr_txt

        instr_txt = instr_dec(tokens, inds)

        # XXX Hmm why replace instr of N toks with a new instr of
        # same # of tokens probably better to try a few times with
        # instructions of lengths drawn from est length
        # distribution

        # replace tokens for just instruction i using the MLM
        (tokens_replaced, prob_repl) = replace_toks(tokens, attn, inds)
        # predicted = decode(tokens_replaced)
        lpr = pp(inds, tokens, tokens_replaced, prob_repl)
        new_instr_txt = instr_dec(tokens_replaced, inds)

        print(f"log(p(new)/p(old)) = {lpr:.2f}\n")
        print(new_instr_txt)

        print(f"original:  {instr_txt}")
        print(f"predicted: {new_instr_txt}")

        # print(f"Considering replacing each of {inds}")

        # for ind in inds:
        #     # try replacing just this ind, starting with tokens_replaced
        #     (new_tokens_replaced, new_prob_repl) = replace_tok(tokens_replaced, attn, ind)
        #     old_id = tokens_replaced[ind]
        #     new_id = new_tokens_replaced[ind]
        #     # token did not change
        #     if new_id == old_id:
        #         continue
        #     old_p = new_prob_repl[ind][old_id]
        #     new_p = new_prob_repl[ind][new_id]
        #     if new_p < old_p:
        #         breakpoint()
        #     assert (new_p > old_p)
        #     new_predicted = decode(new_tokens_replaced)
        #     print(f"with tok {ind} replaced: {new_p/old_p} improvement")
        #     lpr = pp(inds, tokens_replaced, new_tokens_replaced, new_prob_repl)
        #     print(f"log(p(new)/p(old)) = {lpr:.2f}")
        #     print(f"better:    {new_predicted}")
        #     tokens_replaced = new_tokens_replaced
        #     prob_repl = new_prob_repl

        lpr = pp(inds, tokens, tokens_replaced, prob_repl)
        print(f"FINAL log(p(new)/p(old)) = {lpr:.2f}")
        lpri.append((instr_txt, new_instr_txt, lpr))
    return lpri


def remove_braces(text):
    # Matches ' {' followed by any characters (non-greedy) until the next '}'
    return re.sub(r" \{.*?\}", "", text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="predict masked tokens in a piece of disassembly"
    )

    parser.add_argument(
        "-t", "--tokenizer", required=True, help="trained tokenizer file"
    )
    parser.add_argument(
        "-c", "--checkpoint", required=True, help="trained model checkpoint"
    )
    parser.add_argument("elf", help="its an elf filename")
    parser.add_argument("fn", help="its a function name")

    # parser.add_argument(
    #    "input", help="masked disassembly input to fill in (in pretokenized form)"
    # )

    arguments = parser.parse_args()

    tok = tokenizer.load(arguments.tokenizer)
    model = TransformerEncoderForMaskedLM.load_from_checkpoint(arguments.checkpoint)

    pad_token = "[PAD]"
    mask_token = "[MASK]"
    next_token = "[NEXT]"
    pad_token_id = tok.token_to_id(pad_token)
    mask_token_id = tok.token_to_id(mask_token)
    next_token_id = tok.token_to_id(next_token)

    # this is the context window (number of tokens)
    window = 512

    disassembly = []

    if "sym" in arguments.fn:
        r = r2pipe.open(arguments.elf)
        r.cmd("aa")
        d = json.loads(r.cmd(f"pdfj @ {arguments.fn}"))

        buf = None
        for op in d["ops"]:
            byts = bytes.fromhex(op["bytes"])
            if buf is None:
                buf = byts
            else:
                buf += byts

        code_max = 65536
        r = r2pipe.open(f"malloc://{code_max}", flags=["-2"])
        r.cmd("s 0")
        r.cmd(f"r {len(buf)}")
        r.cmd("wx " + (" ".join([f"{i:02x}" for i in buf])))
        r.cmd("aa")
        d = json.loads(r.cmd("pdfj"))

        print("ORIGINAL DISASSEMBLY FROM R2")
        if "ops" in d.keys():
            for i in range(len(d["ops"])):
                dis = d["ops"][i]["disasm"]
                print(dis)
                disassembly.append(dis)
    else:
        SKIP_TOKENS = [
            InstructionTextTokenType.StackVariableToken,
            InstructionTextTokenType.TagToken,
        ]

        bv = binaryninja.load(arguments.elf)

        fn = bv.get_functions_by_name(arguments.fn)[0]
        buf = bv.read(fn.start, fn.total_bytes)

        base_addr = 0
        bv = binaryninja.BinaryView.new(buf)
        bv.arch = Architecture["x86_64"]
        bv.platform = bv.arch.standalone_platform
        bv.add_entry_point(base_addr)
        bv.create_user_function(base_addr)
        bv.update_analysis_and_wait()

        fn = bv.get_function_at(base_addr)
        for block in sorted(fn.basic_blocks, key=lambda b: b.start):
            for line in block.disassembly_text:
                disasm_str = "".join(
                    token.text for token in line.tokens if token.type not in SKIP_TOKENS
                )
                disasm_str = " ".join(disasm_str.strip().split())
                if any("{" in token.text for token in line.tokens):
                    disasm_str = remove_braces(disasm_str)
                if "sub_0" in disasm_str:
                    idx = next(
                        (
                            i
                            for i, token in enumerate(line.tokens)
                            if token.text == "sub_0"
                        ),
                        -1,
                    )
                    disasm_str = disasm_str.replace(
                        "sub_0", str(line.tokens[idx].value)
                    )
                if disasm_str == "retn":
                    disasm_str = disasm_str[:-1]
                disassembly.append(disasm_str)

    num_insns = len(disassembly)
    disassembly = "\n".join(disassembly)

    pretokens = pretokenize(disassembly).split()
    num_tokens = len(pretokens)

    print("\nINPUT PRETOKENS")
    for pret in pretokens:
        print(f"{pret} ", end="")
    print("\n\n")

    # num_tokens = sum(encoded.attention_mask)

    instr_to_inds = {}
    for i in range(num_insns):
        instr_to_inds[i] = get_insn_inds_pretokens(pretokens, i)

    for i in range(0, num_insns, 20):
        # for i in range(240,num_insns,20):

        nt = 0
        instrs = []
        j = i
        pt_txt = ""
        # grab instructions until we fill the context window
        while True:
            if j == num_insns:
                break
            il = len(instr_to_inds[j]) + 1
            if nt + il > window - 5:
                break
            ind1 = instr_to_inds[j][0]
            ind2 = instr_to_inds[j][-1]
            this_instr = " ".join(pretokens[ind1 : ind2 + 1])
            pt_txt += this_instr + " [NEXT] "
            nt += il
            j += 1

        pt_txt = pt_txt[:-8]
        print("\n-----------------------------------------------------------------")
        print(f"WINDOW is {i}..{j} instructions which is {nt} tokens")
        # print(i, j, nt)
        print(pt_txt)
        # breakpoint()

        lpri = anomaly_map(pt_txt)
        with open(f"window-{i}-{j}", "w") as w:
            for instr, new_instr, lpr in lpri:
                w.write(f"{lpr:.3f}: {instr} XXX {new_instr}\n")
