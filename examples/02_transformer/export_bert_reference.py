"""
Exports a randomly initialized Hugging Face BertModel (without pooler) for the comparison with the MDL model `Bert`
(model.dlm): its weights under the MDL weight paths, random inputs, and the reference output, in one safetensors file.

    python export_bert_reference.py --out data/bert_tiny.safetensors              # the tiny model of the tests
    python export_bert_reference.py --base --out bert_base.safetensors            # BERT-base (vocabulary 30522)
    python export_bert_reference.py --lengths 7 4 --out data/bert_tiny_padded.safetensors  # padded sequences

Keys: the MDL weight paths (float32), `input.ids`, `input.types`, `input.positions` (int32, [batch, seq]),
`reference.output` (float32, [batch, seq, hidden]), and with --lengths `input.mask` (bool, [batch, seq]; true for the
first `length` tokens of each sequence, Hugging Face's attention_mask).
"""

import argparse

import torch
from safetensors.torch import save_file
from transformers import BertConfig, BertModel


def mdl_weights(model: BertModel, layers: int) -> dict[str, torch.Tensor]:
    """Maps the Hugging Face parameters onto the MDL weight paths of `Bert` (Linear.W is [out, in] like HF)."""
    hf = {name: p.detach().float().contiguous() for name, p in model.named_parameters()}
    weights = {
        "embeddings.word.table": hf["embeddings.word_embeddings.weight"],
        "embeddings.position.table": hf["embeddings.position_embeddings.weight"],
        "embeddings.tokenType.table": hf["embeddings.token_type_embeddings.weight"],
        "embeddings.norm.gamma": hf["embeddings.LayerNorm.weight"],
        "embeddings.norm.beta": hf["embeddings.LayerNorm.bias"],
    }
    linears = {
        "weightQuery": "attention.self.query",
        "weightKey": "attention.self.key",
        "weightValue": "attention.self.value",
        "weightOut": "attention.output.dense",
        "weightIntermed": "intermediate.dense",
        "weightIntermedOut": "output.dense",
    }
    norms = {"mhaNorm": "attention.output.LayerNorm", "ffnNorm": "output.LayerNorm"}
    for i in range(layers):
        for mdl, path in linears.items():
            weights[f"encoder.layers.{i}.{mdl}.W"] = hf[f"encoder.layer.{i}.{path}.weight"]
            weights[f"encoder.layers.{i}.{mdl}.b"] = hf[f"encoder.layer.{i}.{path}.bias"]
        for mdl, path in norms.items():
            weights[f"encoder.layers.{i}.{mdl}.gamma"] = hf[f"encoder.layer.{i}.{path}.weight"]
            weights[f"encoder.layers.{i}.{mdl}.beta"] = hf[f"encoder.layer.{i}.{path}.bias"]
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base", action="store_true", help="BERT-base sizes (overrides the size options)")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--inner", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--vocab", type=int, default=100)
    parser.add_argument("--positions", type=int, default=16)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lengths", type=int, nargs="+", help="valid length of each sequence (padding after it)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.base:
        config = BertConfig()  # 768 hidden, 12 heads, 3072 inner, 12 layers, 30522 tokens, 512 positions
    else:
        config = BertConfig(
            hidden_size=args.hidden,
            num_attention_heads=args.heads,
            intermediate_size=args.inner,
            num_hidden_layers=args.layers,
            vocab_size=args.vocab,
            max_position_embeddings=args.positions,
        )
    model = BertModel(config, add_pooling_layer=False).eval()

    ids = torch.randint(0, config.vocab_size, (args.batch, args.seq))
    types = torch.randint(0, config.type_vocab_size, (args.batch, args.seq))
    positions = torch.arange(args.seq).expand(args.batch, args.seq)
    mask = None
    if args.lengths:
        if len(args.lengths) != args.batch:
            parser.error("--lengths needs one length per sequence (--batch)")
        mask = torch.arange(args.seq)[None, :] < torch.tensor(args.lengths)[:, None]
    with torch.no_grad():
        output = model(
            input_ids=ids,
            token_type_ids=types,
            position_ids=positions,
            attention_mask=None if mask is None else mask.to(torch.long),
        ).last_hidden_state

    tensors = mdl_weights(model, config.num_hidden_layers)
    tensors["input.ids"] = ids.to(torch.int32).contiguous()
    tensors["input.types"] = types.to(torch.int32).contiguous()
    tensors["input.positions"] = positions.to(torch.int32).contiguous()
    tensors["reference.output"] = output.float().contiguous()
    if mask is not None:
        tensors["input.mask"] = mask.contiguous()
    save_file(tensors, args.out)
    print(f"wrote {args.out}: {len(model_weights := mdl_weights(model, config.num_hidden_layers))} weights, "
          f"output {tuple(output.shape)}" + ("" if mask is None else f", lengths {args.lengths}"))


if __name__ == "__main__":
    main()
