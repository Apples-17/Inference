import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eagle_qwen import EagleConfig, EagleDraftModel, FeatureFusion
from eagle_qwen.data_gen import generate_rollouts, load_sharegpt_ultrachat_prompts
from eagle_qwen.train import train_step, TTTBatch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--sharegpt", default=None)
    ap.add_argument("--ultrachat", default=None)
    ap.add_argument("--out", default="eagle_draft.pt")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seq_len", type=int, default=2048)
    args = ap.parse_args()

    device = "cuda"
    cfg = EagleConfig(target_model_name=args.model, train_seq_len=args.seq_len)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    target_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device
    ).eval()
    for p in target_model.parameters():
        p.requires_grad_(False)  # PS constraint: target weights are frozen/unchanged

    draft_model = EagleDraftModel(target_model, cfg).to(device).to(torch.bfloat16)
    fusion = draft_model.fusion  # trainable fusion lives inside the draft model already

    optimizer = torch.optim.AdamW(
        list(draft_model.trainable_parameters()), lr=cfg.lr, betas=cfg.betas
    )

    prompts = load_sharegpt_ultrachat_prompts(args.sharegpt, args.ultrachat)
    if not prompts:
        raise SystemExit("Provide --sharegpt and/or --ultrachat data paths (see data_gen.py).")

    step = 0
    for example in generate_rollouts(target_model, tokenizer, prompts, cfg, max_new_tokens=args.seq_len):
        if step >= args.steps:
            break
        batch = TTTBatch(input_ids=example.input_ids.unsqueeze(0).to(device))
        loss = train_step(target_model, draft_model, fusion, batch, cfg, optimizer)
        step += 1
        if step % 50 == 0:
            print(f"step {step:6d}  loss {loss:.4f}")
        if step % 2000 == 0:
            torch.save(draft_model.state_dict(), args.out)

    torch.save(draft_model.state_dict(), args.out)
    print(f"Saved draft model to {args.out}")


if __name__ == "__main__":
    main()
