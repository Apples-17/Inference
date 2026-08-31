"""
Training data for the draft model. Per the paper's Implementation
section (page 6): "We call the target model to generate responses
rather than using a fixed dataset" -- i.e. train on the TARGET MODEL'S
OWN output distribution (ShareGPT/UltraChat-200K prompts, Qwen3-4B's
own greedy/sampled completions), not on the raw human ShareGPT text.
This matters because the draft model's job is to mimic THIS model's
behavior, not generic human text.
"""
from dataclasses import dataclass
from typing import Iterator, List
import torch


@dataclass
class RolloutExample:
    input_ids: torch.Tensor  # (T+1,) prompt + Qwen3-4B's own greedy continuation


@torch.no_grad()
def generate_rollouts(target_model, tokenizer, prompts: List[str], cfg,
                       max_new_tokens: int = 512) -> Iterator[RolloutExample]:
    device = next(target_model.parameters()).device
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        # non-thinking mode, matching the PS's target inference configuration
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        enc = tokenizer(text, return_tensors="pt").to(device)
        out_ids = target_model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )[0]
        seq = out_ids[: cfg.train_seq_len + 1]
        if seq.shape[0] < 8:
            continue
        yield RolloutExample(input_ids=seq)


def load_sharegpt_ultrachat_prompts(sharegpt_path: str = None, ultrachat_path: str = None) -> List[str]:
    """
    Loader stub: the paper trains on ~68K ShareGPT + ~464K UltraChat-200K
    first-turn prompts (page 6). Point this at local copies of those
    datasets (e.g. downloaded via `datasets.load_dataset`) and extract
    the first human turn of each conversation as the prompt text.
    """
    prompts: List[str] = []
    if sharegpt_path:
        import json
        with open(sharegpt_path) as f:
            data = json.load(f)
        for conv in data:
            turns = conv.get("conversations") or conv.get("conversation") or []
            if turns:
                prompts.append(turns[0].get("value", turns[0].get("content", "")))
    if ultrachat_path:
        import json
        with open(ultrachat_path) as f:
            for line in f:
                row = json.loads(line)
                msgs = row.get("data") or row.get("messages") or []
                if msgs:
                    prompts.append(msgs[0] if isinstance(msgs[0], str) else msgs[0].get("content", ""))
    return prompts
