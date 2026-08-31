from typing import Optional
import torch

from .features import TargetFeatureExtractor
from .tree import DraftTree
from .speculative import verify_tree_greedy


class EagleGenerator:
    def __init__(self, target_model, draft_model, cfg, tokenizer):
        self.target_model = target_model
        self.draft_model = draft_model
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.extractor = TargetFeatureExtractor(target_model, cfg)
        self.tree_builder = DraftTree(cfg)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256,
                 eos_token_id: Optional[int] = None) -> torch.Tensor:
        device = input_ids.device
        eos_token_id = eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id

        out = self.target_model(input_ids=input_ids, use_cache=True)
        past_key_values = out.past_key_values
        feats = self.extractor.pop()
        g = self.draft_model.fusion(feats.low[:, -1:, :], feats.mid[:, -1:, :], feats.high[:, -1:, :])
        last_token = out.logits[:, -1, :].float().argmax(dim=-1)  # (1,)
        prev_token_id = int(input_ids[0, -1])

        generated = [int(last_token[0])]
        cur_pos = input_ids.shape[1]  # next absolute position id to assign

        while len(generated) < max_new_tokens:
            root_token_tensor = torch.tensor([[prev_token_id]], device=device)
            root_a = self.draft_model.build_input(g, root_token_tensor)
            root_a, _ = self.draft_model.forward_layer(
                root_a,
                position_ids=torch.tensor([[cur_pos - 1]], device=device),
                attention_mask=None,  # single token, nothing to mask
                use_cache=False,
            )

            tree = self.tree_builder.build(
                self.draft_model, root_a=root_a,
                root_token_id=torch.tensor(generated[-1]),
                position_id_start=cur_pos,
            )

            result = verify_tree_greedy(
                self.target_model, tree, prefix_len=cur_pos,
                past_key_values=past_key_values,
                prefix_last_hidden_argmax_token=generated[-1],
            )

            generated.extend(result.accepted_token_ids)
            n_new = len(result.accepted_token_ids)
            cur_pos += n_new

            last_id = torch.tensor([[result.accepted_token_ids[-1]]], device=device)
            out2 = self.target_model(input_ids=last_id, position_ids=torch.tensor([[cur_pos - 1]], device=device),
                                      past_key_values=past_key_values, use_cache=True)
            past_key_values = out2.past_key_values
            feats = self.extractor.pop()
            g = self.draft_model.fusion(feats.low, feats.mid, feats.high)
            prev_token_id = result.accepted_token_ids[-1]

            if eos_token_id is not None and eos_token_id in result.accepted_token_ids:
                idx = result.accepted_token_ids.index(eos_token_id)
                generated = generated[: len(generated) - n_new + idx + 1]
                break

        return torch.tensor([generated], device=device)
