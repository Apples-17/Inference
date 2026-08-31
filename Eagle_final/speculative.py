#(Bascially plan is to verify the entire tree in forward pass and then finds the longest paths that agrees with what Qwen will generate.)

from dataclasses import dataclass
from typing import List, Tuple
import torch

from .tree import FlatTree

@dataclass
class VerifyResult:
    accepted_token_ids: List[int]      # All tokens we're allowed to append to the actual output
    accepted_leaf_index: int           # tree node index of the last accepted draft token (-1 if none accepted)
    bonus_token_id: int                #This is the target model's next guaranteed-correct greedy token after the accepted draft prefix.
    n_accepted_from_draft: int         #how many of accepted_token_ids came from the draft (for accounting)


@torch.no_grad()
def verify_tree_greedy(target_model, tree:FlatTree, prefix_len:int, past_key_values, prefix_last_hidden_argmax_token:int)->VerifyResult:
    device=tree.token_ids.device 
    n=tree.token_ids.shape[1]

    neg=torch.finfo(torch.float32).min # ig this line does nothing here...
    prefix_block = torch.zeros(1, 1, n, prefix_len, device=device)  # attend fully to prefix
    full_mask = torch.cat([prefix_block, tree.attn_mask], dim=-1)   # (1,1,N, prefix_len+N)

    out = target_model(
        input_ids=tree.token_ids,
        position_ids=tree.position_ids,
        attention_mask=full_mask,
        past_key_values=past_key_values,
        use_cache=True,
    )
    logits = out.logits  # (1, N, V)
    target_argmax = logits.float().argmax(dim=-1)[0]  

    best_path, best_len = [], -1
    for path in tree.retrieve_paths:
        matched=[]
        prev_pred=prefix_last_hidden_argmax_token
        ok=True
        for node_idx in path:
            node_token=tree.nodes[node_idx].token_id
            if node_token!=prev_pred:
                ok=False
                break
            matched.append(node_idx)
            prev_pred=int(target_argmax[node_idx])
        if ok and len(matched)>best_len:
            best_len=len(matched)
            best_path=matched
            best_next_pred=prev_pred
        elif not ok:
            if len(matched) > best_len:
                best_len=len(matched)
                best_path=matched
                best_next_pred=int(target_argmax[matched[-1]]) if matched else prefix_last_hidden_argmax_token

    if best_len <= 0:
        best_path = []
        best_next_pred = prefix_last_hidden_argmax_token

    accepted_ids = [tree.nodes[i].token_id for i in best_path]
    bonus = best_next_pred #this next token is guaranteed correct: it's the target's own argmax

    return VerifyResult(
        accepted_token_ids=accepted_ids + [bonus],
        accepted_leaf_index=(best_path[-1] if best_path else -1),
        bonus_token_id=bonus,
        n_accepted_from_draft=len(accepted_ids),
    )

@torch.no_grad()
def rejection_sample(draft_probs: torch.Tensor, target_probs: torch.Tensor,
                      draft_token: int) -> Tuple[bool, int]:
    p=target_probs[draft_token].item()
    q=draft_probs[draft_token].item()
    accept_prob=min(1.0,p/max(q, 1e-8))
    if torch.rand(()).item()<accept_prob:
        return True, draft_token
    residual=(target_probs-draft_probs).clamp(min=0.0)
    residual=residual/residual.sum().clamp(min=1e-8)
    replacement=torch.multinomial(residual, 1).item()
    return False, replacement
