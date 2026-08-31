from dataclasses import dataclass, field
from typing import List, Optional
import torch
import torch.nn.functional as F


@dataclass
class TreeNode:
    token_id: int
    parent: Optional["TreeNode"]
    depth: int
    log_prob: float           
    a: torch.Tensor           
    index: int = -1 


class DraftTree:
    def __init__(self, cfg):
        self.cfg=cfg

    @torch.no_grad()
    def build(self, draft_model, root_a: torch.Tensor, root_token_id: torch.Tensor,
              position_id_start: int) -> "FlatTree":
        cfg=self.cfg
        device=root_a.device
        root=TreeNode(token_id=int(root_token_id), parent=None, depth=0,
                         log_prob=0.0, a=root_a)
        frontier=[root]
        all_nodes:List[TreeNode]=[]

        for depth in range(cfg.draft_tree_depth):
            if not frontier:
                break
            a_in = torch.cat([n.a for n in frontier], dim=1)                     # (1,F,H)
            tok_in = torch.tensor([[n.token_id for n in frontier]], device=device)  # (1,F)
            x = draft_model.build_input(a_in, tok_in)
            pos_ids = torch.tensor(
                [[position_id_start + depth] * len(frontier)], device=device
            )
            mask = _independent_mask(len(frontier), device, x.dtype)
            a_out, _ =draft_model.forward_layer(x, pos_ids, mask, use_cache=False)
            logits =draft_model.logits_from_a(a_out)                            # (1,F,V)
            logp =F.log_softmax(logits.float(), dim=-1)
            topk_logp, topk_ids=logp.topk(cfg.draft_tree_topk, dim=-1)         # (1,F,topk)

            new_frontier=[]
            for f_idx, node in enumerate(frontier):
                node_a_out=a_out[:, f_idx:f_idx + 1, :]
                for k in range(cfg.draft_tree_topk):
                    child=TreeNode(
                        token_id=int(topk_ids[0, f_idx, k]),
                        parent=node,
                        depth=depth + 1,
                        log_prob=node.log_prob + float(topk_logp[0, f_idx, k]),
                        a=node_a_out,
                    )
                    new_frontier.append(child)

            all_nodes.extend(new_frontier)
            new_frontier.sort(key=lambda n: n.log_prob, reverse=True)
            budget_left = cfg.draft_total_tokens -1 -len(all_nodes) + len(new_frontier)
            keep=max(0, min(len(new_frontier), cfg.draft_total_tokens))
            frontier=new_frontier[:keep]
            if len(all_nodes)>=cfg.draft_total_tokens:
                break

        all_nodes.sort(key=lambda n: n.log_prob, reverse=True)
        all_nodes = all_nodes[: cfg.draft_total_tokens - 1]
        return FlatTree.from_nodes(root, all_nodes, position_id_start, device)


def _independent_mask(n, device, dtype):
    neg = torch.finfo(dtype).min
    m = torch.full((1, 1, n, n), neg, device=device, dtype=dtype)
    m[..., torch.arange(n), torch.arange(n)] = 0.0
    return m


@dataclass
class FlatTree:
    token_ids:torch.Tensor        
    position_ids:torch.Tensor # (1, N) absolute position id of each node
    attn_mask:torch.Tensor       # (1,1,N,N) additive mask: node i attends to node j iff j is an ancestor of i (or i==j)
    retrieve_paths:List[List[int]]  #root-to-leaf index paths, for extracting candidate sequences
    nodes:List[TreeNode]

    @staticmethod
    def from_nodes(root: TreeNode, nodes: List[TreeNode], position_id_start: int, device)->"FlatTree":
        for i, n in enumerate(nodes):
            n.index = i
        n_all=len(nodes)
        token_ids=torch.tensor([[n.token_id for n in nodes]], device=device)
        position_ids=torch.tensor([[position_id_start + n.depth for n in nodes]], device=device)

        neg=torch.finfo(torch.float32).min
        mask=torch.full((n_all, n_all), neg)
        for n in nodes:
            j=n
            while j is not None:
                if (j.index>=0):
                    mask[n.index, j.index] = 0.0
                j=j.parent
        attn_mask=mask.view(1, 1, n_all, n_all).to(device)

        children_of={n.index: [] for n in nodes}
        has_children=set()
        for n in nodes:
            if n.parent is not None and n.parent.index>=0:
                children_of[n.parent.index].append(n.index)
                has_children.add(n.parent.index)
        leaves=[n.index for n in nodes if n.index not in has_children]

        paths=[]
        idx_to_node={n.index: n for n in nodes}
        for leaf in leaves:
            path=[]
            cur=idx_to_node[leaf]
            while cur is not None and cur.index>=0:
                path.append(cur.index)
                cur=cur.parent
            paths.append(list(reversed(path)))

        return FlatTree(token_ids=token_ids, position_ids=position_ids,
                         attn_mask=attn_mask, retrieve_paths=paths, nodes=nodes)
