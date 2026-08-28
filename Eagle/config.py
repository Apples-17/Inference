from dataclasses import dataclass

@dataclass
class EagleConfig:
    target_model_name:str="Qwen/Qwen3-4B"

    #feature fusion
    low_layer_frac:float=0.20    
    mid_layer_frac:float=0.50    
    high_layer_frac:float=1.00   

    #draft model itself
    draft_num_layers:int=1      
    draft_hidden_size:int= None  

    #training-time test 
    ttt_steps: int=3           
    train_seq_len: int=2048
    lr:float=5e-5
    betas:tuple=(0.9, 0.95) 
    grad_clip:float=0.5

    # dynamic draft tree (mainly in Eagle 2, I think was reused in Eagle 3)
    draft_tree_depth:int=6       
    draft_tree_topk:int=10       
    draft_total_tokens:int=48    

    #decoding / losslessness 
    temperature:float=0.0       # for greedy
    greedy_lossless:bool=True

    def __post_init__(self):
        if self.draft_tree_topk > self.draft_total_tokens:
            raise ValueError("draft_tree_topk must be <= draft_total_tokens")
