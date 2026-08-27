# target+draft+speculative decoding algorithm

import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


TARGET_MODEL = "Qwen/Qwen3-4B"
DRAFT_MODEL = "Qwen/Qwen3-0.6B"

DEVICE = "cuda"
DTYPE = torch.float16

GAMMA = 4
MAX_NEW_TOKENS = 100

# Loading Tokeniser
tokenizer = AutoTokenizer.from_pretrained(
    TARGET_MODEL,
    trust_remote_code=True,
)

# Loading target model
target_model = AutoModelForCausalLM.from_pretrained(
    TARGET_MODEL,
    torch_dtype=DTYPE,
    trust_remote_code=True,
).to(DEVICE)

target_model.eval()

#Loading draft model
draft_model = AutoModelForCausalLM.from_pretrained(
    DRAFT_MODEL,
    torch_dtype=DTYPE,
    trust_remote_code=True,
).to(DEVICE)

draft_model.eval()

# Draft model proposes Gamma tokens
# This fn generates gamma tokens and return tensor shape [1, gamma]
@torch.inference_mode()
def draft_tokens(input_ids, gamma):
    current_ids = input_ids
    drafted = []

    for _ in range(gamma):
        outputs = draft_model(
            input_ids=current_ids,
            use_cache=False,
        )

        next_logits = outputs.logits[:, -1, :]

        next_token = torch.argmax(
            next_logits,
            dim=-1,
            keepdim=True,
        )

        drafted.append(next_token)

        current_ids = torch.cat(
            [current_ids, next_token],
            dim=-1,
        )

    return torch.cat(drafted, dim=-1)

# Target verfication
# Here, we verify all draft tokens with the target model in 1 forward pass.
# Return target model's greedy prediction.
@torch.inference_mode()
def verify_draft(input_ids, draft_ids):
    combined_ids = torch.cat(
        [input_ids, draft_ids],
        dim=-1,
    )

    outputs = target_model(
        input_ids=combined_ids,
        use_cache=False,
    )

    logits = outputs.logits

    prefix_len = input_ids.shape[1]
    gamma = draft_ids.shape[1]

    verification_logits = logits[:,prefix_len - 1 : prefix_len + gamma,:]

    target_tokens = torch.argmax(
        verification_logits,
        dim=-1,
    )
    return target_tokens

# Acceptance
# Accept the longest prefic where draft and target agrees.
# Returns output tokens and number accepeted
def accept_draft(draft_ids, target_tokens):
    gamma = draft_ids.shape[1]
    num_accepted = 0

    for i in range(gamma):
        draft_token = draft_ids[0, i]
        target_token = target_tokens[0, i]

        if draft_token == target_token:
            num_accepted += 1
        else:
            break

    accepted = draft_ids[:, :num_accepted]

    correction_token = target_tokens[:,num_accepted : num_accepted + 1]

    output_tokens = torch.cat(
        [accepted, correction_token],
        dim=-1,
    )

    return output_tokens, num_accepted

@torch.inference_mode()
def speculative_generate(prompt, max_new_tokens=100, gamma=4):
    input_ids=tokenizer(
        prompt,
        return_tensors="pt"
    ).input_ids.to(DEVICE)
    generated = 0
    while generated < max_new_tokens:
        #Mq proposes γ tokens
        draft_ids=draft_tokens(
            input_ids,
            gamma,
        )

        #Mp verifies γ tokens and gives bonus
        target_tokens=verify_draft(
            input_ids,
            draft_ids,
        )

        #Accept longest matching prefix
        new_tokens, num_accepted = accept_draft(
            draft_ids,
            target_tokens,
        )

        #Don't exceed requested output length
        remaining=(max_new_tokens-generated)
        new_tokens =new_tokens[:,:remaining]

        # Append them
        input_ids=torch.cat(
            [input_ids, new_tokens],
            dim=-1,
        )

        eos_positions = (new_tokens[0] == tokenizer.eos_token_id).nonzero()

        if eos_positions.numel() > 0:
            first_eos = eos_positions[0].item()

            new_tokens = new_tokens[:, :first_eos + 1]

            input_ids = torch.cat(
                [input_ids, new_tokens],
                dim=-1,
            )

            generated += new_tokens.shape[1]
            break

        input_ids = torch.cat(
            [input_ids, new_tokens],
            dim=-1,
        )


        generated+=new_tokens.shape[1]

    return input_ids

# For correctness test
@torch.inference_mode()
def baseline_greedy_generate(prompt, max_new_tokens=100):
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
    ).input_ids.to(DEVICE)

    generated_ids = input_ids

    for _ in range(max_new_tokens):
        outputs = target_model(
            input_ids=generated_ids,
            use_cache=False,
        )

        next_logits = outputs.logits[:, -1, :]

        next_token = torch.argmax(
            next_logits,
            dim=-1,
            keepdim=True,
        )

        generated_ids = torch.cat(
            [generated_ids, next_token],
            dim=-1,
        )

        if next_token.item() == tokenizer.eos_token_id:
            break

    return generated_ids

if __name__ == "__main__":
    prompt = "The capital of France is"

    baseline_ids = baseline_greedy_generate(
        prompt,
        max_new_tokens=50,
    )

    speculative_ids = speculative_generate(
        prompt,
        max_new_tokens=50,
        gamma=GAMMA,
    )

    print("Baseline:")
    print(
        tokenizer.decode(
            baseline_ids[0],
            skip_special_tokens=True,
        )
    )

    print("Speculative:")
    print(
        tokenizer.decode(
            speculative_ids[0],
            skip_special_tokens=True,
        )
    )

    print(
        "Exact token match:",
        torch.equal(
            baseline_ids,
            speculative_ids,
        ),
    )

    assert torch.equal(
        baseline_ids,
        speculative_ids,
    ), "Speculative decoding does not match baseline!"

    print("Correctness test passed!")