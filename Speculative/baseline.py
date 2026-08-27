import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
# Ordinary Qwen3-4B generation

TARGET_MODEL = "Qwen/Qwen3-4B"
DEVICE = "cuda"
DTYPE = torch.float16
MAX_NEW_TOKENS = 100

tokenizer =AutoTokenizer.from_pretrained(
    TARGET_MODEL,
    trust_remote_code=True,
)

target_model = AutoModelForCausalLM.from_pretrained(
    TARGET_MODEL,
    torch_dtype=DTYPE,
    trust_remote_code=True,
).to(DEVICE)

target_model.eval()


# Vanilla greedy decoding
@torch.inference_mode()
def greedy_generate(prompt, max_new_tokens=MAX_NEW_TOKENS):
    inputs=tokenizer(
        prompt,
        return_tensors="pt",
    )
    input_ids=inputs.input_ids.to(DEVICE)

    generated_ids=input_ids
    for _ in range(max_new_tokens):
        outputs = target_model(
            input_ids=generated_ids,
            use_cache=False,
        )
        next_logits=outputs.logits[:, -1, :]
        next_token=torch.argmax(
            next_logits,
            dim=-1,
            keepdim=True,
        )
        generated_ids=torch.cat(
            [generated_ids, next_token],
            dim=-1,
        )

        # stop if EOS is generated
        if next_token.item()==tokenizer.eos_token_id:
            break
    return generated_ids


if __name__ == "__main__":

    prompt = "The capital of France is"

    _ = greedy_generate(
        prompt,
        max_new_tokens=10,
    )
    torch.cuda.synchronize()
    # Benchmark
    print("Baseline:")
    torch.cuda.synchronize()
    start=time.perf_counter()
    output_ids=greedy_generate(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    torch.cuda.synchronize()
    end=time.perf_counter()

    elapsed=end - start
    # Decode output
    output_text=tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )
    prompt_ids=tokenizer(
        prompt,
        return_tensors="pt",
    ).input_ids
    num_generated_tokens=(
        output_ids.shape[1]
        - prompt_ids.shape[1]
    )
    print("OUTPUT:")
    print(output_text)

    print("Generated tokens:", num_generated_tokens)
    print(f"Time: {elapsed:.4f} s")

    if num_generated_tokens>0:
        print(
            f"Latency/token: "
            f"{elapsed / num_generated_tokens * 1000:.3f} ms"
        )

        print(
            f"Throughput: "
            f"{num_generated_tokens / elapsed:.2f} tokens/s"
        )