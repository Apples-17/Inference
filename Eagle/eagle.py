import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

TARGET_MODEL = "Qwen/Qwen3-4B"
DEVICE = "cuda"

class EagleDecoder:
    def __init__(self):
        # Load target model
        self.tokenizer=AutoTokenizer.from_pretrained(
            TARGET_MODEL,
            trust_remote_code=True,
        )

        self.target=AutoModelForCausalLM.from_pretrained(
            TARGET_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        self.target.eval()
        print("TARGET MODEL LOADED")

    @torch.inference_mode()
    def prefill(self,prompt:str):
        inputs=self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        input_ids =inputs["input_ids"].to(self.target.device)
        attention_mask = inputs["attention_mask"].to(self.target.device)

        outputs = self.target(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True, # Eagle arch
            use_cache=True,
            return_dict=True,
        )

        logits=outputs.logits
        hidden_states= outputs.hidden_states
        kv_cache= outputs.past_key_values

        print("Input shape:")
        print(input_ids.shape)

        print("Logits shape:")
        print(logits.shape)

        print("Number of hidden-state tensors:")
        print(len(hidden_states))

        print("Hidden-state shapes:")
        for i, h in enumerate(hidden_states):
            print(i, h.shape)

        return {
            "input_ids":input_ids,
            "logits":logits,
            "hidden_states":hidden_states,
            "kv_cache": kv_cache,
        }


if __name__ == "__main__":
    decoder = EagleDecoder()

    state = decoder.prefill(
        "The capital of France is"
    )