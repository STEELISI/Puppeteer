from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer # type: ignore
import torch

model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")  # type: ignore
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium") # type: ignore

class NeuralDialogEngine():
    def __init__(self) -> None:
        self._chat_history: List[str] = []
        self._chat_history_ids = None
        self._chat_history_lens: List[int] = []

    def reset(self):
        self._chat_history = []
        self._chat_history_ids = None
        self._chat_history_lens = []

    def append_chat_history(self, text, text_ids=None) -> None:
        self._chat_history.append(text)
        if text_ids == None:
            text_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
        self._chat_history_lens.append(text_ids.shape[-1])
        if self._chat_history_ids == None:
            self._chat_history_ids = text_ids
        else:
            self._chat_history_ids = torch.cat([self._chat_history_ids, text_ids], dim=-1)

    def generate_response(self, prior_turns=1) -> str:
        if self._chat_history_ids == None:
            raise TypeError("Can not generate response, chat_history_ids is None")
        prev_len = sum(self._chat_history_lens[-prior_turns:])
        # print(prev_len)
        fully_generated_ids = model.generate(self._chat_history_ids[:, -prev_len:], max_length=1000, pad_token_id=tokenizer.eos_token_id)
        # print("fully_generated_ids: {}, shape: {}".format(fully_generated_ids, fully_generated_ids.shape))
        generated_ids = fully_generated_ids[:, prev_len:][0].reshape(1, -1)
        # print("generated_ids: {}, shape: {}".format(generated_ids, generated_ids.shape))
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # print("DialoGPT: {}".format(generated_text))
        self.append_chat_history(generated_text, generated_ids)
        return generated_text

