import openai
from typing import List, Tuple

openai.api_key = "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"
model_engine = "gpt-3.5-turbo"

persona = "You are a victim of scam."
command = "Please extend the conversation below by one turn. Your answer will be in one sentence and follow this format: victim: {answer}.\n"

class ChatGPTEngine():
    def __init__(self) -> None:
        self._chat_history: List[Tuple[str, str]] = []

    def append_chat_history(self, text, role) -> None:
        if role == "attacker":
            self._chat_history.append(("scammer", text))
        elif role == "victim":
            self._chat_history.append(("victim", text))

    def get_question(self) -> str:
        history = ""
        for r, m in self._chat_history:
            history += "{}: {}\n".format(r, m)
        context = command + "\n" + history
        return context

    def generate_response(self) -> str:
        question = self.get_question()
        print("question", question)
        response_dict = openai.ChatCompletion.create(
                model = model_engine,
                messages = [{"role": "system", "content": persona}, {"role": "user", "content": question}]
                )
        response = response_dict["choices"][0]["message"]["content"] #type: ignore
        # exclude "1: " in front of the response
        response = response.split(": ")[-1]

        used_model = response_dict["model"] #type: ignore
        n_completion_tokens = response_dict["usage"]["completion_tokens"] #type: ignore
        n_prompt_tokens = response_dict["usage"]["prompt_tokens"] #type: ignore
        n_total_tokens = response_dict["usage"]["total_tokens"] #type: ignore

        print("response: {}, used_model: {}, n_completion_tokens: {}, n_prompt_tokens: {}, n_total_tokens: {}"\
                .format(response, used_model, n_completion_tokens, n_prompt_tokens, n_total_tokens))

        return response

    def reset(self):
        self._chat_history = []
