"""
OpenRouter-based drop-in replacement for local Qwen2.5-7B-Instruct.

Provides OpenRouterTokenizer and OpenRouterModel classes that mimic the
HuggingFace tokenizer/model API used by the judge() function in:
  complex_plot.py, complex_landscape.py, human_interaction.py,
  motion_order_understanding.py

Usage:
    from vbench2.openrouter_qwen import OpenRouterTokenizer, OpenRouterModel

    tokenizer = OpenRouterTokenizer()
    model = OpenRouterModel(api_key=os.environ["OPENROUTER_API_KEY"])

    # Then pass to judge(prompt, sys_prompt, tokenizer, model) unchanged.
"""

import os
import json
import requests


# Sentinel key for tracking chat messages between apply_chat_template and generate
_OPENROUTER_MESSAGES = "__openrouter_messages__"


class _FakeInputIds(list):
    """A list subclass that also carries the original text for OpenRouter."""
    def __init__(self, text):
        super().__init__([[]])  # looks like [[]] so len(input_ids[0]) == 0
        self._text = text


class _FakeModelInputs:
    """Mimics the dict-like object returned by tokenizer(..., return_tensors='pt').

    The judge() code does:
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(**model_inputs, ...)

    **model_inputs unpacks via keys()/getitem, passing input_ids and attention_mask
    to model.generate().
    """

    def __init__(self, text):
        self._text = text
        self.input_ids = _FakeInputIds(text)

    def to(self, device):
        return self

    def keys(self):
        return ["input_ids", "attention_mask"]

    def __getitem__(self, key):
        if key == "input_ids":
            return self.input_ids
        if key == "attention_mask":
            return None
        raise KeyError(key)


class OpenRouterTokenizer:
    """Mimics HuggingFace tokenizer interface for OpenRouter usage."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        """Encode messages as JSON string instead of tokenizing."""
        return json.dumps({_OPENROUTER_MESSAGES: messages})

    def __call__(self, texts, return_tensors=None):
        """Mimics tokenizer([text], return_tensors='pt')."""
        return _FakeModelInputs(texts[0])

    def batch_decode(self, generated_ids, skip_special_tokens=True):
        """generated_ids is a list of response strings from OpenRouter."""
        return generated_ids


class OpenRouterModel:
    """Mimics HuggingFace model interface, routing generate() to OpenRouter API."""

    def __init__(self, api_key=None, model_name="qwen/qwen-2.5-7b-instruct"):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model_name = model_name
        self.device = "cpu"  # placeholder for .to(model.device)
        self._api_url = "https://openrouter.ai/api/v1/chat/completions"

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        """Route generation to OpenRouter API.

        The judge() code calls:
            model.generate(**model_inputs, do_sample=False, temperature=0, max_new_tokens=512)

        model_inputs unpacks to input_ids=_FakeInputIds, attention_mask=None.
        _FakeInputIds carries the original JSON text with chat messages.
        """
        # Extract the stored messages from input_ids
        text = None
        if hasattr(input_ids, '_text'):
            text = input_ids._text

        if text is None:
            raise ValueError("Could not extract messages from generate() input_ids")

        # Parse the messages from our JSON encoding
        try:
            parsed = json.loads(text)
            messages = parsed[_OPENROUTER_MESSAGES]
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback: treat as raw user message
            messages = [{"role": "user", "content": str(text)}]

        # Call OpenRouter API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 512,
        }

        response = requests.post(self._api_url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()

        content = result["choices"][0]["message"]["content"]

        # The judge() code does:
        #   generated_ids = [output_ids[len(input_ids):]
        #                    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        #
        # model_inputs.input_ids = _FakeInputIds (a list containing [[]])
        # So zip iterates once: input_ids=[], output_ids=content
        # len([]) = 0, content[0:] = content  (string slicing works!)
        #
        # Then: tokenizer.batch_decode([content]) returns [content]
        # And: [content][0] = content (the response text)
        return [content]
