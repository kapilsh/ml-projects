import torch
import pytest
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from gpt import GPT2


def test_model_correctness():
    hf_model = GPT2LMHeadModel.from_pretrained('gpt2', resume_download=None)
    hf_model.eval()

    gpt2_model = GPT2.from_pretrained()
    gpt2_model.eval()

    torch.random.manual_seed(42)
    model_input = torch.randint(0, 50257, (1, 30))
    gpt2_output = gpt2_model(model_input)
    hf_output = hf_model(model_input)

    assert torch.allclose(hf_output.logits, gpt2_output, atol=1e-4)
