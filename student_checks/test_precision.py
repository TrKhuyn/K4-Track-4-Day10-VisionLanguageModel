import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from lab_runtime import float32_logits_hook


def test_grpo_logits_are_fp32_without_changing_frozen_fp16_head():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head=torch.nn.Linear(16,512,bias=False,dtype=torch.float16)
            self.lm_head.weight.requires_grad_(False)
        def forward(self,hidden):
            return CausalLMOutputWithPast(logits=self.lm_head(hidden))
    model=Model()
    original=model.lm_head.weight.detach().clone()
    model.register_forward_hook(float32_logits_hook)
    hidden=torch.randn(1,16,dtype=torch.float16,requires_grad=True)
    output=model(hidden)
    assert output.logits.dtype==torch.float32
    assert model.lm_head.weight.dtype==torch.float16
    assert torch.equal(model.lm_head.weight,original)
    output.logits.log_softmax(-1).sum().backward()
    assert torch.isfinite(hidden.grad).all()
    assert model.lm_head.weight.grad is None
