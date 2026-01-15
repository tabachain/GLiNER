import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from huggingface_hub import PyTorchModelHubMixin
from transformers.modeling_outputs import BaseModelOutput

# 互換性のためのRMSNorm定義
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * output

class LiquidMambaLayer(nn.Module):
    def __init__(self, d_model, d_state=128):
        super().__init__()
        self.mamba = Mamba2(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
        self.norm = RMSNorm(d_model)
    def forward(self, x):
        return self.norm(x + self.mamba(x))

class LiquidAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = RMSNorm(d_model)
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)

class BidirectionalLiquidMamba2(nn.Module, PyTorchModelHubMixin):
    def __init__(self, d_model=512, n_layers=16, d_state=128, vocab_size=64402, **kwargs):
        super().__init__()
        # configの受け渡しに対応
        self.config = kwargs.get("config", None)
        if self.config:
             # configオブジェクトから値を取得、もしくはデフォルト値
             d_model = getattr(self.config, "hidden_size", d_model) # GLiNER config uses hidden_size
             d_model = getattr(self.config, "d_model", d_model)     # If d_model is explicitly set
             vocab_size = getattr(self.config, "vocab_size", vocab_size)
        
        # LFM2 config.json の layer_types を完全再現
        self.layer_map = ["conv", "conv", "attn", "conv", "conv", "attn", "conv", "conv", "attn", "conv", "attn", "conv", "attn", "conv", "attn", "conv"]
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.fwd_layers = nn.ModuleList([LiquidMambaLayer(d_model, d_state) if t=="conv" else LiquidAttentionLayer(d_model) for t in self.layer_map])
        self.bwd_layers = nn.ModuleList([LiquidMambaLayer(d_model, d_state) if t=="conv" else LiquidAttentionLayer(d_model) for t in self.layer_map])
        
        # output_projはGLiNERのbackboneとしては不要かもしれないが、PreTrainedの互換性のために残すか、
        # GLiNERは `last_hidden_state` を使うので、ここでは `x` をそのまま返す形に修正する。
        # オリジナルのコードには output_proj があった。
        self.output_proj = nn.Linear(d_model, 1024) 

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embedding(input_ids)
        for f_layer, b_layer in zip(self.fwd_layers, self.bwd_layers):
            f_out = f_layer(x)
            x_flipped = torch.flip(x, [1])
            b_out = b_layer(x_flipped)
            x = f_out + torch.flip(b_out, [1])
        
        # GLiNERのEncoderは (batch, seq_len, hidden_size) を期待している
        # original: return self.output_proj(x) -> (batch, seq_len, 1024)
        # d_model=512 だが output_proj で 1024 になっている。
        # GLiNERの hidden_size 設定と合わせる必要がある。
        
        last_hidden_state = self.output_proj(x)
        
        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=None,
            attentions=None
        )
