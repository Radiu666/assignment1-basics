import torch
import torch.nn as nn
import torch.nn.init as init
from einops import rearrange, einsum, reduce, repeat

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        variance = 1 / (self.in_features + self.out_features)
        std = variance ** 0.5
        init.trunc_normal_(self.W, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x):
        return einsum(x, self.W, '... input, output input -> ... output')

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.W = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        std = 1
        init.trunc_normal_(self.W, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W[x]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.scale = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.rsqrt(rms + self.eps)
        x = x * rms * self.scale
        return x.to(in_type)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def SiLU(self, x):
        return x * torch.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.SiLU(self.w1(x)) * self.w3(x))

class ROPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        t = torch.arange(max_seq_len, device=device)
        freqs = torch.einsum('i,j->ij', t, freqs) # max_seq_len x (d_k/2)
        self.register_buffer('cos_cache', torch.cos(freqs))
        self.register_buffer('sin_cache', torch.sin(freqs))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        new_even = x_even * cos - x_odd * sin
        new_odd = x_even * sin + x_odd * cos
        stacked = torch.stack((new_even, new_odd), dim=-1)
        res = stacked.flatten(start_dim=-2)
        return res

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_num = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - max_num)
    x_exp_sum = x_exp.sum(dim=dim, keepdim=True)
    return x_exp / x_exp_sum

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    d_k = Q.shape[-1]
    scores = einsum(Q, K, '... query d_k, ... keys d_k -> ... query keys') / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))
    return softmax(scores, dim=-1) @ V

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads

        self.w_q = Linear(d_model, d_model)
        self.w_k = Linear(d_model, d_model)
        self.w_v = Linear(d_model, d_model)
        self.w_o = Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        Q = rearrange(Q, 'batch seq (head d_k) -> batch head seq d_k', head = self.num_heads)
        K = rearrange(K, 'batch seq (head d_k) -> batch head seq d_k', head = self.num_heads)
        V = rearrange(V, 'batch seq (head d_k) -> batch head seq d_k', head = self.num_heads)

        mask = torch.tril(torch.ones((x.size(1), x.size(1)), device=x.device)).bool()
        attention_output = scaled_dot_product_attention(Q, K, V, mask)
        attention_output = rearrange(attention_output, 'batch head seq d_k -> batch seq (head d_k)')
        output = self.w_o(attention_output)
        return output

class MultiHeadSelfAttentionWithROPE(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, theta):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        assert self.d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = self.d_model // num_heads
        self.w_q = Linear(d_model, d_model)
        self.w_k = Linear(d_model, d_model)
        self.w_v = Linear(d_model, d_model)
        self.w_o = Linear(d_model, d_model)
        self.ROPE = ROPE(theta, self.d_k, max_seq_len)
    
    def forward(self, x, token_positions):
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        Q = rearrange(Q, 'batch seq (head d_k) -> batch head seq d_k', head=self.num_heads)
        K = rearrange(K, 'batch seq (head d_k) -> batch head seq d_k', head=self.num_heads)
        V = rearrange(V, 'batch seq (head d_k) -> batch head seq d_k', head=self.num_heads)

        # pos = token_positions.unsqueeze(1).expand(-1, self.num_heads, -1)
        pos = repeat(token_positions, 'b s -> b h s', h=self.num_heads)
        Q = self.ROPE(Q, pos)
        K = self.ROPE(K, pos)

        mask = torch.tril(torch.ones(x.size(1), x.size(1), device=x.device)).bool()
        attention_output = scaled_dot_product_attention(Q, K, V, mask)
        attention_output = rearrange(attention_output, 'batch head seq d_k -> batch seq (head d_k)')
        output = self.w_o(attention_output)
        return output
    
class TransfomerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = self.d_model / self.num_heads
        self.RMSNorm1 = RMSNorm(d_model=self.d_model)
        self.RMSNorm2 = RMSNorm(d_model=self.d_model)
        self.MHA_ROPE = MultiHeadSelfAttentionWithROPE(self.d_model, self.num_heads, self.max_seq_len, self.theta)
        self.FFN = SwiGLU(self.d_model, self.d_ff)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
        n1_x = self.RMSNorm1(x)
        mha_x = self.MHA_ROPE(n1_x, token_positions)
        x1 = x + mha_x
        n2_x = self.RMSNorm2(x1)
        ffn_x = self.FFN(n2_x)
        return x1 + ffn_x
    
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_size, d_model, num_layers, num_heads, d_ff, rope_theta):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.token_embedding = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransfomerBlock(d_model, num_heads, d_ff, context_size, rope_theta) for _ in range(num_layers)
        ])
        self.rmsnorm = RMSNorm(d_model)
        self.output_projection = Linear(d_model, vocab_size)

    def forward(self, x):
        x_embedding = self.token_embedding(x)
        for layer in self.layers:
            x_embedding = layer(x_embedding)
        x_embedding = self.rmsnorm(x_embedding)
        logits = self.output_projection(x_embedding)
        return logits

if __name__ == "__main__":
    # # Test the Linear layer
    # batch_size = 2
    # in_features = 3
    # out_features = 4
    # x = torch.randn(batch_size, in_features)
    # linear_layer = Linear(in_features, out_features)
    # output = linear_layer(x)
    # print("Input shape:", x.shape)
    # print("Output shape:", output.shape)

    # # Test the RMSNorm layer
    # batch_size = 2
    # d_model = 4
    # x = torch.randn(batch_size, d_model)
    # rmsnorm_layer = RMSNorm(d_model)
    # output = rmsnorm_layer(x)
    # print("Input shape:", x.shape)
    # print("Output shape:", output.shape)

    # # Test the ROPE layer
    # batch_size = 2
    # d_k = 4
    # max_seq_len = 10
    # theta = 10000.0
    # x = torch.randn(batch_size, max_seq_len, d_k)
    # token_positions = torch.arange(max_seq_len).unsqueeze(0).expand(batch_size, -1)
    # rope_layer = ROPE(theta, d_k, max_seq_len)
    # output = rope_layer(x, token_positions)
    # print("Input shape:", x.shape)
    
    # Test the softmax function
    # x = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    # softmax_output = softmax(x, dim=-1)
    # print("Softmax output:", softmax_output)

    # Test the scaled dot product attention
    # batch_size = 2
    # num_heads = 2
    # seq_len = 3
    # d_k = 4
    # Q = torch.randn(batch_size, num_heads, seq_len, d_k)
    # K = torch.randn(batch_size, num_heads, seq_len, d_k)
    # V = torch.randn(batch_size, num_heads, seq_len, d_k)
    # mask = torch.tensor([[[[True, True, False], [True, True, True], [False, True, True]], [[True, False, True], [False, True, True], [True, True, True]]], [[[True, True, True], [True, True, False], [True, False, True]], [[False, True, True], [True, True, True], [True, True, False]]]])
    # attention_output = scaled_dot_product_attention(Q, K, V, mask)
    # print("Attention output shape:", attention_output.shape)
    
    # Test the MultiHeadSelfAttention layer
    # batch_size = 2
    # d_model = 8
    # num_heads = 2
    # seq_len = 5
    # x = torch.randn(batch_size, seq_len, d_model)
    # mha_layer = MultiHeadSelfAttention(d_model, num_heads)
    # output = mha_layer(x)
    # print("Input shape:", x.shape)
    # print("Output shape:", output.shape)

    # Test the MultiHeadSelfAttentionWithROPE layer
    batch_size = 2
    d_model = 8
    num_heads = 2
    seq_len = 5
    max_seq_len = 10
    theta = 10000.0
    x = torch.randn(batch_size, seq_len, d_model)
    token_positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    mha_rope_layer = MultiHeadSelfAttentionWithROPE(d_model, num_heads, max_seq_len, theta)
    output = mha_rope_layer(x, token_positions)
    print("Input shape:", x.shape)
    print("Over")