import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockDiagonal(nn.Module):
    """Independent linear projections for each recurrent head."""

    def __init__(self, in_features, out_features, num_blocks):
        super().__init__()
        if in_features % num_blocks or out_features % num_blocks:
            raise ValueError("in_features and out_features must be divisible by num_blocks")
        self.num_blocks = num_blocks
        block_in = in_features // num_blocks
        block_out = out_features // num_blocks
        self.blocks = nn.ModuleList(
            nn.Linear(block_in, block_out) for _ in range(num_blocks)
        )

    def forward(self, x):
        return torch.cat(
            [block(part) for block, part in zip(self.blocks, x.chunk(self.num_blocks, dim=-1))],
            dim=-1,
        )


class sLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, proj_factor=4 / 3):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.layer_norm = nn.LayerNorm(input_size)
        self.Wz = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wi = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wf = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wo = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Rz = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Ri = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Rf = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Ro = BlockDiagonal(hidden_size, hidden_size, num_heads)

        projected_size = int(hidden_size * proj_factor)
        self.group_norm = nn.GroupNorm(num_heads, hidden_size)
        self.up_proj_left = nn.Linear(hidden_size, projected_size)
        self.up_proj_right = nn.Linear(hidden_size, projected_size)
        self.down_proj = nn.Linear(projected_size, input_size)

    def initial_state(self, batch_size, device, dtype):
        zeros = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        return tuple(zeros.clone() for _ in range(4))

    def forward(self, x, prev_state):
        h_prev, c_prev, n_prev, m_prev = prev_state
        x_norm = self.layer_norm(x)
        z = torch.tanh(self.Wz(x_norm) + self.Rz(h_prev))
        o = torch.sigmoid(self.Wo(x_norm) + self.Ro(h_prev))
        i_tilde = self.Wi(x_norm) + self.Ri(h_prev)
        f_tilde = self.Wf(x_norm) + self.Rf(h_prev)

        m_t = torch.maximum(f_tilde + m_prev, i_tilde)
        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + m_prev - m_t)
        c_t = f * c_prev + i * z
        n_t = f * n_prev + i
        h_t = o * c_t / n_t.clamp_min(1e-6)

        normalized = self.group_norm(h_t)
        gated = self.up_proj_left(normalized) * F.gelu(self.up_proj_right(normalized))
        output = x + self.down_proj(gated)
        return output, (h_t, c_t, n_t, m_t)


class mLSTMBlock(nn.Module):
    """Multi-head mLSTM with a per-sample matrix memory.

    C has shape [batch, heads, head_size, head_size]. No operation reduces over
    the batch dimension, so samples cannot affect one another.
    """

    def __init__(self, input_size, hidden_size, num_heads, proj_factor=2):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.layer_norm = nn.LayerNorm(input_size)
        self.q_proj = nn.Linear(input_size, hidden_size)
        self.k_proj = nn.Linear(input_size, hidden_size)
        self.v_proj = nn.Linear(input_size, hidden_size)
        # One exponential input/forget gate and one output gate per head.
        self.i_proj = nn.Linear(input_size, num_heads)
        self.f_proj = nn.Linear(input_size, num_heads)
        self.o_proj = nn.Linear(input_size, num_heads)

        projected_size = int(hidden_size * proj_factor)
        self.group_norm = nn.GroupNorm(num_heads, hidden_size)
        self.up_proj_left = nn.Linear(hidden_size, projected_size)
        self.up_proj_right = nn.Linear(hidden_size, projected_size)
        self.down_proj = nn.Linear(projected_size, input_size)

    def initial_state(self, batch_size, device, dtype):
        h = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        c = torch.zeros(
            batch_size,
            self.num_heads,
            self.head_size,
            self.head_size,
            device=device,
            dtype=dtype,
        )
        n = torch.zeros(
            batch_size, self.num_heads, self.head_size, device=device, dtype=dtype
        )
        m = torch.zeros(batch_size, self.num_heads, 1, device=device, dtype=dtype)
        return h, c, n, m

    def forward(self, x, prev_state):
        _, c_prev, n_prev, m_prev = prev_state
        batch_size = x.size(0)
        x_norm = self.layer_norm(x)

        def heads(projection):
            return projection(x_norm).view(batch_size, self.num_heads, self.head_size)

        q = heads(self.q_proj)
        k = heads(self.k_proj) / (self.head_size**0.5)
        v = heads(self.v_proj)
        i_tilde = self.i_proj(x_norm).unsqueeze(-1)
        f_tilde = self.f_proj(x_norm).unsqueeze(-1)
        o = torch.sigmoid(self.o_proj(x_norm)).unsqueeze(-1)

        m_t = torch.maximum(f_tilde + m_prev, i_tilde)
        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + m_prev - m_t)

        # Covariance update v k^T: [B, H, Dh, 1] @ [B, H, 1, Dh].
        c_t = f.unsqueeze(-1) * c_prev + i.unsqueeze(-1) * (
            v.unsqueeze(-1) * k.unsqueeze(-2)
        )
        n_t = f * n_prev + i * k
        numerator = torch.matmul(c_t, q.unsqueeze(-1)).squeeze(-1)
        denominator = torch.maximum(
            torch.abs((n_t * q).sum(dim=-1, keepdim=True)),
            torch.ones((), device=x.device, dtype=x.dtype),
        )
        h_heads = o * numerator / denominator
        h_t = h_heads.reshape(batch_size, self.hidden_size)

        normalized = self.group_norm(h_t)
        gated = self.up_proj_left(normalized) * F.silu(self.up_proj_right(normalized))
        output = x + self.down_proj(gated)
        return output, (h_t, c_t, n_t, m_t)


class xLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_heads,
        layers,
        batch_first=True,
        proj_factor_slstm=4 / 3,
        proj_factor_mlstm=2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        blocks = []
        for layer_type in layers:
            if layer_type == "s":
                blocks.append(
                    sLSTMBlock(input_size, hidden_size, num_heads, proj_factor_slstm)
                )
            elif layer_type == "m":
                blocks.append(
                    mLSTMBlock(input_size, hidden_size, num_heads, proj_factor_mlstm)
                )
            else:
                raise ValueError(f"Invalid layer type {layer_type!r}; expected 's' or 'm'")
        self.layers = nn.ModuleList(blocks)

    def _initial_state(self, batch_size, device, dtype):
        return [
            layer.initial_state(batch_size, device=device, dtype=dtype)
            for layer in self.layers
        ]

    def forward(self, x, state=None):
        if not self.batch_first:
            x = x.transpose(0, 1)
        batch_size, seq_len, _ = x.shape
        if state is None:
            state = self._initial_state(batch_size, x.device, x.dtype)
        if len(state) != len(self.layers):
            raise ValueError("state must contain one state tuple per xLSTM layer")

        outputs = []
        states = list(state)
        for t in range(seq_len):
            x_t = x[:, t]
            for layer_idx, layer in enumerate(self.layers):
                x_t, states[layer_idx] = layer(x_t, states[layer_idx])
            outputs.append(x_t)
        output = torch.stack(outputs, dim=1)
        if not self.batch_first:
            output = output.transpose(0, 1)
        return output, states


if __name__ == "__main__":
    model = xLSTM(16, 32, 4, ["s", "m", "s"], batch_first=True)
    output, _ = model(torch.randn(2, 10, 16))
    print("Output size:", output.size())
