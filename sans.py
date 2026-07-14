import torch.nn as nn
import torch.nn.functional as F
import torch

class StackedAttentionNets(nn.Module):
    def __init__(self, d=768, k=768):
        super(StackedAttentionNets, self).__init__()
        self.ff_image = nn.Linear(d, k)
        self.ff_ques = nn.Linear(d, k)
        self.dropout = nn.Dropout(0.3)
        self.ff_attention = nn.Linear(k, 1)
    def forward(self, vi, vq):
        hi = self.ff_image(vi)
        hq = self.ff_ques(vq)
        ha = F.gelu(hi + hq)
        ha = self.dropout(ha)
        ha = self.ff_attention(ha).squeeze(dim=2)
        pi = F.softmax(ha, dim=1)
        vi_attended = (pi.unsqueeze(dim=2) * vi).sum(dim=1)
        u = vi_attended + vq.squeeze(1)
        return u


class FeedForward(nn.Module):
    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


class GLUGate(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.tanh_proj = nn.Linear(2 * d_model, d_model)
        self.sigmoid_proj = nn.Linear(2 * d_model, d_model)

    def forward(self, concat_features, fallback):
        limited = torch.tanh(self.tanh_proj(concat_features))
        gate = torch.sigmoid(self.sigmoid_proj(concat_features))
        return limited * gate + (1.0 - gate) * fallback


class SAGLayer(nn.Module):
    """Self-attention gated unit from MAGM Eq. (3)-(9)."""

    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = FeedForward(d_model=d_model, dropout=dropout)
        self.gate = GLUGate(d_model=d_model)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.gate_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, question_features, key_padding_mask=None):
        attn_features, _ = self.self_attn(
            question_features,
            question_features,
            question_features,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x1 = self.attn_norm(question_features + self.dropout(attn_features))
        x2 = self.ffn_norm(x1 + self.dropout(self.ffn(x1)))
        x3 = torch.cat([x2, x1], dim=-1)
        return self.gate_norm(self.gate(x3, x2))


class SGAGLayer(nn.Module):
    """Self-guided attention gated unit from MAGM Eq. (10)-(15)."""

    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = FeedForward(d_model=d_model, dropout=dropout)
        self.gate = GLUGate(d_model=d_model)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.gate_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_features, question_features, question_padding_mask=None):
        attended, _ = self.cross_attn(
            image_features,
            question_features,
            question_features,
            key_padding_mask=question_padding_mask,
            need_weights=False,
        )
        y1 = self.attn_norm(image_features + self.dropout(attended))
        y2 = self.ffn_norm(y1 + self.dropout(self.ffn(y1)))
        y3 = torch.cat([y2, image_features], dim=-1)
        return self.gate_norm(self.gate(y3, y2))


class AttentionPooling(nn.Module):
    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, features, padding_mask=None):
        logits = self.score(features).squeeze(-1)
        if padding_mask is not None:
            logits = logits.masked_fill(padding_mask, torch.finfo(logits.dtype).min)
        weights = F.softmax(logits, dim=1)
        return torch.sum(weights.unsqueeze(-1) * features, dim=1)


class AdaptiveGatedFusion(nn.Module):
    """Attention pooling plus adaptive gated fusion from MAGM Eq. (18)-(26)."""

    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        self.question_pool = AttentionPooling(d_model=d_model, dropout=dropout)
        self.image_pool = AttentionPooling(d_model=d_model, dropout=dropout)
        self.question_proj = nn.Linear(d_model, d_model)
        self.image_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, question_features, image_features, question_padding_mask=None):
        pooled_question = self.question_pool(question_features, question_padding_mask)
        pooled_image = self.image_pool(image_features)

        h_q = torch.sigmoid(pooled_question)
        h_i = torch.sigmoid(pooled_image)
        h = (h_q * h_q) + (h_i * h_i)
        z = torch.sigmoid(h)
        z_q = z * h_q
        z_i = (1.0 - z) * h_i
        return self.norm(self.question_proj(z_q) + self.image_proj(z_i))
    
if __name__=="__main__":
    from config import config
    from features_extraction import ImageEmbedding, QuesEmbedding
    from data_processing import build_dataloaders

    image_model = ImageEmbedding(output_size=config.d_model).to(config.DEVICE)
    ques_model = QuesEmbedding(output_size=config.d_model).to(config.DEVICE)

    train_loader, _, _ = build_dataloaders()
    for batch in train_loader:
        images, questions, answers = batch
        if torch.cuda.is_available():
            images = images.to(config.DEVICE)
            questions = questions

        with torch.no_grad():
            image_embeddings = image_model(images)
            ques_embeddings = ques_model(questions)
        break

    image_embeddings = image_embeddings.reshape(config.BATCH_SIZE, config.d_model, -1).permute(0, 2, 1)
    ques_embeddings = ques_embeddings.unsqueeze(1)

    san_model = StackedAttentionNets(d=config.d_model, k=768).to(config.DEVICE)
    img_text_att = san_model(image_embeddings.to(config.DEVICE), ques_embeddings.to(config.DEVICE))
    
    print(f"features combined with SANs size: {img_text_att.size()}")
