from config import config
import torch.nn as nn
import torch
from transformers import AutoTokenizer
from features_extraction import ImageEmbedding, QuesEmbedding
from sans import StackedAttentionNets
from xlstm_decoder import xLSTM

torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(config.TEXT_DIR)
vocab = tokenizer.get_vocab()

class DualGatedFusion(nn.Module):
    def __init__(self, d_model=768, dropout=0.1):
        super(DualGatedFusion, self).__init__()
        self.image_proj = nn.Linear(d_model, d_model)
        self.text_proj = nn.Linear(d_model, d_model)
        self.image_gate = nn.Linear(d_model * 2, d_model)
        self.text_gate = nn.Linear(d_model * 2, d_model)
        self.norm_img = nn.LayerNorm(d_model)
        self.norm_txt = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_tokens, question_token):
        # image_tokens: [B, N, D], question_token: [B, 1, D]
        question_expanded = question_token.expand(-1, image_tokens.size(1), -1)
        image_for_gate = torch.cat([image_tokens, question_expanded], dim=-1)
        text_for_gate = torch.cat([question_expanded, image_tokens], dim=-1)

        g_img = torch.sigmoid(self.image_gate(image_for_gate))
        g_txt = torch.sigmoid(self.text_gate(text_for_gate))

        fused_image_tokens = g_img * self.image_proj(image_tokens) + (1.0 - g_img) * self.text_proj(question_expanded)
        fused_text_tokens = g_txt * self.text_proj(question_expanded) + (1.0 - g_txt) * self.image_proj(image_tokens)

        fused_image_tokens = self.norm_img(self.dropout(fused_image_tokens))
        fused_text_token = self.norm_txt(self.dropout(fused_text_tokens.mean(dim=1, keepdim=True)))
        return fused_image_tokens, fused_text_token

class VQAModel(nn.Module):
    def __init__(self, vocab_size=len(vocab), output_size=768, d_model=768,
                 num_heads=8, hidden_size=768,num_att_layers=4, use_dual_gating=False):
        super(VQAModel, self).__init__()
        self.use_dual_gating = use_dual_gating
        self.image_model = ImageEmbedding(output_size=output_size).to(config.DEVICE)
        self.ques_model = QuesEmbedding(output_size=output_size).to(config.DEVICE)
        self.dual_gated_fusion = DualGatedFusion(d_model=d_model).to(config.DEVICE)
        self.san_model = nn.ModuleList(
                        [StackedAttentionNets(d=d_model, k=768) for _ in range(num_att_layers)]).to(config.DEVICE)

        self.decoder = xLSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_heads=num_heads,
            layers=['m']
        ).to(config.DEVICE)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(config.MAX_LEN, d_model)
        self.context_norm = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, vocab_size)
        )

    def _extract_image_tensor(self, image_outputs):
        """
        Normalize HuggingFace vision outputs to a tensor.
        """
        if torch.is_tensor(image_outputs):
            return image_outputs

        if hasattr(image_outputs, "last_hidden_state") and image_outputs.last_hidden_state is not None:
            return image_outputs.last_hidden_state

        if hasattr(image_outputs, "pooler_output") and image_outputs.pooler_output is not None:
            return image_outputs.pooler_output

        raise TypeError(
            f"Unsupported image output type from encoder: {type(image_outputs)}"
        )

    def encode_context(self, images, questions):
        image_outputs = self.image_model(images.to(config.DEVICE))
        image_embeddings = self._extract_image_tensor(image_outputs)
        batch_size = images.size(0)

        if image_embeddings.dim() == 3:
            # [batch, seq_len, hidden]
            image_embedds = image_embeddings
        else:
            # [batch, hidden] -> [batch, 1, hidden]
            image_embedds = image_embeddings.reshape(batch_size, 768, -1).permute(0, 2, 1)

        ques_embeddings = self.ques_model(questions)
        ques_embedds = ques_embeddings.unsqueeze(1)

        if self.use_dual_gating:
            image_embedds, ques_embedds = self.dual_gated_fusion(
                image_embedds.to(config.DEVICE),
                ques_embedds.to(config.DEVICE),
            )

        # Each SAN hop updates the query used by the following hop.
        query = ques_embedds.to(config.DEVICE)
        for att_layer in self.san_model:
            attended = att_layer(image_embedds.to(config.DEVICE), query)
            query = attended.unsqueeze(1)
        return self.context_norm(query.squeeze(1))

    def decode(self, context, decoder_input_ids):
        seq_len = decoder_input_ids.size(1)
        if seq_len > config.MAX_LEN:
            raise ValueError(f"Decoder length {seq_len} exceeds MAX_LEN={config.MAX_LEN}")
        positions = torch.arange(seq_len, device=decoder_input_ids.device).unsqueeze(0)
        decoder_inputs = (
            self.token_embedding(decoder_input_ids)
            + self.position_embedding(positions)
            + context.unsqueeze(1)
        )
        out, _ = self.decoder(decoder_inputs)
        return self.mlp(out)

    def forward(self, images, questions, decoder_input_ids=None, max_len=config.MAX_LEN):
        context = self.encode_context(images, questions)
        if decoder_input_ids is None:
            decoder_input_ids = torch.full(
                (images.size(0), max_len),
                tokenizer.eos_token_id,
                dtype=torch.long,
                device=images.device,
            )
        return self.decode(context, decoder_input_ids.to(context.device))

    @torch.no_grad()
    def generate(self, images, questions, max_len=config.MAX_LEN):
        context = self.encode_context(images, questions)
        batch_size = images.size(0)
        current_tokens = torch.full(
            (batch_size,),
            tokenizer.eos_token_id,
            dtype=torch.long,
            device=context.device,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=context.device)
        state = None
        generated = []

        for position in range(max_len):
            pos = torch.full(
                (batch_size,), position, dtype=torch.long, device=context.device
            )
            step_input = (
                self.token_embedding(current_tokens)
                + self.position_embedding(pos)
                + context
            ).unsqueeze(1)
            step_output, state = self.decoder(step_input, state)
            next_tokens = self.mlp(step_output[:, -1]).argmax(dim=-1)
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, tokenizer.eos_token_id),
                next_tokens,
            )
            generated.append(next_tokens)
            finished |= next_tokens.eq(tokenizer.eos_token_id)
            current_tokens = next_tokens
            if finished.all():
                break

        return torch.stack(generated, dim=1)
