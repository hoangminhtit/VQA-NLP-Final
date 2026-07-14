from config import config
from transformers import AutoImageProcessor, AutoTokenizer, BertModel, CLIPVisionModel
import torch
import torch.nn as nn
from xlstm_decoder import xLSTM

class ImageEmbedding(nn.Module):
    def __init__(self, output_size=config.d_model):
        super(ImageEmbedding, self).__init__()
        self.process = AutoImageProcessor.from_pretrained(config.IMG_DIR)
        self.model = CLIPVisionModel.from_pretrained(config.IMG_DIR)
        self.model.requires_grad_(False)
        self.register_buffer(
            "image_mean",
            torch.tensor(self.process.image_mean).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor(self.process.image_std).view(1, 3, 1, 1),
        )

    def train(self, mode=True):
        # The vision backbone is intentionally frozen; keep dropout disabled too.
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, image):
        # Inputs are already resized tensors in [0, 1]. Normalize directly on the
        # active device and avoid moving CUDA tensors back through a CPU processor.
        pixel_values = image.to(config.DEVICE)
        pixel_values = (pixel_values - self.image_mean) / self.image_std
        with torch.no_grad():
            # Preserve patch tokens for spatial attention instead of returning one
            # pooled image vector from get_image_features().
            outputs = self.model(
                pixel_values=pixel_values
            ).last_hidden_state
        return outputs

class QuesEmbedding(nn.Module):
    def __init__(
        self,
        input_size=config.d_model,
        output_size=config.d_model,
        return_sequence=False,
    ):
        super(QuesEmbedding, self).__init__()
        self.return_sequence = return_sequence
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_DIR)
        self.text_model = BertModel.from_pretrained(config.TEXT_DIR)
        self.xlstm = xLSTM(
            input_size=input_size,
            hidden_size=output_size,
            num_heads=8,
            layers=['m']
        )


    def forward(self, ques, return_attention_mask=False):
        if isinstance(ques, tuple):
            ques = list(ques)
        elif isinstance(ques, str):
            ques = [ques]

        tokenized_input = self.tokenizer(
            ques,
            return_tensors='pt',
            padding='max_length',
            max_length=config.MAX_LEN,
            truncation=True
        )

        ques = self.text_model(**tokenized_input.to(config.DEVICE)).last_hidden_state

        output, _ = self.xlstm(ques)
        if self.return_sequence:
            if return_attention_mask:
                return output, tokenized_input["attention_mask"].to(output.device)
            return output

        mask = tokenized_input["attention_mask"].to(output.device).unsqueeze(-1)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        if return_attention_mask:
            return pooled, tokenized_input["attention_mask"].to(output.device)
        return pooled
    
class AnsEmbedding(nn.Module):
    def __init__(self, input_size=config.d_model):
        super(AnsEmbedding, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_DIR)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.sep_token
        self.bos_token_id = self.tokenizer.cls_token_id
        self.eos_token_id = self.tokenizer.sep_token_id

    def prepare_decoder_batch(self, ans, max_length=config.MAX_LEN):
        if isinstance(ans, tuple):
            ans = list(ans)
        elif isinstance(ans, str):
            ans = [ans]

        encoded = self.tokenizer(
            ans,
            add_special_tokens=False,
            padding=False,
            max_length=max_length - 1,
            truncation=True,
        )
        batch_size = len(ans)
        decoder_inputs = torch.full(
            (batch_size, max_length),
            self.bos_token_id,
            dtype=torch.long,
            device=config.DEVICE,
        )
        labels = torch.full(
            (batch_size, max_length), -100, dtype=torch.long, device=config.DEVICE
        )
        for row, token_ids in enumerate(encoded["input_ids"]):
            targets = token_ids + [self.eos_token_id]
            length = len(targets)
            labels[row, :length] = torch.tensor(targets, device=config.DEVICE)
            if length > 1:
                decoder_inputs[row, 1:length] = torch.tensor(
                    targets[:-1], device=config.DEVICE
                )
        return decoder_inputs, labels

    def forward(self, ans):
        _, labels = self.prepare_decoder_batch(ans)
        return labels
    
if __name__=="__main__":
    from data_processing import build_dataloaders
    image_model = ImageEmbedding(output_size=config.d_model).to(config.DEVICE)
    ques_model = QuesEmbedding(output_size=config.d_model).to(config.DEVICE)
    ans_model = AnsEmbedding().to(config.DEVICE)

    train_loader, _, _ = build_dataloaders()
    for batch in train_loader:
        images, questions, answers = batch
        if torch.cuda.is_available():
            images = images.to(config.DEVICE)
            questions = questions
            answers = answers

        with torch.no_grad():
            image_embeddings = image_model(images)
            ques_embeddings = ques_model(questions)
            ans_tokens = ans_model(answers)
        break

    image_embeddings = image_embeddings.reshape(config.BATCH_SIZE, config.d_model, -1).permute(0, 2, 1)
    ques_embeddings = ques_embeddings.unsqueeze(1)
    
    print(f"image embeddings size: {image_embeddings.size()}")
    print(f"question embeddings size: {ques_embeddings.size()}")
    print(f"answer token size: {ans_tokens.size()}")
