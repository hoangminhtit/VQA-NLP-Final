import argparse
import os
import math
import logging
import string
from datetime import datetime
from collections import Counter
import torch
from PIL import Image
import torchvision.transforms as transforms
from config import config
from transformers import AutoTokenizer
from vqa_model import VQAModel
from data_processing import build_dataloaders
from features_extraction import AnsEmbedding
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, brevity_penalty

PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(text):
    if text is None:
        return ""
    normalized = text.lower().strip()
    normalized = normalized.translate(PUNCT_TABLE)
    normalized = " ".join(normalized.split())
    return normalized


def accuracy_score(prediction, ground_truth):
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def add_file_logger(prefix):
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", f"{prefix}_{timestamp}.log")
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logging.getLogger().addHandler(handler)
    logger.info("File logging enabled: %s", log_path)
    return log_path

tokenizer = AutoTokenizer.from_pretrained(config.TEXT_DIR)

def predict_batch(model, image_paths, questions, device):
    model.eval()

    with torch.no_grad():
        transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
        ])

        images = [transform(Image.open(image_path).convert("RGB")) for image_path in image_paths]
        images = torch.stack(images).to(device)  # Stack thành batch

        predicted_tokens = model.generate(images, questions)
        predicted_sentences = []

        for i in range(len(questions)):
            sentence_predicted = torch.argmax(predicted_tokens[i], dim=-1)
            predicted_sentence = tokenizer.decode(sentence_predicted, skip_special_tokens=True).strip()
            predicted_sentences.append(predicted_sentence)

    for i in range(len(questions)):
        print(f"Image {i+1}: {image_paths[i]}")
        print(f"Question: {questions[i]}")
        print(f"Predicted Answer: {predicted_sentences[i]}")
        print("-" * 50)
    
    return predicted_sentences


def load_model(checkpoint_path, device, use_dual_gating=False):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = VQAModel(use_dual_gating=use_dual_gating).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


def evaluate_test(model, ans_model, data_loader, device, batch_size):
    logger.info("Test Pipeline: start evaluation")
    model.eval()
    total_loss = 0.0
    total_bleu_1 = 0.0
    total_bleu_2 = 0.0
    total_bleu_3 = 0.0
    total_bleu_4 = 0.0
    total_bleu_scores = 0.0
    total_accuracy = 0.0
    total_f1 = 0.0
    num_batches = 0
    inference_logged = 0
    criterion = torch.nn.CrossEntropyLoss()
    smoother = SmoothingFunction()

    with torch.no_grad():
        for batch in data_loader:
            images, questions, answers = batch
            images = images.to(device)
            decoder_inputs, ans_embedds = ans_model.prepare_decoder_batch(answers)
            teacher_forced_logits = model(images, questions, decoder_inputs).float()
            generated_tokens = model.generate(images, questions)

            references = [answer.strip() for answer in answers]
            hypotheses = []
            for i in range(len(images)):
                predicted_sentence = tokenizer.decode(generated_tokens[i], skip_special_tokens=True)
                hypotheses.append(predicted_sentence.strip())

                if inference_logged < 50:
                    logger.info(
                        "Test inference %03d | question=%r | reference=%r | prediction=%r",
                        inference_logged + 1,
                        questions[i],
                        references[i],
                        hypotheses[-1],
                    )
                    inference_logged += 1

            tokenized_references = [[normalize_text(ref).split()] for ref in references]
            tokenized_hypotheses = [normalize_text(hyp).split() for hyp in hypotheses]
            bleu_score_1 = corpus_bleu(tokenized_references, tokenized_hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoother.method1)
            bleu_score_2 = corpus_bleu(tokenized_references, tokenized_hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother.method1)
            bleu_score_3 = corpus_bleu(tokenized_references, tokenized_hypotheses, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoother.method1)
            bleu_score_4 = corpus_bleu(tokenized_references, tokenized_hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother.method1)

            c = sum(len(hypothesis) for hypothesis in tokenized_hypotheses)
            r = sum(len(reference[0]) for reference in tokenized_references)
            bp = brevity_penalty(c, r)
            bleu_scores_all = [bleu_score_1, bleu_score_2, bleu_score_3, bleu_score_4]
            valid_bleu_scores = [score for score in bleu_scores_all if score > 0]
            bleu_score_total = bp * math.exp(sum(math.log(score) for score in valid_bleu_scores) / len(valid_bleu_scores)) if valid_bleu_scores else 0

            batch_accuracy = 0.0
            batch_f1 = 0.0
            for pred, ref in zip(hypotheses, references):
                batch_accuracy += accuracy_score(pred, ref)
                batch_f1 += f1_score(pred, ref)
            current_batch_size = len(images)
            total_accuracy += batch_accuracy / current_batch_size
            total_f1 += batch_f1 / current_batch_size

            loss = criterion(
                teacher_forced_logits.reshape(-1, teacher_forced_logits.size(-1)),
                ans_embedds.reshape(-1),
            )
            total_loss += loss.item()
            total_bleu_1 += bleu_score_1
            total_bleu_2 += bleu_score_2
            total_bleu_3 += bleu_score_3
            total_bleu_4 += bleu_score_4
            total_bleu_scores += bleu_score_total
            num_batches += 1
            logger.info(
                "Test batch | batch=%d | size=%d | loss=%.4f | accuracy=%.4f | f1=%.4f",
                num_batches,
                current_batch_size,
                loss.item(),
                batch_accuracy / current_batch_size,
                batch_f1 / current_batch_size,
            )

    if num_batches == 0:
        logger.info("Test Pipeline: no full batch found, returning zeros")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    avg_loss = total_loss / num_batches
    avg_bleu_1 = total_bleu_1 / num_batches
    avg_bleu_2 = total_bleu_2 / num_batches
    avg_bleu_3 = total_bleu_3 / num_batches
    avg_bleu_4 = total_bleu_4 / num_batches
    avg_bleu_scores = total_bleu_scores / num_batches
    avg_accuracy = total_accuracy / num_batches
    avg_f1 = total_f1 / num_batches
    logger.info(
        "Test Pipeline: done | loss=%.4f | bleu1=%.4f | bleu2=%.4f | bleu3=%.4f | bleu4=%.4f | bleu=%.4f | accuracy=%.4f | f1=%.4f",
        avg_loss,
        avg_bleu_1,
        avg_bleu_2,
        avg_bleu_3,
        avg_bleu_4,
        avg_bleu_scores,
        avg_accuracy,
        avg_f1,
    )
    return avg_loss, avg_bleu_1, avg_bleu_2, avg_bleu_3, avg_bleu_4, avg_bleu_scores, avg_accuracy, avg_f1

# image = "/Users/duyhoang/Documents/Research/VQAG/code/sample/synpic40127.jpg"
# question = "What is the MR weighting in this image?"

# image_paths = [image, image, image, image]
# questions = [
#     question,
#     "What is the condition of the patient?",
#     "What kind of scan is shown in this image?",
#     "Is there a fracture visible in the scan?"
# ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the ExVQA model")
    parser.add_argument("--checkpoint", type=str, default=config.CHECKPOINT_PATH, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default=config.DATASET_NAME, help="Hugging Face dataset name")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--use-dual-gating", action="store_true", help="Enable DualGatedFusion for ablation")
    args = parser.parse_args()

    log_path = add_file_logger("test")

    model = load_model(args.checkpoint, device=config.DEVICE, use_dual_gating=args.use_dual_gating)

    _, _, test_loader = build_dataloaders(args.dataset, batch_size=args.batch_size)
    ans_model = AnsEmbedding().to(config.DEVICE)

    metrics = evaluate_test(model, ans_model, test_loader, device=config.DEVICE, batch_size=args.batch_size)
    logger.info("Final test metrics: %s", metrics)
    print(f"Test log (including first 50 inferences) saved to: {log_path}")

    # predicted_answers = predict_batch(model, image_paths, questions, device=config.DEVICE)
