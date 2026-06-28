import argparse
import os
import logging
import math
import string
from datetime import datetime
from collections import Counter
import torch
import random
import numpy as np
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import torch.optim as optim
from vqa_model import VQAModel
from config import config
from data_processing import build_dataloaders
from features_extraction import AnsEmbedding
from utils.training_plots import plot_training_history, save_training_history
from transformers import AutoTokenizer
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

def evaluate_loss(model, ans_model, data_loader, criterion, device, batch_size=4):
    logger.info("Evaluate Model: start validation pass")
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in data_loader:
            images, questions, answers = batch
            if len(images) > 0:
                images = images.to(device)

                decoder_inputs, ans_embedds = ans_model.prepare_decoder_batch(answers)
                predicted_tokens = model(images, questions, decoder_inputs).float()

                loss = criterion(predicted_tokens.reshape(-1, predicted_tokens.size(-1)), ans_embedds.reshape(-1))
                total_loss += loss.item()
                num_batches += 1

    if num_batches == 0:
        logger.info("Evaluate Model: no full batch found, returning 0.0 loss")
        return 0.0
    avg_val_loss = total_loss / num_batches
    logger.info("Evaluate Model: done | avg_val_loss=%.4f | batches=%d", avg_val_loss, num_batches)
    return avg_val_loss


def train_model(
    model,
    ans_model,
    train_loader,
    tokenizer,
    criterion,
    optimizer,
    device,
    num_epochs=14,
    print_every=500,
    batch_size=4,
    val_loader=None,
    early_stopping_patience=None,
    checkpoint_path=None,
    scheduler=None,
):
    smoother = SmoothingFunction()

    history = {
        "train_loss": [],
        "val_loss": [],
        "bleu1": [],
        "bleu2": [],
        "bleu3": [],
        "bleu4": [],
        "bleu_total": [],
        "accuracy": [],
        "f1": [],
    }

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        logger.info("Train Model: start epoch %d/%d", epoch + 1, num_epochs)
        model.train()
        total_loss = 0.0
        total_bleu_1, total_bleu_2, total_bleu_3, total_bleu_4, total_bleu_scores = 0.0, 0.0, 0.0, 0.0, 0.0
        total_accuracy = 0.0
        total_f1 = 0.0

        for batch_idx, batch in enumerate(train_loader):
            images, questions, answers = batch
            if len(images) > 0:
                images = images.to(device)
                current_batch_size = len(images)
                decoder_inputs, ans_embedds = ans_model.prepare_decoder_batch(answers)
                predicted_tokens = model(images, questions, decoder_inputs).float()

                references = [answer.strip() for answer in answers]
                hypotheses = []

                for i in range(current_batch_size):
                    valid_length = int(ans_embedds[i].ne(-100).sum().item())
                    sentence_predicted = torch.argmax(
                        predicted_tokens[i, :valid_length], axis=1
                    )
                    predicted_sentence = tokenizer.decode(sentence_predicted, skip_special_tokens=True)
                    hypotheses.append(predicted_sentence.strip())

                # BLEU score calculations
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

                total_bleu_1 += bleu_score_1
                total_bleu_2 += bleu_score_2
                total_bleu_3 += bleu_score_3
                total_bleu_4 += bleu_score_4
                total_bleu_scores += bleu_score_total

                batch_accuracy = 0.0
                batch_f1 = 0.0
                for pred, ref in zip(hypotheses, references):
                    batch_accuracy += accuracy_score(pred, ref)
                    batch_f1 += f1_score(pred, ref)
                total_accuracy += batch_accuracy / current_batch_size
                total_f1 += batch_f1 / current_batch_size

                # Loss calculation
                loss = criterion(predicted_tokens.reshape(-1, predicted_tokens.size(-1)), ans_embedds.reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                total_loss += loss.item()

                if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                    logger.info(
                        "Train batch | epoch=%d/%d | batch=%d/%d | size=%d | loss=%.4f",
                        epoch + 1,
                        num_epochs,
                        batch_idx + 1,
                        len(train_loader),
                        current_batch_size,
                        loss.item(),
                    )

                # Log periodically
                if (batch_idx + 1) % print_every == 0:
                    avg_loss_so_far = total_loss / (batch_idx + 1)
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Avg Loss So Far: {avg_loss_so_far:.4f}")
                    print(
                        f"BLEU@1: {bleu_score_1:.4f}, BLEU@2: {bleu_score_2:.4f}, BLEU@3: {bleu_score_3:.4f}, "
                        f"BLEU@4: {bleu_score_4:.4f}, BLEU-SCORES: {bleu_score_total:.4f}, "
                        f"Accuracy: {batch_accuracy / current_batch_size:.4f}, F1-score: {batch_f1 / current_batch_size:.4f}"
                    )

        avg_epoch_loss = total_loss / len(train_loader)
        avg_bleu_1 = total_bleu_1 / len(train_loader)
        avg_bleu_2 = total_bleu_2 / len(train_loader)
        avg_bleu_3 = total_bleu_3 / len(train_loader)
        avg_bleu_4 = total_bleu_4 / len(train_loader)
        avg_bleu_scores = total_bleu_scores / len(train_loader)
        avg_accuracy = total_accuracy / len(train_loader)
        avg_f1 = total_f1 / len(train_loader)

        history["train_loss"].append(avg_epoch_loss)
        history["bleu1"].append(avg_bleu_1)
        history["bleu2"].append(avg_bleu_2)
        history["bleu3"].append(avg_bleu_3)
        history["bleu4"].append(avg_bleu_4)
        history["bleu_total"].append(avg_bleu_scores)
        history["accuracy"].append(avg_accuracy)
        history["f1"].append(avg_f1)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Average Loss: {avg_epoch_loss:.4f}, "
              f"Average BLEU@1: {avg_bleu_1:.4f}, Average BLEU@2: {avg_bleu_2:.4f}, "
              f"Average BLEU@3: {avg_bleu_3:.4f}, Average BLEU@4: {avg_bleu_4:.4f}, "
              f"Average BLEU-Scores: {avg_bleu_scores:.4f}, Average Accuracy: {avg_accuracy:.4f}, "
              f"Average F1-score: {avg_f1:.4f}\n")
        logger.info(
            "Train Model: end epoch %d/%d | loss=%.4f | bleu1=%.4f | bleu2=%.4f | bleu3=%.4f | bleu4=%.4f | accuracy=%.4f | f1=%.4f",
            epoch + 1, num_epochs, avg_epoch_loss, avg_bleu_1, avg_bleu_2, avg_bleu_3, avg_bleu_4, avg_accuracy, avg_f1
        )

        if val_loader is not None:
            val_loss = evaluate_loss(model, ans_model, val_loader, criterion, device, batch_size=batch_size)
            history["val_loss"].append(val_loss)
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                if checkpoint_path:
                    checkpoint_dir = os.path.dirname(checkpoint_path)
                    if checkpoint_dir:
                        os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"Saved best checkpoint to: {checkpoint_path}")
                    logger.info("Checkpoint: saved best model to %s", checkpoint_path)
            else:
                epochs_no_improve += 1
                logger.info(
                    "Evaluate Model: no improvement | best_val_loss=%.4f | current_val_loss=%.4f | patience=%d/%s",
                    best_val_loss,
                    val_loss,
                    epochs_no_improve,
                    str(early_stopping_patience) if early_stopping_patience is not None else "None",
                )

            if early_stopping_patience is not None and epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                logger.info("Train Model: early stopping triggered at epoch %d", epoch + 1)
                break
        else:
            history["val_loss"].append(float("nan"))

    return history


if __name__=="__main__":
    logger.info("Train Pipeline: parse arguments")
    parser = argparse.ArgumentParser(description="Train the ExVQA model")
    parser.add_argument("--dataset", type=str, default=config.DATASET_NAME, help="Hugging Face dataset name")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--early-stopping", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for AdamW optimizer")
    parser.add_argument("--checkpoint", type=str, default=config.CHECKPOINT_PATH, help="Path to save best checkpoint")
    parser.add_argument("--use-dual-gating", action="store_true", help="Enable DualGatedFusion for ablation")
    args = parser.parse_args()

    log_path = add_file_logger("train")

    logger.info("Load model: tokenizer from %s", config.TEXT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_DIR)
    logger.info("Load model: VQAModel")
    model = VQAModel(use_dual_gating=args.use_dual_gating).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    logger.info(
        "Load data: dataset=%s | batch_size=%d | learning_rate=%g",
        args.dataset,
        args.batch_size,
        args.learning_rate,
    )
    train_loader, val_loader, _ = build_dataloaders(args.dataset, batch_size=args.batch_size)
    logger.info("Load scheduler: linear warmup")
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=max(1, int(len(train_loader) * args.epochs)),
    )

    logger.info("Load model: answer embedding module")
    ans_model = AnsEmbedding().to(config.DEVICE)
    logger.info("Train Model: start")
    training_history = train_model(
        model,
        ans_model,
        train_loader,
        tokenizer,
        criterion,
        optimizer,
        device=config.DEVICE,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        val_loader=val_loader,
        early_stopping_patience=args.early_stopping,
        checkpoint_path=args.checkpoint,
        scheduler=scheduler,
    )
    history_path = os.path.join("outputs", "training_history.json")
    save_training_history(training_history, history_path)
    plot_training_history(training_history, output_dir="outputs")
    logger.info("Train Pipeline: saved training curves and metrics in outputs/")
    logger.info("Train Pipeline: completed")
    print(f"Training log saved to: {log_path}")
