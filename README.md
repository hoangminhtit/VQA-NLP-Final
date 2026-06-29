# ExVQA

This is the source code of the published paper CAEE_110439 ( ExVQA: a novel stacked attention networks with extended long short-term memory model for visual question answering) on the Computers and Electrical Engineering Journal.
[Link paper](https://www.sciencedirect.com/science/article/pii/S0045790625003829)

ExVQA is a Visual Question Answering (VQA) model that combines image and text embeddings with advanced neural network architectures, including Stacked Attention Networks (SANs) and xLSTM decoders, to generate answers to questions based on input images.


---

## System Requirements

- **Python**: 3.10
- **Dependencies**: Listed in `requirements.txt`
- **Hardware**: A GPU is recommended for training and inference.

---

## Environment Setup

1. **Clone the Repository**  
   Open your terminal and run:
   ```bash
   git clone https://github.com/Hovohoangduy/ExVQA.git
   cd ExVQA
   ```

2. **Install Dependencies**  
   Use `pip` to install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## Project Structure

- `config.py`: Configuration file for device, batch size, and model parameters.
- `data_processing.py`: Handles dataset loading and preprocessing.
- `features_extraction.py`: Defines image and question embedding models.
- `sans.py`: Implements Stacked Attention Networks (SANs) for image-text attention.
- `xlstm_decoder.py`: Contains the xLSTM decoder for sequence generation.
- `vqa_model.py`: Combines all components into the VQA model.
- `train.py`: Script for training the VQA model.
- `test.py`: Script for testing and inference.

---

## Training the Model

To train the ExVQA model, run the following command:
```bash
python train.py --dataset flaviagiammarino/path-vqa --epochs 5 --early-stopping 3
```

### Training Parameters
- **Dataset**: `--dataset`: **flaviagiammarino/vqa-rad** and **flaviagiammarino/path-vqa** (default: `flaviagiammarino/path-vqa`).
- **Batch Size**: `--batch-size` (default: 4).
- **Number of Epochs**: `--epochs` (default: 5).
- **Early Stopping**: `--early-stopping` (default: 3).
- **Learning Rate**: Configurable in `train.py` (default: 0.0001).

---

## Testing the Model

To test the model on new images and questions, use the `test.py` script:
```bash
python test.py
```

```bash
python test.py \
  --checkpoint checkpoints/best_model.pt \
  --dataset flaviagiammarino/path-vqa \
  --batch-size 12 \
  --use-dual-gating
```

Training writes timestamped progress logs to `logs/train_*.log`. Testing writes
metrics to `logs/test_*.log` and records the question, reference answer, and
generated prediction for the first 50 test samples.

> The corrected decoder uses matrix-memory mLSTM states and autoregressive token
> generation. Checkpoints produced by the older vector-memory implementation are
> not architecture-compatible and must be retrained.

### Example Usage
Uncomment the example code in `test.py` to test the model with sample images and questions.

---

## BLEU Score Evaluation

The training script calculates BLEU scores (BLEU@1, BLEU@2, BLEU@3, BLEU@4) to evaluate the model's performance. These scores are printed after each epoch.

---

## Dataset

The dataset should be structured with `train`, `validation`, and `test` splits. Update the `dataset` path in `data_processing.py` to point to your dataset.
