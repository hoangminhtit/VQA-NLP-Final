import torch

class config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 4
    DATASET_NAME = "flaviagiammarino/path-vqa"
    CHECKPOINT_PATH = "checkpoints/best_model.pt"
    IMG_DIR = 'openai/clip-vit-base-patch32'
    TEXT_DIR = 'bert-base-uncased'
    MAX_LEN = 64
    d_model = 768
