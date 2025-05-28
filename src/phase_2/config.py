"""
Configuration file for Phase 2 model training, inference, and evaluation.
Defines dataset paths, model paths, training hyperparameters, and device setup.
"""
import os
import torch

# DATASET PATHS
TRAIN_DATASET_PATH = "../../data/phase_2/"
TEST_TEXTS_FILE = "../../data/phase_2/test_texts.txt"
ENTITY_OUTPUT_FILE = (
    "../../data/predictions/phase_2/predictions.entities.txt"
)
RELATION_OUTPUT_FILE = (
    "../../data/predictions/phase_2/predictions.relations.txt"
)

# MODEL PATHS
# BASE_MODEL = "answerdotai/ModernBERT-base"
BASE_MODEL = "../../ModernBERT_checkpoint/"
MODEL_SAVE_PATH = (
    "../../JointModel_Checkpoint/phase_2/relation_adapter_weighted_best.pt"
)
TEST_MODEL_PATH = (
    "../../JointModel_Checkpoint/phase_2/relation_adapter_weighted_best.pt"
)
CHECKPOINT_PATH = (
    "../../EntityModel_checkpoint/model.safetensors"
)

PRETRAINED_MODEL_PATH = "../../FewShot_checkpoint/few_shot_finetuned_model_with_val_ner_relation_finetuned.pt"

# TRAINING CONFIGURATIONS/HYPERPARAMETERS
MAX_LENGTH = 128
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 2e-5
PATIENCE = 5
TRAIN_SIZE = 0.8
RANDOM_SEED = 42

# MODEL CONFIGURATIONS
HIDDEN_DROPOUT_PROB = 0.4
ATTENTION_PROBS_DROPOUT_PROB = 0.3
MAX_RELATIONS = 10


# DEVICE
if torch.cuda.device_count() > 1:
    DEVICE = torch.device("cuda:1")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

# CREATE NECESSARY DIRECTORIES
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(ENTITY_OUTPUT_FILE), exist_ok=True)
os.makedirs(os.path.dirname(RELATION_OUTPUT_FILE), exist_ok=True)
