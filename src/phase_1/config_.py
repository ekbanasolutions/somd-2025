"""
Configuration file for Phase 1 model training and evaluation.
Defines model paths, training hyperparameters, and other constants used throughout the pipeline.
"""
BASE_MODEL = "answerdotai/ModernBERT-base"
CHECKPOINT_PATH = "../../CustomModel_checkpoint/model.safetensors"
TRAIN_SIZE = 0.8
RANDOM_STATE = 42
MAX_RELATION = 10
BATCH_SIZE = 16
MAX_LENGTH = 128
TEST_BATCH_SIZE = 1
JOINT_CHECKPOINT_PATH = "../../JointModel_Checkpoint/phase_1/best_model.pth"

NUM_EPOCH = 50
LEARNING_RATE = 2e-5
PATIENCE = 10
