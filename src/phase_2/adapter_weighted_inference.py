"""
Adapter-weighted inference for relation classification.
"""

import torch
import sys
import os
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file

from .config import (
    TEST_TEXTS_FILE,
    DEVICE,
    BASE_MODEL,
    CHECKPOINT_PATH,
    MAX_LENGTH,
    TEST_MODEL_PATH as ADAPTER_MODEL_PATH,
    ENTITY_OUTPUT_FILE, 
    RELATION_OUTPUT_FILE,
)
from .model import EntityModel
from .relation_model import RelationModel
from .dataloader import (
    num_labels_entity,
    num_labels_relation,
    label2id_entity,
    label2id_relation,
    tokenizer,
)

torch.set_float32_matmul_precision("high")


# TextDataset class from text_dataset.py
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, "r", encoding="utf-8") as f:
            sentences = f.readlines()
        self.sentences = [sentence.split() for sentence in sentences]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        encoding = self.tokenizer(
            sentence,
            padding=True,
            truncation=True,
            return_tensors="pt",
            is_split_into_words=True,
        )

        # Pad word_ids to have a fixed length
        word_ids = encoding.word_ids()
        word_ids = [
            w if w is not None else -1 for w in word_ids
        ]  # Replace None with -1
        word_ids = torch.tensor(word_ids, dtype=torch.long)  # Convert to tensor

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "word_ids": word_ids,  # Fixed-length tensor
        }


# save_inference_results function from text_dataset.py
def save_inference_results(model, val_loader, entity_output_file, relation_output_file):
    """Save inference results to output files"""
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Make sure output files don't exist (to avoid appending to existing files)
    for file_path in [entity_output_file, relation_output_file]:
        if os.path.exists(file_path):
            os.remove(file_path)

    entity_labels_all = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_ids = batch["word_ids"].to(device)

            # Forward pass
            outputs = model(
                input_ids,
                attention_mask,
                entity_labels=None,
                relation_labels=None,
                word_ids=word_ids,
                inference=True,
            )

            # Extract entity and relation inference lists
            entity_inference_list = outputs["entity_inference_list"]
            inference_list = outputs["inference_list"]

            entity_labels_all.extend(entity_inference_list)

            relation_list = []

            for relation in inference_list:
                sub_relation = relation.split("\t")
                if len(sub_relation) == 3 and sub_relation[0] != "":
                    relation_list.append(relation)

            # Save entity predictions to file
            with open(entity_output_file, "a", encoding="utf-8") as f_entity:
                for entity_labels in entity_inference_list:
                    # Convert entity labels to a space-separated string
                    entity_str = " ".join(entity_labels)
                    f_entity.write(entity_str + "\n")

            # Save relation predictions to file
            with open(relation_output_file, "a", encoding="utf-8") as f_relation:
                # Convert relations to a semicolon-separated string
                relation_str = ";".join(relation_list)
                f_relation.write(relation_str + "\n")

    return entity_labels_all


def main():
    print("Running inference with weighted relation adapter model...")
    print(f"Loading test data from: {TEST_TEXTS_FILE}")
    
    # Load tokenizer and create dataset
    test_dataset = TextDataset(TEST_TEXTS_FILE, tokenizer)
    
    # Create DataLoader
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Load the base model first
    print(f"Loading base model from: {CHECKPOINT_PATH}")
    custom_model = EntityModel(BASE_MODEL, num_labels_entity).to(DEVICE)
    state_dict = load_file(CHECKPOINT_PATH)
    custom_model.load_state_dict(state_dict)
    
    # Initialize the relation model
    model = RelationModel(
        custom_model=custom_model,
        num_entity_labels=num_labels_entity,
        num_relation_labels=num_labels_relation,
        entity_label2id=label2id_entity,
        relation_label2id=label2id_relation,
        tokenizer=tokenizer,
    ).to(DEVICE)
    
    # Add adapter layers before loading checkpoint
    num_relation_classes = model.fc2_relation.out_features
    print(f"Adding simple relation adapter with {num_relation_classes} relation classes")
    
    # Create the adapter with the same architecture used in training
    model.relation_logits_adapter = torch.nn.Sequential(
        # First layer - wider bottleneck for more expressive power
        torch.nn.Linear(num_relation_classes, num_relation_classes * 3),
        torch.nn.LayerNorm(num_relation_classes * 3),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.05),
        torch.nn.Linear(num_relation_classes * 3, num_relation_classes * 2),
        torch.nn.LayerNorm(num_relation_classes * 2),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.05),
        torch.nn.Linear(num_relation_classes * 2, num_relation_classes),
    ).to(DEVICE)
    
    print("Using enhanced deeper 3-layer adapter with targeted class weighting")
    
    # Load the adapter-enhanced model weights
    print(f"Loading weighted adapter model from: {ADAPTER_MODEL_PATH}")
    if os.path.exists(ADAPTER_MODEL_PATH):
        model.load_state_dict(torch.load(ADAPTER_MODEL_PATH))
    else:
        print(f"WARNING: Adapter model not found at {ADAPTER_MODEL_PATH}")
        print("Using base model with adapter architecture but without trained weights")
    
    # Set model to evaluation mode
    model.eval()
    
    # Patch the model's forward method to apply adapter enhancement
    patch_model_for_adapter_inference(model)
    
    # Run inference and save results
    print("Running inference and saving results...")
    entity_labels = save_inference_results(
        model, test_loader, ENTITY_OUTPUT_FILE, RELATION_OUTPUT_FILE
    )
    
    print("Inference complete!")
    print(f"Entity predictions saved to: {ENTITY_OUTPUT_FILE}")
    print(f"Relation predictions saved to: {RELATION_OUTPUT_FILE}")
    
    return entity_labels


def patch_model_for_adapter_inference(model):
    """Patch the model to use the relation adapter during inference"""
    
    # Helper function to apply adapter to a single logit
    def apply_adapter_enhancement(logit):
        """Apply adapter enhancement to a relation logit"""
        enhanced = model.relation_logits_adapter(logit)
        return logit + enhanced  # Residual connection
    
    # Store original forward method
    original_forward = model.forward
    
    # Create patched method that applies adapter
    def patched_forward(input_ids, attention_mask, entity_labels=None, 
                       relation_labels=None, word_ids=None, inference=True):
        # Call original method to get outputs
        outputs = original_forward(
            input_ids, attention_mask, entity_labels, relation_labels, 
            word_ids, inference
        )
        
        # Check if this inference has relation logits
        if "relation_logits" in outputs and isinstance(outputs["relation_logits"], list):
            enhanced_logits = []
            
            # Apply adapter to each relation logit
            for logit in outputs["relation_logits"]:
                # Apply adapter enhancement
                enhanced_logit = apply_adapter_enhancement(logit)
                enhanced_logits.append(enhanced_logit)
            
            # Replace original logits with enhanced ones
            outputs["relation_logits"] = enhanced_logits
        
        return outputs
    
    # Apply patch
    model.forward = patched_forward
    print("Model patched to use relation adapter during inference")


if __name__ == "__main__":
    main() 
