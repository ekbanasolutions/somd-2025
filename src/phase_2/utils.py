import torch
from torch.nn import functional as F
from transformers import (
    AutoTokenizer,
    AutoConfig,
)
from config import BASE_MODEL
from dataloader import label2id_entity

config = AutoConfig.from_pretrained(BASE_MODEL)
print("Config loaded successfully.")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_prefix_space=True)


# Align labels with tokenized words
def align_labels(tokenized_inputs, word_labels):
    aligned_labels = []
    word_ids = tokenized_inputs.word_ids()

    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)  # Ignore special tokens
        else:
            aligned_labels.append(label2id_entity[word_labels[word_idx]])

    return aligned_labels


def pad_relation_labels(relation_labels, max_relations=10, pad_value=[11, 0, 0]):
    """
    Pad relation_labels to a fixed maximum number of relations.
    Args:
        relation_labels (torch.Tensor or list): List of relation labels for a batch.
        max_relations (int): Maximum number of relations to pad to.
        pad_value (list): Padding value for relations (e.g., [11, 0, 0] for no relation)
    Returns:
        Padded relation_labels as a tensor.
    """
    if isinstance(relation_labels, torch.Tensor):
        relation_labels = relation_labels.tolist()  # Convert tensor to list of lists

    padded_labels = []
    for labels in relation_labels:
        if len(labels) < max_relations:
            # Pad with [11, 0, 0] if fewer relations than max_relations
            padded = labels + [pad_value] * (max_relations - len(labels))
        else:
            # Truncate if more relations than max_relations
            padded = labels[:max_relations]
        padded_labels.append(padded)

    # Flatten and convert to integers to work with torch.tensor
    padded_labels = [
        [
            int(item)
            for sublist in label
            for item in (sublist if isinstance(sublist, list) else [sublist])
        ]
        for label in padded_labels
    ]
    return torch.tensor(padded_labels, dtype=torch.long)


def custom_data_collator(batch):
    """
    Custom collator to handle padded relation_labels and include word_ids.
    Args:
        batch (list): List of samples from the dataset.
    Returns:
        collated_batch (dict): Collated batch with padded relation_labels and word_ids.
    """
    collated_batch = {}
    fixed_keys = [
        "input_ids",
        "attention_mask",
        "entity_labels",
        "word_ids",
    ]  # Add word_ids to fixed keys

    # Stack fixed-size tensors
    for key in fixed_keys:
        collated_values = []
        for sample in batch:
            value = sample[key]
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)
            collated_values.append(value)
        collated_batch[key] = torch.stack(collated_values)

    # Pad relation_labels to a fixed size (assuming max_relations=10)
    max_relations = 10
    relation_labels_padded = [
        F.pad(
            torch.tensor(sample["relation_labels"]),
            (
                0,
                0,
                0,
                max_relations - len(sample["relation_labels"]),
            ),  # Use len() for dynamic length
            value=11,
        )
        for sample in batch
    ]
    # Stack padded relation_labels
    collated_batch["relation_labels"] = torch.stack(relation_labels_padded)

    return collated_batch


class FocalLossWithLabelSmoothing(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smoothing=0.1):
        super(FocalLossWithLabelSmoothing, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits, targets):
        # Apply label smoothing
        num_classes = logits.size(-1)

        mask = targets != 11  # Exclude label 11

        logits = logits[mask]  # Mask logits based on valid targets
        targets = targets[mask]  # Mask targets

        # Check if there are any valid targets after masking
        if targets.numel() == 0:
            raise ValueError(
                "No valid labels left after masking. Please check your dataset."
            )

        # Step 3: Compute one-hot encoding of targets
        one_hot_targets = torch.zeros_like(logits).float().to(logits.device)
        one_hot_targets.scatter_(1, targets.unsqueeze(1), 1)

        # Step 4: Apply label smoothing
        one_hot_targets = one_hot_targets * (1 - self.smoothing) + (
            self.smoothing / num_classes
        )

        # Step 5: Compute cross entropy loss
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # Reshape focal_loss to match the shape of one_hot_targets
        focal_loss = focal_loss.view(
            -1, 1
        )  # Flatten focal_loss to match the batch size

        # Step 6: Apply label smoothing by performing the dot product
        loss = torch.sum(
            focal_loss * one_hot_targets.view(-1, num_classes)
        )  # Match shapes

        return loss.mean()

def align_labels(tokenized_inputs, word_labels, label2id_entity):
    aligned_labels = []
    word_ids = tokenized_inputs.word_ids()

    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)  # Ignore special tokens
        else:
            # Check if word_idx is within bounds of word_labels
            if word_idx < len(word_labels):
                try:
                    aligned_labels.append(label2id_entity[word_labels[word_idx]])
                except KeyError:
                    # If label not found in label2id_entity, use -100
                    aligned_labels.append(-100)
            else:
                # Handle out of bounds case by using a default label (-100 for ignored index)
                aligned_labels.append(-100)

    return aligned_labels