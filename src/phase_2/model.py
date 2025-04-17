import torch.nn as nn
import torch
from torch.nn import init
from transformers import AutoConfig, AutoModel

from .config import (
    BASE_MODEL,
    HIDDEN_DROPOUT_PROB,
    ATTENTION_PROBS_DROPOUT_PROB,
    CHECKPOINT_PATH,
)
from .dataloader import num_labels_entity
from safetensors.torch import load_file


class CustomModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.config.hidden_dropout_prob = HIDDEN_DROPOUT_PROB
        self.config.attention_probs_dropout_prob = ATTENTION_PROBS_DROPOUT_PROB
        self.config.output_hidden_states = True  # Get all hidden states 768-dim embeddings

        self.base_model = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.base_model.config.hidden_size

        self.norm = nn.LayerNorm(hidden_size)  # Adaptive normalization
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.act1 = nn.SiLU()  # Swish ActivatiATTENTION_PROBS_DROPOUT_PROBon
        self.dropout1 = nn.Dropout(0.5)  # Increased dropout
        self.norm2 = nn.LayerNorm(hidden_size // 2)  # Additional LayerNorm
        self.fc2 = nn.Linear(hidden_size // 2, num_labels)

        # Attention-based weighted aggregation for last 4 hidden layers
        self.attn_weights = nn.Parameter(torch.ones(4))  # Learnable weights

        # Xavier initialization for the classification layer
        init.xavier_uniform_(self.fc2.weight)
        if self.fc2.bias is not None:
            init.zeros_(self.fc2.bias)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states  # Tuple of all hidden states

        # Weighted sum of last 4 layers instead of simple mean
        hidden_states = torch.stack(
            all_hidden_states[-4:], dim=-1
        )  # Shape: (batch, seq, hidden, 4)
        weights = torch.softmax(self.attn_weights, dim=0)  # Normalize attention weights, it seems that all weights are 1, meaning its same as taking mean.
        x = torch.sum(hidden_states * weights, dim=-1)  # Weighted sum instead of mean
        x = self.norm(x)  # Apply first LayerNorm

        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.norm2(x)  # Additional LayerNorm before classification
        logits = self.fc2(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=-100, label_smoothing=0.1
            )
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits, "hidden_states": all_hidden_states[-2]}


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    Args:
        alpha (float): Weighting factor in range (0,1) to balance positive vs negative examples.
        gamma (float): Focusing parameter that reduces loss contribution from easy examples.
        reduction (str): 'mean', 'sum' or 'none'. Defaults to 'mean'.
    """

    def __init__(self, alpha=0.25, gamma=2, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction="none", ignore_index=-100
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    """Dice Loss for NER tasks.

    Args:
        smooth (float): Smoothing factor to prevent division by zero.
        ignore_index (int): Index to ignore in loss calculation.
    """

    def __init__(self, smooth=1.0, ignore_index=-100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        mask = (targets != self.ignore_index).float()
        targets_one_hot = nn.functional.one_hot(
            torch.clamp(targets, min=0), num_classes=inputs.size(1)
        ).float()

        intersection = (inputs * targets_one_hot * mask.unsqueeze(-1)).sum()
        union = (inputs + targets_one_hot) * mask.unsqueeze(-1)

        dice = (2.0 * intersection + self.smooth) / (union.sum() + self.smooth)
        return 1 - dice


class CustomVaghawan(nn.Module):
    """Enhanced Custom Model with multiple loss functions for NER and relation classification.

    Args:
        model_name (str): Name of the pretrained model to use
        num_labels (int): Number of output labels
        loss_type (str): Type of loss function to use ('ce', 'focal', 'dice', 'combined')
        class_weights (torch.Tensor, optional): Weights for each class for weighted loss
        focal_alpha (float): Alpha parameter for focal loss
        focal_gamma (float): Gamma parameter for focal loss
        dice_smooth (float): Smoothing factor for dice loss
    """

    def __init__(
        self,
        model_name,
        num_labels,
        loss_type="ce",
        class_weights=None,
        focal_alpha=0.25,
        focal_gamma=2,
        dice_smooth=1.0,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.config.hidden_dropout_prob = HIDDEN_DROPOUT_PROB
        self.config.attention_probs_dropout_prob = ATTENTION_PROBS_DROPOUT_PROB
        self.config.output_hidden_states = True

        self.base_model = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.base_model.config.hidden_size

        # Model architecture
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.act1 = nn.SiLU()
        self.dropout1 = nn.Dropout(0.5)
        self.norm2 = nn.LayerNorm(hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_labels)

        # Attention weights for layer aggregation
        self.attn_weights = nn.Parameter(torch.ones(4))

        # Loss function setup
        self.loss_type = loss_type
        self.class_weights = class_weights
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss(smooth=dice_smooth)

        # Initialize weights
        init.xavier_uniform_(self.fc2.weight)
        if self.fc2.bias is not None:
            init.zeros_(self.fc2.bias)

    def compute_loss(self, logits, labels):
        """Compute the specified loss function.

        Args:
            logits (torch.Tensor): Model predictions
            labels (torch.Tensor): Ground truth labels

        Returns:
            torch.Tensor: Computed loss value
        """
        if self.loss_type == "focal":
            return self.focal_loss(logits, labels)
        elif self.loss_type == "dice":
            return self.dice_loss(logits, labels)
        elif self.loss_type == "combined":
            focal = self.focal_loss(logits, labels)
            dice = self.dice_loss(logits, labels)
            return 0.5 * focal + 0.5 * dice
        else:  # 'ce' (Cross Entropy)
            return nn.CrossEntropyLoss(
                weight=self.class_weights, ignore_index=-100, label_smoothing=0.1
            )(logits.view(-1, self.config.num_labels), labels.view(-1))

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor, optional): Attention mask
            labels (torch.Tensor, optional): Ground truth labels

        Returns:
            dict: Contains loss (if labels provided), logits, and hidden states
        """
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states

        # Weighted sum of last 4 layers
        hidden_states = torch.stack(all_hidden_states[-4:], dim=-1)
        weights = torch.softmax(self.attn_weights, dim=0)
        x = torch.sum(hidden_states * weights, dim=-1)
        x = self.norm(x)

        # Classification head
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.norm2(x)
        logits = self.fc2(x)

        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels)

        return {"loss": loss, "logits": logits, "hidden_states": all_hidden_states[-2]}


custom_model = CustomModel(BASE_MODEL, num_labels_entity).to("cuda")
state_dict = load_file(CHECKPOINT_PATH)
custom_model.load_state_dict(state_dict)
