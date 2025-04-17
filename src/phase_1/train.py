import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
)
from datasets import Dataset as HFDataset
import numpy as np
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report as seqeval_classification
from sklearn.metrics import classification_report as sklearn_classification
from seqeval.scheme import IOB2
import warnings
from tqdm import tqdm
import os
from safetensors.torch import load_file

from sklearn.model_selection import train_test_split
from datasets import Dataset as HFDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from transformers import PreTrainedTokenizer


from torch.optim import AdamW
import torch
from tqdm import tqdm

from .config_ import (
    BASE_MODEL,
    CHECKPOINT_PATH,
    TRAIN_SIZE,
    RANDOM_STATE,
    MAX_RELATION,
    BATCH_SIZE,
    MAX_LENGTH,
    JOINT_CHECKPOINT_PATH,
    NUM_EPOCH,
    LEARNING_RATE,
    PATIENCE
)

# Suppress warnings
warnings.filterwarnings("ignore")

# Device setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = BASE_MODEL
checkpoint_path = CHECKPOINT_PATH

config = AutoConfig.from_pretrained(model_name)
print("Config loaded successfully.")

tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)


def load_data(text_file, entity_file, relation_file, max_sentences=None):
    with open(text_file, "r", encoding="utf-8") as f:
        sentences = f.readlines()
    with open(entity_file, "r", encoding="utf-8") as f:
        entity_labels = f.readlines()
    with open(relation_file, "r", encoding="utf-8") as f:
        relation_labels = f.readlines()

    unique_labels_entity = sorted(
        set(label for labels in entity_labels for label in labels.strip().split())
    )

    unique_labels_relation = sorted(
        set(
            label
            for labels in relation_labels
            for label_tuple in labels.strip().split(";")
            if len(label_tuple.split()) >= 3
            for label, _, _ in [label_tuple.split()]
        )
    )

    label2id_entity = {label: i for i, label in enumerate(unique_labels_entity)}
    id2label_entity = {i: label for label, i in label2id_entity.items()}

    label2id_relation = {label: i for i, label in enumerate(unique_labels_relation)}
    label2id_relation[""] = 11  # Add a default label for no relation
    id2label_relation = {i: label for label, i in label2id_relation.items()}

    if max_sentences is not None:
        sentences = sentences[:max_sentences]
        entity_labels = entity_labels[:max_sentences]
        relation_labels = relation_labels[:max_sentences]

    sentences = [s.split() for s in sentences]
    entity_labels = [e.split() for e in entity_labels]
    relation_labels = [
        [tuple(rel.split()) for rel in rel_str.split(";") if rel]
        for rel_str in relation_labels
    ]

    sentence_lengths = [len(sentence) for sentence in sentences]

    match_lengths = [
        len(sentence) == len(entity)
        for sentence, entity, relation in zip(sentences, entity_labels, relation_labels)
    ]

    matching_indices = [i for i, match in enumerate(match_lengths) if match]

    return (
        sentences,
        entity_labels,
        relation_labels,
        label2id_entity,
        id2label_entity,
        label2id_relation,
        id2label_relation,
        sentence_lengths,
        match_lengths,
        matching_indices,
    )

(
    sentences,
    entity_labels,
    relation_labels,
    label2id_entity,
    id2label_entity,
    label2id_relation,
    id2label_relation,
    sentence_lengths,
    match_lengths,
    matching_indices,
) = load_data(
    "sentences.txt",
    "entities.txt",
    "relations.txt",
)

# Number of labels
num_labels_entity = len(label2id_entity)
num_labels_relation = len(label2id_relation) 


def align_labels(tokenized_inputs, word_labels):
    aligned_labels = []
    word_ids = tokenized_inputs.word_ids()

    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)  # Ignore special tokens
        else:
            aligned_labels.append(label2id_entity[word_labels[word_idx]])

    return aligned_labels


class RelationDataset(Dataset):
    def __init__(
        self,
        sentences,
        entity_labels,
        relation_labels,
        tokenizer,
        entity_label2id,
        relation_label2id,
        max_relations=MAX_RELATION,
    ):
        self.sentences = sentences
        self.entity_labels = entity_labels
        self.relation_labels = relation_labels
        self.tokenizer = tokenizer
        self.entity_label2id = entity_label2id
        self.relation_label2id = relation_label2id
        self.max_relations = max_relations 

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        word_labels = self.entity_labels[idx]
        relation_labels = self.relation_labels[idx]

        tokenized_inputs = self.tokenizer(
            sentence,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            is_split_into_words=True,
        )

        entity_labels = align_labels(tokenized_inputs, word_labels)

        relation_labels_aligned = self.align_relation_labels(
            idx, tokenized_inputs, relation_labels
        )

        word_ids_padded = [
            wid if wid is not None else -1 for wid in tokenized_inputs.word_ids()
        ]

        return {
            "input_ids": torch.tensor(tokenized_inputs["input_ids"]),
            "attention_mask": torch.tensor(tokenized_inputs["attention_mask"]),
            "entity_labels": torch.tensor(entity_labels),
            "relation_labels": relation_labels_aligned,
            "word_ids": torch.tensor(word_ids_padded),  # Add word_ids to the output
        }

    def align_relation_labels(self, idx, tokenized_inputs, relation_labels):
        aligned_relation_labels = []
        word_ids = tokenized_inputs.word_ids()
        input_ids = tokenized_inputs["input_ids"]

        for relation in relation_labels:
            if isinstance(relation, tuple):
                relation = "\t".join(map(str, relation))
            parts = relation.split()
            if len(parts) != 3:
                continue

            relation_type, entity_1, entity_2 = parts
            try:
                entity_1_idx = int(entity_1)
                entity_2_idx = int(entity_2)
            except ValueError:
                continue

            relation_id = self.relation_label2id.get(relation_type, -1)
            aligned_relation_labels.append([relation_id, entity_1_idx, entity_2_idx])

        if not aligned_relation_labels:
            return [[11, 0, 0]] 
        return aligned_relation_labels 

    def get_token_index(self, word_ids, entity_idx):
        for idx, word_id in enumerate(word_ids):
            if word_id == entity_idx:
                return idx
        return None


dataset = RelationDataset(
    sentences,
    entity_labels,
    relation_labels,
    tokenizer,
    label2id_entity,
    label2id_relation,
)

train_size = TRAIN_SIZE
train_indices, val_indices = train_test_split(
    list(range(len(dataset))), test_size=1 - train_size, random_state=RANDOM_STATE
)

train_dataset = HFDataset.from_dict(
    {
        "input_ids": [dataset[i]["input_ids"] for i in train_indices],
        "attention_mask": [dataset[i]["attention_mask"] for i in train_indices],
        "entity_labels": [dataset[i]["entity_labels"] for i in train_indices],
        "relation_labels": [dataset[i]["relation_labels"] for i in train_indices],
        "word_ids": [dataset[i]["word_ids"] for i in train_indices],  
    }
)

val_dataset = HFDataset.from_dict(
    {
        "input_ids": [dataset[i]["input_ids"] for i in val_indices],
        "attention_mask": [dataset[i]["attention_mask"] for i in val_indices],
        "entity_labels": [dataset[i]["entity_labels"] for i in val_indices],
        "relation_labels": [dataset[i]["relation_labels"] for i in val_indices],
        "word_ids": [dataset[i]["word_ids"] for i in val_indices],  
    }
)


def custom_data_collator(batch):
    collated_batch = {}
    fixed_keys = [
        "input_ids",
        "attention_mask",
        "entity_labels",
        "word_ids",
    ]  
    for key in fixed_keys:
        collated_values = []
        for sample in batch:
            value = sample[key]
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)
            collated_values.append(value)
        collated_batch[key] = torch.stack(collated_values)

    max_relations = 10
    relation_labels_padded = [
        F.pad(
            torch.tensor(sample["relation_labels"]),
            (0, 0, 0, max_relations - len(sample["relation_labels"])),
            value=11,
        )
        for sample in batch
    ]
    collated_batch["relation_labels"] = torch.stack(relation_labels_padded)

    return collated_batch


train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_data_collator
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_data_collator
)


class CustomModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.config.hidden_dropout_prob = 0.4
        self.config.attention_probs_dropout_prob = 0.3
        self.config.output_hidden_states = True

        self.base_model = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.base_model.config.hidden_size

        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size // 2, num_labels)

        init.xavier_uniform_(self.fc2.weight)
        if self.fc2.bias is not None:
            init.zeros_(self.fc2.bias)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states
        second_last_hidden = all_hidden_states[-2]

        x = self.fc1(second_last_hidden)
        x = self.act1(x)
        x = self.dropout1(x)
        logits = self.fc2(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits, "hidden_states": all_hidden_states[-2]}


custom_model = CustomModel(model_name, num_labels=num_labels_entity).to(device)

state_dict_custom = load_file(checkpoint_path)
custom_model.load_state_dict(state_dict_custom)

num_labels = len(relation_labels)

class RelationModel(nn.Module):
    def __init__(self, custom_model, num_entity_labels, num_relation_labels, entity_label2id, relation_label2id, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.entity_label2id = entity_label2id
        self.num_label_entity = num_entity_labels
        self.num_label_relation = num_relation_labels

        self.id2label_entity = {v: k for k, v in entity_label2id.items()}
        self.id2label_relation = {v: k for k, v in relation_label2id.items()}

        self.base_model = custom_model
        self.tokenizer = tokenizer

        hidden_size = self.base_model.config.hidden_size

        # Relation extraction layers
        self.fc1_relation = nn.Linear(hidden_size * 3, hidden_size // 2)
        self.act1_relation = nn.GELU()
        self.dropout1_relation = nn.Dropout(0.4)  # Increased dropout to 0.5
        self.fc2_relation = nn.Linear(hidden_size // 2, num_relation_labels)

    def get_tokenized_index(self, entity_idx, word_ids):
        for token_idx, word_id in enumerate(word_ids):
            if word_id == entity_idx:
                return token_idx
        return None

    def forward(self, input_ids, attention_mask=None, entity_labels=None, relation_labels=None, word_ids=None, inference=False):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs["hidden_states"]
        entity_logits = outputs["logits"]

        batch_size = input_ids.shape[0]
        entity_inference_list = []
        relation_logits_list = []
        relation_labels_list = []
        inference_list = []

        logits_labels = torch.argmax(entity_logits, dim=-1)

        for i in range(batch_size):
            valid_mask = (attention_mask[i] == 1) & (word_ids[i] != -1)
            valid_token_indices = torch.where(valid_mask)[0]
            cleaned_logits_labels = logits_labels[i, valid_token_indices]

            unique_word_ids = []
            word_to_token_idx = {}
            for idx, wid in enumerate(word_ids[i].tolist()):
                if wid != -1 and wid not in unique_word_ids:
                    unique_word_ids.append(wid)
                    word_to_token_idx[wid] = idx

            word_entity_labels = []
            for wid in unique_word_ids:
                first_token_idx = word_to_token_idx[wid]
                if first_token_idx in valid_token_indices.tolist():
                    relative_idx = (valid_token_indices == first_token_idx).nonzero(as_tuple=True)[0].item()
                    label_id = cleaned_logits_labels[relative_idx].item()
                    word_entity_labels.append(self.id2label_entity[label_id])
            entity_inference_list.append(word_entity_labels)

            entities = [idx for idx, entity in enumerate(entity_inference_list[i]) if entity != "O"]
            num_entities = len(entities)
            cls_rep = hidden_states[i, 0].unsqueeze(0)

            for j in range(num_entities):
                for k in range(num_entities):
                    if j != k:
                        entity_1_idx, entity_2_idx = entities[j], entities[k]

                        if entity_inference_list[i][entity_1_idx] == "O" or entity_inference_list[i][entity_2_idx] == "O":
                            continue

                        entity_1_tokenized_idx = self.get_tokenized_index(entity_1_idx, word_ids[i])
                        entity_2_tokenized_idx = self.get_tokenized_index(entity_2_idx, word_ids[i])

                        entity_pair = torch.cat([
                            cls_rep,
                            hidden_states[i, entity_1_tokenized_idx].unsqueeze(0),
                            hidden_states[i, entity_2_tokenized_idx].unsqueeze(0)
                        ], dim=-1).unsqueeze(0)

                        # Relation extraction layers
                        relation_x = self.fc1_relation(entity_pair)
                        relation_x = self.act1_relation(relation_x)
                        relation_x = self.dropout1_relation(relation_x)
                        relation_logits_current = self.fc2_relation(relation_x)

                        relation_probs = torch.softmax(relation_logits_current, dim=-1)
                        max_prob, predicted_label_id = torch.max(relation_probs, dim=-1)

                        # Adjusted threshold (experiment with 0.5-0.7)
                        threshold = 0.5
                        predicted_label_id = predicted_label_id.item()
                        if max_prob.item() < threshold:
                            predicted_label_id = 11

                        predicted_label_name = self.id2label_relation[predicted_label_id]

                        if predicted_label_name != "":
                            inference_list.append(f"{predicted_label_name}\t{entity_1_idx}\t{entity_2_idx}")

                        if relation_labels is not None:
                            gt_label = 11
                            for rel in relation_labels[i]:
                                if rel[1] == entity_1_idx and rel[2] == entity_2_idx:
                                    gt_label = rel[0]
                                    break

                            relation_logits_list.append(relation_logits_current.squeeze(0))
                            relation_labels_list.append(gt_label)

        if relation_logits_list:
            relation_logits = torch.cat(relation_logits_list, dim=0)
            relation_labels_tensor = torch.tensor(relation_labels_list, device=input_ids.device)
            loss_relation = F.cross_entropy(relation_logits, relation_labels_tensor)  # Label smoothing added
        else:
            loss_relation = torch.tensor(0.0, device=input_ids.device)

        if entity_labels is not None:
            loss_entity = F.cross_entropy(entity_logits.view(-1, self.num_label_entity), entity_labels.view(-1))
        else:
            loss_entity = torch.tensor(0.0, device=input_ids.device)

        total_loss = loss_entity + loss_relation

        return {
            "loss": total_loss,
            "entity_logits": entity_logits,
            "relation_logits": relation_logits_list,
            "relation_labels": relation_labels_tensor if relation_labels_list else None,
            "loss_entity": loss_entity,
            "loss_relation": loss_relation,
            "inference_list": inference_list,
            "entity_inference_list": entity_inference_list,
        }

model = RelationModel(
    custom_model,
    num_labels_entity,
    num_labels_relation,
    label2id_entity,
    label2id_relation,
    tokenizer,
).to("cuda")

def train_model(model, train_loader, val_loader, num_epochs=10, lr=2e-5, patience=3):
    model.to(device)

    entity_optimizer = AdamW(model.base_model.parameters(), lr=lr)
    relation_optimizer = AdamW(model.fc1_relation.parameters(), lr=lr)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, entity_loss_total, relation_loss_total = 0, 0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            entity_labels = batch["entity_labels"].to(device)
            relation_labels = batch["relation_labels"].to(device)
            word_ids = batch["word_ids"].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask, entity_labels, relation_labels, word_ids)
            entity_loss = outputs["loss_entity"]
            relation_loss = outputs["loss_relation"]

            # Combine losses
            total_batch_loss = entity_loss + relation_loss

            # Backward pass and optimization
            entity_optimizer.zero_grad()
            relation_optimizer.zero_grad()
            total_batch_loss.backward()  # Single backward pass
            entity_optimizer.step()
            relation_optimizer.step()

            # Track losses
            batch_loss = total_batch_loss.item()
            total_loss += batch_loss
            entity_loss_total += entity_loss.item()
            relation_loss_total += relation_loss.item()
            
        # Compute average losses for the epoch
        avg_loss = total_loss / len(train_loader)
        avg_entity_loss = entity_loss_total / len(train_loader)
        avg_relation_loss = relation_loss_total / len(train_loader)


        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f} | Entity Loss: {avg_entity_loss:.4f} | Relation Loss: {avg_relation_loss:.4f}")

        # Validation
        val_loss, val_entity_f1, val_relation_f1 = evaluate(model, val_loader)

        # Early stopping
        if val_loss <= best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), JOINT_CHECKPOINT_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

def evaluate(model, val_loader):
    model.eval()
    total_loss, entity_loss_total, relation_loss_total = 0, 0, 0

    all_true_entities, all_pred_entities = [], []
    all_true_relations, all_pred_relations = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            entity_labels = batch["entity_labels"].to(device)
            relation_labels = batch["relation_labels"]
            word_ids = batch["word_ids"].to(device)

            # Forward pass
            outputs = model(
                input_ids, attention_mask, entity_labels, relation_labels, word_ids
            )
            entity_loss = outputs.get("loss_entity", torch.tensor(0.0)).item()
            relation_loss = outputs.get("loss_relation", torch.tensor(0.0)).item()

            total_loss += (entity_loss + relation_loss) / 2
            entity_loss_total += entity_loss
            relation_loss_total += relation_loss

            # Entity predictions
            entity_preds = torch.argmax(outputs["entity_logits"], dim=-1).cpu().numpy()
            entity_labels = entity_labels.cpu().numpy()
            attention_mask = attention_mask.cpu().numpy()

            # Align predictions with ground truth by removing padding
            for seq_idx, (true_seq, pred_seq, mask) in enumerate(
                zip(entity_labels, entity_preds, attention_mask)
            ):
                # Remove padding tokens based on attention_mask
                true_seq = true_seq[mask == 1]
                pred_seq = pred_seq[mask == 1]
                pred_seq = pred_seq[1:-1]  # Remove first [CLS] and last tokens [SEP]

                # Convert label IDs to label names
                true_entities = [
                    id2label_entity[label] for label in true_seq if label != -100
                ]
                pred_entities = [
                    id2label_entity[int(label)] for label in pred_seq if label != -100
                ]

                all_true_entities.append(true_entities)
                all_pred_entities.append(pred_entities)

            # Relation predictions
            if "relation_logits" in outputs and outputs["relation_logits"] is not None:
                pred_rel = outputs["relation_logits"]
                true_rel = outputs["relation_labels"]

                # Check if pred_rel is empty or true_rel is None
                if not pred_rel or true_rel is None:
                    print(f"Warning: pred_rel is empty or true_rel is None for batch {batch_idx}")
                    continue  
                
                for rel_idx, (pred, true) in enumerate(zip(pred_rel, true_rel)):
                    pred_label_id = torch.argmax(pred).item()
                    true_label_id = true.item()

                    pred_rel_label = id2label_relation[pred_label_id]
                    true_rel_label = id2label_relation[true_label_id]

                    all_pred_relations.append(pred_rel_label)
                    all_true_relations.append(true_rel_label)


    # Compute entity F1 score
    entity_f1 = seqeval_classification(
        all_true_entities,
        all_pred_entities,
        mode="strict",
        scheme=IOB2,
        output_dict=True,
    )["macro avg"]["f1-score"]

    # Compute relation F1 score
    if all_true_relations:
        relation_f1 = sklearn_classification(
            all_true_relations, all_pred_relations, output_dict=True
        )["macro avg"]["f1-score"]
    else:
        relation_f1 = 0.0

    avg_loss = total_loss / len(val_loader)
    print(
        f"Validation Loss: {avg_loss:.4f} | Entity F1: {entity_f1:.4f} | Relation F1: {relation_f1:.4f}"
    )
    return avg_loss, entity_f1, relation_f1

train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCH, lr=LEARNING_RATE, patience=PATIENCE)
