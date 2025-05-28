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
import warnings
import os
from safetensors.torch import load_file

from sklearn.model_selection import train_test_split
from datasets import Dataset as HFDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from transformers import PreTrainedTokenizer

from config_ import (
    BASE_MODEL,
    CHECKPOINT_PATH,
    MAX_RELATION,
    MAX_LENGTH,
    TEST_BATCH_SIZE,
    JOINT_CHECKPOINT_PATH
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

    non_matching_indices = [i for i, match in enumerate(match_lengths) if not match]

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
    "../../data/phase_1/sentences.txt",
    "../../data/phase_1/entities.txt",
    "../../data/phase_1/relations.txt",
)


num_labels_entity = len(label2id_entity)
num_labels_relation = len(label2id_relation) 


def align_labels(tokenized_inputs, word_labels):
    aligned_labels = []
    word_ids = tokenized_inputs.word_ids()

    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)
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
        max_relations=10,
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
            max_length=128,
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

    max_relations = MAX_RELATION
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

class EntityModel(nn.Module):
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


custom_model = EntityModel(model_name, num_labels=num_labels_entity).to(device)

state_dict_custom = load_file(CHECKPOINT_PATH)
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

        word_ids = encoding.word_ids()
        word_ids = [
            w if w is not None else -1 for w in word_ids
        ]  
        word_ids = torch.tensor(word_ids, dtype=torch.long)  # Convert to tensor

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "word_ids": word_ids,  
        }


def save_inference_results(model, val_loader, entity_output_file, relation_output_file):
    model.eval()
    model.to(device)

    entity_labels_all = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_ids = batch["word_ids"].to(device)

            outputs = model(
                input_ids,
                attention_mask,
                entity_labels=None,
                relation_labels=None,
                word_ids=word_ids,
                inference=True,
            )

            entity_inference_list = outputs["entity_inference_list"]
            inference_list = outputs["inference_list"]

            entity_labels_all.extend(entity_inference_list)

            relation_list = []

            for relation in inference_list:
                sub_relation = relation.split("\t")
                if len(sub_relation) == 3 and sub_relation[0] != "":
                    relation_list.append(relation)

            with open(entity_output_file, "a", encoding="utf-8") as f_entity:
                for entity_labels in entity_inference_list:
                    entity_str = " ".join(entity_labels)
                    f_entity.write(entity_str + "\n")

                    print(entity_str)

            with open(relation_output_file, "a", encoding="utf-8") as f_relation:
                relation_str = ";".join(relation_list)
                f_relation.write(relation_str + "\n")

    return entity_labels_all


checkpoint = torch.load(JOINT_CHECKPOINT_PATH)
model.load_state_dict(checkpoint)
model.to(device)

test_texts_file = "../../data/phase_1/test_texts.txt"
entity_output_file = "../../data/predictions/phase_1/predictions.entities.txt"
relation_output_file = "../../data/predictions/phase_1/predictions.relations.txt"

open(entity_output_file, "w+", encoding="utf-8").close()
open(relation_output_file, "w+", encoding="utf-8").close()

test_dataset = TextDataset(test_texts_file, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

entity_labels = save_inference_results(
    model, test_loader, entity_output_file, relation_output_file
)

print(f"Entity predictions saved to: {entity_output_file}")
print(f"Relation predictions saved to: {relation_output_file}")

test_texts_file = "../../data/phase_2/test_texts.txt"
entity_output_file = "../../data/predictions/phase_2/predictions.entities.txt"
relation_output_file = "../../data/predictions/phase_2/predictions.relations.txt"

open(entity_output_file, "w+", encoding="utf-8").close()
open(relation_output_file, "w+", encoding="utf-8").close()

test_dataset = TextDataset(test_texts_file, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

entity_labels = save_inference_results(
    model, test_loader, entity_output_file, relation_output_file
)

print(f"Entity predictions saved to: {entity_output_file}")
print(f"Relation predictions saved to: {relation_output_file}")

