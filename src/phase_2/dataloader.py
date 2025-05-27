"""
Data loading utilities for Phase 2: loads, processes, and splits datasets for relation extraction tasks.
"""
import os
from config import (
    TRAIN_DATASET_PATH,
    TRAIN_SIZE,
    RANDOM_SEED,
    MAX_RELATIONS,
    BASE_MODEL,
)
from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from relation_dataset import RelationDataset

os.environ["TOKENIZERS_PARALLELISM"] = (
    "false" 
)

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

    sentences = [s.strip().split() for s in sentences]
    entity_labels = [e.strip().split() for e in entity_labels]
    relation_labels = [
        [tuple(rel.split()) for rel in rel_str.split(";") if rel]
        for rel_str in relation_labels
    ]

    return (
        sentences,
        entity_labels,
        relation_labels,
        label2id_entity,
        id2label_entity,
        label2id_relation,
        id2label_relation,
    )

(
    sentences,
    entity_labels,
    relation_labels,
    label2id_entity,
    id2label_entity,
    label2id_relation,
    id2label_relation,
) = load_data(
    os.path.join(TRAIN_DATASET_PATH, "sentences.txt"),
    os.path.join(TRAIN_DATASET_PATH, "entities.txt"),
    os.path.join(TRAIN_DATASET_PATH, "relations.txt"),
    # max_sentences=100
)

num_labels_entity = len(label2id_entity)
num_labels_relation = len(label2id_relation)  

print(f"Number of entity labels: {num_labels_entity}")
print(f"Number of relation labels: {num_labels_relation}")
print("Train Dataset loaded successfully.")

print("\nRelation Labels:")
for rel_id, rel_label in id2label_relation.items():
    print(f"ID {rel_id}: {rel_label}")


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_prefix_space=True)

# Create dataset
dataset = RelationDataset(
    sentences,
    entity_labels,
    relation_labels,
    tokenizer,
    label2id_entity,
    label2id_relation,
    max_relations=MAX_RELATIONS,
)

# Split dataset
train_indices, val_indices = train_test_split(
    list(range(len(dataset))), test_size=1 - TRAIN_SIZE, random_state=RANDOM_SEED
)

# Create train and validation datasets
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
