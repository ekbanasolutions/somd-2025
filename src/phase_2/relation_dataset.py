import torch
from torch.utils.data import Dataset
from .utils import align_labels

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
        self.max_relations = max_relations  # Maximum number of relations per sample

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

        # Align entity labels with tokenized inputs
        entity_labels = align_labels(
            tokenized_inputs, word_labels, self.entity_label2id
        )

        # Align relation labels with tokenized inputs
        relation_labels_aligned = self.align_relation_labels(
            idx, tokenized_inputs, relation_labels
        )

        # Pad word_ids to match max_length (128), replacing None with -1
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
            return [[11, 0, 0]]  # Return as a list of lists
        return aligned_relation_labels  # Return as a list of lists

    def get_token_index(self, word_ids, entity_idx):
        for idx, word_id in enumerate(word_ids):
            if word_id == entity_idx:
                return idx
        return None
