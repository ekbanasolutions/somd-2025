import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizer

from .dataloader import (
    num_labels_entity,
    num_labels_relation,
    label2id_entity,
    label2id_relation,
    tokenizer,
)
from .model import custom_model

class RelationModel(nn.Module):
    def __init__(
        self,
        custom_model,
        num_entity_labels,
        num_relation_labels,
        entity_label2id,
        relation_label2id,
        tokenizer: PreTrainedTokenizer,
    ):
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
        self.act1_relation = nn.ReLU()
        self.dropout1_relation = nn.Dropout(0.3)
        self.fc2_relation = nn.Linear(hidden_size // 2, num_relation_labels)

    def get_tokenized_index(self, entity_idx, word_ids):
        for token_idx, word_id in enumerate(word_ids):
            if word_id == entity_idx:
                return token_idx
        return None

    def forward(
        self,
        input_ids,
        attention_mask=None,
        entity_labels=None,
        relation_labels=None,
        word_ids=None,
        inference=False,
    ):
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
            word_ids_tensor = torch.as_tensor(word_ids[i]).to(attention_mask.device) #added by V
            valid_mask = (attention_mask[i] == 1) & (word_ids_tensor != -1)
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
                    relative_idx = (
                        (valid_token_indices == first_token_idx)
                        .nonzero(as_tuple=True)[0]
                        .item()
                    )
                    label_id = cleaned_logits_labels[relative_idx].item()
                    word_entity_labels.append(self.id2label_entity[label_id])
            entity_inference_list.append(word_entity_labels)

            entities = [
                idx
                for idx, entity in enumerate(entity_inference_list[i])
                if entity != "O"
            ]
            num_entities = len(entities)
            cls_rep = hidden_states[i, 0].unsqueeze(0)

            for j in range(num_entities):
                for k in range(num_entities):
                    if j != k:
                        entity_1_idx, entity_2_idx = entities[j], entities[k]

                        if (
                            entity_inference_list[i][entity_1_idx] == "O"
                            or entity_inference_list[i][entity_2_idx] == "O"
                        ):
                            continue

                        entity_1_tokenized_idx = self.get_tokenized_index(
                            entity_1_idx, word_ids[i]
                        )
                        entity_2_tokenized_idx = self.get_tokenized_index(
                            entity_2_idx, word_ids[i]
                        )

                        entity_pair = torch.cat(
                            [
                                cls_rep,
                                hidden_states[i, entity_1_tokenized_idx].unsqueeze(0),
                                hidden_states[i, entity_2_tokenized_idx].unsqueeze(0),
                            ],
                            dim=-1,
                        ).unsqueeze(0)
                        
                        relation_x = self.fc1_relation(entity_pair)
                        relation_x = self.act1_relation(relation_x)
                        relation_x = self.dropout1_relation(relation_x)
                        relation_logits_current = self.fc2_relation(
                            relation_x
                        )  # Shape: (1, num_relation_labels)
                        relation_probs = torch.softmax(
                            relation_logits_current, dim=-1
                        )  # Convert to probabilities
                        max_prob, predicted_label_id = torch.max(relation_probs, dim=-1)

                        # Assign label only if probability > 85%
                        predicted_label_id = predicted_label_id.item()
                        if max_prob.item() < 0.80:
                            predicted_label_id = 11

                        predicted_label_name = self.id2label_relation[
                            predicted_label_id
                        ]

                        if predicted_label_name != "":
                            inference_list.append(
                                f"{predicted_label_name}\t{entity_1_idx}\t{entity_2_idx}"
                            )

                        if relation_labels is not None:
                            gt_label = 11
                            for rel in relation_labels[i]:
                                if rel[1] == entity_1_idx and rel[2] == entity_2_idx:
                                    gt_label = rel[0]
                                    break

                            relation_logits_list.append(
                                relation_logits_current.squeeze(0)
                            )
                            relation_labels_list.append(gt_label)

        if relation_logits_list:
            relation_logits = torch.cat(relation_logits_list, dim=0)
            relation_labels_tensor = torch.tensor(
                relation_labels_list, device=input_ids.device
            )
            loss_relation = F.cross_entropy(relation_logits, relation_labels_tensor)
        else:
            loss_relation = torch.tensor(0.0, device=input_ids.device)

        if entity_labels is not None:
            loss_entity = F.cross_entropy(
                entity_logits.view(-1, self.num_label_entity), entity_labels.view(-1)
            )
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


# Initialize the relation model
model = RelationModel(
    custom_model,
    num_labels_entity,
    num_labels_relation,
    label2id_entity,
    label2id_relation,
    tokenizer,
).to("cuda")
