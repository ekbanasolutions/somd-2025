"""
Training and evaluation script for relation extraction with class-weighted adapter layers in Phase 2.
Handles dataset loading, class weighting, model setup, and evaluation.
"""
import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import classification_report as sklearn_classification
from torch.optim.lr_scheduler import OneCycleLR

# Import configuration, models, and dataset-related functions/classes
from config import BASE_MODEL, MAX_RELATIONS, RANDOM_SEED, MODEL_SAVE_PATH, PRETRAINED_MODEL_PATH
from .model import CustomModel
from .relation_model import RelationModel
from .dataloader import load_data, num_labels_entity, num_labels_relation
from .relation_dataset import RelationDataset
from .utils import custom_data_collator

print("Creating a relation-specific adapter model with class weighting...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_prefix_space=True)
merged_sentences_file = (
    "../../data/phase_2/sentences.txt"
)
merged_entities_file = (
    "../../data/phase_2/entities.txt"
)
merged_relations_file = (
    "../../data/phase_2/relations.txt"
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
    merged_sentences_file,
    merged_entities_file,
    merged_relations_file,
    max_sentences=None,
)

print(f"Loaded dataset with {len(sentences)} sentences")
print(f"Relation labels: {label2id_relation}")

relation_counts = Counter()
for rels in relation_labels:
    for rel in rels:
        if not rel: 
            relation_counts[11] += 1  
        else:
            rel_type, _, _ = rel  
            if rel_type in label2id_relation:
                rel_id = label2id_relation[rel_type]
                relation_counts[rel_id] += 1
            else:
                print(f"Warning: Unknown relation type: {rel_type}")

print("\nRelation class distribution:")
for rel_id, count in relation_counts.items():
    rel_name = id2label_relation[rel_id] if rel_id in id2label_relation else "Unknown"
    print(f"{rel_name}: {count}")

num_samples = sum(relation_counts.values())
num_classes = len(label2id_relation)

# Create support-aware class weights
class_weights = torch.ones(num_classes)
zero_support_classes = set(range(num_classes)) - set(relation_counts.keys())

high_support_threshold = 100
low_support_threshold = 20

high_support_classes = [rel_id for rel_id, count in relation_counts.items() if count >= high_support_threshold]
low_support_classes = [rel_id for rel_id, count in relation_counts.items() if 0 < count < low_support_threshold]
medium_support_classes = [rel_id for rel_id, count in relation_counts.items() 
                         if count >= low_support_threshold and count < high_support_threshold]

print(f"\nHigh support classes: {[id2label_relation[i] for i in high_support_classes]}")
print(f"Medium support classes: {[id2label_relation[i] for i in medium_support_classes]}")
print(f"Low support classes: {[id2label_relation[i] for i in low_support_classes]}")
print(f"Zero support classes: {[id2label_relation[i] for i in zero_support_classes]}")

for rel_id, count in relation_counts.items():
    # Base weight using gentler inverse frequency
    base_weight = (num_samples / (count * num_classes)) ** 0.25
    
    # Get relation name for targeted weighting
    rel_name = id2label_relation[rel_id]
    
    # Fine-tuned targeting based on current performance metrics
    if rel_name == "Developer_of":
        # Keep the good precision (0.8889), need better recall (0.400)
        class_weights[rel_id] = base_weight * 2.5
    elif rel_name == "Citation_of":
        # Decent balanced performance (0.6074) - slight boost
        class_weights[rel_id] = base_weight * 1.5
    elif rel_name == "Version_of":
        # Already good performance (0.6707) - modest boost to maintain
        class_weights[rel_id] = base_weight * 1.5
    elif rel_name == "PlugIn_of":
        # Perfectly balanced (0.6154/0.6154) - keep as is
        class_weights[rel_id] = base_weight * 1.4
    elif rel_name == "URL_of":
        # Very poor performance (0.2632) - needs major boost
        class_weights[rel_id] = base_weight * 3.5
    elif rel_name == "Abbreviation_of":
        # Good performance (0.6667) - maintain with slight boost
        class_weights[rel_id] = base_weight * 1.3
    elif rel_name == "AlternativeName_of":
        # Perfect precision but low recall (0.3810) - boost recall
        class_weights[rel_id] = base_weight * 2.2
    elif rel_name == "Release_of":
        # Perfect precision but low recall (0.4615) - boost recall
        class_weights[rel_id] = base_weight * 2.0
    elif rel_name == "Extension_of":
        # Zero performance despite having support - extreme boost
        class_weights[rel_id] = base_weight * 3.5
    elif rel_id in high_support_classes:
        # Other high support classes - minimal boost
        class_weights[rel_id] = base_weight * 1.0
    elif rel_id in medium_support_classes:
        # Medium support - minimal boost
        class_weights[rel_id] = base_weight * 0.9
    elif rel_id in low_support_classes:
        # Low support - slightly reduced
        class_weights[rel_id] = base_weight * 0.8

# For zero support classes (License_of, Specification_of)
for rel_id in zero_support_classes:
    # Very low weight for zero-support classes
    class_weights[rel_id] = 0.01

class_weights = class_weights / (class_weights.mean() * 0.75)
class_weights = class_weights.to("cuda")

print("\nComputed support-aware class weights:")
for rel_id, weight in enumerate(class_weights):
    rel_name = id2label_relation[rel_id]
    count = relation_counts.get(rel_id, 0)
    print(f"{rel_name}: {weight.item():.4f} (count: {count})")

class_weights = None
dataset = RelationDataset(
    sentences,
    entity_labels,
    relation_labels,
    tokenizer,
    label2id_entity,    
    label2id_relation,  
    max_relations=MAX_RELATIONS,
)

indices = list(range(len(dataset)))
train_indices, val_indices = train_test_split(
    indices, test_size=0.2, random_state=RANDOM_SEED
)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

train_dataloader = DataLoader(
    train_dataset, batch_size=8, shuffle=True, collate_fn=custom_data_collator
)
val_dataloader = DataLoader(
    val_dataset, batch_size=8, shuffle=False, collate_fn=custom_data_collator
)

custom_model = CustomModel(BASE_MODEL, num_labels_entity)
model = RelationModel(
    custom_model,
    num_labels_entity,
    num_labels_relation,
    label2id_entity,
    label2id_relation,
    tokenizer,
).to("cuda")

checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location="cuda")
model.load_state_dict(checkpoint)
print(f"Loaded pre-trained model from {PRETRAINED_MODEL_PATH}")

hidden_size = model.base_model.config.hidden_size
relation_hidden_size = model.fc1_relation.out_features 
num_relation_classes = model.fc2_relation.out_features

print(f"Number of relation classes: {num_relation_classes}")

model.relation_logits_adapter = torch.nn.Sequential(
    torch.nn.Linear(num_relation_classes, num_relation_classes * 3),
    torch.nn.LayerNorm(num_relation_classes * 3),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.05),
    torch.nn.Linear(num_relation_classes * 3, num_relation_classes * 2),
    torch.nn.LayerNorm(num_relation_classes * 2),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.05),
    torch.nn.Linear(num_relation_classes * 2, num_relation_classes),
).to("cuda")

def patch_forward_method():
    """
    Patch the forward method of RelationModel to use the relation adapters.
    This implementation enhances only the relation classification part.
    """
    original_forward = model.forward
    
    def new_forward(input_ids, attention_mask, entity_labels=None, relation_labels=None, 
                   word_ids=None, inference=False):
        """
        Enhanced forward method that applies relation adapters after the standard forward pass.
        Works by enhancing relation logits directly.
        """
        outputs = original_forward(input_ids, attention_mask, entity_labels, 
                                 relation_labels, word_ids, inference)
        
        if "relation_logits" in outputs and isinstance(outputs["relation_logits"], list) and len(outputs["relation_logits"]) > 0:
            enhanced_logits = []
            
            for logit in outputs["relation_logits"]:
                enhanced = model.relation_logits_adapter(logit)
                enhanced_logit = logit + enhanced
                enhanced_logits.append(enhanced_logit)
            
            outputs["relation_logits"] = enhanced_logits
            
            if relation_labels is not None and not inference and "relation_labels" in outputs:
                relation_labels_tensor = outputs["relation_labels"]
                
                if enhanced_logits and relation_labels_tensor is not None:
                    batch_losses = []
                    
                    for i, (logit, label) in enumerate(zip(enhanced_logits, relation_labels_tensor)):
                        if not isinstance(label, torch.Tensor):
                            label = torch.tensor(label, device=input_ids.device)
                        
                        if not hasattr(label, 'shape') or label.shape == torch.Size([]):
                            label = label.unsqueeze(0)
                            
                        if len(logit.shape) == 1:
                            logit = logit.unsqueeze(0)
                            
                        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
                        example_loss = loss_fn(logit, label)
                        batch_losses.append(example_loss)
                    
                    if batch_losses:
                        loss_relation = torch.stack(batch_losses).mean()
                        
                        outputs["loss_relation"] = loss_relation
                        outputs["loss"] = outputs.get("loss_entity", torch.tensor(0.0).to(input_ids.device)) + loss_relation
        
        return outputs
    
    model.forward = new_forward
    print("RelationModel forward method patched to use relation logits adapter with class weighting")

patch_forward_method()

for param in model.parameters():
    param.requires_grad = False

for param in model.relation_logits_adapter.parameters():
    param.requires_grad = True

for param in model.fc2_relation.parameters():
    param.requires_grad = True

print("\nTrainable Parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}, Shape: {param.shape}")

adapter_params = [
    {'params': model.relation_logits_adapter.parameters(), 'lr': 3e-4},  # Higher learning rate
    {'params': model.fc2_relation.parameters(), 'lr': 1e-4}
]

optimizer = optim.AdamW(adapter_params, weight_decay=0.005)  # Lower weight decay

num_epochs = 20
total_steps = len(train_dataloader) * num_epochs

scheduler = OneCycleLR(
    optimizer,
    max_lr=3e-4,  
    total_steps=total_steps,
    pct_start=0.1,  
    anneal_strategy='cos',  
    div_factor=10.0,  
    final_div_factor=100.0, 
)

def evaluate(model, val_loader, id2label_entity, id2label_relation):
    """
    Evaluate the model on the validation dataset, focusing on relation extraction.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    total_loss = 0.0
    all_true_relations, all_pred_relations = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            entity_labels = batch["entity_labels"].to(device)
            relation_labels = batch["relation_labels"]
            word_ids = batch["word_ids"].to(device)
            
            outputs = model(
                input_ids,
                attention_mask,
                entity_labels,
                relation_labels,
                word_ids,
                inference=True
            )
            
            relation_loss = outputs.get("loss_relation", torch.tensor(0.0)).item()
            total_loss += relation_loss
            
            if "relation_logits" in outputs and outputs["relation_logits"] is not None:
                pred_rel = outputs["relation_logits"]
                true_rel = outputs["relation_labels"]
                
                if isinstance(pred_rel, list):
                    for pred, true in zip(pred_rel, true_rel):
                        pred_label_id = torch.argmax(pred).item()
                        true_label_id = true.item()
                        
                        pred_rel_label = id2label_relation[pred_label_id]
                        true_rel_label = id2label_relation[true_label_id]
                        
                        all_pred_relations.append(pred_rel_label)
                        all_true_relations.append(true_rel_label)
    
    if all_true_relations:
        report = sklearn_classification(
            all_true_relations, all_pred_relations, output_dict=True
        )
        relation_f1 = report["macro avg"]["f1-score"]
        
        print("\nRelation Classification Report:")
        print(sklearn_classification(
            all_true_relations, all_pred_relations, 
            labels=list(id2label_relation.values()),
            zero_division=0
        ))
    else:
        relation_f1 = 0.0
    
    avg_loss = total_loss / len(val_loader)
    
    print(f"Validation - Relation Loss: {avg_loss:.4f}, Relation F1: {relation_f1:.4f}")
    return avg_loss, relation_f1

print("\nStarting training...")
best_relation_f1 = 0.0
patience = 5
no_improvement_count = 0

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    for batch in train_dataloader:
        input_ids = torch.as_tensor(batch["input_ids"]).to("cuda")
        attention_mask = torch.as_tensor(batch["attention_mask"]).to("cuda")
        entity_labels = torch.as_tensor(batch["entity_labels"]).to("cuda")
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            entity_labels=entity_labels,
            relation_labels=batch["relation_labels"],
            word_ids=batch["word_ids"]
        )
        
        loss = outputs["loss_relation"]
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_dataloader)
    val_loss, relation_f1 = evaluate(model, val_dataloader, id2label_entity, id2label_relation)
    
    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, "
          f"Relation F1: {relation_f1:.4f}")
    
    # Early stopping check
    if relation_f1 > best_relation_f1:
        best_relation_f1 = relation_f1
        no_improvement_count = 0
        
        # Save the best model
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"New best model saved with Relation F1: {relation_f1:.4f}")
    else:
        no_improvement_count += 1
        
    if no_improvement_count >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

torch.save(model.state_dict(), MODEL_SAVE_PATH)

print("\nTraining complete!")
print(f"Best Relation F1: {best_relation_f1:.4f}")
print(f"Final model saved to {MODEL_SAVE_PATH}")
