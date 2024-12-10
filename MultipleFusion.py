import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dgl.nn.pytorch import GraphConv
from tqdm import tqdm
from transformers import BertTokenizer
from VocabularyGraph import VocabularyGraph
from custom_dataset import CustomDataset
from torch.utils.data import DataLoader  
from vgcn_bert import VGCNBertDynMM
import dgl

def collate_fn(batch):
    input_ids, attention_masks, graphs, char_ids, labels = zip(*batch)
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    batched_graph = dgl.batch(graphs)
    char_ids = [torch.tensor(c) for c in char_ids]
    char_ids = torch.nn.utils.rnn.pad_sequence(char_ids, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return input_ids, attention_masks, batched_graph, char_ids, labels

def evaluate(model, data_loader, device, num_classes=2):
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids, attention_mask, graph, char_ids, labels = batch
            input_ids, attention_mask, graph, char_ids, labels = (
                input_ids.to(device, non_blocking=True),
                attention_mask.to(device, non_blocking=True),
                graph.to(device, non_blocking=True),
                char_ids.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )

            logits = model(input_ids, attention_mask, graph, char_ids)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
            y_probs.extend(F.softmax(logits, dim=1).cpu().numpy())

    avg_loss = total_loss / len(data_loader)

    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=[f"Class {i}" for i in range(num_classes)], 
                yticklabels=[f"Class {i}" for i in range(num_classes)])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")

    # Plot ROC for each class
    y_probs = np.array(y_probs)
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(np.array(y_true) == i, y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f}) for class {i}")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for Class {i}")
        plt.legend(loc="lower right")
        plt.savefig(f"roc_curve_class_{i}.png")

    # Print overall metrics
    print(f"Evaluation: Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Print class-specific metrics
    for i in range(num_classes):
        class_indices = np.where(np.array(y_true) == i)[0]
        class_y_true = np.array(y_true)[class_indices]
        class_y_pred = np.array(y_pred)[class_indices]

        # Compute binary precision, recall, and F1 for the current class
        class_precision = precision_score(np.array(y_true) == i, np.array(y_pred) == i, zero_division=0)
        class_recall = recall_score(np.array(y_true) == i, np.array(y_pred) == i, zero_division=0)
        class_f1 = f1_score(np.array(y_true) == i, np.array(y_pred) == i, zero_division=0)

        print(f"Class {i} Metrics: Precision = {class_precision:.4f}, Recall = {class_recall:.4f}, F1 Score = {class_f1:.4f}")

    return accuracy, precision, recall, f1


def run_layerwise_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    bert_model_dir = "./model"
    num_classes = 2
    gcn_in_dim = 768
    gcn_hidden_dim = 128
    char_vocab_size = 256
    char_emb_dim = 128
    learning_rate = 2e-5
    num_epochs = 5
    batch_size = 16

    tokenizer = BertTokenizer.from_pretrained(bert_model_dir)
    vocab_size = len(tokenizer.get_vocab())
    vocab_graph = VocabularyGraph(vocab_size)

    # Data loading
    train_dataset = CustomDataset("Data/Grambedding_dataset/Train.csv", tokenizer, vocab_size, gcn_in_dim, train=True,sample_fraction=0.0375)
    val_dataset = CustomDataset("Data/Grambedding_dataset/Train.csv", tokenizer, vocab_size, gcn_in_dim, train=False,sample_fraction=0.15)

    # Optimized DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=4)

    results = []
    for layer_idx in [1, 6, 12]:
        model = VGCNBertDynMM(bert_model_dir, gcn_in_dim, gcn_hidden_dim, num_classes, char_vocab_size, char_emb_dim, layer_idx=layer_idx)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            print(f"Layer {layer_idx}, Epoch {epoch + 1}/{num_epochs}")
            model.train()
            total_train_loss = 0
            for batch in tqdm(train_loader, desc="Training", leave=False):
                input_ids, attention_mask, graph, char_ids, labels = batch
                input_ids, attention_mask, graph, char_ids, labels = (
                    input_ids.to(device, non_blocking=True),
                    attention_mask.to(device, non_blocking=True),
                    graph.to(device, non_blocking=True),
                    char_ids.to(device, non_blocking=True),
                    labels.to(device, non_blocking=True),
                )

                optimizer.zero_grad()
                logits = model(input_ids, attention_mask, graph, char_ids)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
            print(f"Train Loss: {total_train_loss / len(train_loader):.4f}")

        accuracy, precision, recall, f1 = evaluate(model, val_loader, device, num_classes)
        results.append((layer_idx, accuracy, precision, recall, f1))

    # Visualization
    layers, accuracies, precisions, recalls, f1s = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.plot(layers, accuracies, label="Accuracy", marker="o")
    plt.plot(layers, precisions, label="Precision", marker="x")
    plt.plot(layers, recalls, label="Recall", marker="s")
    plt.plot(layers, f1s, label="F1 Score", marker="d")
    plt.title("Layer-wise Performance of VGCNBertDynMM Model")
    plt.xlabel("BERT Layer Index")
    plt.ylabel("Metrics")
    plt.legend()
    plt.grid()
    plt.savefig("layerwise_performance.png")

    print("Layer-wise Experiment Results:")
    for layer, acc, prec, rec, f1 in results:
        print(f"Layer {layer}: Accuracy = {acc:.4f}, Precision = {prec:.4f}, Recall = {rec:.4f}, F1 Score = {f1:.4f}")

if __name__ == "__main__":
    run_layerwise_experiment()

