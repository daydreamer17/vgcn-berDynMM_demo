import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from transformers import BertTokenizer
from tqdm import tqdm
from custom_dataset import CustomDataset
from VocabularyGraph import VocabularyGraph
from vgcn_bert import VGCNBertDynMM
from vgcn_bertNoDynMM import VGCNBertDynMM_NoDynMM

# 自定义数据合并函数
def collate_fn(batch):
    input_ids, attention_masks, graphs, char_ids, labels = zip(*batch)
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    batched_graph = dgl.batch(graphs)
    char_ids = [torch.tensor(c) for c in char_ids]
    char_ids = torch.nn.utils.rnn.pad_sequence(char_ids, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return input_ids, attention_masks, batched_graph, char_ids, labels

# 评估函数
def evaluate(model, data_loader, device, num_classes=2, use_dynmm=True):
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

            if use_dynmm:
                logits = model(input_ids, attention_mask, graph, char_ids)
            else:
                logits = model(input_ids, attention_mask, graph,char_ids)

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
    cm_filename = f"confusion_matrix_{'with_dynmm' if use_dynmm else 'without_dynmm'}.png"
    plt.savefig(cm_filename)

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
        roc_filename = f"roc_curve_class_{i}_{'with_dynmm' if use_dynmm else 'without_dynmm'}.png"
        plt.savefig(roc_filename)

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

# 实验函数
def run_ablation_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model_dir = './model'  # 指定你的BERT模型路径
    num_classes = 2  # 分类数量
    gcn_in_dim = 768  # GCN输入维度
    gcn_hidden_dim = 128  # GCN隐藏层维度
    char_vocab_size = 256  # 字符词汇表大小
    char_emb_dim = 128  # 字符嵌入维度
    learning_rate = 2e-5
    num_epochs = 5
    batch_size = 16

    # 数据加载器
    tokenizer = BertTokenizer.from_pretrained(bert_model_dir)
    vocab_size = len(tokenizer.get_vocab())
    vocab_graph = VocabularyGraph(vocab_size)
    train_dataset = CustomDataset('Data/Grambedding_dataset/Train.csv', tokenizer, vocab_size, gcn_in_dim, train=True, sample_fraction=0.08)
    val_dataset = CustomDataset('Data/Grambedding_dataset/Train.csv', tokenizer, vocab_size, gcn_in_dim, train=False, sample_fraction=0.05)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn, pin_memory=True)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 模型设置
    model_with_dynmm = VGCNBertDynMM(bert_model_dir, gcn_in_dim, gcn_hidden_dim, num_classes, char_vocab_size, char_emb_dim).to(device)
    optimizer_with_dynmm = optim.Adam(model_with_dynmm.parameters(), lr=learning_rate)

    model_without_dynmm = VGCNBertDynMM_NoDynMM(bert_model_dir, gcn_in_dim, gcn_hidden_dim, num_classes, char_vocab_size, char_emb_dim).to(device)
    optimizer_without_dynmm = optim.Adam(model_without_dynmm.parameters(), lr=learning_rate)

    results = {}

    # 训练去除 DynMM 的模型
    print("\nTraining Model without DynMM...")
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        model_without_dynmm.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc="Training", leave=False):
            input_ids, attention_mask, graphs, char_ids, labels = batch
            input_ids, attention_mask, graphs, char_ids, labels = (
                input_ids.to(device, non_blocking=True),
                attention_mask.to(device, non_blocking=True),
                graphs.to(device, non_blocking=True),
                char_ids.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )

            optimizer_without_dynmm.zero_grad()
            logits = model_without_dynmm(input_ids, attention_mask, graphs, char_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer_without_dynmm.step()

            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        val_accuracy, val_precision, val_recall, val_f1 = evaluate(model_without_dynmm, val_loader, device, use_dynmm=False)
        print(f"Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")

    results["without_dynmm"] = {"train_loss": train_loss, "val_accuracy": val_accuracy, "val_precision": val_precision, "val_recall": val_recall, "val_f1": val_f1}

    # 训练包含 DynMM 的模型
    print("Training Model with DynMM...")
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        model_with_dynmm.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc="Training", leave=False):
            input_ids, attention_mask, graphs, char_ids, labels = batch
            input_ids, attention_mask, graphs, char_ids, labels = (
                input_ids.to(device, non_blocking=True),
                attention_mask.to(device, non_blocking=True),
                graphs.to(device, non_blocking=True),
                char_ids.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )

            optimizer_with_dynmm.zero_grad()
            logits = model_with_dynmm(input_ids, attention_mask, graphs, char_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer_with_dynmm.step()

            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        val_accuracy, val_precision, val_recall, val_f1 = evaluate(model_with_dynmm, val_loader, device, use_dynmm=True)
        print(f"Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")

    results["with_dynmm"] = {"train_loss": train_loss, "val_accuracy": val_accuracy, "val_precision": val_precision, "val_recall": val_recall, "val_f1": val_f1}

    return results

if __name__ == '__main__':
    results = run_ablation_experiment()

    print("\nAblation Experiment Results:")
    print(f"With DynMM: {results['with_dynmm']}")
    print(f"Without DynMM: {results['without_dynmm']}")
