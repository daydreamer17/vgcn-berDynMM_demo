# main.py

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import BertTokenizer
from VocabularyGraph import VocabularyGraph
from CharacterGraph import CharacterGraph
from custom_dataset import CustomDataset
# 将原本对 vgcn_bert 的引用改为无 DynMM 版本
# from vgcn_bert import VGCNBertDynMM
from vgcn_bertNoDynMM import VGCNBertNoDynMM
import dgl

def collate_fn(batch):
    """
    自定义的 collate_fn，用于 DataLoader。

    Args:
        batch (list of tuples): 每个元素为 (input_ids, attention_mask, subgraph_word, subgraph_char, char_ids, label)。

    Returns:
        tuple: (input_ids, attention_masks, batched_graph_word, batched_graph_char, char_ids, labels)
    """
    input_ids, attention_masks, graphs_word, graphs_char, char_ids, labels = zip(*batch)
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    batched_graph_word = dgl.batch(graphs_word)
    batched_graph_char = dgl.batch(graphs_char)
    char_ids = torch.nn.utils.rnn.pad_sequence(char_ids, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return input_ids, attention_masks, batched_graph_word, batched_graph_char, char_ids, labels

def evaluate(model, data_loader, device, num_classes=2):
    """
    评估模型性能。

    Args:
        model (nn.Module): 要评估的模型（已去除 DynMM）。
        data_loader (DataLoader): 数据加载器。
        device (torch.device): 设备。
        num_classes (int): 分类类别数。

    Returns:
        tuple: (accuracy, precision, recall, f1)
    """
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids, attention_mask, graph_word, graph_char, char_ids, labels = batch
            input_ids, attention_mask, graph_word, graph_char, char_ids, labels = (
                input_ids.to(device, non_blocking=True),
                attention_mask.to(device, non_blocking=True),
                graph_word.to(device, non_blocking=True),
                graph_char.to(device, non_blocking=True),
                char_ids.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )

            # 前向传播 (无 DynMM 模型)
            logits = model(input_ids, attention_mask, graph_word, graph_char, char_ids)

            # 计算损失
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()

            # 预测结果
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
    plt.title("Confusion Matrix (No DynMM)")
    plt.savefig("confusion_matrix.png")
    plt.close()

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
        plt.title(f"ROC Curve (No DynMM) for Class {i}")
        plt.legend(loc="lower right")
        plt.savefig(f"roc_curve_class_{i}.png")
        plt.close()

    # Print overall metrics
    print(f"\nEvaluation: Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Print class-specific metrics
    for i in range(num_classes):
        class_precision = precision_score(np.array(y_true) == i, np.array(y_pred) == i, zero_division=0)
        class_recall = recall_score(np.array(y_true) == i, np.array(y_pred) == i, zero_division=0)
        class_f1 = f1_score(np.array(y_true) == i, np.array(y_pred) == i, zero_division=0)
        print(f"Class {i} Metrics: Precision = {class_precision:.4f}, Recall = {class_recall:.4f}, F1 Score = {class_f1:.4f}")

    return accuracy, precision, recall, f1

def run_layerwise_experiment():
    """
    运行分层实验，训练和评估“无 DynMM”模型在不同 BERT 层的表现。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    bert_model_dir = "./model"  # 使用预训练的 BERT 模型名称或路径
    num_classes = 2
    gcn_in_dim_word = 768  # 单词级图的节点特征维度
    gcn_hidden_dim_word = 128
    gcn_in_dim_char = 128  # 字符级图的节点特征维度
    gcn_hidden_dim_char = 64
    char_vocab_size = 256
    char_emb_dim = 128
    learning_rate = 2e-5
    num_epochs = 10
    batch_size = 32

    tokenizer = BertTokenizer.from_pretrained(bert_model_dir)
    vocab_size = len(tokenizer.get_vocab())

    # 数据加载
    train_dataset = CustomDataset(
        csv_file="Data/Grambedding_dataset/Train.csv",
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        gcn_in_dim_word=gcn_in_dim_word,
        gcn_in_dim_char=gcn_in_dim_char,
        use_char=True,           # 如果数据集需要字符级处理
        char_vocab_size=char_vocab_size,
        train=True,
        test_size=0.2,
        sample_fraction=0.0125
    )
    val_dataset = CustomDataset(
        csv_file="Data/Grambedding_dataset/Train.csv",
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        gcn_in_dim_word=gcn_in_dim_word,
        gcn_in_dim_char=gcn_in_dim_char,
        use_char=True,
        char_vocab_size=char_vocab_size,
        train=False,
        test_size=0.2,
        sample_fraction=0.05
    )

    # 优化的数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        pin_memory=True, 
        num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        pin_memory=True, 
        num_workers=4
    )

    results = []
    # 测试您在 [1, 6, 12] 等不同 BERT 层下的表现
    for layer_idx in [1, 6, 12]:
        print(f"\n--- Training 'No DynMM' Model with BERT layer {layer_idx} ---")
        # 改为使用无 DynMM 的模型类
        model = VGCNBertNoDynMM(
            bert_model_name=bert_model_dir,
            gcn_in_dim_word=gcn_in_dim_word,
            gcn_hidden_dim_word=gcn_hidden_dim_word,
            gcn_in_dim_char=gcn_in_dim_char,
            gcn_hidden_dim_char=gcn_hidden_dim_char,
            num_classes=num_classes,
            char_vocab_size=char_vocab_size,
            char_emb_dim=char_emb_dim,
            layer_idx=layer_idx
        )
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            print(f"\nLayer {layer_idx}, Epoch {epoch + 1}/{num_epochs}")
            model.train()
            total_train_loss = 0
            for batch in tqdm(train_loader, desc="Training", leave=False):
                input_ids, attention_mask, graph_word_batch, graph_char_batch, char_ids, labels = batch
                input_ids, attention_mask, graph_word_batch, graph_char_batch, char_ids, labels = (
                    input_ids.to(device, non_blocking=True),
                    attention_mask.to(device, non_blocking=True),
                    graph_word_batch.to(device, non_blocking=True),
                    graph_char_batch.to(device, non_blocking=True),
                    char_ids.to(device, non_blocking=True),
                    labels.to(device, non_blocking=True),
                )

                optimizer.zero_grad()
                logits = model(input_ids, attention_mask, graph_word_batch, graph_char_batch, char_ids)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)
            print(f"Train Loss: {avg_train_loss:.4f}")

        # 评估模型
        accuracy, precision, recall, f1 = evaluate(model, val_loader, device, num_classes)
        results.append((layer_idx, accuracy, precision, recall, f1))

    # 可视化
    layers, accuracies, precisions, recalls, f1s = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.plot(layers, accuracies, label="Accuracy", marker="o")
    plt.plot(layers, precisions, label="Precision", marker="x")
    plt.plot(layers, recalls, label="Recall", marker="s")
    plt.plot(layers, f1s, label="F1 Score", marker="d")
    plt.title("Layer-wise Performance of VGCNBertNoDynMM Model")
    plt.xlabel("BERT Layer Index")
    plt.ylabel("Metrics")
    plt.legend()
    plt.grid()
    plt.savefig("layerwise_performance_no_dynmm.png")
    plt.close()

    print("\nLayer-wise Experiment Results (No DynMM):")
    for layer, acc, prec, rec, f1 in results:
        print(f"Layer {layer}: Accuracy = {acc:.4f}, Precision = {prec:.4f}, Recall = {rec:.4f}, F1 Score = {f1:.4f}")
        
if __name__ == "__main__":
    run_layerwise_experiment()
