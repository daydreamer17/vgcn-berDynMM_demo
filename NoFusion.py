# NoFusion.py

import logging
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import BertTokenizer
from VocabularyGraph import VocabularyGraph
from custom_dataset import CustomDataset
from vgcn_bertNoChar import VGCNBertDynMM
import dgl
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

def collate_fn(batch):
    """
    自定义的 collate_fn，用于 DataLoader。

    Args:
        batch (list of tuples): 每个元素为 (input_ids, attention_mask, subgraph_word, label)。

    Returns:
        tuple: (input_ids, attention_masks, batched_graph_word, labels)
    """
    input_ids, attention_masks, graphs_word, labels = zip(*batch)
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    batched_graph_word = dgl.batch(graphs_word)
    labels = torch.stack(labels)
    return input_ids, attention_masks, batched_graph_word, labels

def evaluate(model, data_loader, device, num_classes=2):
    """
    评估模型性能。

    Args:
        model (nn.Module): 要评估的模型。
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
            input_ids, attention_mask, graph_word, labels = batch
            input_ids, attention_mask, graph_word, labels = (
                input_ids.to(device, non_blocking=True),
                attention_mask.to(device, non_blocking=True),
                graph_word.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )

            logits = model(input_ids, attention_mask, graph_word)
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
    plt.savefig("confusion_matrix_withoutcharcnn.png")
    plt.close()

    # Plot ROC for each class
    y_true_binary = np.eye(num_classes)[y_true]
    y_probs = np.array(y_probs)
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f}) for class {i}")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for Class {i}")
        plt.legend(loc="lower right")
        plt.savefig(f"roc_curve_class_{i}_withoutcharcnn.png")
        plt.close()

    # Print overall metrics
    print(f"Evaluation: Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Print class-specific metrics
    for i in range(num_classes):
        class_precision = precision_score(np.array(y_true) == i, np.array(y_pred) == i, zero_division=0)
        class_recall = recall_score(np.array(y_true) == i, np.array(y_pred) == i, zero_division=0)
        class_f1 = f1_score(np.array(y_true) == i, np.array(y_pred) == i, zero_division=0)
        print(f"Class {i} Metrics: Precision = {class_precision:.4f}, Recall = {class_recall:.4f}, F1 Score = {class_f1:.4f}")

    return accuracy, precision, recall, f1

def run_layerwise_experiment():
    """
    运行分层实验，训练和评估模型在不同 BERT 层的表现。
    """
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    bert_model_dir = "./model"  # 使用预训练的 BERT 模型名称或路径
    num_classes = 2
    gcn_in_dim_word = 768  # 单词级图的节点特征维度
    gcn_hidden_dim_word = 128
    # char_vocab_size 和 char_emb_dim 已移除
    learning_rate = 2e-5
    num_epochs = 10
    batch_size = 16

    tokenizer = BertTokenizer.from_pretrained(bert_model_dir)

    vocab_size = len(tokenizer.get_vocab())

    # 数据加载
    try:
        train_dataset = CustomDataset(
            csv_file="Data/Grambedding_dataset/Train.csv",
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            gcn_in_dim_word=gcn_in_dim_word,
            use_char=False,  # 消融实验，不使用字符相关数据
            # gcn_in_dim_char 和 char_vocab_size 不需要传递
            train=True,
            test_size=0.2,
            sample_fraction=0.0125
        )
        val_dataset = CustomDataset(
            csv_file="Data/Grambedding_dataset/Train.csv",
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            gcn_in_dim_word=gcn_in_dim_word,
            use_char=False,  # 消融实验，不使用字符相关数据
            # gcn_in_dim_char 和 char_vocab_size 不需要传递
            train=False,
            test_size=0.2,
            sample_fraction=0.05
        )
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        return

    # 优化的数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        pin_memory=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        pin_memory=True, 
        num_workers=4
    )

    results = []
    for layer_idx in [1, 6, 12]:
        logger.info(f"--- Training with BERT layer {layer_idx} ---")
       
        model = VGCNBertDynMM(
                bert_model_name=bert_model_dir,
                gcn_in_dim_word=gcn_in_dim_word,
                gcn_hidden_dim_word=gcn_hidden_dim_word,
                num_classes=num_classes,
                layer_idx=layer_idx
        )
        model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()


        for epoch in range(num_epochs):
            logger.info(f"Layer {layer_idx}, Epoch {epoch + 1}/{num_epochs}")
            model.train()
            total_train_loss = 0
            for batch in tqdm(train_loader, desc="Training", leave=False):
                input_ids, attention_mask, graph_word_batch, labels = batch
                input_ids, attention_mask, graph_word_batch, labels = (
                    input_ids.to(device, non_blocking=True),
                    attention_mask.to(device, non_blocking=True),
                    graph_word_batch.to(device, non_blocking=True),
                    labels.to(device, non_blocking=True),
                )

                optimizer.zero_grad()
                logits = model(input_ids, attention_mask, graph_word_batch)
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
    plt.title("Layer-wise Performance of VGCNBertDynMM Model")
    plt.xlabel("BERT Layer Index")
    plt.ylabel("Metrics")
    plt.legend()
    plt.grid()
    plt.savefig("layerwise_performance_withoutcharcnn.png")
    plt.close()

    logger.info("\nLayer-wise Experiment Results:")
    for layer, acc, prec, rec, f1 in results:
        logger.info(f"Layer {layer}: Accuracy = {acc:.4f}, Precision = {prec:.4f}, Recall = {rec:.4f}, F1 Score = {f1:.4f}")

if __name__ == "__main__":
    run_layerwise_experiment()




