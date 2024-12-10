import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
from vgcn_bert import VGCNBertDynMM
from VocabularyGraph import VocabularyGraph
import dgl
from tqdm import tqdm


def collate_fn(batch):
    input_ids, attention_masks, graphs, char_ids, labels = zip(*batch)

    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    batched_graph = dgl.batch(graphs)

    # 对 char_ids 进行填充，使得它们具有相同的长度
    char_ids = [torch.tensor(c) for c in char_ids]
    char_ids = torch.nn.utils.rnn.pad_sequence(char_ids, batch_first=True, padding_value=0)

    labels = torch.stack(labels)

    return input_ids, attention_masks, batched_graph, char_ids, labels



def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in tqdm(data_loader, desc="Training", leave=False):
        # 解包 char_ids
        input_ids, attention_mask, graph, char_ids, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        graph = graph.to(device)
        char_ids = char_ids.to(device)  # 将 char_ids 也传输到设备
        labels = labels.to(device)

        # 将 char_ids 传递给模型
        logits = model(input_ids, attention_mask, graph, char_ids)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)



def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            # 解包 char_ids
            input_ids, attention_mask, graph, char_ids, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            graph = graph.to(device)
            char_ids = char_ids.to(device)  # 将 char_ids 也传输到设备
            labels = labels.to(device)

            # 将 char_ids 传递给模型
            logits = model(input_ids, attention_mask, graph, char_ids)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()

    accuracy = correct / len(data_loader.dataset)
    return total_loss / len(data_loader), accuracy



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bert_model_dir = './model'
    num_classes = 2
    gcn_in_dim = 768
    gcn_hidden_dim = 128
    char_vocab_size = 256  # 假设字符词汇大小为256，请根据你的数据集进行调整
    char_emb_dim = 128  # 假设字符嵌入维度为128，请根据你的需求进行调整
    learning_rate = 2e-5
    num_epochs = 5
    batch_size = 16

    tokenizer = BertTokenizer.from_pretrained(bert_model_dir)
    vocab_size = len(tokenizer.get_vocab())
    vocab_graph = VocabularyGraph(vocab_size)

    train_dataset = CustomDataset('Data/Mendeley_dataset/Train_data.csv', tokenizer, vocab_size, gcn_in_dim)
    val_dataset = CustomDataset('Data/Mendeley_dataset/Test_data.csv', tokenizer, vocab_size, gcn_in_dim)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 使用 VGCNBertDynMM，确保提供所有必需的参数
    model = VGCNBertDynMM(bert_model_dir, gcn_in_dim, gcn_hidden_dim, num_classes, char_vocab_size, char_emb_dim)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')


if __name__ == "__main__":
    main()




