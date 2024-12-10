import torch
import torch.nn as nn
from transformers import BertModel
from dgl.nn.pytorch import GraphConv

class CharCNN(nn.Module):
    def __init__(self, char_vocab_size, char_emb_dim, kernel_sizes, num_kernels):
        super(CharCNN, self).__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, char_emb_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=char_emb_dim, out_channels=num_kernels, kernel_size=k) for k in kernel_sizes
        ])
        self.highway = nn.Linear(len(kernel_sizes) * num_kernels, 768)  # 投影到与BERT输出相同的维度

    def forward(self, char_ids):
        x = self.char_embedding(char_ids).permute(0, 2, 1)  # 转换维度用于CNN输入
        conv_outputs = [torch.relu(conv(x)) for conv in self.convs]
        pooled_outputs = [torch.max(output, dim=2)[0] for output in conv_outputs]
        cnn_output = torch.cat(pooled_outputs, dim=1)  # 将不同卷积核的输出拼接
        cnn_output = self.highway(cnn_output)  # 投影到768维
        return cnn_output


class VGCNBertDynMM(nn.Module):
    def __init__(self, bert_model_name, gcn_in_dim, gcn_hidden_dim, num_classes, char_vocab_size, char_emb_dim, layer_idx=-1):
        super(VGCNBertDynMM, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name, output_hidden_states=True)
        self.char_cnn = CharCNN(char_vocab_size, char_emb_dim, kernel_sizes=[3, 4, 5], num_kernels=50)
        self.layer_idx = layer_idx

        self.gcn_layer = GraphConv(gcn_in_dim + 768, gcn_hidden_dim, allow_zero_in_degree=True)
        self.gcn_fc = nn.Linear(gcn_hidden_dim, self.bert.config.hidden_size)

        self.gating_layer = nn.Linear(self.bert.config.hidden_size * 2, 2)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, graph, char_ids):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if self.layer_idx == -1:
            bert_output = bert_outputs.pooler_output
        else:
            bert_output = bert_outputs.hidden_states[self.layer_idx][:, 0, :]

        char_output = self.char_cnn(char_ids)

        gcn_features = torch.cat([graph.ndata['feat'], char_output.unsqueeze(0).mean(1).repeat(len(graph.ndata['feat']), 1)], dim=1)
        gcn_output = self.gcn_layer(graph, gcn_features)
        gcn_output = gcn_output.mean(dim=0)
        gcn_output = gcn_output.unsqueeze(0).expand(bert_output.size(0), -1)

        gcn_output = self.gcn_fc(gcn_output)

        combined_output = torch.cat([bert_output, gcn_output], dim=1)
        gate = torch.sigmoid(self.gating_layer(combined_output))
        bert_weight = gate[:, 0].unsqueeze(1)
        gcn_weight = gate[:, 1].unsqueeze(1)

        fused_output = bert_weight * bert_output + gcn_weight * gcn_output
        logits = self.fc(fused_output)
        return logits









