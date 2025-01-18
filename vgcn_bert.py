# vgcn_bert.py

import torch
import torch.nn as nn
from transformers import BertModel
from dgl.nn.pytorch import GraphConv

class CharCNN(nn.Module):
    def __init__(self, char_vocab_size, char_emb_dim, kernel_sizes, num_kernels):
        """
        初始化 CharCNN 类。

        Args:
            char_vocab_size (int): 字符词汇表的大小。
            char_emb_dim (int): 字符嵌入的维度。
            kernel_sizes (list of int): 卷积核的大小列表。
            num_kernels (int): 每种卷积核的数量。
        """
        super(CharCNN, self).__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, char_emb_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=char_emb_dim, out_channels=num_kernels, kernel_size=k) for k in kernel_sizes
        ])
        self.highway = nn.Linear(len(kernel_sizes) * num_kernels, 768)  # 投影到与BERT输出相同的维度

    def forward(self, char_ids):
        """
        前向传播。

        Args:
            char_ids (torch.Tensor): 字符 ID，形状 [batch_size, seq_length]。

        Returns:
            torch.Tensor: CharCNN 的输出，形状 [batch_size, 768]。
        """
        x = self.char_embedding(char_ids).permute(0, 2, 1)  # 转换维度用于CNN输入
        conv_outputs = [torch.relu(conv(x)) for conv in self.convs]
        pooled_outputs = [torch.max(output, dim=2)[0] for output in conv_outputs]
        cnn_output = torch.cat(pooled_outputs, dim=1)  # 将不同卷积核的输出拼接
        cnn_output = self.highway(cnn_output)  # 投影到768维
        return cnn_output

class VGCNBertDynMM(nn.Module):
    def __init__(self, bert_model_name, gcn_in_dim_word, gcn_hidden_dim_word,
                 gcn_in_dim_char, gcn_hidden_dim_char, num_classes, char_vocab_size,
                 char_emb_dim, layer_idx=-1):
        """
        初始化 VGCNBertDynMM 模型。

        Args:
            bert_model_name (str): 预训练 BERT 模型的名称或路径。
            gcn_in_dim_word (int): 单词级 GCN 的输入维度。
            gcn_hidden_dim_word (int): 单词级 GCN 的隐藏维度。
            gcn_in_dim_char (int): 字符级 GCN 的输入维度。
            gcn_hidden_dim_char (int): 字符级 GCN 的隐藏维度。
            num_classes (int): 分类的类别数。
            char_vocab_size (int): 字符词汇表大小。
            char_emb_dim (int): 字符嵌入的维度。
            layer_idx (int): BERT 的层索引，-1 表示使用 pooler_output。
        """
        super(VGCNBertDynMM, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name, output_hidden_states=True)
        self.char_cnn = CharCNN(char_vocab_size, char_emb_dim, kernel_sizes=[3, 4, 5], num_kernels=50)
        self.layer_idx = layer_idx

        # 单词级 GCN
        self.gcn_layer_word = GraphConv(gcn_in_dim_word + 768, gcn_hidden_dim_word, allow_zero_in_degree=True)
        self.gcn_fc_word = nn.Linear(gcn_hidden_dim_word, self.bert.config.hidden_size)

        # 字符级 GCN
        self.gcn_layer_char = GraphConv(gcn_in_dim_char + 768, gcn_hidden_dim_char, allow_zero_in_degree=True)
        self.gcn_fc_char = nn.Linear(gcn_hidden_dim_char, self.bert.config.hidden_size)

        # 融合层
        self.gating_layer = nn.Linear(self.bert.config.hidden_size * 3, 3)  # BERT, GCN_word, GCN_char
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, graph_word, graph_char, char_ids):
        """
        前向传播。

        Args:
            input_ids (torch.Tensor): BERT 输入 ID，形状 [batch_size, seq_length]。
            attention_mask (torch.Tensor): BERT 注意力掩码，形状 [batch_size, seq_length]。
            graph_word (dgl.DGLGraph): 单词级图。
            graph_char (dgl.DGLGraph): 字符级图。
            char_ids (torch.Tensor): 字符 ID，形状 [batch_size, char_seq_length]。

        Returns:
            torch.Tensor: 分类 logits，形状 [batch_size, num_classes]。
        """
        # BERT 输出
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if self.layer_idx == -1:
            bert_output = bert_outputs.pooler_output  # [batch_size, hidden_size]
        else:
            bert_output = bert_outputs.hidden_states[self.layer_idx][:, 0, :]  # [batch_size, hidden_size]

        # CharCNN 输出
        char_output = self.char_cnn(char_ids)  # [batch_size, 768]

        # 单词级 GCN 处理
        # 将 BERT 和 CharCNN 的输出与图的节点特征拼接
        gcn_features_word = torch.cat([graph_word.ndata['feat'], char_output.mean(dim=0).unsqueeze(0).repeat(graph_word.number_of_nodes(), 1)], dim=1)
        gcn_output_word = self.gcn_layer_word(graph_word, gcn_features_word)  # [num_nodes, hidden_dim]
        gcn_output_word = gcn_output_word.mean(dim=0)  # [hidden_dim]
        gcn_output_word = self.gcn_fc_word(gcn_output_word).unsqueeze(0).repeat(bert_output.size(0), 1)  # [batch_size, hidden_size]

        # 字符级 GCN 处理
        gcn_features_char = torch.cat([graph_char.ndata['feat'], char_output.mean(dim=0).unsqueeze(0).repeat(graph_char.number_of_nodes(), 1)], dim=1)
        gcn_output_char = self.gcn_layer_char(graph_char, gcn_features_char)  # [num_nodes, hidden_dim]
        gcn_output_char = gcn_output_char.mean(dim=0)  # [hidden_dim]
        gcn_output_char = self.gcn_fc_char(gcn_output_char).unsqueeze(0).repeat(bert_output.size(0), 1)  # [batch_size, hidden_size]

        # 特征融合
        combined_output = torch.cat([bert_output, gcn_output_word, gcn_output_char], dim=1)  # [batch_size, hidden_size * 3]
        gate = torch.sigmoid(self.gating_layer(combined_output))  # [batch_size, 3]
        bert_weight = gate[:, 0].unsqueeze(1)
        gcn_word_weight = gate[:, 1].unsqueeze(1)
        gcn_char_weight = gate[:, 2].unsqueeze(1)

        fused_output = bert_weight * bert_output + gcn_word_weight * gcn_output_word + gcn_char_weight * gcn_output_char  # [batch_size, hidden_size]
        logits = self.fc(fused_output)  # [batch_size, num_classes]
        return logits









