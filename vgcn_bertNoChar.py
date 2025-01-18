# vgcn_bertNoChar.py

import torch
import torch.nn as nn
from transformers import BertModel
from dgl.nn.pytorch import GraphConv
import dgl


class VGCNBertDynMM(nn.Module):
    def __init__(self, bert_model_name, gcn_in_dim_word, gcn_hidden_dim_word,
                 num_classes, layer_idx=-1):
        """
        初始化 VGCNBertDynMM 模型。

        Args:
            bert_model_name (str): 预训练 BERT 模型的名称或路径。
            gcn_in_dim_word (int): 单词级 GCN 的输入维度。
            gcn_hidden_dim_word (int): 单词级 GCN 的隐藏维度。
            num_classes (int): 分类的类别数。
            layer_idx (int): BERT 的层索引，-1 表示使用 pooler_output。
        """
        super(VGCNBertDynMM, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name, output_hidden_states=True)
        self.layer_idx = layer_idx

        # 单词级 GCN
        self.gcn_layer_word = GraphConv(gcn_in_dim_word + self.bert.config.hidden_size, 
                                       gcn_hidden_dim_word, allow_zero_in_degree=True)
        self.gcn_fc_word = nn.Linear(gcn_hidden_dim_word, self.bert.config.hidden_size)

        # 融合层
        self.gating_layer = nn.Linear(self.bert.config.hidden_size * 2, 2)  # BERT, GCN_word
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, graph_word):
        """
        前向传播。

        Args:
            input_ids (torch.Tensor): BERT 输入 ID，形状 [batch_size, seq_length]。
            attention_mask (torch.Tensor): BERT 注意力掩码，形状 [batch_size, seq_length]。
            graph_word (dgl.DGLGraph): 批量单词级图。

        Returns:
            torch.Tensor: 分类 logits，形状 [batch_size, num_classes]。
        """
        # BERT 输出
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if self.layer_idx == -1:
            bert_output = bert_outputs.pooler_output  # [batch_size, hidden_size]
        else:
            bert_output = bert_outputs.hidden_states[self.layer_idx][:, 0, :]  # [batch_size, hidden_size]

        # 获取每个图的节点数
        num_nodes_per_graph = graph_word.batch_num_nodes()  # [batch_size]

        # 扩展 BERT 输出以匹配图节点
        bert_output_expanded = bert_output.repeat_interleave(num_nodes_per_graph, dim=0)  # [total_nodes, hidden_size]

        # 拼接图节点特征与 BERT 输出
        gcn_features_word = torch.cat([graph_word.ndata['feat'], bert_output_expanded], dim=1)  # [total_nodes, gcn_in_dim_word + hidden_size]

        # 通过 GCN 层
        gcn_output_word = self.gcn_layer_word(graph_word, gcn_features_word)  # [total_nodes, gcn_hidden_dim_word]
        gcn_output_word = torch.relu(gcn_output_word)
        # 聚合每个图的 GCN 输出（如取平均）
        graph_word.ndata['gcn_hidden'] = gcn_output_word
        gcn_output_word_graph = dgl.mean_nodes(graph_word, 'gcn_hidden')  # [batch_size, gcn_hidden_dim_word]

        # 通过线性层调整维度
        gcn_output_word_graph = self.gcn_fc_word(gcn_output_word_graph)  # [batch_size, hidden_size]

        # 特征融合
        combined_output = torch.cat([bert_output, gcn_output_word_graph], dim=1)  # [batch_size, hidden_size * 2]
        gate = torch.sigmoid(self.gating_layer(combined_output))  # [batch_size, 2]
        bert_weight = gate[:, 0].unsqueeze(1)  # [batch_size, 1]
        gcn_word_weight = gate[:, 1].unsqueeze(1)  # [batch_size, 1]

        fused_output = bert_weight * bert_output + gcn_word_weight * gcn_output_word_graph  # [batch_size, hidden_size]
        logits = self.fc(fused_output)  # [batch_size, num_classes]
        return logits
