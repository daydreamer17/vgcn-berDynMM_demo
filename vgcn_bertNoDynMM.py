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
            nn.Conv1d(in_channels=char_emb_dim, out_channels=num_kernels, kernel_size=k) 
            for k in kernel_sizes
        ])
        # 投影到与BERT输出相同的维度 (768)
        self.highway = nn.Linear(len(kernel_sizes) * num_kernels, 768)

    def forward(self, char_ids):
        """
        前向传播。

        Args:
            char_ids (torch.Tensor): 字符 ID，形状 [batch_size, seq_length]。

        Returns:
            torch.Tensor: CharCNN 的输出，形状 [batch_size, 768]。
        """
        x = self.char_embedding(char_ids).permute(0, 2, 1)  # [batch_size, emb_dim, seq_len]
        conv_outputs = [torch.relu(conv(x)) for conv in self.convs]  # 每个Conv1d输出形如 [batch_size, num_kernels, seq_len']
        pooled_outputs = [torch.max(output, dim=2)[0] for output in conv_outputs]  
        # pooled_outputs里每个元素形如 [batch_size, num_kernels]
        cnn_output = torch.cat(pooled_outputs, dim=1)  # [batch_size, num_kernels * len(kernel_sizes)]
        cnn_output = self.highway(cnn_output)          # [batch_size, 768]
        return cnn_output

class VGCNBertNoDynMM(nn.Module):
    """
    去除 DynMM 后的模型：
    - 保留 BERT、CharCNN、词级/字符级 GCN
    - 不再使用 gating_layer 动态加权
    - 改为简单拼接后再用一个 linear layer (fusion_fc) 将拼接向量降维到 768
    - 最后用 self.fc 进行分类
    """
    def __init__(self, bert_model_name, gcn_in_dim_word, gcn_hidden_dim_word,
                 gcn_in_dim_char, gcn_hidden_dim_char, num_classes,
                 char_vocab_size, char_emb_dim, layer_idx=-1):
        """
        初始化 VGCNBertNoDynMM 模型。

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
        super(VGCNBertNoDynMM, self).__init__()
        
        # 1) BERT
        self.bert = BertModel.from_pretrained(bert_model_name, output_hidden_states=True)
        self.layer_idx = layer_idx

        # 2) CharCNN
        self.char_cnn = CharCNN(
            char_vocab_size=char_vocab_size,
            char_emb_dim=char_emb_dim,
            kernel_sizes=[3, 4, 5],
            num_kernels=50
        )

        # 3) 词级 GCN
        self.gcn_layer_word = GraphConv(
            in_feats=(gcn_in_dim_word + 768),
            out_feats=gcn_hidden_dim_word,
            allow_zero_in_degree=True
        )
        self.gcn_fc_word = nn.Linear(gcn_hidden_dim_word, self.bert.config.hidden_size)

        # 4) 字符级 GCN
        self.gcn_layer_char = GraphConv(
            in_feats=(gcn_in_dim_char + 768),
            out_feats=gcn_hidden_dim_char,
            allow_zero_in_degree=True
        )
        self.gcn_fc_char = nn.Linear(gcn_hidden_dim_char, self.bert.config.hidden_size)

        # ★ 移除 gating_layer，改用简单拼接后线性层降维 (fusion_fc)
        # dynmm_removed: no gating_layer
        # hidden_size * 3 -> hidden_size
        self.fusion_fc = nn.Linear(self.bert.config.hidden_size * 3, self.bert.config.hidden_size)

        # 最终分类层
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
        # 1) BERT 输出
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if self.layer_idx == -1:
            bert_output = bert_outputs.pooler_output  # [batch_size, hidden_size]
        else:
            # 取指定层的 [CLS] 向量
            bert_output = bert_outputs.hidden_states[self.layer_idx][:, 0, :]  # [batch_size, hidden_size]

        # 2) CharCNN 输出
        char_output = self.char_cnn(char_ids)  # [batch_size, 768]

        # 3) 单词级 GCN 处理
        #    注意：代码中以 char_output.mean(dim=0) 进行拼接，这里保持不变
        gcn_features_word = torch.cat([
            graph_word.ndata['feat'],
            char_output.mean(dim=0).unsqueeze(0).repeat(graph_word.number_of_nodes(), 1)
        ], dim=1)
        gcn_output_word = self.gcn_layer_word(graph_word, gcn_features_word)  # [num_nodes, gcn_hidden_dim_word]
        gcn_output_word = gcn_output_word.mean(dim=0)  # [gcn_hidden_dim_word]
        gcn_output_word = self.gcn_fc_word(gcn_output_word).unsqueeze(0).repeat(bert_output.size(0), 1)  
        # [batch_size, hidden_size]

        # 4) 字符级 GCN 处理
        gcn_features_char = torch.cat([
            graph_char.ndata['feat'],
            char_output.mean(dim=0).unsqueeze(0).repeat(graph_char.number_of_nodes(), 1)
        ], dim=1)
        gcn_output_char = self.gcn_layer_char(graph_char, gcn_features_char)  # [num_nodes, gcn_hidden_dim_char]
        gcn_output_char = gcn_output_char.mean(dim=0)  # [gcn_hidden_dim_char]
        gcn_output_char = self.gcn_fc_char(gcn_output_char).unsqueeze(0).repeat(bert_output.size(0), 1)
        # [batch_size, hidden_size]

        # ★ 5) 去除 DynMM 动态加权，改用简单拼接 + 线性降维
        #    拼接 bert_output, gcn_output_word, gcn_output_char
        combined_output = torch.cat([bert_output, gcn_output_word, gcn_output_char], dim=1)
        # [batch_size, hidden_size * 3]

        # 线性降维到 hidden_size (768)，也可以视需要添加激活函数
        fused_output = self.fusion_fc(combined_output)  # [batch_size, hidden_size]

        # 6) 分类层
        logits = self.fc(fused_output)  # [batch_size, num_classes]
        return logits










