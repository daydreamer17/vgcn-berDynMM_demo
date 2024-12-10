import torch
import torch.nn as nn
from transformers import BertModel
from dgl.nn.pytorch import GraphConv

class VGCNBertDynMM_NoCharCNN(nn.Module):
    def __init__(self, bert_model_name, gcn_in_dim, gcn_hidden_dim, num_classes, layer_idx=-1):
        super(VGCNBertDynMM_NoCharCNN, self).__init__()
        # BERT部分
        self.bert = BertModel.from_pretrained(bert_model_name, output_hidden_states=True)
        self.layer_idx = layer_idx

        # GCN部分
        self.gcn_layer = GraphConv(768, gcn_hidden_dim, allow_zero_in_degree=True)
        self.gcn_fc = nn.Linear(gcn_hidden_dim, self.bert.config.hidden_size)

        # 门控机制
        self.gating_layer = nn.Linear(self.bert.config.hidden_size * 2, 2)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, graph):
        # BERT编码
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if self.layer_idx == -1:
            bert_output = bert_outputs.pooler_output
        else:
            bert_output = bert_outputs.hidden_states[self.layer_idx][:, 0, :]

        # GCN部分
        gcn_features = graph.ndata['feat']  # 这里没有CharCNN的输出了，直接使用图的特征
        gcn_output = self.gcn_layer(graph, gcn_features)
        gcn_output = gcn_output.mean(dim=0)  # 对GCN的输出进行池化
        gcn_output = gcn_output.unsqueeze(0).expand(bert_output.size(0), -1)

        gcn_output = self.gcn_fc(gcn_output)

        # 融合BERT输出和GCN输出
        combined_output = torch.cat([bert_output, gcn_output], dim=1)
        gate = torch.sigmoid(self.gating_layer(combined_output))
        bert_weight = gate[:, 0].unsqueeze(1)
        gcn_weight = gate[:, 1].unsqueeze(1)

        # 最终输出
        fused_output = bert_weight * bert_output + gcn_weight * gcn_output
        logits = self.fc(fused_output)
        return logits
