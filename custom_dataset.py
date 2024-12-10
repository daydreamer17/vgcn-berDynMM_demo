import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer
import dgl
from collections import defaultdict
from VocabularyGraph import VocabularyGraph  # 从单独的文件导入 VocabularyGraph
from sklearn.model_selection import train_test_split  # 导入 train_test_split

def label_to_int(label):
    """
    将标签从字符串转换为整数。如果你有多个标签，请调整映射关系
    例如：将 "positive" 转换为 1，将 "negative" 转换为 0
    """
    label_mapping = {'malicious': 0, 'benign': 1}
    return label_mapping.get(label, -1)  # 返回 -1 作为默认值，防止未匹配标签

class CustomDataset(Dataset):
    def __init__(self, csv_file, tokenizer, vocab_size, gcn_in_dim, max_length=128, npmi_threshold=0.2, window_size=5, train=True, test_size=0.2,sample_fraction=None):
        """
        csv_file: 数据集文件路径
        tokenizer: BERT tokenizer，用于将文本转化为BERT输入
        vocab_size: 词汇表的大小
        gcn_in_dim: GCN输入的维度
        max_length: BERT输入的最大长度
        npmi_threshold: NPMI阈值，决定词之间是否建立边
        window_size: 滑动窗口大小，控制词共现范围
        train: 是否为训练集
        test_size: 测试集的比例
        sample_fraction: 采样的比例
        """
        self.data = pd.read_csv(csv_file)  # 先读取整个数据集

        if sample_fraction is not None:
            self.data = self.data.sample(frac=sample_fraction, random_state=42)  # 随机采样部分数据

        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.gcn_in_dim = gcn_in_dim
        self.npmi_threshold = npmi_threshold
        self.window_size = window_size

        # 将标签列转换为整数
        self.data['label'] = self.data['label'].apply(label_to_int)

        # 划分训练集和测试集
        train_data,test_data = train_test_split(self.data, test_size=test_size, random_state=42, stratify=self.data['label'])

        if train:
            self.data = train_data
        else:
            self.data = test_data

        # 初始化 VocabularyGraph
        self.vocab_graph = VocabularyGraph(self.vocab_size, window_size, npmi_threshold)
        self._build_vocabulary_graph()

        print("Unique labels in the dataset:", self.data['label'].unique())

    def _build_vocabulary_graph(self):
        for text in self.data['url']:
            tokens = self.tokenizer.tokenize(text)
            self.vocab_graph.process_sentence(tokens)
        self.global_graph = self.vocab_graph.build_graph(
            {token: i for i, token in enumerate(self.tokenizer.get_vocab())})

    def _extract_subgraph(self, token_list):
        subgraph = dgl.node_subgraph(self.global_graph,
                                     [self.tokenizer.convert_tokens_to_ids(token) for token in token_list if
                                      token in self.tokenizer.get_vocab()])
        subgraph.ndata['feat'] = torch.randn(subgraph.number_of_nodes(), self.gcn_in_dim)
        return subgraph

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['url']
        label = int(self.data.iloc[idx]['label'])

        # BERT 预处理
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        input_ids = encoding['input_ids'].squeeze(0)  # 去除批次维度
        attention_mask = encoding['attention_mask'].squeeze(0)

        # 提取子图
        token_list = self.tokenizer.convert_ids_to_tokens(input_ids)
        subgraph = self._extract_subgraph(token_list)

        # 这里需要生成 char_ids，如果你的模型需要它
        # 生成 char_ids 的示例：这取决于你的 tokenizer 和定义 char_ids 的方式
        char_ids = self._generate_char_ids(text)  # 你需要根据需要实现这个方法

        return input_ids, attention_mask, subgraph, char_ids, torch.tensor(label, dtype=torch.long)

    def _generate_char_ids(self, text):
        # 实现将文本转换为字符 ID 的逻辑
        # 这是一个占位符实现
        return torch.tensor([ord(c) for c in text], dtype=torch.long)






