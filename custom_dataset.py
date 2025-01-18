# custom_dataset.py

import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer
import dgl
from VocabularyGraph import VocabularyGraph
from CharacterGraph import CharacterGraph
from sklearn.model_selection import train_test_split

def label_to_int(label):
    """
    将标签从字符串转换为整数。

    Args:
        label (str): 原始标签。

    Returns:
        int: 转换后的整数标签。
    """
    label_mapping = {'bad': 0, 'good': 1}
    return label_mapping.get(label, -1)  # 返回 -1 作为默认值，防止未匹配标签

class CustomDataset(Dataset):
    def __init__(self, csv_file, tokenizer, vocab_size, gcn_in_dim_word, 
                 gcn_in_dim_char=None, char_vocab_size=None, use_char=False,
                 npmi_threshold=0.2, window_size=5, max_length=128, 
                 train=True, test_size=0.2, sample_fraction=None):
        """
        初始化 CustomDataset 类。

        Args:
            csv_file (str): 数据集文件路径。
            tokenizer (BertTokenizer): BERT 分词器。
            vocab_size (int): 词汇表的大小。
            gcn_in_dim_word (int): 单词级图的节点特征维度。
            gcn_in_dim_char (int, optional): 字符级图的节点特征维度。默认为 None。
            char_vocab_size (int, optional): 字符词汇表大小。默认为 None。
            use_char (bool): 是否使用字符相关数据。默认为 False。
            npmi_threshold (float): NPMI 阈值，决定词/字符之间是否建立边。
            window_size (int): 滑动窗口大小，控制词/字符共现范围。
            max_length (int): BERT 输入的最大长度。
            train (bool): 是否为训练集。
            test_size (float): 测试集的比例。
            sample_fraction (float or None): 采样的比例。默认为 None，表示不进行采样。
        """
        self.use_char = use_char
        self.data = pd.read_csv(csv_file)  # 读取整个数据集

        if sample_fraction is not None:
            self.data = self.data.sample(frac=sample_fraction, random_state=42)  # 随机采样部分数据

        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.gcn_in_dim_word = gcn_in_dim_word
        self.npmi_threshold = npmi_threshold
        self.window_size = window_size

        if self.use_char:
            if gcn_in_dim_char is None or char_vocab_size is None:
                raise ValueError("gcn_in_dim_char and char_vocab_size must be provided when use_char is True.")
            self.gcn_in_dim_char = gcn_in_dim_char
            self.char_vocab_size = char_vocab_size

        # 将标签列转换为整数
        self.data['label'] = self.data['label'].apply(label_to_int)

        # 划分训练集和测试集
        train_data, test_data = train_test_split(
            self.data, test_size=test_size, random_state=42, stratify=self.data['label']
        )

        if train:
            self.data = train_data
        else:
            self.data = test_data

        # 初始化 VocabularyGraph
        self.vocab_graph = VocabularyGraph(self.vocab_size, window_size, npmi_threshold)
        self._build_vocab_graph()

        if self.use_char:
            # 初始化 CharacterGraph
            self.char_graph = CharacterGraph(self.char_vocab_size, window_size=3, npmi_threshold=0.2)
            self._build_char_graph()

    def _build_vocab_graph(self):
        """
        构建全局的单词级图。
        """
        for text in self.data['url']:
            tokens = self.tokenizer.tokenize(text)
            self.vocab_graph.process_sentence(tokens)
        self.global_graph_word = self.vocab_graph.build_graph(
            {token: i for i, token in enumerate(self.tokenizer.get_vocab())}
        )
        # 初始化节点特征
        self.global_graph_word.ndata['feat'] = torch.randn(
            self.global_graph_word.number_of_nodes(), self.gcn_in_dim_word
        )

    def _build_char_graph(self):
        """
        构建全局的字符级图。
        """
        for text in self.data['url']:
            url_chars = list(text)
            self.char_graph.process_url(url_chars)
        self.global_graph_char = self.char_graph.build_graph(
            {chr(i): i for i in range(256)}  # 假设字符是ASCII
        )
        # 初始化节点特征
        self.global_graph_char.ndata['feat'] = torch.randn(
            self.global_graph_char.number_of_nodes(), self.gcn_in_dim_char
        )

    def _extract_subgraph_word(self, token_list):
        """
        根据词列表提取单词级子图。

        Args:
            token_list (list of str): 词列表。

        Returns:
            dgl.DGLGraph: 提取的子图。
        """
        node_ids = [
            self.tokenizer.convert_tokens_to_ids(token) 
            for token in token_list 
            if token in self.tokenizer.get_vocab()
        ]
        node_ids = list(filter(lambda x: x != self.tokenizer.pad_token_id, node_ids))
        if not node_ids:
            node_ids = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)]
        subgraph = dgl.node_subgraph(self.global_graph_word, node_ids)
        return subgraph

    def _extract_subgraph_char(self, char_list):
        """
        根据字符列表提取字符级子图。

        Args:
            char_list (list of str): 字符列表。

        Returns:
            dgl.DGLGraph: 提取的子图。
        """
        node_ids = [ord(c) for c in char_list if ord(c) < 256]
        if not node_ids:
            node_ids = [ord(' ')]  # 空格作为默认字符
        subgraph = dgl.node_subgraph(self.global_graph_char, node_ids)
        return subgraph

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取一个样本的数据。

        Args:
            idx (int): 样本索引。

        Returns:
            tuple: 
                - 如果 use_char=True:
                    (input_ids, attention_mask, subgraph_word, subgraph_char, char_ids, label)
                - 如果 use_char=False:
                    (input_ids, attention_mask, subgraph_word, label)
        """
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

        # 提取单词级子图
        token_list = self.tokenizer.convert_ids_to_tokens(input_ids)
        subgraph_word = self._extract_subgraph_word(token_list)

        if self.use_char:
            # 提取字符级子图
            char_list = list(text)
            subgraph_char = self._extract_subgraph_char(char_list)

            # 生成 char_ids
            char_ids = self._generate_char_ids(text)

            return input_ids, attention_mask, subgraph_word, subgraph_char, char_ids, torch.tensor(label, dtype=torch.long)
        else:
            return input_ids, attention_mask, subgraph_word, torch.tensor(label, dtype=torch.long)

    def _generate_char_ids(self, text):
        """
        将文本转换为字符 ID。

        Args:
            text (str): 输入文本。

        Returns:
            torch.Tensor: 字符 ID 的张量。
        """
        # 实现将文本转换为字符 ID 的逻辑
        # 例如，将每个字符转换为其ASCII码，或使用预定义的字符词汇表
        return torch.tensor([ord(c) for c in text], dtype=torch.long)






