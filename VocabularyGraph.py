# VocabularyGraph.py

import torch
from collections import defaultdict
import dgl
import math

class VocabularyGraph:
    def __init__(self, vocab_size, window_size=5, npmi_threshold=0.2):
        """
        初始化 VocabularyGraph 类。

        Args:
            vocab_size (int): 词汇表的大小。
            window_size (int): 滑动窗口的大小，用于统计词共现。
            npmi_threshold (float): NPMI 阈值，决定词之间是否建立边。
        """
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.npmi_threshold = npmi_threshold
        self.word_count = defaultdict(int)
        self.co_occurrence = defaultdict(int)
        self.total_windows = 0

    def process_sentence(self, sentence_tokens):
        """
        处理一个句子的词汇，统计词频和共现频率。

        Args:
            sentence_tokens (list of str): 句子的词列表。
        """
        n = len(sentence_tokens)
        for i in range(n):
            self.word_count[sentence_tokens[i]] += 1
            window_end = min(i + self.window_size, n)
            window_size = window_end - i - 1
            for j in range(i + 1, window_end):
                pair = tuple(sorted([sentence_tokens[i], sentence_tokens[j]]))
                self.co_occurrence[pair] += 1
            self.total_windows += window_size  # 每个词的窗口数

    def compute_npmi(self):
        """
        计算每对词的 NPMI 值。

        Returns:
            dict: 词对到其 NPMI 值的映射。
        """
        npmi_values = defaultdict(float)
        for (word_i, word_j), co_count in self.co_occurrence.items():
            p_i = self.word_count[word_i] / self.total_windows
            p_j = self.word_count[word_j] / self.total_windows
            p_ij = co_count / self.total_windows
            if p_ij > 0 and p_i > 0 and p_j > 0:
                npmi = -math.log(p_ij) / (math.log(p_i) + math.log(p_j))
                npmi_values[(word_i, word_j)] = npmi
        return npmi_values

    def build_graph(self, vocab):
        """
        根据 NPMI 值构建 DGL 图。

        Args:
            vocab (dict): 词汇表，词到索引的映射。

        Returns:
            dgl.DGLGraph: 构建好的图。
        """
        npmi_values = self.compute_npmi()
        graph = dgl.graph(([], []))
        graph.add_nodes(self.vocab_size)

        src_nodes = []
        dst_nodes = []
        weights = []
        for (word_i, word_j), npmi in npmi_values.items():
            if npmi >= self.npmi_threshold:
                src_nodes.extend([vocab[word_i], vocab[word_j]])
                dst_nodes.extend([vocab[word_j], vocab[word_i]])
                weights.extend([npmi, npmi])  # 双向边的权重相同

        if src_nodes and dst_nodes:
            graph.add_edges(src_nodes, dst_nodes)
            graph.edata['weight'] = torch.tensor(weights, dtype=torch.float32)

        graph = dgl.add_self_loop(graph)
        return graph
