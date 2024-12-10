import numpy as np
import pandas as pd
from collections import defaultdict
import dgl
import torch


class VocabularyGraph:
    def __init__(self, vocab_size, window_size=5, npmi_threshold=0.2):
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.npmi_threshold = npmi_threshold
        self.word_count = defaultdict(int)
        self.co_occurrence = defaultdict(int)
        self.total_windows = 0

    def process_sentence(self, sentence_tokens):
        """
        使用滑动窗口处理句子并计算共现频率
        """
        n = len(sentence_tokens)
        for i in range(n):
            self.word_count[sentence_tokens[i]] += 1
            for j in range(i + 1, min(i + self.window_size, n)):
                pair = tuple(sorted([sentence_tokens[i], sentence_tokens[j]]))
                self.co_occurrence[pair] += 1
        self.total_windows += 1

    def compute_npmi(self):
        """
        根据共现频率计算词对的 NPMI 值
        """
        npmi_values = defaultdict(float)
        for (word_i, word_j), co_occurrence_count in self.co_occurrence.items():
            p_i = self.word_count[word_i] / self.total_windows
            p_j = self.word_count[word_j] / self.total_windows
            p_ij = co_occurrence_count / self.total_windows
            npmi = -np.log(p_ij) / (np.log(p_i) + np.log(p_j))
            npmi_values[(word_i, word_j)] = npmi
        return npmi_values

    def build_graph(self, vocab):
        """
        根据 NPMI 构建词汇图
        """
        npmi_values = self.compute_npmi()
        # 使用 dgl.graph 替换 dgl.DGLGraph
        graph = dgl.graph(([], []))  # 初始化一个空图
        graph.add_nodes(self.vocab_size)

        for (word_i, word_j), npmi in npmi_values.items():
            if npmi >= self.npmi_threshold:
                # 使用 add_edges 替换 add_edge
                graph.add_edges(vocab[word_i], vocab[word_j])
                graph.add_edges(vocab[word_j], vocab[word_i])
        # 为每个节点添加自环
        graph = dgl.add_self_loop(graph)

        return graph
