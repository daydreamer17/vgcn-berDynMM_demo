# CharacterGraph.py

import torch
from collections import defaultdict
import dgl
import math

class CharacterGraph:
    def __init__(self, char_vocab_size, window_size=3, npmi_threshold=0.2):
        """
        初始化 CharacterGraph 类。

        Args:
            char_vocab_size (int): 字符词汇表的大小。
            window_size (int): 滑动窗口的大小，用于统计字符共现。
            npmi_threshold (float): NPMI 阈值，决定字符之间是否建立边。
        """
        self.char_vocab_size = char_vocab_size
        self.window_size = window_size
        self.npmi_threshold = npmi_threshold
        self.char_count = defaultdict(int)
        self.co_occurrence = defaultdict(int)
        self.total_windows = 0

    def process_url(self, url_chars):
        """
        处理一个 URL 的字符，统计字符频率和共现频率。

        Args:
            url_chars (list of str): URL 的字符列表。
        """
        n = len(url_chars)
        for i in range(n):
            self.char_count[url_chars[i]] += 1
            window_end = min(i + self.window_size, n)
            window_size = window_end - i - 1
            for j in range(i + 1, window_end):
                pair = tuple(sorted([url_chars[i], url_chars[j]]))
                self.co_occurrence[pair] += 1
            self.total_windows += window_size  # 每个字符的窗口数

    def compute_npmi(self):
        """
        计算每对字符的 NPMI 值。

        Returns:
            dict: 字符对到其 NPMI 值的映射。
        """
        npmi_values = defaultdict(float)
        for (char_i, char_j), co_count in self.co_occurrence.items():
            p_i = self.char_count[char_i] / self.total_windows
            p_j = self.char_count[char_j] / self.total_windows
            p_ij = co_count / self.total_windows
            if p_ij > 0 and p_i > 0 and p_j > 0:
                npmi = -math.log(p_ij) / (math.log(p_i) + math.log(p_j))
                npmi_values[(char_i, char_j)] = npmi
        return npmi_values

    def build_graph(self, char_vocab):
        """
        根据 NPMI 值构建 DGL 图。

        Args:
            char_vocab (dict): 字符词汇表，字符到索引的映射。

        Returns:
            dgl.DGLGraph: 构建好的图。
        """
        npmi_values = self.compute_npmi()
        graph = dgl.graph(([], []))
        graph.add_nodes(self.char_vocab_size)

        src_nodes = []
        dst_nodes = []
        weights = []
        for (char_i, char_j), npmi in npmi_values.items():
            if npmi >= self.npmi_threshold:
                src_nodes.extend([char_vocab[char_i], char_vocab[char_j]])
                dst_nodes.extend([char_vocab[char_j], char_vocab[char_i]])
                weights.extend([npmi, npmi])  # 双向边的权重相同

        if src_nodes and dst_nodes:
            graph.add_edges(src_nodes, dst_nodes)
            graph.edata['weight'] = torch.tensor(weights, dtype=torch.float32)

        graph = dgl.add_self_loop(graph)
        return graph
