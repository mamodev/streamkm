import math
import numpy as np
from typing import Union, Sequence


class ContingencyMatrix:
    def __init__(
        self,
        true_clusters: Union[Sequence[int], np.ndarray],
        pred_clusters: Union[Sequence[int], np.ndarray],
    ):
        # Convert to numpy arrays
        true_clusters = np.asarray(true_clusters, dtype=int)
        pred_clusters = np.asarray(pred_clusters, dtype=int)

        if true_clusters.shape[0] != pred_clusters.shape[0]:
            raise ValueError("true_clusters and pred_clusters must have same length")

        self.total = true_clusters.shape[0]
        self.k_true = true_clusters.max() + 1
        self.k_pred = pred_clusters.max() + 1

        # contingency matrix shape (k_true, k_pred)
        self.contingency = np.zeros((self.k_true, self.k_pred), dtype=int)

        for ti, pj in zip(true_clusters, pred_clusters):
            self.contingency[ti, pj] += 1

        self.true_cluster_sizes = self.contingency.sum(axis=1)  # a[i]
        self.predicted_cluster_sizes = self.contingency.sum(axis=0)  # b[j]

    # -------------------- Helpers --------------------

    def n(self, i: int, j: int) -> int:
        return self.contingency[i, j]

    def a(self, i: int) -> int:
        return self.true_cluster_sizes[i]

    def b(self, j: int) -> int:
        return self.predicted_cluster_sizes[j]

    def N(self) -> int:
        return self.total

    @staticmethod
    def comb2(x: int) -> float:
        return 0.0 if x < 2 else x * (x - 1) / 2.0

    # -------------------- Metrics --------------------

    def purity(self) -> float:
        return np.sum(np.max(self.contingency, axis=0)) / self.N()

    def adjusted_rand_index(self) -> float:
        sum_index = np.sum([self.comb2(nij) for nij in self.contingency.flatten()])
        sum_a = np.sum([self.comb2(ai) for ai in self.true_cluster_sizes])
        sum_b = np.sum([self.comb2(bj) for bj in self.predicted_cluster_sizes])

        nC2 = self.comb2(self.N())
        expected_index = (sum_a * sum_b) / nC2 if nC2 > 0 else 0.0
        max_index = 0.5 * (sum_a + sum_b)

        denom = (max_index - expected_index)
        return (sum_index - expected_index) / denom if denom != 0 else 0.0

    def normalized_mutual_info(self) -> float:
        Nf = float(self.N())
        mi = 0.0
        for i in range(self.k_true):
            for j in range(self.k_pred):
                nij = self.contingency[i, j]
                if nij > 0:
                    mi += (nij / Nf) * math.log(
                        (nij * Nf) / (self.a(i) * self.b(j))
                    )

        hu = -np.sum(
            [
                (ai / Nf) * math.log(ai / Nf)
                for ai in self.true_cluster_sizes if ai > 0
            ]
        )
        hv = -np.sum(
            [
                (bj / Nf) * math.log(bj / Nf)
                for bj in self.predicted_cluster_sizes if bj > 0
            ]
        )
        return 0.0 if hu == 0 or hv == 0 else mi / math.sqrt(hu * hv)

    def fowlkes_mallows(self) -> float:
        tp = np.sum([self.comb2(nij) for nij in self.contingency.flatten()])
        fp = np.sum([self.comb2(bj) for bj in self.predicted_cluster_sizes]) - tp
        fn = np.sum([self.comb2(ai) for ai in self.true_cluster_sizes]) - tp
        return 0.0 if tp == 0 else tp / math.sqrt((tp + fp) * (tp + fn))