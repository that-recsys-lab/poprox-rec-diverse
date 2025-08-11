import numpy as np
import torch
from lenskit.pipeline import Component
from pydantic import BaseModel
from sklearn.preprocessing import OneHotEncoder

from poprox_concepts.domain import CandidateSet, InterestProfile, RecommendationList
from poprox_recommender.pytorch.datachecks import assert_tensor_size
from poprox_recommender.pytorch.decorators import torch_inference


class MMREncodingConfig(BaseModel):
    theta: float = 0.8
    num_slots: int = 10


class MMREncodingDiversifier(Component):
    config: MMREncodingConfig

    @torch_inference
    def __call__(self, candidate_articles: CandidateSet, interest_profile: InterestProfile) -> RecommendationList:
        if candidate_articles.scores is None:
            recommended = candidate_articles.articles
        else:
            topic_vectors = self._create_topic_vectors(candidate_articles.articles)
            similarity_matrix = compute_similarity_matrix(topic_vectors)
            scores = torch.as_tensor(candidate_articles.scores).to(similarity_matrix.device)
            article_indices = mmr_diversification(
                scores, similarity_matrix, theta=self.config.theta, topk=self.config.num_slots
            )
            recommended = [candidate_articles.articles[int(idx)] for idx in article_indices]
        return RecommendationList(articles=recommended)

    def _create_topic_vectors(self, articles) -> torch.Tensor:
        num_articles = len(articles)
        rows: list[list[str]] = []
        row_article_index: list[int] = []
        for i, article in enumerate(articles):
            article_topics = sorted({mention.entity.name for mention in article.mentions})
            for topic in article_topics:
                rows.append([topic])
                row_article_index.append(i)

        if len(rows) == 0:
            return torch.zeros((num_articles, 1), dtype=torch.float32)
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded_rows = encoder.fit_transform(rows)

        num_topics = encoded_rows.shape[1]
        topic_matrix = np.zeros((num_articles, num_topics), dtype=np.float32)
        for r, art_idx in enumerate(row_article_index):
            topic_matrix[art_idx] += encoded_rows[r]

        topic_matrix = (topic_matrix > 0).astype(np.float32)

        return torch.from_numpy(topic_matrix)


def compute_similarity_matrix(topic_vectors: torch.Tensor) -> torch.Tensor:
    num_values = len(topic_vectors)
    # M is (n, k), where n = # articles & k = # topics
    # M M^T is (n, n) matrix of pairwise dot products (topic overlap)
    similarity_matrix = topic_vectors @ topic_vectors.T
    assert_tensor_size(similarity_matrix, num_values, num_values, label="sim-matrix", prefix=False)
    return similarity_matrix


def mmr_diversification(rewards, similarity_matrix, theta: float, topk: int):
    # MR_i = \theta * reward_i - (1 - \theta)*max_{j \in S} sim(i, j) # S us
    # R is all candidates (not selected yet)

    # final recommendation (topk index) - initialize to invalid indexes
    S = torch.full((topk,), -1, dtype=torch.int32)
    # first recommended item
    S[0] = rewards.argmax()

    for k in range(1, topk):
        # find the best combo of reward and max sim to existing item
        # first, let's pare the matrix: candidates on rows, selected items on cols
        Sset = S[S >= 0]
        M = similarity_matrix[:, Sset]

        # for each target item, we want to find the *max* simialrity to an existing.
        # we do this by taking the max of each row.
        scores, _maxes = torch.max(M, axis=1)
        assert_tensor_size(scores, len(rewards), label="scores", prefix=False)

        # now, we want to compute θ*r - (1-θ)*s. let's do that *in-place* using
        # this scores vector. To start, multiply by θ-1 (-(1-θ)):
        scores *= theta - 1

        # with this, we can add theta * rewards in-place:
        scores.add_(rewards, alpha=theta)

        # now, we're looking for the *max* score in this list. we can do this
        # in two steps. step 1: clear the items we already have:
        scores[Sset] = -torch.inf
        # step 2: find the largest value
        S[k] = torch.argmax(scores)

    return S[S >= 0].tolist()
