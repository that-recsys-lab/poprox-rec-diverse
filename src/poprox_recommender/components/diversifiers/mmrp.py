# Add imports for entropy calculation
import numpy as np
import torch
from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import CandidateSet, InterestProfile, RecommendationList
from poprox_recommender.pytorch.datachecks import assert_tensor_size
from poprox_recommender.pytorch.decorators import torch_inference


class MMRPConfig(BaseModel):
    theta: float = 0.8
    num_slots: int = 10


def calculate_beta(interest_profile: InterestProfile) -> float:
    click_history = getattr(interest_profile, "click_history", [])
    all_topics = []
    for article in click_history:
        topics = getattr(article, "topics", [])
        all_topics.extend(topics)

    if all_topics:
        unique_topics, counts = np.unique(all_topics, return_counts=True)
        topic_weights = counts / counts.sum()
        entropy = -np.sum(topic_weights * np.log(topic_weights + 1e-12))  # epsilon for log(0)
        if len(unique_topics) > 1:
            entropy /= np.log(len(unique_topics))
        beta = entropy
    else:
        beta = 1.0

    return beta


class MMRPDiversifier(Component):
    config: MMRPConfig

    @torch_inference
    def __call__(self, candidate_articles: CandidateSet, interest_profile: InterestProfile) -> RecommendationList:
        beta = calculate_beta(interest_profile)
        theta = self.config.theta * beta

        if candidate_articles.scores is None:
            recommended = candidate_articles.articles
        else:
            similarity_matrix = compute_similarity_matrix(candidate_articles.embeddings)

            scores = torch.as_tensor(candidate_articles.scores).to(similarity_matrix.device)
            article_indices = mmrp_diversification(scores, similarity_matrix, theta=theta, topk=self.config.num_slots)
            recommended = [candidate_articles.articles[int(idx)] for idx in article_indices]

        return RecommendationList(articles=recommended)


def compute_similarity_matrix(todays_article_vectors):
    num_values = len(todays_article_vectors)
    # M is (n, k), where n = # articles & k = embed. dim.
    # M M^T is (n, n) matrix of pairwise dot products
    similarity_matrix = todays_article_vectors @ todays_article_vectors.T
    assert_tensor_size(similarity_matrix, num_values, num_values, label="sim-matrix", prefix=False)
    return similarity_matrix


def mmrp_diversification(rewards, similarity_matrix, theta: float, topk: int):
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
