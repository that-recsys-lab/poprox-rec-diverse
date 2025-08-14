# Add imports for entropy calculation
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import CandidateSet, InterestProfile, RecommendationList
from poprox_recommender.pytorch.datachecks import assert_tensor_size
from poprox_recommender.pytorch.decorators import torch_inference
from poprox_recommender.topics import find_topic

logger = logging.getLogger(__name__)


class MMRPConfig(BaseModel):
    theta: float = 0.8
    num_slots: int = 10


def collect_beta_data(
    interest_profile: InterestProfile,
    unique_topics: np.ndarray | None,
    topic_counts: np.ndarray | None,
    beta: float,
    theta: float,
) -> dict:
    """Collect beta data for logging and analysis."""
    user_id = getattr(interest_profile, "user_id", "unknown")
    profile_id = getattr(interest_profile, "profile_id", "unknown")
    click_topic_counts = getattr(interest_profile, "click_topic_counts", {})
    if not click_topic_counts and unique_topics is not None and topic_counts is not None:
        click_topic_counts = {}
        for topic, count in zip(unique_topics, topic_counts):
            click_topic_counts[str(topic)] = int(count)

    # if click_topic_counts is None:
    #     click_topic_counts = {}

    user_id_str = str(user_id) if user_id != "unknown" else "unknown"
    profile_id_str = str(profile_id) if profile_id != "unknown" else "unknown"

    return {
        "user_id": user_id_str,
        "profile_id": profile_id_str,
        "beta": beta,
        "original_theta": 0.8,  # Default theta from MMRPConfig
        "modified_theta": theta,
        "click_count": click_topic_counts,
    }


def save_beta_to_file(beta_data: dict, output_dir: str = "outputs/mind-subset/nrms_topic_mmr_personalized"):
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        beta_file = output_path / "beta_values.json"

        existing_data = []
        if beta_file.exists():
            try:
                with open(beta_file, "r") as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
        existing_data.append(beta_data)
        with open(beta_file, "w") as f:
            json.dump(existing_data, f, indent=2)

    except Exception as e:
        logger.error(f"Error saving beta data: {e}")


def calculate_beta(
    interest_profile: InterestProfile, interacted_articles: CandidateSet
) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    click_history = getattr(interest_profile, "click_history", [])
    clicked_article_ids = [click.article_id for click in click_history]
    past_articles = interacted_articles.articles
    all_topics = []

    for article_id in clicked_article_ids:
        topics = find_topic(past_articles, article_id)
        if topics is not None:
            all_topics.extend(topics)

    if all_topics:
        unique_topics, counts = np.unique(all_topics, return_counts=True)
        topic_weights = counts / counts.sum()
        entropy = -np.sum(topic_weights * np.log(topic_weights + 1e-12))  # epsilon for log(0)
        # if len(unique_topics) > 1:
        #     entropy /= np.log(len(unique_topics))
        beta = entropy
        return beta, unique_topics, counts
    else:
        beta = 1.0
        return beta, None, None


class MMRPDiversifier(Component):
    config: MMRPConfig

    @torch_inference
    def __call__(
        self, candidate_articles: CandidateSet, interest_profile: InterestProfile, interacted_articles: CandidateSet
    ) -> RecommendationList:
        beta, unique_topics, topic_counts = calculate_beta(interest_profile, interacted_articles)
        theta = self.config.theta * beta

        beta_data = collect_beta_data(interest_profile, unique_topics, topic_counts, beta, theta)
        output_dir = os.environ.get("POPROX_OUTPUT_DIR", "outputs/mind-subset/nrms_topic_mmr_personalized")
        save_beta_to_file(beta_data, output_dir)

        if candidate_articles.scores is None:
            recommended = candidate_articles.articles
        else:
            similarity_matrix = compute_similarity_matrix(candidate_articles.embeddings)

            scores = torch.as_tensor(candidate_articles.scores).to(similarity_matrix.device)
            article_indices = mmrp_diversification(scores, similarity_matrix, theta=theta, topk=self.config.num_slots)
            recommended = [candidate_articles.articles[int(idx)] for idx in article_indices]

        return RecommendationList(articles=recommended)


def compute_similarity_matrix(todays_article_vectors: torch.Tensor) -> torch.Tensor:
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
