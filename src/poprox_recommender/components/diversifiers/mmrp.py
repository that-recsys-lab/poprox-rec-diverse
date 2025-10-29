# Add imports for entropy calculation
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from lenskit.pipeline import Component
from pydantic import BaseModel
from scipy.spatial import distance

from poprox_concepts.domain import CandidateSet, InterestProfile, RecommendationList
from poprox_recommender.pytorch.datachecks import assert_tensor_size
from poprox_recommender.pytorch.decorators import torch_inference

logger = logging.getLogger(__name__)


class MMRPConfig(BaseModel):
    theta: float = 0.8
    num_slots: int = 10


def collect_beta_data(
    interest_profile: InterestProfile,
    topic_interest_probability_profile: np.ndarray | None,
    topic_availability_probability_profile: np.ndarray | None,
    beta: float,
    original_theta: float,
    theta: float,
) -> dict:
    """Collect beta data for logging and analysis."""
    onboarding = getattr(interest_profile, "onboarding_topics", "unknown")
    profile_id = getattr(interest_profile, "profile_id", "unknown")
    profile_id_str = str(profile_id) if profile_id != "unknown" else "unknown"
    if onboarding and hasattr(onboarding[0], "account_id"):
        account_id_str = str(onboarding[0].account_id)
    else:
        account_id_str = "unknown"

    return {
        "account_id": account_id_str,
        "profile_id": profile_id_str,
        "beta": beta,
    }


def save_beta_to_file(beta_data: dict, output_dir: str = "outputs/poprox/nrms_topic_mmr_personalized"):
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


def compute_topic_dist(interest_profile):
    topic_preferences: dict[str, int] = defaultdict(int)
    for interest in interest_profile.onboarding_topics:
        topic_preferences[interest.entity_name] = interest.preference
    normalized_topic_prefs = {k: v / 14 for k, v in topic_preferences.items()}  # changed
    return normalized_topic_prefs


def calculate_beta(
    interest_profile: InterestProfile, interacted_articles: CandidateSet
) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    mu = 0.1265
    sigma = 0.0446

    topic_interest_dist = compute_topic_dist(interest_profile)
    topic_interest_probability_profile = list(
        topic_interest_dist.values()
    )  # some of them have 13 topics instead of 14 so padding with 0

    if len(topic_interest_probability_profile) < 14:
        topic_interest_probability_profile.extend([0] * (14 - len(topic_interest_probability_profile)))
    topic_interest_probability_profile = np.array(topic_interest_probability_profile)
    topic_availability_probability_profile = np.array([1 / 14] * 14)

    # high beta_raw = distance is high = topic pref is far from uniform = low diversity tolerace = low diversity
    # low beta_raw = distance is low = topic pref is close to uniform = high diversity tolerace = high diversity

    beta_raw = distance.jensenshannon(topic_interest_probability_profile, topic_availability_probability_profile)

    # high beta_raw = +ve beta
    # low beta_raw = -ve beta
    beta = ((beta_raw) - mu) / sigma  # z-score

    return beta, topic_interest_probability_profile, topic_availability_probability_profile


class MMRPDiversifier(Component):
    config: MMRPConfig

    @torch_inference
    def __call__(
        self, candidate_articles: CandidateSet, interest_profile: InterestProfile, interacted_articles: CandidateSet
    ) -> RecommendationList:
        beta, topic_interest_probability_profile, topic_availability_probability_profile = calculate_beta(
            interest_profile, interacted_articles
        )

        # high theta = low diversity
        # low theta = high diversity

        # what does theta >1 mean?
        theta = self.config.theta * (1 + (beta * 0.5))  # theta_p change
        theta = np.clip(theta, 0, 1)  # keeping theta between 0-1

        beta_data = collect_beta_data(
            interest_profile,
            topic_interest_probability_profile,
            topic_availability_probability_profile,
            beta,
            self.config.theta,
            theta,
        )
        output_dir = os.environ.get("POPROX_OUTPUT_DIR", "outputs/poprox/nrms_topic_mmr_personalized")
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
