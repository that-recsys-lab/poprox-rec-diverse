# pyright: basic

import torch
from lenskit.pipeline import PipelineBuilder

from poprox_concepts import CandidateSet, InterestProfile
from poprox_concepts.domain import RecommendationList
from poprox_recommender.components.diversifiers.mmr import MMRDiversifier
from poprox_recommender.components.embedders import NRMSArticleEmbedder
from poprox_recommender.components.embedders.article import NRMSArticleEmbedderConfig
from poprox_recommender.components.embedders.user import NRMSUserEmbedder, NRMSUserEmbedderConfig
from poprox_recommender.components.embedders.user_topic_prefs import UserOnboardingConfig, UserOnboardingEmbedder
from poprox_recommender.components.joiners.score import ScoreFusion
from poprox_recommender.components.rankers.topk import TopkRanker
from poprox_recommender.components.scorers.article import ArticleScorer
from poprox_recommender.paths import model_file_path


def to_recommendation_list(x):
    if x is None or not hasattr(x, "articles") or x.articles is None:
        return RecommendationList(articles=[])
    return RecommendationList(articles=x.articles)


def add_embeddings(fused: CandidateSet, embedded: CandidateSet) -> CandidateSet:
    # Copy embeddings from embedded candidates to fused set
    emb_map = {a.article_id: emb for a, emb in zip(embedded.articles, embedded.embeddings)}
    embeddings = torch.stack([emb_map[a.article_id] for a in fused.articles])
    return CandidateSet(articles=fused.articles, scores=fused.scores, embeddings=embeddings)


def configure(builder: PipelineBuilder, num_slots: int, device: str):
    i_candidates = builder.create_input("candidate", CandidateSet)
    i_clicked = builder.create_input("clicked", CandidateSet)
    i_profile = builder.create_input("profile", InterestProfile)

    ae_config = NRMSArticleEmbedderConfig(
        model_path=model_file_path("nrms-mind/news_encoder.safetensors"), device=device
    )
    e_candidates = builder.add_component("candidate-embedder", NRMSArticleEmbedder, ae_config, article_set=i_candidates)
    e_clicked = builder.add_component(
        "history-NRMSArticleEmbedder", NRMSArticleEmbedder, ae_config, article_set=i_clicked
    )

    ue_config = NRMSUserEmbedderConfig(model_path=model_file_path("nrms-mind/user_encoder.safetensors"), device=device)
    e_user = builder.add_component(
        "user-embedder",
        NRMSUserEmbedder,
        ue_config,
        candidate_articles=e_candidates,
        clicked_articles=e_clicked,
        interest_profile=i_profile,
    )
    ue_config2 = UserOnboardingConfig(
        model_path=model_file_path("nrms-mind/user_encoder.safetensors"),
        device=device,
        embedding_source="static",
        topic_embedding="nrms",
    )
    e_user2 = builder.add_component(
        "user-embedder2",
        UserOnboardingEmbedder,
        ue_config2,
        candidate_articles=e_candidates,
        clicked_articles=e_clicked,
        interest_profile=i_profile,
    )

    n_scorer = builder.add_component("scorer", ArticleScorer, candidate_articles=e_candidates, interest_profile=e_user)
    n_scorer2 = builder.add_component(
        "scorer2", ArticleScorer, candidate_articles=e_candidates, interest_profile=e_user2
    )
    fusion = builder.add_component(
        "fusion", ScoreFusion, {"combiner": "avg"}, candidates1=n_scorer, candidates2=n_scorer2
    )

    fused_with_embeddings = builder.add_component(
        "fused_with_embeddings", add_embeddings, fused=fusion, embedded=e_candidates
    )

    builder.add_component("ranker", TopkRanker, {"num_slots": num_slots}, candidate_articles=fused_with_embeddings)

    builder.add_component(
        "reranker",
        MMRDiversifier,
        {"num_slots": num_slots, "theta": 0.7},
        candidate_articles=fused_with_embeddings,
        interest_profile=e_user,
    )

    # The final recommender component that the API expects
    builder.add_component(
        "recommender",
        MMRDiversifier,
        {"num_slots": num_slots, "theta": 0.7},
        candidate_articles=fused_with_embeddings,
        interest_profile=e_user,
    )
