# pyright: basic

from lenskit.pipeline import PipelineBuilder

from poprox_concepts import CandidateSet, InterestProfile
from poprox_concepts.domain import RecommendationList
from poprox_recommender.components.diversifiers.mmr import MMRConfig, MMRDiversifier
from poprox_recommender.components.embedders import NRMSArticleEmbedder
from poprox_recommender.components.embedders.article import NRMSArticleEmbedderConfig
from poprox_recommender.components.embedders.topic_wise_user import UserOnboardingConfig, UserOnboardingEmbedder
from poprox_recommender.components.embedders.user import NRMSUserEmbedder, NRMSUserEmbedderConfig
from poprox_recommender.components.joiners.score import ScoreFusion
from poprox_recommender.components.scorers.article import ArticleScorer
from poprox_recommender.paths import model_file_path

##TODO:
# allow weigths for the scores (1/-1)


def create_scored_candidates(fused_scores: CandidateSet, embedded_candidates: CandidateSet) -> CandidateSet:
    """Create a new CandidateSet with fused scores and embeddings from embedded candidates."""
    return CandidateSet(
        articles=fused_scores.articles, scores=fused_scores.scores, embeddings=embedded_candidates.embeddings
    )


def recommendation_list_to_candidate_set(recommendation_list):
    """Convert RecommendationList back to CandidateSet for TopkRanker."""
    return CandidateSet(articles=recommendation_list.articles)


def final_recommender(x):
    if x is None or not hasattr(x, "articles") or x.articles is None:
        return RecommendationList(articles=[])
    return x


def configure(builder: PipelineBuilder, num_slots: int, device: str):
    # standard practice is to put these calls in this order, to reuse logic
    # Define pipeline inputs
    i_candidates = builder.create_input("candidate", CandidateSet)
    i_clicked = builder.create_input("clicked", CandidateSet)
    i_profile = builder.create_input("profile", InterestProfile)

    # Embed candidate and clicked articles
    ae_config = NRMSArticleEmbedderConfig(
        model_path=model_file_path("nrms-mind/news_encoder.safetensors"), device=device
    )
    e_candidates = builder.add_component("candidate-embedder", NRMSArticleEmbedder, ae_config, article_set=i_candidates)
    e_clicked = builder.add_component(
        "history-NRMSArticleEmbedder", NRMSArticleEmbedder, ae_config, article_set=i_clicked
    )

    # Embed the user (historical clicks)
    ue_config = NRMSUserEmbedderConfig(model_path=model_file_path("nrms-mind/user_encoder.safetensors"), device=device)
    e_user = builder.add_component(
        "user-embedder",
        NRMSUserEmbedder,
        ue_config,
        candidate_articles=e_candidates,
        clicked_articles=e_clicked,
        interest_profile=i_profile,
    )

    # Embed the user (topics)
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

    # Score and rank articles (history)
    n_scorer = builder.add_component("scorer", ArticleScorer, candidate_articles=e_candidates, interest_profile=e_user)

    # Score and rank articles (topics)
    n_scorer2 = builder.add_component(
        "scorer2", ArticleScorer, candidate_articles=builder.node("candidate-embedder"), interest_profile=e_user2
    )

    # Combine click and topic scoring
    fusion = builder.add_component(
        "fusion", ScoreFusion, {"combiner": "avg"}, candidates1=n_scorer, candidates2=n_scorer2
    )

    scored_candidates = builder.add_component(
        "scored-candidates", create_scored_candidates, fused_scores=fusion, embedded_candidates=e_candidates
    )

    # Apply MMR diversification to the scored candidates
    def apply_mmr(candidate_articles, interest_profile):
        try:
            config = MMRConfig(num_slots=num_slots, theta=0.7)
            result = MMRDiversifier(config)(candidate_articles, interest_profile)
            if result is None or not hasattr(result, "articles") or result.articles is None:
                return RecommendationList(articles=[])
            return result
        except Exception:
            return RecommendationList(articles=[])

    n_reranker = builder.add_component(
        "reranker",
        apply_mmr,
        candidate_articles=scored_candidates,
        interest_profile=e_user,
    )

    # Final recommendation using the MMR diversified results directly, with a safety wrapper
    builder.add_component("recommender", final_recommender, candidate_articles=n_reranker)
