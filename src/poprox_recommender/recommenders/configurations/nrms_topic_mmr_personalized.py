# pyright: basic

from lenskit.pipeline import PipelineBuilder

from poprox_concepts import CandidateSet, InterestProfile
from poprox_recommender.components.diversifiers.mmr import MMRDiversifier
from poprox_recommender.components.embedders import NRMSArticleEmbedder, NRMSUserEmbedder
from poprox_recommender.components.embedders.article import EmbeddingCopier, NRMSArticleEmbedderConfig
from poprox_recommender.components.embedders.user import NRMSUserEmbedderConfig
from poprox_recommender.components.filters.topic import TopicFilter
from poprox_recommender.components.joiners.fill import FillRecs
from poprox_recommender.components.joiners.score import ScoreFusion
from poprox_recommender.components.rankers.topk import TopkRanker
from poprox_recommender.components.samplers.uniform import UniformSampler
from poprox_recommender.components.scorers.article import ArticleScorer
from poprox_recommender.paths import model_file_path


def configure(builder: PipelineBuilder, num_slots: int, device: str):
    # Create input nodes for the three main data sources
    # These will receive data from the API request
    i_candidates = builder.create_input("candidate", CandidateSet)
    i_clicked = builder.create_input("clicked", CandidateSet)
    i_profile = builder.create_input("profile", InterestProfile)

    # Configure the NRMS article embedding model
    # This converts article text (headline, body) into dense vector representations
    ae_config = NRMSArticleEmbedderConfig(
        model_path=model_file_path("nrms-mind/news_encoder.safetensors"), device=device
    )

    # Create article embeddings for candidate articles
    # Input: Raw article text → Output: Dense vectors (e.g., 400-dimensional)
    e_candidates = builder.add_component("candidate-embedder", NRMSArticleEmbedder, ae_config, article_set=i_candidates)

    # Create article embeddings for user's clicked articles (same model, different data)
    # Input: User's click history → Output: Dense vectors for personalization
    e_clicked = builder.add_component(
        "history-NRMSArticleEmbedder", NRMSArticleEmbedder, ae_config, article_set=i_clicked
    )

    # Configure the NRMS user embedding model: user representation from clicked articles and profile
    ue_config = NRMSUserEmbedderConfig(model_path=model_file_path("nrms-mind/user_encoder.safetensors"), device=device)

    # Create user embedding by combining clicked article embeddings with profile
    # Input: Clicked article embeddings + user profile → Output: User vector representation
    e_user = builder.add_component(
        "user-embedder",
        NRMSUserEmbedder,
        ue_config,
        candidate_articles=e_candidates,  # Needed for attention mechanism
        clicked_articles=e_clicked,
        interest_profile=i_profile,
    )

    # Score candidate articles based on similarity to user embedding
    # Input: Candidate embeddings + user embedding → Output: Relevance scores (0-1)
    n_scorer = builder.add_component("scorer", ArticleScorer, candidate_articles=e_candidates, interest_profile=e_user)

    # Create initial ranking based on relevance scores (not used in final output)
    # Input: Scored articles → Output: Top-K ranked articles
    _n_topk = builder.add_component("ranker", TopkRanker, {"num_slots": num_slots}, candidate_articles=n_scorer)

    # Personalized scoring: create topic-aligned candidates and score them
    n_topic_filter = builder.add_component(
        "topic-filter", TopicFilter, candidate=i_candidates, interest_profile=i_profile
    )

    # Embed the topic-filtered candidates so they can be scored
    e_topic_filtered = builder.add_component(
        "topic-filtered-embedder", NRMSArticleEmbedder, ae_config, article_set=n_topic_filter
    )

    n_topic_scorer = builder.add_component(
        "topic-scorer", ArticleScorer, candidate_articles=e_topic_filtered, interest_profile=e_user
    )

    n_personalized = builder.add_component(
        "personalized-fusion",
        ScoreFusion,
        {
            "combiner": "weighted_avg",
            "weights": [0.7, 0.3],  # 70% click history, 30% topic alignment
        },
        candidates1=n_scorer,
        candidates2=n_topic_scorer,
    )

    # Copy embeddings to the fused candidate set for MMR diversification
    n_personalized_with_embeddings = builder.add_component(
        "embedding-copier", EmbeddingCopier, candidate_set=e_candidates, selected_set=n_personalized
    )

    # Apply MMR (Maximal Marginal Relevance) diversification
    # Balances relevance (70%) with diversity (30%) to prevent echo chambers
    # Input: Scored articles + user embedding → Output: Diversified article list
    n_reranker = builder.add_component(
        "reranker",
        MMRDiversifier,
        {"num_slots": num_slots, "theta": 0.7},  # θ=0.7: 70% relevance, 30% diversity
        candidate_articles=n_personalized_with_embeddings,  # Use personalized scores instead of base scores
        interest_profile=e_user,  # User embedding for diversity calculation
    )

    # Input: Topic-filtered candidates + all candidates → Output: Sampled candidate set
    # Creates a UniformSampler that combines two candidate sets
    # candidates1: Topic-filtered articles (from existing topic filter)
    # candidates2: Original full candidate set (fallback)
    # Samples uniformly from both sets to ensure coverage
    n_sampler = builder.add_component("sampler", UniformSampler, candidates1=n_topic_filter, candidates2=i_candidates)

    # Input: MMR diversified results + topic-sampled results → Output: Final recommendation list
    # Creates the final FillRecs (Fill Recommendations) component
    # recs1: MMR-diversified results (primary recommendations)
    # recs2: Topic-filtered + sampled results (fallback recommendations)
    # Combines both to fill exactly num_slots recommendations
    builder.add_component("recommender", FillRecs, {"num_slots": num_slots}, recs1=n_reranker, recs2=n_sampler)
