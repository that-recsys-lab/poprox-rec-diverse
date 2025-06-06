from collections import defaultdict

from lenskit.pipeline import Component
from pydantic import BaseModel

from poprox_concepts.domain import CandidateSet


class ScoreFusionConfig(BaseModel):
    combiner: str = "sum"


class ScoreFusion(Component):
    config: ScoreFusionConfig

    def __call__(self, candidates1: CandidateSet, candidates2: CandidateSet) -> CandidateSet:
        combined_score = defaultdict(float)
        combined_article = {}

        if candidates1.scores is not None:
            for article, score in zip(candidates1.articles, candidates1.scores):
                article_id = article.article_id
                combined_score[article_id] += score
                combined_article[article_id] = article

        if candidates2.scores is not None:
            for article, score in zip(candidates2.articles, candidates2.scores):
                article_id = article.article_id
                combined_score[article_id] += score
                combined_article[article_id] = article

        if self.config.combiner == "avg":
            denominator = 2
        else:
            denominator = 1

        merged_scores = []
        merged_articles = []

        for key, score in combined_score.items():
            merged_articles.append(combined_article[key])
            merged_scores.append(score / denominator)

        return CandidateSet(articles=merged_articles, scores=merged_scores)
