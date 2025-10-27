"""
Support for loading POPROX data for evaluation.
"""

# pyright: basic
from __future__ import annotations

import json
import logging
from typing import Generator
from uuid import UUID

import numpy as np
import pandas as pd

from poprox_concepts import AccountInterest, Article, Click, Entity, InterestProfile, Mention
from poprox_concepts.api.recommendations import RecommendationRequestV2
from poprox_concepts.domain import CandidateSet
from poprox_recommender.data.eval import EvalData
from poprox_recommender.paths import project_root

logger = logging.getLogger(__name__)
TEST_REC_COUNT = 10


class PoproxData(EvalData):
    def __init__(self, archive: str = "POPROX"):
        (
            articles_df,
            mentions_df,
            newsletters_df,
            clicks_df,
            clicked_articles_df,
            clicked_mentions_df,
            interests_df,
        ) = load_poprox_frames(archive)

        rng = np.random.default_rng()  # get numpy Generator
        perm = np.arange(len(newsletters_df))
        rng.shuffle(perm)
        self.newsletters_df = newsletters_df.iloc[perm]

        # index data frames for quick lookup of users & articles
        self.mentions_df = mentions_df
        self.articles_df = articles_df.set_index("article_id", drop=False)
        if not self.articles_df.index.unique:
            logger.warning("article data has non-unique index")

        self.clicks_df = clicks_df
        self.clicked_mentions_df = clicked_mentions_df
        self.clicked_articles_df = clicked_articles_df.set_index("article_id", drop=False)
        if not self.clicked_articles_df.index.unique:
            logger.warning("clicked article data has non-unique index")

        self.interests_df = interests_df
        self.name = "POPROX"

    @property
    def n_profiles(self) -> int:
        return len(self.newsletters_df["newsletter_id"].unique())

    @property
    def n_articles(self) -> int:
        return self.articles_df.shape[0]

    def profile_truth(self, newsletter_id: UUID) -> pd.DataFrame | None:
        # Create one row per clicked article with this newsletter_id
        # Returned dataframe must have an "item_id" column containing the clicked article ids
        # and the "item_id" column must be the index of the dataframe
        # There must also be a "rating" columns
        newsletter_clicks = self.clicks_df[self.clicks_df["newsletter_id"] == str(newsletter_id)]
        clicked_items = newsletter_clicks["article_id"].unique()
        return pd.DataFrame({"item_id": clicked_items, "rating": [1.0] * len(clicked_items)}).set_index("item_id")

    def iter_profiles(self) -> Generator[RecommendationRequestV2]:
        newsletters_df = self.newsletters_df.copy()
        newsletters_df["account_id_alias"] = newsletters_df["profile_id"].astype(str)

        unique_accounts_df = newsletters_df.drop_duplicates(subset=["account_id_alias"]).reset_index(drop=True)
        logger.info(f"Found {len(unique_accounts_df)} unique accounts for recommendation generation")

        for _, row in unique_accounts_df.iterrows():
            newsletter_id = row["newsletter_id"]
            profile_id = row["profile_id"]
            newsletter_created_at = row["created_at"]

            # Filter clicks to those before the newsletter
            profile_clicks_df = self.clicks_df.loc[self.clicks_df["profile_id"] == profile_id]
            # TODO: Change `timestamp` to `clicked_at` in the export
            filtered_clicks_df = profile_clicks_df[profile_clicks_df["clicked_at"] < newsletter_created_at]

            # Create Article and Click objects from dataframe rows
            clicks = []
            past_articles = []
            for article_row in filtered_clicks_df.itertuples():
                article = self.lookup_clicked_article(article_row.article_id)
                if article:
                    past_articles.append(article)

                    clicks.append(
                        Click(
                            article_id=article_row.article_id,
                            newsletter_id=article_row.newsletter_id,
                            timestamp=article_row.clicked_at,
                        )
                    )

            interests = self.interests_df.loc[self.interests_df["account_id"] == profile_id]
            topics = [
                AccountInterest(
                    account_id=profile_id,
                    entity_id=interest.entity_id,
                    entity_name=interest.entity_name,
                    preference=interest.preference,
                )
                for interest in interests.itertuples()
            ]

            profile = InterestProfile(
                profile_id=newsletter_id,
                click_history=clicks,
                onboarding_topics=topics,
            )
            profile.account_id_alias = str(profile_id)
            candidate_articles = []
            newsletter_date = newsletter_created_at.date()

            for article_row in self.articles_df[
                self.articles_df["created_at"].apply(lambda c: c.date()) == newsletter_date
            ].itertuples():
                candidate_articles.append(self.lookup_candidate_article(article_row.article_id))

            yield RecommendationRequestV2(
                candidates=CandidateSet(articles=candidate_articles),
                interacted=CandidateSet(articles=past_articles),
                interest_profile=profile,
                num_recs=TEST_REC_COUNT,
            )

    def lookup_candidate_article(self, article_id: UUID):
        article_row = self.articles_df.loc[str(article_id)]
        mention_rows = self.mentions_df[self.mentions_df["article_id"] == article_row.article_id]
        return self.convert_row_to_article(article_row, mention_rows)

    def lookup_article(self, *, id: str | None = None, uuid: UUID | None = None):
        """
        Lookup article by either id or uuid. This method matches the interface expected by the evaluation metrics.
        For POPROX data, we only support UUID lookups, so the id parameter is ignored.
        """
        if uuid is None:
            if id:
                # Convert string id to UUID if needed - for POPROX we expect UUIDs
                try:
                    uuid = UUID(id)
                except ValueError:
                    raise ValueError(f"Invalid UUID format: {id}")
            else:
                raise ValueError("must provide one of uuid, id")

        # Use the existing lookup_candidate_article method
        return self.lookup_candidate_article(uuid)

    def lookup_clicked_article(self, article_id: UUID):
        try:
            article_row = self.clicked_articles_df.loc[str(article_id)]
            mention_rows = self.clicked_mentions_df[self.clicked_mentions_df["article_id"] == article_row.article_id]
            return self.convert_row_to_article(article_row, mention_rows)
        except Exception as _:
            print(f"Did not find the clicked article with id {str(article_id)}")
            return None

    def convert_row_to_article(self, article_row, mention_rows):
        mentions = [
            Mention(
                mention_id=row.mention_id,
                article_id=row.article_id,
                source=row.source,
                relevance=row.relevance,
                entity=Entity(**json.loads(row.entity)) if row.entity else None,
            )
            for row in mention_rows.itertuples()
        ]

        return Article(
            article_id=article_row.article_id,
            headline=article_row.headline,
            subhead=article_row.subhead,
            body=article_row.body,
            published_at=article_row.published_at,
            mentions=mentions,
            source="AP",
            external_id="",
            raw_data=json.loads(article_row.raw_data) if article_row.raw_data else None,
        )


def load_poprox_frames(archive: str = "POPROX"):
    data = project_root() / "data"
    logger.info("loading POPROX data from %s", archive)

    newsletters_df = pd.read_parquet(data / "POPROX" / "newsletters.parquet")

    articles_df = pd.read_parquet(data / "POPROX" / "articles.parquet")
    mentions_df = pd.read_parquet(data / "POPROX" / "mentions.parquet")

    clicks_df = pd.read_parquet(data / "POPROX" / "clicks.parquet")
    clicked_articles_df = pd.read_parquet(data / "POPROX" / "clicked" / "articles.parquet")
    clicked_mentions_df = pd.read_parquet(data / "POPROX" / "clicked" / "mentions.parquet")

    interests_df = pd.read_parquet(data / "POPROX" / "interests.parquet")

    return (
        articles_df,
        mentions_df,
        newsletters_df,
        clicks_df,
        clicked_articles_df,
        clicked_mentions_df,
        interests_df,
    )
