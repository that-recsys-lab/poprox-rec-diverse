# Simulates a request to the recommender without requiring Serverless
import warnings

from poprox_concepts.api.recommendations.v4 import RecommendationRequestV4, RecommendationResponseV4
from poprox_recommender.api.main import root
from poprox_recommender.paths import project_root
from poprox_recommender.topics import extract_general_topics

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    with open(project_root() / "tests/request_data/onboarding.json", "r") as req_file:
        raw_json = req_file.read()
        req = RecommendationRequestV4.model_validate_json(raw_json)

    event_nrms_topics_fb_filter = {
        "body": raw_json,
        "queryStringParameters": {"pipeline": "nrms_topics_fb_filter"},
        "isBase64Encoded": False,
    }
    event_nrms_mmr = {
        "body": raw_json,
        "queryStringParameters": {"pipeline": "nrms_topic_mmr"},
        "isBase64Encoded": False,
    }
    event_nrms_mmr_personalized = {
        "body": raw_json,
        "queryStringParameters": {"pipeline": "nrms_topic_mmr_personalized"},
        "isBase64Encoded": False,
    }

    response_nrms_topics_fb_filter = root(req.model_dump(), pipeline="nrms_topics_fb_filter")
    response_nrms_topics_fb_filter = RecommendationResponseV4.model_validate(response_nrms_topics_fb_filter)

    response_nrms_mmr = root(req.model_dump(), pipeline="nrms_topic_mmr")
    response_nrms_mmr = RecommendationResponseV4.model_validate(response_nrms_mmr)

    response_nrms_mmr_personalized = root(req.model_dump(), pipeline="nrms_topic_mmr_personalized")
    response_nrms_mmr_personalized = RecommendationResponseV4.model_validate(response_nrms_mmr_personalized)

    print("\n")
    print(f"{event_nrms_topics_fb_filter['queryStringParameters']['pipeline']}")

    for idx, article in enumerate(
        [impression.article for impression in response_nrms_topics_fb_filter.recommendations.impressions]
    ):
        article_topics = extract_general_topics(article)
        print(f"{idx + 1}. {article.headline} {article_topics}")

    print("\n")
    print(f"{event_nrms_mmr['queryStringParameters']['pipeline']}")

    for idx, article in enumerate([impression.article for impression in response_nrms_mmr.recommendations.impressions]):
        article_topics = extract_general_topics(article)
        print(f"{idx + 1}. {article.headline} {article_topics}")

    print("\n")
    print(f"{event_nrms_mmr_personalized['queryStringParameters']['pipeline']}")

    for idx, article in enumerate(
        [impression.article for impression in response_nrms_mmr_personalized.recommendations.impressions]
    ):
        article_topics = extract_general_topics(article)
        print(f"{idx + 1}. {article.headline} {article_topics}")
