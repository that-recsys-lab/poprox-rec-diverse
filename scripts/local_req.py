# Simulates a request to the recommender without requiring Serverless
import warnings

from poprox_concepts.api.recommendations.v2 import RecommendationRequestV2, RecommendationResponseV2
from poprox_recommender.api.main import root
from poprox_recommender.paths import project_root
from poprox_recommender.topics import extract_general_topics

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    with open(project_root() / "tests/request_data/onboarding.json", "r") as req_file:
        raw_json = req_file.read()
        req = RecommendationRequestV2.model_validate_json(raw_json)

    event_topic_score = {
        "body": raw_json,
        "queryStringParameters": {"pipeline": "nrms_topic_scores"},
        "isBase64Encoded": False,
    }
    event_nrms_mmr = {
        "body": raw_json,
        "queryStringParameters": {"pipeline": "nrms_topic_mmr"},
        "isBase64Encoded": False,
    }
    # event_mmr = {
    #     "body": raw_json,
    #     "queryStringParameters": {"pipeline": "mmr"},
    #     "isBase64Encoded": False,
    # }
    event_nrms_mmr_personalized = {
        "body": raw_json,
        "queryStringParameters": {"pipeline": "nrms_topic_mmr_personalized"},
        "isBase64Encoded": False,
    }
    event_mmrp = {
        "body": raw_json,
        "queryStringParameters": {"pipeline": "mmrp"},
        "isBase64Encoded": False,
    }

    response_topic_score = root(req.model_dump(), pipeline="nrms_topic_scores")
    response_topic_score = RecommendationResponseV2.model_validate(response_topic_score)

    response_nrms_mmr = root(req.model_dump(), pipeline="nrms_topic_mmr")
    response_nrms_mmr = RecommendationResponseV2.model_validate(response_nrms_mmr)

    # response_mmr = root(req.model_dump(), pipeline="mmr")
    # response_mmr = RecommendationResponseV2.model_validate(response_mmr)

    response_nrms_mmr_personalized = root(req.model_dump(), pipeline="nrms_topic_mmr_personalized")
    response_nrms_mmr_personalized = RecommendationResponseV2.model_validate(response_nrms_mmr_personalized)

    response_mmrp = root(req.model_dump(), pipeline="mmrp")
    response_mmrp = RecommendationResponseV2.model_validate(response_mmrp)

    print("\n")
    print(f"{event_topic_score['queryStringParameters']['pipeline']}")

    for idx, article in enumerate(response_topic_score.recommendations.articles):
        article_topics = extract_general_topics(article)
        print(f"{idx + 1}. {article.headline} {article_topics}")

    print("\n")
    print(f"{event_nrms_mmr['queryStringParameters']['pipeline']}")

    for idx, article in enumerate(response_nrms_mmr.recommendations.articles):
        article_topics = extract_general_topics(article)
        print(f"{idx + 1}. {article.headline} {article_topics}")

    # print("\n")
    # print(f"{event_mmr['queryStringParameters']['pipeline']}")

    # for idx, article in enumerate(response_mmr.recommendations.articles):
    #     article_topics = extract_general_topics(article)
    #     print(f"{idx + 1}. {article.headline} {article_topics}")

    print("\n")
    print(f"{event_nrms_mmr_personalized['queryStringParameters']['pipeline']}")

    for idx, article in enumerate(response_nrms_mmr_personalized.recommendations.articles):
        article_topics = extract_general_topics(article)
        print(f"{idx + 1}. {article.headline} {article_topics}")

    print("\n")
    print(f"{event_mmrp['queryStringParameters']['pipeline']}")

    for idx, article in enumerate(response_mmrp.recommendations.articles):
        article_topics = extract_general_topics(article)
        print(f"{idx + 1}. {article.headline} {article_topics}")
