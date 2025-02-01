import pytest
from services.topic_agent import TopicAgent

@pytest.fixture
def agent():
    return TopicAgent()

def test_process_query_returns_results(agent):
    user_query = "plastic recycling innovations"
    results = agent.process_query(user_query)
    # Check that results are returned in the expected format (a list of dicts)
    assert isinstance(results, list)
    if results:
        assert "display_name" in results[0]
        assert "description" in results[0]

def test_exclude_topics(agent):
    initial_excluded = len(agent.get_state_summary().get("excluded_topics_count", 0))
    agent.exclude_topics(["topic1", "topic2"])
    new_excluded = agent.get_state_summary()["excluded_topics_count"]
    assert new_excluded >= initial_excluded + 2
