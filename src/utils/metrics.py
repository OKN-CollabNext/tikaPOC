from prometheus_client import Counter, Histogram, start_http_server
import time
import os

# Start the Prometheus metrics server (for example, on port 8000)
METRICS_PORT = int(os.getenv("PROMETHEUS_PORT", "8000"))
start_http_server(METRICS_PORT)

# Define some metrics
QUERY_COUNTER = Counter('topic_agent_queries_total', 'Total number of queries processed by TopicAgent')
QUERY_LATENCY = Histogram('topic_agent_query_latency_seconds', 'Latency of TopicAgent query processing')
ERROR_COUNTER = Counter('topic_agent_errors_total', 'Total errors in TopicAgent operations')

def record_query(func):
    """Decorator to record query processing metrics."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        QUERY_COUNTER.inc()
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            ERROR_COUNTER.inc()
            raise e
        finally:
            duration = time.time() - start_time
            QUERY_LATENCY.observe(duration)
    return wrapper
