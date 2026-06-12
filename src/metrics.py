from prometheus_client import Counter, Gauge, Histogram

connections_current = Gauge(
    "asr_connections_current",
    "Number of active WebSocket connections",
)

processing_latency = Histogram(
    "asr_processing_latency_ms",
    "ASR processing latency in milliseconds",
    buckets=[50, 100, 200, 500, 1000, 2000, 5000, 10000],
)

queue_depth = Gauge(
    "asr_queue_depth",
    "Number of pending tasks in the ITN queue",
)

segments_total = Counter(
    "asr_segments_total",
    "Total number of processed speech segments",
)

errors_total = Counter(
    "asr_errors_total",
    "Total number of processing errors",
    ["error_type"],
)
