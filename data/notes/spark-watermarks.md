# Spark Structured Streaming Watermarks

A watermark defines how long Spark should wait for late-arriving events.

For example, a watermark of 10 minutes means Spark can clean up state for
windows that are sufficiently older than the latest observed event time.

Watermarks are commonly used with event-time aggregations and deduplication.
They help prevent streaming state from growing without limits.

A watermark does not automatically discard every event that is more than
10 minutes late. The exact behavior depends on the aggregation, window
boundary, trigger, and current maximum observed event time.