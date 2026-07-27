# Spark Streaming Checkpoints

Spark Structured Streaming checkpoints store query progress and state
information.

A checkpoint allows a streaming query to recover after the process restarts.
It may contain information about processed source offsets, stateful operators,
and commit progress.

A streaming application should use a stable checkpoint location and should not
share the same checkpoint directory across unrelated queries.