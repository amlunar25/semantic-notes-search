# Kafka Consumer Offsets

A Kafka consumer offset represents the position of a consumer within a
partition.

Offsets allow consumers to continue processing from a previously recorded
position after a restart.

When offsets are committed only after successful processing, the consumer can
reduce the risk of losing messages. However, processing may happen again after
a failure, so downstream operations should be idempotent.