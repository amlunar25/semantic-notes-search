# Snowflake Clustering

Snowflake stores table data in immutable micro-partitions.

Clustering helps Snowflake prune micro-partitions when queries filter on
relevant columns. Automatic clustering can maintain clustering over time,
but it consumes credits.

Clustering keys should be selected based on common selective filters and
large-table query patterns rather than added to every table.