# Airflow Task Retries

Airflow tasks can be configured with retries and retry delays.

Retries are useful for temporary failures such as unavailable APIs, transient
database connections, network interruptions, and rate limiting.

Tasks should normally be idempotent so that running the same task more than
once does not create duplicate data or inconsistent state.

Exponential backoff can reduce pressure on external systems during outages.