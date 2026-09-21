# Agentic benchmark task: bounded model-gate retries

The lifecycle controller makes OpenAI-compatible model-gate requests. A short
HTTP 503, HTTP 429, connection error, or timeout can currently fail a provider
wake even when the endpoint becomes ready on the next request.

Implement bounded retry behavior for `Controller._request_json`.

Requirements:

1. Read the existing controller, configuration validation, tests, and user
   documentation before you edit code.
2. Retry only transient failures: HTTP 408, 429, 500, 502, 503, and 504;
   `urllib.error.URLError`; and `TimeoutError`.
3. Do not retry other HTTP 4xx responses or invalid JSON returned by a
   successful HTTP response.
4. Use two optional top-level configuration fields:
   `model_gate_attempts` and `model_gate_backoff_seconds`. Defaults are 3
   attempts and 1.0 seconds. Attempts must be a positive integer, not a bool.
   Backoff must be a non-negative int or float, not a bool.
5. Use fixed backoff between attempts. Tests must be able to patch
   `time.sleep`; do not add real delays to tests.
6. Preserve the request body and authorization behavior on each attempt.
7. After the final failure, raise `LifecycleError` with a useful message that
   includes the request URL and attempt count. Preserve exception chaining.
8. Add focused unit tests. Update `config/example.json` and the most relevant
   user documentation.
9. Run the focused tests and the complete Python test suite. Do not stop until
   tests pass.

Do not change provider fallback semantics. Do not add a new dependency.
