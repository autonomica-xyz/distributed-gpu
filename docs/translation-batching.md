# Translation batching

DGPU exposes two independent bulk translation contracts:

- `translate-text-batch/libretranslate-batch-v1` sends an ordered,
  same-language batch to a warm pinned LibreTranslate service. It amortizes a
  lease, claim, HTTP request, artifact upload/download, and job state machine
  across up to 64 segments and 256,000 total characters. LibreTranslate 1.9.6
  processes the array internally in order; this primarily removes orchestration
  overhead rather than guaranteeing fused GPU inference.
- `translate-batch/argos-*-fast` invokes CTranslate2 `translate_batch` with
  token batching. It provides true inference batching but requires each
  immutable, reviewed language-pair model pack on the worker.

Both contracts are generic. Payloads contain ordered text and schema-bounded
language selections only; producer record IDs and database semantics never
cross the workload trust boundary.

A producer should group only items with the same source and target language,
cap batches by both item count and total text size, retain an ordered local
mapping, and independently hash/revalidate every item before applying results.
A failed batch may be retried as smaller batches or individual
`translate-segment` jobs without changing the DGPU worker contract.

Workers must advertise `libretranslate-batch` separately from
`libretranslate`. This prevents an old agent wheel from claiming a batch route
that it cannot execute during a rolling deployment.
