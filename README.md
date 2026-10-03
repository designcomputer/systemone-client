# systemone-client

Python client for the Ollama **System One** decision endpoint (`POST /v1/systemone`).

System One models classify text, answer yes/no questions, or score text against a
rubric in a single forward pass — no generation, no streaming, one JSON response.
Typical latency on this server: **~70–500 ms** depending on model and state size.

## Requirements

- Python 3.10+ (uses `X | Y` type syntax)
- `requests`
- An Ollama server with the **decision** capability (v0.35.0+; Clef image judging needs v0.35.1+)

```bash
pip install requests
```

## Setup

The server base URL defaults to `http://localhost:11434`. Override it with the
`SYSTEMONE_BASE_URL` environment variable or the `base_url` argument:

```bash
export SYSTEMONE_BASE_URL=http://localhost:11434   # optional
```

## Quickstart

```bash
python example.py
```

```python
from systemone import systemone

response = systemone(
    model="tev1:4b",
    state="Hello World",
    questions={
        "says_hello": {
            "type": "noul",
            "instructions": "Does the state text contain a greeting?",
            "criteria": {
                "true": "The state text contains a greeting.",
                "false": "The state text does not contain a greeting.",
            },
        },
    },
)
print(response.answers["says_hello"].noul)  # ~0.99
```

## Question types

One request can carry any mix of the three types. Answers are **not** passed to
later questions — each is scored independently against the shared state.

| Type | `criteria` shape | Answer fields |
|------|------------------|---------------|
| `noul` | `{"true": ..., "false": ...}` | `.noul` — probability 0–1 that the `true` criterion holds |
| `choice` | `{"label": description, ...}` | `.choice`, `.probabilities` (per label), `.confidence` |
| `score` | `["level 0", "level 1", ...]` | `.score` — value on the 0–(N−1) scale (N = number of rubric levels) |

### Example: all three in one request

```python
response = systemone(
    model="tev1:4b",
    state={"ticket": "I was charged twice. Please refund the extra payment. It's urgent!"},
    questions={
        "refund": {
            "type": "noul",
            "instructions": "Is the customer requesting a refund?",
            "criteria": {"true": "The customer requests a refund.",
                         "false": "No refund is requested."},
        },
        "label": {
            "type": "choice",
            "instructions": "Which label fits this ticket?",
            "criteria": {"billing": "Payments and refunds",
                         "bug": "Software errors",
                         "account": "Login and account access"},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgently does this ticket need a response?",
            "criteria": ["Routine: no time pressure",
                         "Soon: a customer is inconvenienced",
                         "Immediate: a critical service is unavailable"],
        },
    },
)

response.answers["refund"].noul        # 0.9916
response.answers["label"].choice       # "billing"
response.answers["label"].confidence   # 0.8861
response.answers["urgency"].score      # 1.6415  (0–2 scale)
response.usage                         # {'input_tokens': 805, 'output_tokens': 4}
```

## API reference

### `systemone(model, state, questions, *, images=None, keep_alive=None, base_url=None, timeout=120.0) -> SystemOneResponse`

| Parameter | Description |
|-----------|-------------|
| `model` | System One model tag, e.g. `'tev1:4b'`, `'nimble:9b'`, `'clef:27b'` |
| `state` | Nonempty string, or a `dict`/`list` (serialized to JSON text). Never interpreted as chat messages |
| `questions` | Named map of questions; each has `type`, `instructions`, `criteria` |
| `images` | Optional list of base64 PNG/JPEG/WebP strings, shared by all questions in request order. **Clef / Clef Flash only** |
| `keep_alive` | How long the model stays loaded after the request: duration string (`'5m'`) or seconds. `0` unloads immediately; a **negative value keeps it loaded forever**. Default: server setting (usually 5m) |
| `base_url` | Override the server base URL |
| `timeout` | Request timeout in seconds (default 120) |

### `systemone_fallback(model, state, questions, *, fallback_model=None, confidence_threshold=0.6, noul_gray_band=(0.3, 0.7), client=None, **kwargs) -> SystemOneResponse`

Optional confidence-based escalation. **With `fallback_model=None` it behaves
exactly like `systemone()`.** With a fallback model configured:

1. The primary model answers first.
2. If any `choice` confidence is below `confidence_threshold`, or any `noul`
   score falls inside `noul_gray_band`, the whole request re-runs on the
   fallback model.
3. The returned response has `.escalated = True` when the fallback answered.

| Parameter | Description |
|-----------|-------------|
| `fallback_model` | Model to re-run the request on when the primary is uncertain. `None` (default) disables escalation entirely |
| `confidence_threshold` | Escalate `choice` answers with confidence below this (default 0.6) |
| `noul_gray_band` | Escalate `noul` scores inside this band (default (0.3, 0.7)) |
| `client` | Transport to call. Any callable with the `systemone()` signature returning a `SystemOneResponse`. Defaults to this module's `systemone()` |

```python
from systemone import systemone_fallback

# Fast first pass on the small model; escalate only when uncertain.
response = systemone_fallback(
    "tev1:0.8b", state, questions,
    fallback_model="clef:27b",
    confidence_threshold=0.6,
)
if response.escalated:
    print("adjudicated by", response.model)  # clef:27b
```

#### Swapping the transport

The escalation policy is transport-agnostic: `client` can be any function
matching the `systemone()` signature. When the official `ollama` Python
package gains System One support, point the fallback at it without changing
the policy:

```python
from ollama import systemone as official_systemone  # future

response = systemone_fallback(
    "tev1:0.8b", state, questions,
    fallback_model="clef:27b",
    client=official_systemone,  # drop-in swap
)
```

The client must return an object with `.answers` (values exposing the
`noul` / `choice` + `confidence` fields), `.model`, `.usage`, and a settable
`.escalated` attribute — i.e., a `SystemOneResponse` or a thin adapter.
This keeps the routing/escalation logic — the part that is userland policy,
not API plumbing — reusable across client generations.

### Response objects

```
SystemOneResponse
├── model: str                 # model that actually answered
├── answers: dict[str, Answer] # keyed by question name
├── usage: dict[str, int]      # {'input_tokens': ..., 'output_tokens': ...}
└── escalated: bool            # True only from systemone_fallback

NoulAnswer(type, noul)
ChoiceAnswer(type, choice, probabilities, confidence)
ScoreAnswer(type, score)
```

### Errors

| Exception | When |
|-----------|------|
| `SystemOneError` (has `.status_code`, `.message`) | Non-200 from the server: 400 bad request, 404 model not found, 413 body too large, 500 server error (e.g. `non-finite logit`) |
| `ValueError` | Empty state, or request body over the 64 KiB (32 MiB with images) server limit — raised **before** the request is sent |
| `TypeError` | `state` is not a `str`, `dict`, or `list` |

## Server limits (from the Ollama API docs)

- Single JSON response — no streaming, video, tools, or generation controls
- Body ≤ **64 KiB** without images; ≤ **32 MiB** with images (base64 + JSON)
- Input must fit the loaded context window; it is **never truncated**
- Nimble and Tev require two extra token positions for scoring
- Only one large model fits in VRAM at a time — the server evicts the previous
  model, so the first call after a switch pays a cold load (~1.4–11 s)

## Models on the local server and measured performance

Benchmark: noul greeting question, short state (143 input tokens) and long
state (911 input tokens), steady-state mean after warmup.

| Model | Quant | Cold load | Short | Long | Positive | Negative | Pos−Neg margin |
|-------|-------|-----------|-------|------|----------|----------|----------------|
| `clef:27b` | Q4_K_M | ~7.3 s | ~178 ms | ~478 ms | 0.965 | 0.006 | 0.96 |
| `nimble:9b` | Q8_0 | ~4.3 s | ~79 ms | ~108 ms | 0.997 | 0.130 | 0.87 |
| `tev1:4b` | Q8_0 | ~2.2 s | ~68 ms | ~102 ms | 0.996 | 0.031 | 0.96 |
| `tev1:0.8b` | Q8_0 | ~1.4 s | ~67 ms | ~80 ms | 0.688 | 0.290 | 0.40 |
| `clef-flash:9b` | Q8_0 | — | ❌ | ❌ | — | — | broken (see below) |

### Multi-type accuracy (ticket state, all three question types)

| Model | noul (refund) | choice (label, confidence) | score (urgency, 0–2) |
|-------|---------------|------------------------------|------------------------|
| `clef:27b` | 0.991 | billing (0.931) | 1.447 |
| `nimble:9b` | 0.998 | billing (0.870) | 1.067 |
| `tev1:4b` | 0.992 | billing (0.886) | 1.642 |
| `tev1:0.8b` | 0.982 | ⚠️ bug (0.431) — wrong, low confidence | 1.066 |

### Model notes

- **`tev1:4b`** — the sweet spot: fastest practical inference with a sharp
  decision margin (0.96). Default recommendation.
- **`tev1:0.8b`** — marginally faster than the 4b (~1 ms; fixed overhead
  dominates at this size) but calibration degrades sharply (margin 0.40).
  Usable as a cheap first pass *with* escalation, or for high-frequency
  checks where even ~20 ms matters.
- **`nimble:9b`** — good noul calibration, softer on negatives (0.13).
- **`clef:27b`** — tightest calibration overall (0.006 on negatives); the
  adjudicator of choice. Also the only non-flash model with image judging.
- **`clef-flash:9b`** — ❌ broken on this server: every request returns
  `500: Clef: non-finite logit`, deterministic across all inputs. Re-pulling
  did not help (sha256 verifies against the registry, digest unchanged), so
  the registry build itself is suspect. Excluded from benchmarks until fixed.

## Recommended patterns

1. **Hot loop** — pass `keep_alive=-1` so the model never unloads; skip the
   cold-load cost entirely. Use a finite value in production if the server
   must reclaim VRAM when idle.
2. **Escalation** — `systemone_fallback("tev1:4b", ..., fallback_model="clef:27b")`
   gives 4b latency for the easy cases and 27b quality when confidence drops.
3. **Thresholds** — treat `noul` as a calibrated probability (0.5 = flip);
   treat `choice.confidence < 0.6` as "needs a second opinion"; treat
   `score` as a continuous position on your rubric, not a hard bucket.

## Project layout

```
systemone.py   # client: systemone(), systemone_fallback(), answer types, errors
example.py     # runnable demo of all three question types
```

## References

- Decision capability guide: <https://docs.ollama.com/capabilities/decision>
- System One API reference: <https://docs.ollama.com/api/systemone>
- Decision models: <https://ollama.com/search?c=decision>
