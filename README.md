# Symbiotic-Agent
## Overview

**Symbiotic Agent** is a hybrid architecture for the **Rule Interaction Threat (RIT) validation** task in IoT automation. Unlike pure LLM-based agents, it combines:

* **A deterministic symbolic analysis layer** — guaranteed logical coverage and consistent rule reasoning
* **A contextual LLM adjudication layer** — semantic interpretation for ambiguous or context-heavy cases

This *symbiosis* improves precision and robustness, and serves as a scalable blueprint for building reliable agents in **safety-critical** domains.

---

## What is a Rule Interaction Threat (RIT)?

A **Rule Interaction Threat (RIT)** occurs when multiple IoT automation rules interact in a way that can create unsafe or unintended behavior.

This project uses **6 RIT classes**, grouped into **3 larger categories**.

### RIT taxonomy

| Category                  | RIT Type                              | Key Features                                                                                          | Why it matters                                                                   |
| ------------------------- | ------------------------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Action Contradictions** | **WAC** (Weak Action Contradiction)   | Overlapping triggers; overlapping conditions (**at least one action guarded**); contradictory actions | Race conditions where one automation reverses another under some conditions      |
|                           | **SAC** (Strong Action Contradiction) | Overlapping triggers; **no** guarding conditions; contradictory actions                               | Highest risk action conflict—immediate reversal is likely                        |
| **Trigger Cascades**      | **WTC** (Weak Trigger Cascade)        | One rule’s action triggers another rule; conditions **guard** the cascade; resulting action occurs    | Chained behaviors that depend on conditions—still risky and hard to predict      |
|                           | **STC** (Strong Trigger Cascade)      | One rule’s action directly triggers another rule; **no** guarding conditions; immediate cascade       | Fast, uncontrolled cascades that can spiral without checks                       |
| **Condition Cascades**    | **WCC** (Weak Condition Cascade)      | Overlapping triggers; action enables **some but not all** conditions; multiple conditions present     | Enables partial preconditions that can push the system toward undesired states   |
|                           | **SCC** (Strong Condition Cascade)    | Overlapping triggers; action enables **all** conditions; direct enablement                            | Directly enables another rule’s conditions—can reliably cause unsafe activations |

> Example risk (Action Contradictions): a fire-safety automation can be immediately reversed by a scheduled automation.

---

## How benchmarking works (Green ↔ Purple)

This repo contains a two-agent evaluation loop:

* **Green agent** loads the benchmark dataset and iterates over it **row-by-row**.
* For each row, Green extracts the **ruleset** and asks the **Purple agent** to predict which of the **six RIT types** the ruleset belongs to:

  * `WAC`, `SAC`, `WTC`, `STC`, `WCC`, `SCC`

* The A2A Payload of this green agent is structured like `{2-shot prompt} + ===== RULES START ===== + ruleset + ===== RULES END =====`

### Dataset

* The dataset contains **2,496 rulesets** in total. With ** 386** WAC, **418** SAC, **300** WTC, **180** STC, **627** WCC, and **729** SCC.
* Benchmark scope is adjustable via startup config:

  * Evaluate all rows, or only a subset
  * Filter by specific RIT type(s)
  * Limit maximum evaluated rows

---

# Green Agent (Benchmark Runner)

Green Agent is a **benchmark/evaluation agent**. It reads a dataset of rules + gold labels, calls a **Purple agent** to get predictions, and outputs a structured JSON result with metrics, samples, errors, and the resolved configuration.

---

## Quick start

### Minimal request (valid JSON)

```json
{
  "participants": {
    "agent": "http://purple:8080"
  }
}
```

### Typical request

```json
{
  "participants": { "agent": "http://purple:8080" },
  "config": {
    "max_rows": 50,
    "rit_filter": "SAC",
    "row_timeout": 60,
    "max_retries": 2,
    "circuit_breaker_threshold": 5,
    "skip_health_check": false
  }
}

## Request format

Green expects:

```json
{
  "participants": {
    "agent": "http://<purple-agent-url>"
  },
  "config": { }
}
```
### To use scenario.toml in the benchmark repo

[green_agent]
agentbeats_id = "019c178a-0f6f-7852-9fc8-f2aa84ebdf56"
env = {}
[config]
# Add your assessment config under [config]
max_rows = 30
rit_filter = "STC"

### `participants` (required)

* **Type:** object
* **Required key:** `"agent"`

  * This is the Purple agent endpoint Green will call.

### `config` (optional)

* **Type:** object
* If omitted, Green uses defaults.

---

## Config options

| Key                         | Type          |         Default | Meaning                                                                                        |
| --------------------------- | ------------- | --------------: | ---------------------------------------------------------------------------------------------- |
| `max_rows`                  | int           |         `10000` | Max number of **evaluated rows** (rows that pass filtering and get sent to Purple).            |
| `rit_filter`                | string | null |          `null` | Only evaluate rows whose gold label matches this value. Allowed: `WAC,SAC,WTC,STC,WCC,SCC`.    |
| `filter_rit`                | string | null |          `null` | Alias for `rit_filter` (either works).                                                         |
| `row_timeout`               | int (seconds) |            `60` | Timeout per Purple call **per attempt**.                                                       |
| `max_retries`               | int           |             `2` | Retry count after a failure/timeout. Total attempts per row = `max_retries + 1`.               |
| `circuit_breaker_threshold` | int           |             `5` | Stop early if this many failures happen **in a row**.                                          |
| `skip_health_check`         | bool          |         `false` | If `false`, Green pings Purple before starting. If `true`, it skips the ping.                  |

---

## Labels (classification targets)

Green only accepts these labels:

* `WAC`, `SAC`, `WTC`, `STC`, `WCC`, `SCC`

If `rit_filter` is set, it must be one of the above.

---

## Output format

Green emits a **Result** artifact containing a JSON object shaped like:

```json
{
  "metrics": { ... },
  "label_stats": { ... },
  "samples": [ ... ],
  "errors": [ ... ],
  "config_used": { ... },
  "early_termination_reason": "..."
}
```

---

## Output fields explained

### `metrics`

This object summarizes performance and run stats.

| Field                       | Meaning                                                                          |
| --------------------------- | -------------------------------------------------------------------------------- |
| `rows_attempted`            | Rows evaluated (Green tried calling Purple and recorded a row result).           |
| `rows_successful`           | Rows where Green got a Purple response (**includes** `parse_failed`).            |
| `rows_skipped_by_filter`    | Rows skipped because they didn’t match `rit_filter`.                             |
| `rows_scanned`              | Total dataset rows scanned by the CSV reader (includes filtered rows).           |
| `total_matching_in_dataset` | Total dataset rows matching `rit_filter` (only meaningful when a filter is set). |
| `correct`                   | Count where `pred == gold`.                                                      |
| `accuracy`                  | `correct / rows_successful` (note: `parse_failed` reduces accuracy).             |
| `success_rate`              | `rows_successful / rows_attempted`.                                              |
| `total_failures`            | Rows that failed after retries (timeouts/errors).                                |
| `elapsed_seconds`           | Total wall-clock runtime for this run.                                           |

### `label_stats`

Prediction breakdowns.

| Field                 | Meaning                                              |
| --------------------- | ---------------------------------------------------- |
| `pred_counts`         | Count of predictions by predicted label.             |
| `pred_correct_counts` | Count of **correct** predictions by predicted label. |

### `samples`

A list of example row outcomes (up to 50). Each sample typically includes:

| Field              | Type          | Meaning                                                |
| ------------------ | ------------- | ------------------------------------------------------ |
| `row_index`        | int           | Row index in the CSV (0-based).                        |
| `gold`             | string        | Gold label from `gold_column`.                         |
| `pred`             | string | null | Parsed prediction label (or null if parsing failed).   |
| `status`           | string        | One of: `success`, `parse_failed`, `timeout`, `error`. |
| `is_correct`       | bool          | Whether `pred == gold`.                                |
| `response_preview` | string        | Truncated preview of Purple’s raw response.            |
| `error`            | string        | Error message (if any).                                |
| `duration_ms`      | int           | Time spent on the final attempt for that row.          |

### `errors`

Up to 20 error strings, commonly formatted like:

* `"Row <row_index>: <error_message>"`

### `config_used`

The resolved config after defaults are applied, typically including:

* `dataset_path`, `dataset_encrypted`
* `ruleset_column`, `gold_column`, `max_rows`, `rit_filter`
* `row_timeout`, `max_retries`, `circuit_breaker_threshold`
* `allowed_labels` (`WAC`, `SAC`, `WTC`, `STC`, `WCC`, `SCC`)

### `early_termination_reason`

A human-readable reason when Green stops early or wants to explain something, e.g.:

* Circuit breaker triggered
* Dataset had fewer matching rows than requested
* Unexpected runtime error
* No matching rows found

---

## Important metric nuance: filter mode keeps scanning

If `rit_filter` is set, Green may continue scanning the dataset after it has already evaluated `max_rows` rows in order to compute `total_matching_in_dataset`.

That’s why you might see:

* `rows_scanned` much larger than `rows_attempted`
* `total_matching_in_dataset` greater than `max_rows`

---

## Example output snippet

```json
{
  "metrics": {
    "rows_attempted": 50,
    "rows_successful": 50,
    "correct": 47,
    "accuracy": 0.94,
    "elapsed_seconds": 1.23
  },
  "config_used": {
    "rit_filter": "SAC",
    "max_rows": 50,
    "row_timeout": 60
  },
  "early_termination_reason": ""
}
```

