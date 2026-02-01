# Symbiotic-Agent
Project Name: Symbiotic Agent (A Hybrid Symbolic-LLM Agent for IoT Rule Validation)

Description: We present the Symbiotic Agent, a novel hybrid architecture for the Rule Interaction Threat (RIT) validation task. Unlike pure LLM agents, it integrates a deterministic symbolic analysis layer (for guaranteed logical coverage) with a contextual LLM adjudication layer (for semantic reasoning). This symbiosis allows it to achieve state-of-the-art precision and robustness, demonstrating a scalable blueprint for reliable AI agents in safety-critical domains.

Field	Meaning
rows_attempted	Rows actually evaluated (i.e., Green tried calling Purple and recorded a row result).
rows_successful	Rows where Green got a Purple response (includes parse_failed).
rows_skipped_by_filter	Rows skipped because they didn’t match rit_filter.
rows_scanned	Total dataset rows scanned by the CSV reader.
total_matching_in_dataset	Total rows in the dataset matching rit_filter (only meaningful when a filter is set).
correct	Count where pred == gold.
accuracy	correct / rows_successful (note: parse_failed hurts accuracy).
success_rate	rows_successful / rows_attempted.
total_failures	Number of rows that failed after retries (timeouts/errors).
elapsed_seconds	Total wall-clock runtime for this run.
