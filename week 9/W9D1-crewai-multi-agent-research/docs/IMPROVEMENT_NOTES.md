# Baseline vs. web-assisted run

| Dimension | Offline baseline | Web-assisted run |
|---|---|---|
| Evidence | No external sources | Search result titles, URLs and snippets attached |
| Traceability | No citations | Readers can open source URLs |
| Freshness | Not assessed | Depends on live search results and source publication dates |
| Reliability | Reproducible deterministic output | Network/rate limits may trigger clearly identified fallback |
| Remaining risk | Generic prompts, not factual conclusions | Snippets are not full-source verification |

## How to demonstrate improvement
1. Run `python -m research_agent.cli --topic "..." --mode offline`.
2. Run the same topic with `--mode web`.
3. Compare the Sources and Limitations sections.
4. For the CrewAI mode, capture the three agent/task execution logs and reviewed Markdown output.
5. Do not describe fallback results as live search. Open and verify sources before making factual claims.

The offline-to-web difference is evidence traceability, not an automatic guarantee of factual accuracy.
