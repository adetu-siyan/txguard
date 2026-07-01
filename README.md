# TxGuard

Real-time transaction anomaly detection pipeline for Nigerian fintech, built
on Azure (Event Hubs, Stream Analytics, Functions, Azure OpenAI, Cosmos DB).

## Setup

1. Create and activate a virtual environment:

   Windows:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

   Mac/Linux:
   ```
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and fill in your real Event Hubs
   connection string and Event Hub name. Never commit `.env`.

4. Run the simulator:
   ```
   python -m simulator.main
   ```

## Project Structure

```
txguard/
├── simulator/
│   ├── __init__.py
│   ├── config.py          # account population, channels, transaction types
│   ├── generators.py      # normal + anomaly transaction generators
│   ├── eventhub_client.py # Azure Event Hubs producer wrapper
│   └── main.py            # entry point — runs the mixed transaction stream
├── data/
│   └── simulated_transactions.jsonl   # local output log (gitignored)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Status

- [x] Day 1 — Azure resource setup (Event Hubs namespace + Event Hub)
- [x] Day 2 — Simulator: normal traffic generator
- [x] Day 3 — Simulator: anomaly patterns + Event Hubs ingestion
- [ ] Day 4 — Stream Analytics job setup
- [ ] Day 5-6 — Anomaly detection rules
- [ ] Day 7-8 — Azure Functions + severity scoring
- [ ] Day 9-10 — Cosmos DB + account history
- [ ] Day 11-12 — Azure OpenAI reasoning integration
- [ ] Day 13 — Dashboard build + deploy
- [ ] Day 14 — Buffer, polish, demo rehearsal
