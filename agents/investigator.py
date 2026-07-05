import os
import json
import time
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

from groq import Groq

MODEL = "qwen/qwen3.6-27b"
MAX_RETRIES = 3
RETRY_DELAY = 65

REGULATORY_CONTEXT = """
You are a Nigerian AML compliance analyst assistant working within a bank's
compliance department. You operate under the following regulatory framework:

KEY REGULATIONS:
1. Money Laundering (Prevention and Prohibition) Act 2022 (MLPPA 2022)
   - Section 2(1): Cash transactions exceeding ₦5,000,000 (individuals) must be reported to NFIU
   - Section 2(2): Splitting transactions to evade reporting thresholds is a criminal offence
   - Section 3: STRs must be filed with NFIU within 7 days
   - Penalty for failure to file: ₦250,000-₦1,000,000 per day

2. CBN Revised Cash-Related Policies Circular FPRD/DIR/PUB/CIR/001/011 (effective Jan 1 2026)
   - ATM withdrawals capped at ₦100,000/day per customer
   - Weekly cumulative withdrawal cap: ₦500,000 (individuals)
   - Excess withdrawal fee: 3% on amount over limit, split 40% CBN / 60% bank

3. CBN Tiered KYC Framework
   - Tier 1: ₦30,000 daily limit
   - Tier 2: ₦500,000 daily limit
   - Tier 3: unrestricted

IMPORTANT: You DRAFT reports and RECOMMEND actions. The human MLRO makes all
final decisions. No regulatory filing happens without explicit human sign-off.
"""


def get_groq_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError(
            "GROQ_API_KEY not found in environment.\n"
            "Check your .env file contains: GROQ_API_KEY=your_key_here\n"
            "No quotes, no spaces around the = sign."
        )
    return Groq(api_key=key)


def call_groq(client, messages, max_tokens=800):
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,
            )
            content = response.choices[0].message.content

            # Strip Qwen3 chain-of-thought thinking blocks
            # <think>...</think> tags contain internal reasoning, not output
            import re
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

            return content

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str.lower():
                if attempt < MAX_RETRIES - 1:
                    print(f"  Rate limit hit — waiting {RETRY_DELAY}s before retry {attempt + 2}/{MAX_RETRIES}...")
                    time.sleep(RETRY_DELAY)
                else:
                    raise RuntimeError(f"Rate limit exceeded after {MAX_RETRIES} retries: {e}")
            else:
                raise RuntimeError(f"Groq API error: {e}")


def summarize_history(transaction_history):
    if not transaction_history:
        return "No recent transaction history available."
    lines = []
    total = sum(t.get("amount", 0) for t in transaction_history)
    lines.append(f"Total transactions: {len(transaction_history)}")
    lines.append(f"Combined amount: ₦{total:,.2f}")
    lines.append("")
    for t in transaction_history[-10:]:
        lines.append(
            f"  {str(t.get('timestamp', 'N/A'))[:16]} | "
            f"{t.get('account_id', 'N/A')} | "
            f"₦{t.get('amount', 0):>12,.2f} | "
            f"{t.get('type', 'N/A')} via {t.get('channel', 'N/A')}"
        )
    return "\n".join(lines)


def breakdown_by_account(transaction_history, accounts):
    per_account = defaultdict(list)
    for t in transaction_history:
        acc = t.get("account_id")
        if acc in accounts:
            per_account[acc].append(t)
    lines = []
    grand_total = 0
    for acc in accounts:
        txns = per_account.get(acc, [])
        total = sum(t.get("amount", 0) for t in txns)
        grand_total += total
        lines.append(f"  {acc}: {len(txns)} transactions | ₦{total:,.2f} total")
    lines.append(f"\n  COMBINED TOTAL: ₦{grand_total:,.2f}")
    lines.append(f"  VS NFIU THRESHOLD: {grand_total / 5_000_000:.2f}x the ₦5,000,000 limit")
    return "\n".join(lines)


def extract_recommendation(text):
    text_upper = text.upper()
    if "FILE STR" in text_upper:
        return "FILE STR"
    elif "ENHANCED MONITORING" in text_upper:
        return "ENHANCED MONITORING"
    elif "DISMISS" in text_upper:
        return "DISMISS"
    return "ENHANCED MONITORING"


def step1_pattern_analysis(client, alert, transaction_history):
    history_summary = summarize_history(transaction_history)
    messages = [
        {"role": "system", "content": REGULATORY_CONTEXT},
        {"role": "user", "content": f"""
Analyze this flagged account and identify the fraud typology.

ALERT DETAILS:
- Account: {alert['account_id']} (Customer: {alert['customer_id']})
- Account Tier: {alert['account_tier']}
- Flagged Amount: ₦{alert['amount']:,.2f}
- TxGuard Risk Score: {alert['txguard_risk_score']}/100
- Risk Tier: {alert['risk_tier']}
- Triggered Rules: {', '.join(alert['triggered_rules']) if alert['triggered_rules'] else 'ML anomaly detection only'}
- Model Scores: ISO={alert['iso_score']:.3f} | RF={alert['rf_score']:.3f} | GNN={alert['gnn_score']:.3f}
- Timestamp: {alert['timestamp']}

CUSTOMER TRANSACTION HISTORY (last 24 hours, all accounts):
{history_summary}

Identify:
1. The primary fraud typology
2. The key evidence supporting this classification
3. Which specific MLPPA 2022 or CBN regulation is most directly implicated
4. Whether this appears to be single-account or cross-account coordinated activity

Be concise and factual. Use Nigerian regulatory terminology.
"""}
    ]
    return call_groq(client, messages, max_tokens=600)


def step2_cross_account_analysis(client, alert, all_customer_accounts, transaction_history):
    if len(all_customer_accounts) <= 1:
        return None
    account_breakdown = breakdown_by_account(transaction_history, all_customer_accounts)
    messages = [
        {"role": "system", "content": REGULATORY_CONTEXT},
        {"role": "user", "content": f"""
This customer ({alert['customer_id']}) holds {len(all_customer_accounts)} accounts.
Analyze the cross-account activity for evidence of coordinated structuring.

ACCOUNTS: {', '.join(all_customer_accounts)}

PER-ACCOUNT BREAKDOWN (last 24 hours):
{account_breakdown}

MLPPA 2022 Section 2(1) individual threshold: ₦5,000,000
MLPPA 2022 Section 2(2): Splitting across accounts to evade this threshold is a criminal offence.

Calculate:
1. Total combined exposure across all accounts
2. How far above/below the ₦5,000,000 reporting threshold
3. Whether the split pattern is consistent with deliberate structuring
4. The strongest evidence for or against Section 2(2) structuring
"""}
    ]
    return call_groq(client, messages, max_tokens=500)


def step3_risk_assessment(client, alert, pattern_analysis, cross_account_analysis):
    cross_section = f"\nCROSS-ACCOUNT ANALYSIS:\n{cross_account_analysis}" if cross_account_analysis else ""
    messages = [
        {"role": "system", "content": REGULATORY_CONTEXT},
        {"role": "user", "content": f"""
Based on the analysis below, provide a risk assessment and recommended action for the MLRO.

PATTERN ANALYSIS:
{pattern_analysis}
{cross_section}

ALERT SCORE: {alert['txguard_risk_score']}/100 ({alert['risk_tier']} tier)

Provide:
1. OVERALL RISK LEVEL: HIGH / MEDIUM / LOW (justify in one sentence)
2. RECOMMENDED ACTION: Choose exactly one:
   - FILE STR: File Suspicious Transaction Report with NFIU within 7 days
   - ENHANCED MONITORING: Flag for 30-day enhanced monitoring, no STR yet
   - DISMISS: Insufficient evidence, document reason and close
3. REGULATORY BASIS: Cite the specific MLPPA 2022 section or CBN circular
4. URGENCY: Standard (7-day STR window) or Urgent (terrorism-related, 24-hour window)

The MLRO will make the final decision.
"""}
    ]
    return call_groq(client, messages, max_tokens=500)


def step4_draft_str(client, alert, pattern_analysis, cross_account_analysis, risk_assessment):
    cross_section = f"\nCROSS-ACCOUNT FINDINGS:\n{cross_account_analysis}" if cross_account_analysis else ""
    messages = [
        {"role": "system", "content": REGULATORY_CONTEXT},
        {"role": "user", "content": f"""
Draft a Suspicious Transaction Report (STR) narrative for NFIU submission.
This draft requires MLRO review and sign-off before any submission.

SUBJECT ACCOUNT: {alert['account_id']}
CUSTOMER ID: {alert['customer_id']}
ACCOUNT TIER: {alert['account_tier']}
REPORTING DATE: {datetime.utcnow().strftime('%Y-%m-%d')}

INVESTIGATION FINDINGS:
{pattern_analysis}
{cross_section}

RISK ASSESSMENT:
{risk_assessment}

Draft the STR narrative using this structure:
1. SUBJECT DETAILS: Account information and KYC tier
2. TRANSACTION SUMMARY: Key transactions, amounts, dates, channels
3. SUSPICIOUS INDICATORS: Specific behaviors that raised concern
4. REGULATORY BREACH: Which provision was breached and how
5. RECOMMENDED ACTION: What the institution recommends (for MLRO to confirm)

Write in formal regulatory language. Be factual and specific.
End with: "DRAFT — Pending MLRO review and approval before submission."
"""}
    ]
    return call_groq(client, messages, max_tokens=800)


class TxGuardInvestigator:

    def __init__(self):
        self.client = get_groq_client()
        print(f"TxGuard Investigator initialized (model: {MODEL})")

    def investigate(self, alert, transaction_history, all_customer_accounts=None):
        if all_customer_accounts is None:
            all_customer_accounts = [alert["account_id"]]

        print(f"\n{'='*60}")
        print(f"  INVESTIGATING: {alert['account_id']} ({alert['customer_id']})")
        print(f"  Risk Score: {alert['txguard_risk_score']}/100 | {alert['risk_tier']}")
        print(f"{'='*60}")

        report = {
            "alert": alert,
            "customer_id": alert["customer_id"],
            "account_id": alert["account_id"],
            "all_accounts": all_customer_accounts,
            "investigated_at": datetime.utcnow().isoformat() + "Z",
            "model_used": MODEL,
            "steps": {},
            "str_draft": None,
            "final_recommendation": None,
            "mlro_decision": "PENDING",
        }

        print("  Step 1/4: Analyzing fraud pattern...")
        pattern = step1_pattern_analysis(self.client, alert, transaction_history)
        report["steps"]["pattern_analysis"] = pattern
        print("  Done")

        cross_account = None
        if len(all_customer_accounts) > 1:
            print(f"  Step 2/4: Cross-account analysis ({len(all_customer_accounts)} accounts)...")
            cross_account = step2_cross_account_analysis(
                self.client, alert, all_customer_accounts, transaction_history
            )
            report["steps"]["cross_account_analysis"] = cross_account
            print("  Done")
        else:
            print("  Step 2/4: Single account — skipping")
            report["steps"]["cross_account_analysis"] = None

        print("  Step 3/4: Generating risk assessment...")
        risk_assessment = step3_risk_assessment(self.client, alert, pattern, cross_account)
        report["steps"]["risk_assessment"] = risk_assessment
        report["final_recommendation"] = extract_recommendation(risk_assessment)
        print(f"  Done — Recommendation: {report['final_recommendation']}")

        if report["final_recommendation"] == "FILE STR":
            print("  Step 4/4: Drafting STR narrative...")
            str_draft = step4_draft_str(
                self.client, alert, pattern, cross_account, risk_assessment
            )
            report["str_draft"] = str_draft
            print("  Done — PENDING MLRO REVIEW")
        else:
            print(f"  Step 4/4: STR not required ({report['final_recommendation']})")

        return report

    def print_report(self, report):
        alert = report["alert"]
        print(f"\n{'#'*60}")
        print(f"  TXGUARD INVESTIGATION REPORT")
        print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"{'#'*60}")
        print(f"\n  Account:      {report['account_id']}")
        print(f"  Customer:     {report['customer_id']}")
        print(f"  All Accounts: {', '.join(report['all_accounts'])}")
        print(f"  Risk Score:   {alert['txguard_risk_score']}/100 ({alert['risk_tier']})")
        print(f"  Amount:       ₦{alert['amount']:,.2f}")
        print(f"  Rules:        {', '.join(alert['triggered_rules']) or 'ML flag only'}")

        print(f"\n{'─'*60}")
        print("  PATTERN ANALYSIS")
        print(f"{'─'*60}")
        print(report["steps"]["pattern_analysis"])

        if report["steps"].get("cross_account_analysis"):
            print(f"\n{'─'*60}")
            print("  CROSS-ACCOUNT ANALYSIS")
            print(f"{'─'*60}")
            print(report["steps"]["cross_account_analysis"])

        print(f"\n{'─'*60}")
        print("  RISK ASSESSMENT & RECOMMENDATION")
        print(f"{'─'*60}")
        print(report["steps"]["risk_assessment"])

        if report["str_draft"]:
            print(f"\n{'─'*60}")
            print("  STR DRAFT (PENDING MLRO REVIEW)")
            print(f"{'─'*60}")
            print(report["str_draft"])

        print(f"\n{'#'*60}")
        print(f"  FINAL STATUS:   {report['final_recommendation']}")
        print(f"  MLRO DECISION:  {report['mlro_decision']}")
        print(f"  No regulatory filing until MLRO signs off")
        print(f"{'#'*60}\n")

    def save_report(self, report, output_dir="reports"):
        os.makedirs(output_dir, exist_ok=True)
        filename = (
            f"{output_dir}/investigation_"
            f"{report['customer_id']}_"
            f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved to {filename}")
        return filename


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from scorer.unified import TxGuardScorer

    print("Loading TxGuard pipeline...")
    scorer = TxGuardScorer()
    investigator = TxGuardInvestigator()

    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "simulated_transactions.jsonl"
    )

    print(f"Scanning transactions for HIGH alerts...")

    all_transactions = []
    alerts = []
     
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                txn = json.loads(line)
                all_transactions.append(txn)
                alert = scorer.score(txn)
                if alert["risk_tier"] == "HIGH" and alert["txguard_risk_score"] >= 85:
                    alerts.append(alert)
            except Exception:
                continue

            if len(all_transactions) >= 500:
                break
    # with open(data_path, "r") as f:
    #     for line in f:
    #         line = line.strip()
    #         if not line:
    #             continue
    #         try:
    #             txn = json.loads(line)
    #             all_transactions.append(txn)
    #             alert = scorer.score(txn)
    #             if alert["risk_tier"] == "HIGH" and alert["txguard_risk_score"] >= 85:
    #                 alerts.append(alert)
    #         except Exception:
    #             continue

    if not alerts:
        print("No HIGH alerts found.")
    else:
        top_alert = max(alerts, key=lambda a: a["txguard_risk_score"])
        print(f"\nTop alert: {top_alert['account_id']} "
              f"({top_alert['customer_id']}) — "
              f"Score: {top_alert['txguard_risk_score']}/100")

        customer_id = top_alert["customer_id"]
        customer_history = [
            t for t in all_transactions
            if t.get("customer_id") == customer_id
        ]
        all_accounts = list({t["account_id"] for t in customer_history})

        print(f"Customer history: {len(customer_history)} transactions "
              f"across {all_accounts}")

        report = investigator.investigate(top_alert, customer_history, all_accounts)
        investigator.print_report(report)
        investigator.save_report(report)