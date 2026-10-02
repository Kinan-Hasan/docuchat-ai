"""Custom evaluation harness for the RAG pipeline.

Rather than pulling in a heavyweight framework (e.g. RAGAS, which typically
needs a hosted judge LLM), this measures two concrete, explainable metrics
against a hand-labeled eval set:

  - retrieval_hit_rate: did retrieval surface a chunk containing at least
    one expected keyword? (measures the retriever in isolation)
  - answer_keyword_recall: what fraction of expected keywords appear in the
    final generated answer? (measures the full pipeline end-to-end)

This is exactly the kind of harness you can point to on a resume: "built an
automated evaluation suite that measures retrieval accuracy and answer
quality against a labeled question set."

Usage:
    python -m evaluation.evaluate
"""
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.deps import get_pipeline


async def run_evaluation():
    dataset_path = Path(__file__).parent / "eval_dataset.json"
    dataset = json.loads(dataset_path.read_text())
    pipeline = get_pipeline()

    results = []
    for item in dataset:
        question = item["question"]
        keywords = [k.lower() for k in item["expected_keywords"]]

        start = time.time()
        outcome = await pipeline.answer(question)
        latency = time.time() - start

        answer_lower = outcome["answer"].lower()
        retrieved_text = " ".join(s["content"] for s in outcome["sources"]).lower()

        retrieval_hit = any(k in retrieved_text for k in keywords)
        matched = [k for k in keywords if k in answer_lower]
        keyword_recall = len(matched) / len(keywords) if keywords else 0.0

        results.append({
            "question": question,
            "answer": outcome["answer"],
            "retrieval_hit": retrieval_hit,
            "keyword_recall": round(keyword_recall, 2),
            "latency_s": round(latency, 2),
        })

    avg_recall = sum(r["keyword_recall"] for r in results) / len(results)
    hit_rate = sum(r["retrieval_hit"] for r in results) / len(results)
    avg_latency = sum(r["latency_s"] for r in results) / len(results)

    print(f"\n{'='*60}\nEVALUATION SUMMARY\n{'='*60}")
    print(f"Questions evaluated:       {len(results)}")
    print(f"Retrieval hit rate:        {hit_rate:.0%}")
    print(f"Avg answer keyword recall: {avg_recall:.0%}")
    print(f"Avg latency:               {avg_latency:.2f}s")
    print(f"{'='*60}\n")

    for r in results:
        status = "PASS" if r["retrieval_hit"] and r["keyword_recall"] >= 0.5 else "REVIEW"
        print(f"[{status}] {r['question']}")
        print(f"        recall={r['keyword_recall']:.0%}  retrieval_hit={r['retrieval_hit']}  latency={r['latency_s']}s")

    out_path = Path(__file__).parent / "eval_results.json"
    out_path.write_text(json.dumps({
        "summary": {"hit_rate": hit_rate, "avg_keyword_recall": avg_recall, "avg_latency_s": avg_latency},
        "results": results,
    }, indent=2))
    print(f"Full results written to {out_path}")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
