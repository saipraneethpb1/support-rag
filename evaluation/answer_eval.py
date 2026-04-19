"""Answer quality evaluation.

Three metrics, all LLM-as-judge (free tier friendly):

1. FAITHFULNESS
   For each claim in the answer, does the retrieved context support it?
   Score in [0, 1]. Catches hallucinations and made-up facts.

2. ANSWER RELEVANCY
   Does the answer actually address the question asked? An answer can be
   faithful-but-off-topic ("the question was about cancellation; the
   answer is about pricing but everything in it is true"). Score [0, 1].

3. CITATION GROUNDING (cheap non-LLM metric)
   Fraction of answers with zero invented citation markers. If the LLM is
   citing sources that aren't in the context block, it's confabulating.

This complements retrieval_eval.py — that one measures WHETHER the right
doc was retrieved; this measures WHAT THE LLM DID WITH IT.

Usage:
    python -m evaluation.answer_eval
"""
from __future__ import annotations
import asyncio
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

from generation.generator import Generator, GeneratedAnswer
from generation.llm_router import LLMRouter
from observability.logger import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)

GOLDEN_PATH = Path("evaluation/datasets/golden_qa.jsonl")
RESULTS_DIR = Path("evaluation/results")


@dataclass
class AnswerEvalCase:
    query: str


def load_cases() -> list[AnswerEvalCase]:
    cases: list[AnswerEvalCase] = []
    with GOLDEN_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cases.append(AnswerEvalCase(query=obj["query"]))
    return cases


_FAITHFULNESS_PROMPT = """You are evaluating whether an ANSWER is faithful to a CONTEXT.

Instructions:
1. Extract each atomic factual claim the ANSWER makes.
2. For each claim, decide whether it is directly supported by the CONTEXT.
3. Ignore citation markers like [1], [2] — only evaluate the factual content.
4. A claim counts as supported only if the CONTEXT explicitly states or directly implies it. Do NOT give credit for plausible-sounding but unsupported claims.

Return a JSON object ONLY, no other text:
{{
  "claims": [
    {{"claim": "...", "supported": true}},
    ...
  ]
}}

CONTEXT:
{context}

ANSWER:
{answer}
"""


_RELEVANCY_PROMPT = """Score how well the ANSWER addresses the QUESTION on a scale of 0.0 to 1.0.

- 1.0: Directly and completely answers the question.
- 0.7: Addresses the main point but misses some aspects.
- 0.4: Tangentially related, or answers a different question than asked.
- 0.0: Irrelevant or refuses to answer.

Do NOT penalize for brevity if the answer is complete. Do NOT reward verbosity.

Return a JSON object ONLY:
{{"score": <float 0.0..1.0>, "reason": "<one short sentence>"}}

QUESTION: {question}

ANSWER: {answer}
"""


class Judge:
    """LLM-as-judge wrapper. Uses the same router as generation, but low temp."""
    def __init__(self, router: LLMRouter):
        self.router = router

    async def faithfulness(self, context: str, answer: str) -> tuple[float, list[dict]]:
        prompt = _FAITHFULNESS_PROMPT.format(context=context, answer=answer)
        raw = await self.router.complete(prompt, max_tokens=800, temperature=0.0)
        obj = _extract_json(raw)
        if not obj or "claims" not in obj:
            return 0.0, []
        claims = obj["claims"]
        if not claims:
            # No claims -> vacuously faithful (probably an "I don't know" answer)
            return 1.0, []
        supported = sum(1 for c in claims if c.get("supported"))
        return supported / len(claims), claims

    async def relevancy(self, question: str, answer: str) -> float:
        prompt = _RELEVANCY_PROMPT.format(question=question, answer=answer)
        raw = await self.router.complete(prompt, max_tokens=120, temperature=0.0)
        obj = _extract_json(raw)
        if not obj:
            return 0.0
        try:
            return max(0.0, min(1.0, float(obj.get("score", 0.0))))
        except (TypeError, ValueError):
            return 0.0


def _extract_json(text: str) -> dict | None:
    # Strip ```json fences if present
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    # Find outermost {...}
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _build_context_block(answer: GeneratedAnswer) -> str:
    """Rebuild the context block as the LLM saw it, so judge has same grounding."""
    if not answer.retrieval or not answer.retrieval.chunks:
        return "(no context)"
    parts: list[str] = []
    for i, ch in enumerate(answer.retrieval.chunks, start=1):
        parts.append(f"[{i}] {ch.title}\n{ch.text.strip()}")
    return "\n\n".join(parts)


async def evaluate(generator: Generator, judge: Judge, cases: list[AnswerEvalCase]) -> dict:
    results: list[dict] = []
    for case in cases:
        try:
            ans = await generator.generate(case.query, use_cache=False)
        except Exception as e:
            log.exception("generation_failed", query=case.query, error=str(e))
            continue

        context = _build_context_block(ans)
        faith, claims = await judge.faithfulness(context, ans.answer)
        rel = await judge.relevancy(case.query, ans.answer)
        grounded = 1.0 if not ans.audit.invented_markers else 0.0

        results.append({
            "query": case.query,
            "answer": ans.answer,
            "faithfulness": faith,
            "relevancy": rel,
            "citation_grounded": grounded,
            "invented_citations": sorted(ans.audit.invented_markers),
            "coverage": ans.audit.sentence_coverage,
            "claims": claims,
        })
        log.info(
            "case_done",
            query=case.query,
            faithfulness=round(faith, 3),
            relevancy=round(rel, 3),
            grounded=bool(grounded),
        )

    if not results:
        return {"n": 0}

    agg = {
        "n": len(results),
        "faithfulness_mean": statistics.mean(r["faithfulness"] for r in results),
        "relevancy_mean": statistics.mean(r["relevancy"] for r in results),
        "citation_grounded_rate": statistics.mean(r["citation_grounded"] for r in results),
        "hallucination_rate": 1.0 - statistics.mean(r["citation_grounded"] for r in results),
        "coverage_mean": statistics.mean(r["coverage"] for r in results),
        "per_case": results,
    }
    return agg


async def main() -> None:
    cases = load_cases()
    generator = Generator()
    judge = Judge(LLMRouter())

    log.info("answer_eval_start", n=len(cases))
    agg = await evaluate(generator, judge, cases)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "answer_eval.json").write_text(json.dumps(agg, indent=2))

    print("\n" + "=" * 72)
    print(f"{'Metric':<28} {'Score':>10}")
    print("-" * 72)
    print(f"{'Queries evaluated':<28} {agg['n']:>10}")
    print(f"{'Faithfulness (mean)':<28} {agg.get('faithfulness_mean', 0):>10.3f}")
    print(f"{'Answer relevancy (mean)':<28} {agg.get('relevancy_mean', 0):>10.3f}")
    print(f"{'Citation grounded rate':<28} {agg.get('citation_grounded_rate', 0):>10.3f}")
    print(f"{'Hallucination rate':<28} {agg.get('hallucination_rate', 0):>10.3f}")
    print(f"{'Sentence coverage (mean)':<28} {agg.get('coverage_mean', 0):>10.3f}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
