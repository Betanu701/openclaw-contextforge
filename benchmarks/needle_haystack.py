#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Needle:
    key: str
    fact: str
    question: str
    expected: str


@dataclass(frozen=True)
class ConversationTurn:
    turnIndex: int
    role: str
    content: str
    needleKeys: list[str]


def generate_needles(count: int, seed: int) -> list[Needle]:
    rng = random.Random(seed)
    needles: list[Needle] = []
    used_codes: set[str] = set()
    templates = [
        (
            "project-{i}",
            "The confidential project marker {marker} launch code is {code}.",
            "What is the confidential project marker {marker} launch code?",
        ),
        (
            "vault-{i}",
            "The vault marker {marker} for archive is {code}.",
            "What is the vault marker {marker} for archive?",
        ),
        (
            "agent-{i}",
            "Agent dossier marker {marker} has retrieval phrase {code}.",
            "What retrieval phrase belongs to agent dossier marker {marker}?",
        ),
    ]
    for i in range(count):
        marker = f"needle{i:04d}"
        while True:
            code = f"{rng.choice(['ALPHA', 'BETA', 'GAMMA', 'DELTA'])}-{rng.randint(100000, 999999)}"
            if code not in used_codes:
                used_codes.add(code)
                break
        key_template, fact_template, question_template = templates[i % len(templates)]
        key = key_template.format(i=i)
        fact = fact_template.format(i=i, marker=marker, code=code)
        question = question_template.format(i=i, marker=marker)
        needles.append(Needle(key=key, fact=fact, question=question, expected=code))
    return needles


def estimate_word_tokens(text: str) -> int:
    return len(text.split())


def filler_sentence(rng: random.Random, index: int) -> str:
    subjects = ["logistics", "weather", "operations", "inventory", "analysis", "planning"]
    locations = ["north pier", "relay station", "archive room", "training yard", "lab wing"]
    values = ["routine", "ordinary", "non-critical", "background", "ambient"]
    return (
        f"Background note {index}: {rng.choice(subjects)} at the {rng.choice(locations)} "
        f"was {rng.choice(values)} and unrelated to any hidden retrieval key."
    )


def generate_haystack(target_tokens: int, needles: list[Needle], seed: int) -> str:
    rng = random.Random(seed + 7919)
    paragraphs: list[str] = []
    positions = sorted(rng.sample(range(max(target_tokens // 35, len(needles) + 1)), len(needles)))
    needle_by_position = dict(zip(positions, needles))
    token_estimate = 0
    index = 0
    while token_estimate < target_tokens:
        if index in needle_by_position:
            paragraphs.append(needle_by_position[index].fact)
        else:
            paragraphs.append(filler_sentence(rng, index))
        token_estimate = len(" ".join(paragraphs).split())
        index += 1
    return "\n".join(paragraphs)


def random_words(rng: random.Random, count: int) -> str:
    vocabulary = [
        "lumen",
        "cobalt",
        "harbor",
        "matrix",
        "orbit",
        "canvas",
        "deltaic",
        "signal",
        "ember",
        "quartz",
        "vector",
        "lantern",
        "meadow",
        "circuit",
        "ripple",
        "forest",
        "packet",
        "sonnet",
        "granite",
        "violet",
        "binary",
        "anchor",
        "tunnel",
        "mirror",
        "plasma",
        "archive",
        "survey",
        "routine",
        "unrelated",
        "ambient",
    ]
    return " ".join(rng.choice(vocabulary) for _ in range(count))


def distractor_question(rng: random.Random, turn_index: int) -> str:
    topics = [
        "weather routing",
        "inventory sorting",
        "calendar cleanup",
        "fictional city planning",
        "garden layout",
        "training logistics",
    ]
    ticket = f"ticket-{turn_index:04d}-{rng.randint(1000, 9999)}"
    return (
        f"Distractor question: What routine note applies to {rng.choice(topics)} "
        f"for {ticket}? The answer is intentionally irrelevant."
    )


def make_conversation_turn(
    rng: random.Random,
    turn_index: int,
    target_tokens: int,
    needles: list[Needle],
) -> ConversationTurn:
    role = "user" if turn_index % 2 == 0 else "assistant"
    needle_text = " ".join(f"Needle memory: {needle.fact}" for needle in needles)
    heading = (
        f"Turn {turn_index:03d} {role} note. "
        f"{distractor_question(rng, turn_index)} "
    )
    body_tokens = max(8, target_tokens - estimate_word_tokens(heading) - estimate_word_tokens(needle_text))
    content = f"{heading}{random_words(rng, body_tokens)}"
    if needle_text:
        content = f"{content}\n{needle_text}"
    return ConversationTurn(
        turnIndex=turn_index,
        role=role,
        content=content,
        needleKeys=[needle.key for needle in needles],
    )


def native_tail_transcript(turns: list[dict[str, Any]], token_budget: int) -> str:
    selected: list[str] = []
    total_tokens = 0
    for turn in reversed(turns):
        entry = f"{turn['role'].upper()} TURN {turn['turnIndex']}:\n{turn['content']}"
        entry_tokens = estimate_word_tokens(entry)
        if selected and total_tokens + entry_tokens > token_budget:
            break
        selected.append(entry)
        total_tokens += entry_tokens
        if total_tokens >= token_budget:
            break
    return "\n\n".join(reversed(selected))


def make_conversation_dataset(
    turn_count: int,
    tokens_per_turn: int,
    needles: int,
    control_needles: int,
    seed: int,
    early_turns: int,
) -> dict[str, Any]:
    if turn_count < 2:
        raise ValueError("turn_count must be at least 2")
    if needles < 1:
        raise ValueError("needles must be at least 1")
    if control_needles < 0 or control_needles >= needles:
        raise ValueError("control_needles must be >= 0 and less than needles")

    rng = random.Random(seed + 17_017)
    generated_needles = generate_needles(needles, seed)
    early_count = needles - control_needles
    early_window = max(1, min(early_turns, turn_count - control_needles))
    late_start = max(early_window, turn_count - max(control_needles, 1))
    positions: dict[int, list[tuple[Needle, str]]] = {}

    for index, needle in enumerate(generated_needles[:early_count]):
        turn_index = index % early_window
        positions.setdefault(turn_index, []).append((needle, "early"))

    for index, needle in enumerate(generated_needles[early_count:]):
        turn_index = min(turn_count - 1, late_start + index)
        positions.setdefault(turn_index, []).append((needle, "control"))

    tokens_per_turn = max(40, tokens_per_turn)
    turns = [
        make_conversation_turn(
            rng,
            turn_index,
            tokens_per_turn,
            [needle for needle, _placement in positions.get(turn_index, [])],
        )
        for turn_index in range(turn_count)
    ]
    serialized_turns = [asdict(turn) for turn in turns]
    needle_rows: list[dict[str, Any]] = []
    for turn_index, entries in positions.items():
        for needle, placement in entries:
            needle_rows.append({**asdict(needle), "turnIndex": turn_index, "placement": placement})
    needle_rows.sort(key=lambda row: (int(row["turnIndex"]), str(row["key"])))

    dataset = {
        "id": f"conversation-seed{seed}-turns{turn_count}-tpt{tokens_per_turn}",
        "mode": "conversation-decay",
        "turns": turn_count,
        "tokensPerTurn": tokens_per_turn,
        "targetTokens": turn_count * tokens_per_turn,
        "estimatedTokens": estimate_word_tokens(conversation_transcript(serialized_turns)),
        "seed": seed,
        "earlyTurns": early_window,
        "controlNeedles": control_needles,
        "conversation": serialized_turns,
        "needles": needle_rows,
    }
    validate_conversation_dataset(dataset)
    return dataset


def conversation_transcript(turns: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"{turn['role'].upper()} TURN {turn['turnIndex']}:\n{turn['content']}" for turn in turns
    )


def make_dataset(tokens: int, needles: int, seed: int) -> dict[str, Any]:
    generated_needles = generate_needles(needles, seed)
    haystack = generate_haystack(tokens, generated_needles, seed)
    validate_dataset(haystack, generated_needles)
    return {
        "tokens": tokens,
        "seed": seed,
        "haystack": haystack,
        "needles": [asdict(needle) for needle in generated_needles],
    }


def validate_dataset(haystack: str, needles: list[Needle]) -> None:
    seen_questions: set[str] = set()
    for needle in needles:
        if needle.expected not in needle.fact:
            raise AssertionError(f"expected answer not in fact for {needle.key}")
        if needle.fact not in haystack:
            raise AssertionError(f"fact missing from haystack for {needle.key}")
        if needle.question in seen_questions:
            raise AssertionError(f"duplicate question: {needle.question}")
        seen_questions.add(needle.question)
        if haystack.count(needle.expected) != 1:
            raise AssertionError(f"expected answer is not unique in haystack: {needle.expected}")


def validate_conversation_dataset(dataset: dict[str, Any]) -> None:
    transcript = conversation_transcript(dataset["conversation"])
    seen_questions: set[str] = set()
    for raw_needle in dataset["needles"]:
        needle = Needle(
            key=raw_needle["key"],
            fact=raw_needle["fact"],
            question=raw_needle["question"],
            expected=raw_needle["expected"],
        )
        if needle.expected not in needle.fact:
            raise AssertionError(f"expected answer not in fact for {needle.key}")
        if transcript.count(needle.expected) != 1:
            raise AssertionError(f"expected answer is not unique in conversation: {needle.expected}")
        markers = re.findall(r"needle\d{4}", needle.fact)
        for marker in markers:
            if transcript.count(marker) != 1:
                raise AssertionError(f"needle marker is not unique in conversation: {marker}")
        if needle.question in seen_questions:
            raise AssertionError(f"duplicate question: {needle.question}")
        seen_questions.add(needle.question)


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} returned {error.code}: {body}") from error


def run_contextforge_retrieval(
    sidecar_url: str,
    dataset: dict[str, Any],
    namespace: str,
    timeout: float,
) -> dict[str, Any]:
    sidecar_url = sidecar_url.rstrip("/")
    ingest = post_json(
        f"{sidecar_url}/ingest",
        {
            "namespace": {"namespace": namespace, "sessionId": "benchmark"},
            "text": dataset["haystack"],
            "title": f"needle-haystack-{dataset['tokens']}",
            "category": "benchmark",
            "metadata": {"source": "needle_haystack"},
        },
        timeout,
    )
    rows: list[dict[str, Any]] = []
    correct = 0
    for raw_needle in dataset["needles"]:
        started = time.perf_counter()
        recall = post_json(
            f"{sidecar_url}/recall",
            {
                "namespace": {"namespace": namespace, "sessionId": "benchmark"},
                "query": raw_needle["question"],
                "category": "benchmark",
                "maxTokens": 4096,
                "limit": 8,
            },
            timeout,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        context = recall.get("context", "")
        hit = raw_needle["expected"] in context
        correct += int(hit)
        rows.append(
            {
                "key": raw_needle["key"],
                "expected": raw_needle["expected"],
                "sourceHit": hit,
                "latencyMs": latency_ms,
                "sources": [source["id"] for source in recall.get("sources", [])],
                "tokens": recall.get("totalTokens", 0),
            }
        )
    return {
        "mode": "contextforge-retrieval",
        "ingest": ingest,
        "accuracy": correct / max(1, len(dataset["needles"])),
        "correct": correct,
        "total": len(dataset["needles"]),
        "rows": rows,
    }


def run_contextforge_conversation(
    sidecar_url: str,
    dataset: dict[str, Any],
    namespace: str,
    timeout: float,
    max_tokens: int,
    limit: int,
    native_window_tokens: int,
) -> dict[str, Any]:
    sidecar_url = sidecar_url.rstrip("/")
    ingest_started = time.perf_counter()
    ingest = post_json(
        f"{sidecar_url}/ingest",
        {
            "namespace": {"namespace": namespace, "sessionId": dataset["id"]},
            "text": conversation_transcript(dataset["conversation"]),
            "title": dataset["id"],
            "category": "benchmark-conversation",
            "metadata": {
                "source": "needle_haystack_conversation",
                "turns": dataset["turns"],
                "tokensPerTurn": dataset["tokensPerTurn"],
                "estimatedTokens": dataset["estimatedTokens"],
            },
        },
        timeout,
    )

    recent_native_context = native_tail_transcript(dataset["conversation"], native_window_tokens)
    rows: list[dict[str, Any]] = []
    correct = 0
    native_visible = 0
    for raw_needle in dataset["needles"]:
        started = time.perf_counter()
        recall = post_json(
            f"{sidecar_url}/recall",
            {
                "namespace": {"namespace": namespace, "sessionId": dataset["id"]},
                "query": raw_needle["question"],
                "category": "benchmark-conversation",
                "maxTokens": max_tokens,
                "limit": limit,
            },
            timeout,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        sources = recall.get("sources", [])
        source_ids = [source["id"] for source in sources]
        context = recall.get("context", "")
        hit = raw_needle["expected"] in context
        source_rank = answer_source_rank(context, sources, raw_needle["expected"])
        native_contains = raw_needle["expected"] in recent_native_context
        correct += int(hit)
        native_visible += int(native_contains)
        rows.append(
            {
                "key": raw_needle["key"],
                "placement": raw_needle["placement"],
                "turnIndex": raw_needle["turnIndex"],
                "expected": raw_needle["expected"],
                "sourceHit": hit,
                "sourceRank": source_rank,
                "nativeWindowContainsAnswer": native_contains,
                "latencyMs": latency_ms,
                "sources": source_ids,
                "tokens": recall.get("totalTokens", 0),
            }
        )

    return {
        "mode": "contextforge-conversation-decay",
        "namespace": namespace,
        "dataset": {
            "id": dataset["id"],
            "turns": dataset["turns"],
            "tokensPerTurn": dataset["tokensPerTurn"],
            "targetTokens": dataset["targetTokens"],
            "estimatedTokens": dataset["estimatedTokens"],
            "earlyTurns": dataset["earlyTurns"],
            "controlNeedles": dataset["controlNeedles"],
        },
        "ingest": {
            "count": ingest["count"],
            "idsPreview": ingest["ids"][:5],
            "latencyMs": int((time.perf_counter() - ingest_started) * 1000),
        },
        "recallMaxTokens": max_tokens,
        "nativeWindowTokens": native_window_tokens,
        "contextForgeAccuracy": correct / max(1, len(dataset["needles"])),
        "nativeWindowVisibility": native_visible / max(1, len(dataset["needles"])),
        "correct": correct,
        "nativeVisible": native_visible,
        "total": len(dataset["needles"]),
        "rows": rows,
    }


def answer_source_rank(context: str, sources: list[dict[str, Any]], expected: str) -> int | None:
    if expected not in context:
        return None
    starts: list[tuple[int, int]] = []
    for index, source in enumerate(sources):
        marker = f"[{source['id']}]"
        position = context.find(marker)
        if position >= 0:
            starts.append((position, index))
    starts.sort()
    for ordered_index, (start, source_index) in enumerate(starts):
        end = starts[ordered_index + 1][0] if ordered_index + 1 < len(starts) else len(context)
        if expected in context[start:end]:
            return source_index + 1
    return None


def command_generate(args: argparse.Namespace) -> None:
    dataset = make_dataset(args.tokens, args.needles, args.seed)
    output = json.dumps(dataset, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


def command_self_test(args: argparse.Namespace) -> None:
    for tokens in [1_000, 10_000, 50_000]:
        dataset = make_dataset(tokens, args.needles, args.seed)
        validate_dataset(dataset["haystack"], [Needle(**item) for item in dataset["needles"]])
    conversation = make_conversation_dataset(
        turn_count=32,
        tokens_per_turn=250,
        needles=args.needles,
        control_needles=min(2, max(0, args.needles - 1)),
        seed=args.seed,
        early_turns=8,
    )
    validate_conversation_dataset(conversation)
    print("needle_haystack self-test passed")


def command_retrieval(args: argparse.Namespace) -> None:
    if args.dataset:
        dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    else:
        dataset = make_dataset(args.tokens, args.needles, args.seed)
    result = run_contextforge_retrieval(args.sidecar_url, dataset, args.namespace, args.timeout)
    print(json.dumps(result, indent=2))


def command_conversation(args: argparse.Namespace) -> None:
    if args.dataset:
        dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
        validate_conversation_dataset(dataset)
    else:
        dataset = make_conversation_dataset(
            turn_count=args.turns,
            tokens_per_turn=args.tokens_per_turn,
            needles=args.needles,
            control_needles=args.control_needles,
            seed=args.seed,
            early_turns=args.early_turns,
        )
    if args.output_dataset:
        Path(args.output_dataset).write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    run_id = args.run_id or dataset["id"]
    namespace = f"{args.namespace.rstrip('/')}/{run_id}"
    result = run_contextforge_conversation(
        args.sidecar_url,
        dataset,
        namespace,
        args.timeout,
        args.max_tokens,
        args.limit,
        args.native_window_tokens,
    )
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Needle-in-a-haystack benchmark for ContextForge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate a deterministic dataset")
    generate.add_argument("--tokens", type=int, default=10_000)
    generate.add_argument("--needles", type=int, default=10)
    generate.add_argument("--seed", type=int, default=701)
    generate.add_argument("--output")
    generate.set_defaults(func=command_generate)

    self_test = subparsers.add_parser("self-test", help="Validate deterministic generator invariants")
    self_test.add_argument("--needles", type=int, default=10)
    self_test.add_argument("--seed", type=int, default=701)
    self_test.set_defaults(func=command_self_test)

    retrieval = subparsers.add_parser("retrieval", help="Run retrieval-only benchmark against sidecar")
    retrieval.add_argument("--sidecar-url", default="http://localhost:8765")
    retrieval.add_argument("--namespace", default="openclaw/benchmark/needle")
    retrieval.add_argument("--dataset")
    retrieval.add_argument("--tokens", type=int, default=10_000)
    retrieval.add_argument("--needles", type=int, default=10)
    retrieval.add_argument("--seed", type=int, default=701)
    retrieval.add_argument("--timeout", type=float, default=30.0)
    retrieval.set_defaults(func=command_retrieval)

    conversation = subparsers.add_parser(
        "conversation",
        help="Run a multi-turn conversation decay benchmark against sidecar recall",
    )
    conversation.add_argument("--sidecar-url", default="http://localhost:8765")
    conversation.add_argument("--namespace", default="openclaw/benchmark/conversation")
    conversation.add_argument("--run-id")
    conversation.add_argument("--dataset")
    conversation.add_argument("--output-dataset")
    conversation.add_argument("--turns", type=int, default=100)
    conversation.add_argument("--tokens-per-turn", type=int, default=12_000)
    conversation.add_argument("--needles", type=int, default=12)
    conversation.add_argument("--control-needles", type=int, default=2)
    conversation.add_argument("--early-turns", type=int, default=10)
    conversation.add_argument("--seed", type=int, default=701)
    conversation.add_argument("--timeout", type=float, default=30.0)
    conversation.add_argument("--max-tokens", type=int, default=4096)
    conversation.add_argument("--limit", type=int, default=8)
    conversation.add_argument("--native-window-tokens", type=int, default=40_960)
    conversation.set_defaults(func=command_conversation)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
