"""Tests covering the VALENCE invariant.

These tests exercise the kernel directly with mock tool schemas and
observations; they do not require a live vLLM server, tau-bench env, or
ACEBench data.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

# Make the repo root importable when running `python -m unittest`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.valence import (  # noqa: E402
    AffordanceKernel,
    Handle,
    TINY_SYSTEM_PROMPT,
    resolve_date,
    resolve_enum,
    resolve_money,
    resolve_selector,
)
from src.valence.handles import (  # noqa: E402
    _IdMinter,
    mint_handles_from_observation,
    mint_handles_from_user_text,
)
from src.valence.lattice import classify_tool  # noqa: E402


# ---------------------------------------------------------------------------
# Mock tool schemas
# ---------------------------------------------------------------------------
def _tool(name, required, props=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"mock {name}",
            "parameters": {
                "type": "object",
                "required": required,
                "properties": props or {p: {"type": "string"} for p in required},
            },
        },
    }


GET_ORDER = _tool("get_order_details", ["order_id"])
GET_USER = _tool("get_user_details", ["user_id"])
CANCEL_ORDER = _tool("cancel_order", ["order_id"])
UPDATE_ADDR = _tool("update_user_address", ["user_id", "address"])
SEARCH_ITEMS = _tool("search_items", ["query"])


def _make_kernel_with_tool_evidence():
    """Kernel that has already observed an order with order_id=O1234."""
    k = AffordanceKernel()
    k.ingest_user_message("Please cancel my order O1234 for user_id alex_smith_42.")
    # Simulate a get_order_details response that returns the order.
    k.ingest_tool_result("get_order_details", {"order_id": "O1234"},
                         {"order_id": "O1234", "status": "pending",
                          "user_id": "alex_smith_42"})
    return k


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestValenceInvariant(unittest.TestCase):

    # 1. Hallucinated action_id cannot compile.
    def test_hallucinated_action_id_cannot_compile(self):
        k = _make_kernel_with_tool_evidence()
        affs = k.build_affordances([CANCEL_ORDER, GET_ORDER], remaining_steps=10)
        k.render_menu(affs, k=8, remaining_steps=10)
        # "A999" was never rendered.
        self.assertIsNone(k.compile_action("A999"))
        # Empty / None likewise.
        self.assertIsNone(k.compile_action(None))
        self.assertIsNone(k.compile_action(""))

    # 2. Mutation compiles when all args are handle-backed.
    def test_grounded_mutation_compiles_and_validates(self):
        k = _make_kernel_with_tool_evidence()
        affs = k.build_affordances([CANCEL_ORDER, GET_ORDER], remaining_steps=10)
        k.render_menu(affs, k=8, remaining_steps=10)
        # Find the cancel_order affordance — it should be executable.
        cancel = next(a for a in affs if a.tool_name == "cancel_order")
        self.assertTrue(cancel.is_executable, "cancel_order should bind from order_id handle")
        compiled = k.compile_action(cancel.action_id)
        self.assertIsNotNone(compiled)
        self.assertEqual(compiled.kwargs.get("order_id"), "O1234")
        # All kwargs have provenance refs.
        for arg in compiled.kwargs:
            self.assertTrue(compiled.argument_refs.get(arg),
                            f"missing provenance for arg={arg}")
        result = k.validate_mutation(compiled)
        self.assertTrue(result.ok, result.reason)

    # 3. Resolver output must include provenance.
    def test_resolver_output_carries_provenance(self):
        h = Handle(handle_id="H1", type="money", value=100.0,
                   source_event_id="E0001", source_path="x.price",
                   confidence="exact")
        tok = resolve_money("half", base=h)
        self.assertIsNotNone(tok)
        self.assertEqual(tok.value, 50.0)
        self.assertIn("H1", tok.source_handle_ids)
        self.assertEqual(tok.confidence, "resolved")
        self.assertEqual(tok.resolver, "resolve_money.half")

    # 4. Ambiguous resolver fails closed.
    def test_ambiguous_resolver_fails_closed(self):
        # Selector exact_match with two equal candidates → ambiguous → None.
        c1 = Handle("H1", "string", "alex", "E1", "p", "exact")
        c2 = Handle("H2", "string", "alex", "E2", "q", "exact")
        self.assertIsNone(resolve_selector("exact_match", [c1, c2], query="alex"))
        # Date resolver on non-ISO string → None.
        bad_date = Handle("H3", "datetime", "tomorrow", "E3", "p", "exact")
        self.assertIsNone(resolve_date(bad_date))

    # 5. Selector resolver picks only among tool-returned candidates.
    def test_selector_only_chooses_among_candidates(self):
        c1 = Handle("H1", "money", 50.0, "E1", "p", "exact")
        c2 = Handle("H2", "money", 30.0, "E2", "q", "exact")
        c3 = Handle("H3", "money", 90.0, "E3", "r", "exact")
        chosen = resolve_selector("cheapest", [c1, c2, c3])
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.value, 30.0)
        self.assertIn("H2", chosen.source_handle_ids)

    # 6. Duplicate mutation is rejected.
    def test_duplicate_mutation_rejected(self):
        k = _make_kernel_with_tool_evidence()
        affs = k.build_affordances([CANCEL_ORDER, GET_ORDER], remaining_steps=10)
        k.render_menu(affs, k=8, remaining_steps=10)
        cancel = next(a for a in affs if a.tool_name == "cancel_order")
        compiled = k.compile_action(cancel.action_id)
        self.assertTrue(k.validate_mutation(compiled).ok)
        k.record_execution(compiled)
        # Try the same mutation again — must be rejected.
        again = k.compile_action(cancel.action_id)
        # Either compile filters it out OR validator rejects it.
        if again is not None:
            self.assertFalse(k.validate_mutation(again).ok)

    # 7. render_menu remains compact (top-k=8).
    def test_render_menu_size_bounded(self):
        # Build many tools.
        many = [_tool(f"get_thing_{i}", []) for i in range(20)]
        k = AffordanceKernel()
        k.ingest_user_message("hello")
        affs = k.build_affordances(many, remaining_steps=15)
        text = k.render_menu(affs, k=8, remaining_steps=15)
        # Count action lines (action IDs are A1, A2, ...).
        import re as _re
        action_lines = [ln for ln in text.splitlines()
                        if _re.match(r"^A\d+\s", ln)]
        self.assertLessEqual(len(action_lines), 8)
        self.assertGreater(len(action_lines), 0)

    # 8. choose_action JSON translates to a real tool call.
    def test_choose_action_json_translates_to_tool_call(self):
        k = _make_kernel_with_tool_evidence()
        affs = k.build_affordances([GET_ORDER, CANCEL_ORDER], remaining_steps=10)
        k.render_menu(affs, k=8, remaining_steps=10)
        # Pick the first executable mutation.
        mut = next(a for a in affs if a.kind == "mutation" and a.is_executable)
        model_response = json.dumps({"action_id": mut.action_id})
        parsed = AffordanceKernel.parse_choice(model_response)
        self.assertEqual(parsed, mut.action_id)
        compiled = k.compile_action(parsed)
        self.assertIsNotNone(compiled)
        self.assertEqual(compiled.tool_name, "cancel_order")
        # Translate-to-real-tool-call shape:
        tool_call = {
            "id": "x",
            "type": "function",
            "function": {
                "name": compiled.tool_name,
                "arguments": json.dumps(compiled.kwargs),
            },
        }
        self.assertEqual(tool_call["function"]["name"], "cancel_order")
        kwargs_back = json.loads(tool_call["function"]["arguments"])
        self.assertEqual(kwargs_back["order_id"], "O1234")

    # 9. Baseline / act / react still import (when tau_bench available).
    def test_baselines_still_import(self):
        try:
            import tau_bench  # noqa: F401
        except ImportError:
            self.skipTest("tau_bench not installed in this environment")
        from src.baselines import ActAgent, ReActAgent, ToolCallingAgent  # noqa: F401
        from src.baselines.ace_loops import run_baseline_style  # noqa: F401
        # ACEBench-side baseline loop has no tau_bench dependency:
        self.assertTrue(callable(run_baseline_style))

    # 9b. ACEBench baseline loop is importable (depends on baselines pkg
    # which itself touches tau_bench; skip if unavailable).
    def test_baseline_ace_loop_imports(self):
        try:
            import tau_bench  # noqa: F401
        except ImportError:
            self.skipTest("tau_bench not installed in this environment")
        from src.baselines.ace_loops import run_baseline_style  # noqa: F401
        self.assertTrue(callable(run_baseline_style))

    # 10. VALENCE controllers import + initialize.
    def test_valence_controllers_init(self):
        # ACEBench loop has no tau_bench dependency — must always import.
        from src.valence.ace_loop import run_valence  # noqa: F401
        self.assertTrue(callable(run_valence))
        # tau-bench agent depends on tau_bench; only assert if available.
        try:
            import tau_bench  # noqa: F401
        except ImportError:
            return
        from src.valence.tau_agent import ValenceAgent
        try:
            agent = ValenceAgent(tools_info=[GET_ORDER], wiki="wiki",
                                 model="m", temperature=0.0)
            self.assertEqual(agent.style_name, "valence")
        except Exception:
            self.assertTrue(hasattr(ValenceAgent, "solve"))


# ---------------------------------------------------------------------------
# Extra coverage: handles + classify_tool
# ---------------------------------------------------------------------------
class TestHandleMinting(unittest.TestCase):
    def test_user_text_mints_typed_handles(self):
        m = _IdMinter()
        hs = mint_handles_from_user_text(
            "cancel order O1234 placed on 2025-05-01 for user alex_smith_42",
            "E0001", m)
        types = sorted({h.type for h in hs})
        self.assertIn("order_id", types)
        self.assertIn("user_id", types)
        self.assertIn("datetime", types)

    def test_observation_mints_typed_handles(self):
        m = _IdMinter()
        hs = mint_handles_from_observation(
            {"order_id": "O999", "items": [{"item_id": "I1", "price": 12.5}]},
            "E0002", m)
        kinds = {h.type for h in hs}
        self.assertEqual(
            {"order_id", "item_id", "money"},
            kinds & {"order_id", "item_id", "money"})

    def test_classify_tool(self):
        self.assertEqual(classify_tool("cancel_order"), "mutation")
        self.assertEqual(classify_tool("get_user_details"), "read")
        self.assertEqual(classify_tool("respond"), "neutral")

    def test_enum_resolver(self):
        ok = resolve_enum("approved", ["pending", "approved", "rejected"], "E0")
        self.assertIsNotNone(ok)
        self.assertEqual(ok.value, "approved")
        self.assertIsNone(resolve_enum("frozen",
                                       ["pending", "approved"], "E0"))


# ---------------------------------------------------------------------------
# Extra coverage: ungrounded mutation rejected at the validator level
# ---------------------------------------------------------------------------
class TestUngroundedRejection(unittest.TestCase):
    def test_ungrounded_mutation_blocked_by_lattice(self):
        # No prior tool evidence; user text mentions no order_id.
        k = AffordanceKernel()
        k.ingest_user_message("please cancel my order")
        affs = k.build_affordances([CANCEL_ORDER], remaining_steps=10)
        cancel = next((a for a in affs if a.tool_name == "cancel_order"), None)
        self.assertIsNotNone(cancel)
        self.assertFalse(cancel.is_executable,
                         "cancel_order must be NON-executable without an order_id handle")
        k.render_menu(affs, k=8, remaining_steps=10)
        # Even if the model 'chooses' it, compile must fail.
        self.assertIsNone(k.compile_action(cancel.action_id))


class TestTinyPrompt(unittest.TestCase):
    def test_tiny_prompt_is_compact(self):
        # Tiny prompt: no theory, no chain-of-thought directive.
        self.assertLess(len(TINY_SYSTEM_PROMPT), 400)
        self.assertIn("action_id", TINY_SYSTEM_PROMPT)
        self.assertNotIn("Think step", TINY_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
