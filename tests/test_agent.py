from datetime import date

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from trip_planner.agent import PlanningSession, build_agent, make_model
from trip_planner.demo import DemoProvider, demo_parameters
from trip_planner.models import TripParameters


class Decisions:
    """A small model stand-in to test tool-loop routing, not travel search responses."""

    def __init__(self, steps):
        self.steps = iter(steps)
        self.calls = 0

    def bind_tools(self, tools, **kwargs):
        assert {t.name for t in tools} == {
            "validate_trip",
            "resolve_locations",
            "search_flights",
            "search_hotels",
            "compare_options",
            "finish_itinerary",
        }
        return self

    def invoke(self, messages):
        self.calls += 1
        step = next(self.steps)
        if callable(step):
            step = step(messages)
        name, args = step
        return AIMessage(
            content="", tool_calls=[{"name": name, "args": args, "id": f"call-{self.calls}"}]
        )


def test_default_model_uses_anthropic_without_deprecated_sampling(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("TRIP_PLANNER_MODEL", raising=False)
    model = make_model()
    assert model.model == "claude-haiku-5-5"
    assert model.temperature is None
    assert model.max_tokens == 4096
    bound = model.bind_tools(
        PlanningSession(DemoProvider(), date.today()).tools(), parallel_tool_calls=False
    )
    assert bound.kwargs["tool_choice"]["disable_parallel_tool_use"] is True


def choose_last(messages):
    import json

    receipt = json.loads(messages[-1].content)
    return "finish_itinerary", {
        "candidate_id": receipt["candidates"][-1]["candidate_id"],
        "rationale": "Selected this verified package after comparing the tradeoffs.",
    }


def test_model_controls_search_order_research_and_final_selection():
    model = Decisions(
        [
            ("validate_trip", {"parameters": demo_parameters()}),
            ("resolve_locations", {}),
            ("search_hotels", {"sort_by": "lowest_price"}),
            ("search_flights", {"max_stops": 0}),
            ("search_flights", {"sort_by": "price", "max_stops": 1}),
            ("compare_options", {}),
            choose_last,
        ]
    )
    output = build_agent(DemoProvider(), model=model).invoke({"parameters": demo_parameters()})
    result = output["result"]
    assert result["status"] == "success"
    assert result["model_calls"] == 7
    assert [x["tool"] for x in result["agent_trace"]][2:5] == [
        "search_hotels",
        "search_flights",
        "search_flights",
    ]
    assert result["recommendation"]["score"] < result["alternatives"][0]["score"]
    assert "Selected this verified package" in result["report"]
    assert any(isinstance(m, ToolMessage) for m in output["messages"])


def test_missing_parameters_cannot_trigger_search():
    model = Decisions(
        [
            ("validate_trip", {"parameters": {"destination": "Japan"}}),
            ("search_flights", {}),
            ("finish_itinerary", {}),
        ]
    )
    result = build_agent(DemoProvider(), model=model).invoke({"request": "plan a trip to japan"})[
        "result"
    ]
    assert result["status"] == "missing_parameters"
    assert "destination" in result["missing_parameters"]
    assert result["search_counts"] == {"flights": 0, "hotels": 0}


def test_country_validation_preserves_airport_codes():
    assert TripParameters(origin="CAN", destination=" Japan ").origin == "CAN"
    assert TripParameters(destination=" Japan ").destination is None


@pytest.mark.parametrize("missing_usage", [False, True])
def test_usage_sums_all_turns_and_marks_unreported_tokens(missing_usage):
    class MeteredDecisions(Decisions):
        def invoke(self, messages):
            reply = super().invoke(messages)
            if missing_usage and self.calls == 2:
                return reply
            reply.usage_metadata = {
                "input_tokens": 100 * self.calls,
                "output_tokens": 10 * self.calls,
                "total_tokens": 110 * self.calls,
                "input_token_details": {"cache_read": 40},
            }
            return reply

    model = MeteredDecisions(
        [
            ("validate_trip", {"parameters": {}}),
            ("finish_itinerary", {}),
        ]
    )
    result = build_agent(DemoProvider(), model=model).invoke({"request": "Plan a trip"})["result"]
    multiplier = 1 if missing_usage else 3
    assert result["token_usage"] == {
        "input_tokens": 100 * multiplier,
        "output_tokens": 10 * multiplier,
        "total_tokens": 110 * multiplier,
        "reported_model_calls": 1 if missing_usage else 2,
        "complete": not missing_usage,
    }
    assert result["model_usage"][0]["usage"]["input_token_details"]["cache_read"] == 40
    assert f"**Total:** {110 * multiplier} tokens" in result["report"]
    assert ("Partial usage" in result["report"]) == missing_usage


def test_usage_survives_a_later_model_failure():
    class FailingDecisions(Decisions):
        def invoke(self, messages):
            if self.calls:
                raise ValueError("Model unavailable")
            reply = super().invoke(messages)
            reply.usage_metadata = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
            return reply

    model = FailingDecisions([("validate_trip", {"parameters": {}})])
    result = build_agent(DemoProvider(), model=model).invoke({"request": "Plan a trip"})["result"]
    assert result["status"] == "error"
    assert result["model_calls"] == 2
    assert result["token_usage"]["total_tokens"] == 110
    assert result["token_usage"]["complete"] is False
    assert "Partial usage" in result["report"]


def test_model_cannot_finish_with_a_fabricated_candidate():
    model = Decisions(
        [("finish_itinerary", {"candidate_id": "invented", "rationale": "Cheap"})] * 2
    )
    result = build_agent(DemoProvider(), model=model, max_steps=2).invoke(
        {"parameters": demo_parameters()}
    )["result"]
    assert result["status"] == "error"
    assert "limit" in result["message"]
    assert "recommendation" not in result


@pytest.mark.parametrize(
    "patch",
    [
        {"departure_date": "2020-01-01"},
        {"adults": 3, "rooms": 2},
        {"unsupported_reason": "Children unsupported"},
    ],
)
def test_invalid_trip_is_blocked(patch):
    session = PlanningSession(DemoProvider(), date.today())
    assert (
        session.validate(TripParameters(**(demo_parameters() | patch)))["status"]
        == "invalid_parameters"
    )
    assert session.resolve()["status"] == "invalid_parameters"
    assert session.searches == 0


def test_supplied_parameters_override_model_arguments_and_cannot_be_changed():
    session = PlanningSession(
        DemoProvider(), date.today(), parameters=demo_parameters() | {"budget": "1000"}
    )
    session.validate(TripParameters(**demo_parameters()))
    assert str(session.trip.budget) == "1000"
    session.validate(TripParameters(**(demo_parameters() | {"budget": "9000"})))
    assert str(session.trip.budget) == "1000"


def test_search_cache_limits_and_hard_constraint_guards():
    session = PlanningSession(DemoProvider(), date.today(), max_searches=2)
    session.validate(TripParameters(**(demo_parameters() | {"max_stops": 0})))
    session.resolve()
    assert session.search("flights", "price", 1)["status"] == "blocked"
    assert not session.flights
    session.search("flights", "price", 0)
    assert session.search("flights", "price", 0)["cached"] is True
    assert session.search("hotels", "lowest_price")["status"] == "searched"
    assert session.search("hotels", "highest_rating")["status"] == "blocked"


def test_over_budget_packages_cannot_be_selected():
    session = PlanningSession(DemoProvider(), date.today())
    session.validate(TripParameters(**(demo_parameters() | {"budget": "1000"})))
    session.resolve()
    session.search("flights", "price")
    session.search("hotels", "lowest_price")
    comparison = session.compare()
    assert comparison["status"] == "no_options"
    assert not session.candidates
    session.finish(None, "No eligible package")
    assert session.result["budget_shortfall"] == "155"
