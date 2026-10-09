from datetime import UTC, date, datetime

from trip_planner.agent import PlanningSession
from trip_planner.demo import DemoProvider, demo_parameters
from trip_planner.models import TripParameters
from trip_planner.report import render_itinerary


def result_for(parameters):
    provider = DemoProvider()
    session = PlanningSession(provider, date.today())
    session.validate(TripParameters(**parameters))
    session.resolve()
    session.search("flights", "top_flights")
    session.search("hotels", "relevance")
    comparison = session.compare()
    candidate = comparison["candidates"][0]["candidate_id"] if comparison["candidates"] else None
    session.finish(candidate, "A comfortable package within budget.")
    result = session.result | {
        "parameters": session.trip.model_dump(mode="json"),
        "source": provider.source,
        "searched_at": datetime.now(UTC).isoformat(),
        "warnings": provider.warnings,
    }
    result["report"] = render_itinerary(result)
    return result


def test_report_covers_stay_and_total_without_inventing_demo_flights():
    result = result_for(demo_parameters())
    report = result["report"]
    assert "## Stay overview" in report
    assert "7 hotel nights" in report
    assert "2,080.00 USD" in report
    assert "Synthetic" in report
    assert "Flight schedules are unavailable" in report
    assert "## Alternatives" in report


def test_real_segment_schedule_preserves_overnight_arrival():
    result = result_for(demo_parameters())
    result["recommendation"]["flight"]["itineraries"] = [
        {
            "direction": "outbound",
            "segments": [
                {
                    "airline": "ZIPAIR",
                    "flight_number": "ZG 25",
                    "departure_airport": {"id": "SFO", "date": "2027-04-10", "time": "16:50"},
                    "arrival_airport": {"id": "NRT", "date": "2027-04-11", "time": "20:00"},
                }
            ],
        }
    ]
    report = render_itinerary(result)
    assert "ZIPAIR ZG 25" in report
    assert "SFO → NRT" in report
    assert "Apr 10, 2027, 16:50" in report
    assert "Apr 11, 2027, 20:00" in report
    assert "local to the airport" in report


def test_no_options_report_explains_budget_shortfall():
    result = result_for(demo_parameters() | {"budget": "1000"})
    assert "Lowest matching package: **1,155.00 USD**" in result["report"]
    assert "Budget shortfall: **155.00 USD**" in result["report"]
    assert "## Hotel stay" not in result["report"]
