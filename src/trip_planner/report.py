"""Render searched trip data as a human-readable itinerary without another model call."""

from datetime import date, timedelta
from decimal import Decimal


def _text(value) -> str:
    return str(value).replace("\n", " ").replace("\r", " ").replace("|", "\\|")


def _date(value: str) -> str:
    return date.fromisoformat(value).strftime("%b %d, %Y")


def _money(value, currency: str) -> str:
    return f"{Decimal(str(value)):,.2f} {currency}"


def _airport(airport: dict) -> str:
    return _text(airport.get("id") or airport.get("name") or "Airport unspecified")


def _schedule(airport: dict) -> str:
    day = _date(airport["date"]) if airport.get("date") else "Date unspecified"
    return f"{day}, {_text(airport.get('time') or 'time unspecified')}"


def _usage_lines(result: dict) -> list[str]:
    usage = result.get("token_usage")
    if usage is None:
        return []
    lines = [
        "",
        "## LLM token usage",
        "",
        (
            f"**Total:** {usage['total_tokens']:,} tokens "
            f"({usage['input_tokens']:,} input + {usage['output_tokens']:,} output) "
            f"across {result['model_calls']} model turns."
        ),
    ]
    if not usage["complete"]:
        lines += [
            "",
            (
                "Partial usage: some model calls did not return token usage; "
                "the total includes only reported tokens."
            ),
        ]
    return lines


def render_itinerary(result: dict) -> str:
    """Render successful itineraries and useful missing-input/error reports."""
    status = result["status"]
    trip = result.get("parameters", {})
    lines = ["# Trip itinerary report", ""]
    if status != "success":
        lines += [f"**Status:** {status.replace('_', ' ').capitalize()}", ""]
        if result.get("message"):
            lines += [_text(result["message"]), ""]
        if result.get("questions"):
            lines += ["## Details needed", ""]
            lines += [f"- {_text(question)}" for question in result["questions"]]
            lines.append("")
        if result.get("errors"):
            lines += ["## Corrections needed", ""]
            for error in result["errors"]:
                if isinstance(error, dict):
                    lines.append(f"- {_text(error['field'])}: {_text(error['message'])}")
                else:
                    lines.append(f"- {_text(error)}")
            lines.append("")
        if result.get("candidates"):
            lines += ["## Location choices", ""]
            lines += [f"- {_text(c['name'])} ({_text(c['id'])})" for c in result["candidates"]]
            lines.append("")
        if result.get("cheapest_matching_total"):
            currency = trip["currency"]
            lines += [
                f"Lowest matching package: **{_money(result['cheapest_matching_total'], currency)}**.",
                f"Budget shortfall: **{_money(result['budget_shortfall'], currency)}**.",
                "",
            ]
        return "\n".join(lines + _usage_lines(result)).rstrip() + "\n"

    choice = result["recommendation"]
    flight, hotel = choice["flight"], choice["hotel"]
    currency = choice["currency"]
    nights = (date.fromisoformat(hotel["check_out"]) - date.fromisoformat(hotel["check_in"])).days
    lines += [
        f"## {_text(trip['origin'])} → {_text(trip['destination'])}",
        "",
        f"**Travel dates:** {_date(trip['departure_date'])}–{_date(trip['return_date'])}  ",
        f"**Travelers:** {trip['adults']} adult(s) · {hotel['rooms']} room · {nights} hotel nights  ",
        f"**Proposed trip total:** {_money(choice['total'], currency)}  ",
        f"**Budget remaining:** {_money(choice['budget_remaining'], currency)}",
        "",
        "## Flight itinerary",
        "",
        "All times are local to the airport shown.",
        "",
    ]
    legs = flight.get("itineraries", [])
    if not legs:
        lines += ["Flight schedules are unavailable in the synthetic demo.", ""]
    for leg in legs:
        label = "Outbound" if leg.get("direction") == "outbound" else "Return"
        lines += [
            f"### {label}",
            "",
            "| Flight | Route | Departure | Arrival |",
            "| --- | --- | --- | --- |",
        ]
        for segment in leg["segments"]:
            departure, arrival = segment["departure_airport"], segment["arrival_airport"]
            name = " ".join(str(segment.get(k, "")) for k in ["airline", "flight_number"]).strip()
            lines.append(
                f"| {_text(name or 'Flight unspecified')} | {_airport(departure)} → {_airport(arrival)} | {_schedule(departure)} | {_schedule(arrival)} |"
            )
        lines.append("")
    hours, minutes = divmod(flight["duration_minutes"], 60)
    lines += [
        (
            f"**Cabin:** {flight['cabin'].replace('_', ' ').title()} · "
            f"**Maximum stops per direction:** {flight['stops']} · "
            f"**Total flight itinerary time:** {hours}h {minutes}m"
        ),
        "",
        "## Hotel stay",
        "",
        f"**{_text(hotel['name'])}**",
        "",
        f"- Check-in: {_date(hotel['check_in'])}",
        f"- Check-out: {_date(hotel['check_out'])}",
        f"- Stay: {nights} nights, {hotel['rooms']} room, {hotel['adults']} adult(s)",
        f"- Hotel class: {str(hotel['stars']) + ' stars' if hotel.get('stars') else 'Unspecified'}",
        f"- Room: {_text(hotel['room'])}",
        "",
    ]
    if hotel.get("link", "") and hotel["link"].startswith(("https://", "http://")):
        lines += [f"[Hotel website]({hotel['link'].replace(')', '%29')})", ""]

    lines += [
        "## Stay overview",
        "",
        f"- **{_date(trip['departure_date'])}:** Depart {_text(trip['origin'])}.",
        f"- **{_date(flight['arrival_date'])}:** Arrive at your destination.",
        f"- **{_date(hotel['check_in'])}:** Check in to {_text(hotel['name'])}.",
    ]
    first_full_day = date.fromisoformat(hotel["check_in"]) + timedelta(days=1)
    last_full_day = date.fromisoformat(hotel["check_out"]) - timedelta(days=1)
    if first_full_day <= last_full_day:
        lines.append(
            f"- **{_date(first_full_day.isoformat())}–{_date(last_full_day.isoformat())}:** Free time in {_text(trip['destination'])}."
        )
    lines += [
        f"- **{_date(hotel['check_out'])}:** Check out of the hotel.",
        f"- **{_date(flight['return_departure_date'])}:** Take the return flight.",
        "",
        "## Cost breakdown",
        "",
        "| Item | Total |",
        "| --- | ---: |",
        f"| Round-trip flights for {trip['adults']} adults | {_money(flight['total'], currency)} |",
        f"| Hotel for {nights} nights | {_money(hotel['total'], currency)} |",
        f"| **Trip total** | **{_money(choice['total'], currency)}** |",
        f"| Budget | {_money(trip['budget'], currency)} |",
        f"| Remaining | {_money(choice['budget_remaining'], currency)} |",
        "",
        "## Why this option",
        "",
    ]
    weight = trip["comfort_weight"]
    preference = (
        "price" if weight < 0.5 else "comfort" if weight > 0.5 else "a balance of price and comfort"
    )
    lines += [
        _text(result["decision_reason"])
        if result.get("decision_reason")
        else (
            f"This package ranks highest among the searched options for your preference for {preference}. "
            "It meets your budget, cabin, stop limit, hotel class and stay dates. "
            "Comfort is assessed using stops, travel duration and hotel stars."
        ),
        "",
    ]
    alternatives = result.get("alternatives", [])
    if alternatives:
        lines += [
            "## Alternatives",
            "",
            "| Hotel | Flight stops (maximum) | Trip total | Difference |",
            "| --- | ---: | ---: | ---: |",
        ]
        for alternative in alternatives:
            difference = Decimal(alternative["total"]) - Decimal(choice["total"])
            label = "less" if difference < 0 else "more" if difference > 0 else "same price"
            difference_text = (
                f"{_money(abs(difference), currency)} {label}" if difference else label
            )
            lines.append(
                f"| {_text(alternative['hotel']['name'])} | {alternative['flight']['stops']} | {_money(alternative['total'], currency)} | {difference_text} |"
            )
        lines.append("")
    lines += [
        "## Search notes",
        "",
        f"Source: {_text(result['source'])}. Searched at {_text(result['searched_at'])}.",
        "",
    ]
    if result.get("search_counts"):
        counts = result["search_counts"]
        lines += [
            f"Compared {counts['flights']} round-trip flight options and {counts['hotels']} hotel offers.",
            "",
        ]
    lines += [f"- {_text(warning)}" for warning in result.get("warnings", [])]
    lines += ["- This is a proposed itinerary; flights and hotel have not been booked."]
    if result.get("agent_trace"):
        lines += ["", "## Agent actions", ""]
        lines += [
            f"{index}. {step['tool']} — {step['status']}"
            for index, step in enumerate(result["agent_trace"], 1)
        ]
    return "\n".join(lines + _usage_lines(result)).rstrip() + "\n"
