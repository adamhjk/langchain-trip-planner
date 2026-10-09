from itertools import product

from .models import Flight, Hotel, TripParameters


def rank_options(
    trip: TripParameters, flights: list[Flight], hotels: list[Hotel], limit: int = 3
) -> dict:
    """Hard constraints first; deterministic weighted utility for feasible combinations."""
    flights = [
        f
        for f in flights
        if f.currency == trip.currency
        and f.stops <= trip.max_stops
        and f.cabin == trip.cabin
        and f.arrival_date is not None
        and f.arrival_date <= trip.hotel_check_in
        and f.return_departure_date is not None
        and f.return_departure_date >= trip.hotel_check_out
    ]
    hotels = [
        h
        for h in hotels
        if h.currency == trip.currency
        and h.rooms == trip.rooms
        and h.adults == trip.adults
        and h.check_in == trip.hotel_check_in
        and h.check_out == trip.hotel_check_out
        and (not trip.min_hotel_stars or (h.stars or 0) >= trip.min_hotel_stars)
    ]
    candidates = []
    cheapest = None
    shortest_duration = min((f.duration_minutes for f in flights), default=1)
    for flight, hotel in product(flights, hotels):
        total = flight.total + hotel.total
        cheapest = total if cheapest is None else min(cheapest, total)
        if total > trip.budget:
            continue
        # Comfort is a transparent proxy: connections, elapsed time, hotel class.
        connection_score = max(0, 1 - flight.stops / 3)
        duration_score = shortest_duration / flight.duration_minutes
        hotel_score = (hotel.stars / 5) if hotel.stars is not None else 0.4
        comfort = 0.3 * connection_score + 0.3 * duration_score + 0.4 * hotel_score
        price_score = float(1 - total / trip.budget)
        score = (1 - trip.comfort_weight) * price_score + trip.comfort_weight * comfort
        candidates.append(
            {
                "total": str(total),
                "currency": trip.currency,
                "budget_remaining": str(trip.budget - total),
                "score": round(score, 6),
                "comfort_score": round(comfort, 6),
                "price_score": round(price_score, 6),
                "flight": flight.model_dump(mode="json"),
                "hotel": hotel.model_dump(mode="json"),
                "reason": f"{flight.stops} stop(s) per direction at most; {flight.duration_minutes} total flight itinerary minutes; "
                f"hotel class {hotel.stars or 'unknown'}; total {total} {trip.currency} for all travelers and rooms.",
            }
        )
    candidates.sort(
        key=lambda x: (-x["score"], float(x["total"]), x["flight"]["id"], x["hotel"]["id"])
    )
    if not candidates:
        return {
            "status": "no_options",
            "recommendation": None,
            "alternatives": [],
            "message": "No flight/hotel combination meets the budget and hard preferences in the searched inventory.",
            "cheapest_matching_total": str(cheapest) if cheapest is not None else None,
            "budget_shortfall": str(cheapest - trip.budget) if cheapest is not None else None,
        }
    return {
        "status": "success",
        "recommendation": candidates[0],
        "alternatives": candidates[1:limit],
        "feasible_combinations": len(candidates),
    }
