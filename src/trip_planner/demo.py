"""Explicit synthetic search data for an offline workflow demonstration."""

from datetime import date, timedelta

from .models import Flight, Hotel, TripParameters
from .providers import LocationError


def demo_parameters() -> dict:
    departure = date.today() + timedelta(days=60)
    return {
        "origin": "SFO",
        "destination": "Tokyo",
        "departure_date": departure.isoformat(),
        "return_date": (departure + timedelta(days=8)).isoformat(),
        "hotel_check_in": (departure + timedelta(days=1)).isoformat(),
        "hotel_check_out": (departure + timedelta(days=8)).isoformat(),
        "adults": 1,
        "rooms": 1,
        "budget": "2500",
        "currency": "USD",
        "comfort_weight": 0.6,
    }


class DemoProvider:
    source = "Synthetic demo data (not live prices or availability)"

    def __init__(self):
        self.warnings = [self.source]

    def resolve(self, location: str, field: str) -> tuple[str, str]:
        cities = {
            "sfo": ("SFO", "SFO"),
            "san francisco": ("SFO", "SFO"),
            "tokyo": ("TYO", "TYO"),
            "tyo": ("TYO", "TYO"),
        }
        if location.casefold() not in cities:
            raise LocationError(field, "The demo only supports SFO to Tokyo.", [])
        return cities[location.casefold()]

    def flights(
        self, trip: TripParameters, origin: str, destination: str, *, sort_by: str = "top_flights"
    ) -> list[Flight]:
        if origin != "SFO" or destination != "TYO":
            return []
        return [
            Flight(
                id=offer_id,
                total=str(price * trip.adults),
                currency="USD",
                stops=stops,
                duration_minutes=duration,
                cabin="ECONOMY",
                arrival_date=trip.hotel_check_in,
                return_departure_date=trip.return_date,
            )
            for offer_id, price, stops, duration in [
                ("demo-cheap", 700, 1, 2300),
                ("demo-direct", 1100, 0, 1400),
                ("demo-expensive", 1900, 0, 1350),
            ]
        ]

    def hotels(self, trip: TripParameters, city: str, *, sort_by: str = "relevance") -> list[Hotel]:
        nights = (trip.hotel_check_out - trip.hotel_check_in).days
        return [
            Hotel(
                id=offer_id,
                name=name,
                total=str(price * nights * trip.rooms),
                currency="USD",
                stars=stars,
                room="Synthetic standard room",
                check_in=trip.hotel_check_in,
                check_out=trip.hotel_check_out,
                rooms=trip.rooms,
                adults=trip.adults,
            )
            for offer_id, name, price, stars in [
                ("demo-budget", "Demo Budget Inn", 65, 2),
                ("demo-comfort", "Demo Comfort Hotel", 140, 4),
                ("demo-luxury", "Demo Luxury Hotel", 300, 5),
            ]
        ]
