from decimal import Decimal

import pytest

from trip_planner.demo import demo_parameters
from trip_planner.models import TripParameters
from trip_planner.providers import normalize_hotel, normalize_round_trip


def test_round_trip_uses_return_quote_once_for_whole_party():
    trip = TripParameters(**(demo_parameters() | {"adults": 2}))
    outbound = {
        "price": 900,
        "total_duration": 670,
        "flights": [
            {
                "travel_class": "Economy",
                "departure_airport": {"date": trip.departure_date.isoformat()},
                "arrival_airport": {"date": trip.hotel_check_in.isoformat()},
            }
        ],
    }
    inbound = {
        "price": 1250,
        "booking_token": "selected-round-trip",
        "total_duration": 575,
        "flights": [
            {
                "travel_class": "Economy",
                "departure_airport": {"date": trip.return_date.isoformat()},
                "arrival_airport": {"date": trip.return_date.isoformat()},
            }
        ],
    }
    flight = normalize_round_trip(outbound, inbound, trip)
    assert flight.total == Decimal(1250)  # No adding outbound price or multiplying by adults.
    assert flight.duration_minutes == 1245
    assert flight.arrival_date == trip.hotel_check_in
    assert len(flight.itineraries) == 2


def test_hotel_uses_full_stay_tax_inclusive_price():
    trip = TripParameters(**(demo_parameters() | {"adults": 2}))
    property_data = {
        "type": "hotel",
        "property_token": "h1",
        "name": "Example Hotel",
        "price_per_night": {"extracted_price": 100},
        "total_price": {"extracted_price": 840, "extracted_price_before_taxes": 700},
        "extracted_hotel_class": 4,
    }
    hotel = normalize_hotel(property_data, trip)
    assert hotel.total == Decimal(840)  # Already includes nights, taxes and both guests.
    assert hotel.adults == 2
    assert hotel.stars == 4
    del property_data["total_price"]
    with pytest.raises(KeyError):
        normalize_hotel(property_data, trip)
