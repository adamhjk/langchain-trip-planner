"""Read-only SearchAPI adapters for Google Flights and Google Hotels."""

import os
import time
from datetime import date
from decimal import Decimal
from typing import Protocol

import httpx
from pydantic import ValidationError

from .models import Flight, Hotel, TripParameters


class SearchError(Exception):
    pass


class LocationError(SearchError):
    def __init__(self, field: str, message: str, candidates: list[dict]):
        super().__init__(message)
        self.field = field
        self.candidates = candidates


class TravelProvider(Protocol):
    source: str
    warnings: list[str]

    def resolve(self, location: str, field: str) -> tuple[str, str]: ...
    def flights(
        self, trip: TripParameters, origin: str, destination: str, *, sort_by: str = "top_flights"
    ) -> list[Flight]: ...
    def hotels(
        self, trip: TripParameters, city: str, *, sort_by: str = "relevance"
    ) -> list[Hotel]: ...


CABINS = {
    "ECONOMY": "economy",
    "PREMIUM_ECONOMY": "premium_economy",
    "BUSINESS": "business",
    "FIRST": "first_class",
}


def normalize_round_trip(outbound: dict, inbound: dict, trip: TripParameters) -> Flight:
    """Return search prices cover the whole round trip and requested party, not one leg."""
    segments = [outbound["flights"], inbound["flights"]]
    allowed = CABINS[trip.cabin].replace("_", " ")
    labels = {allowed, "first"} if trip.cabin == "FIRST" else {allowed}
    for leg in segments:
        if not leg or any(s["travel_class"].casefold() not in labels for s in leg):
            raise ValueError("Incomplete or mixed-cabin itinerary")
    if date.fromisoformat(segments[0][0]["departure_airport"]["date"]) != trip.departure_date:
        raise ValueError("Unexpected departure date")
    if date.fromisoformat(segments[1][0]["departure_airport"]["date"]) != trip.return_date:
        raise ValueError("Unexpected return date")
    return Flight(
        id=inbound["booking_token"],
        total=Decimal(str(inbound["price"])),
        currency=trip.currency,
        cabin=trip.cabin,
        stops=max(len(leg) - 1 + sum(len(s.get("stops", [])) for s in leg) for leg in segments),
        duration_minutes=int(outbound["total_duration"]) + int(inbound["total_duration"]),
        itineraries=[
            {"direction": "outbound", "segments": segments[0]},
            {"direction": "return", "segments": segments[1]},
        ],
        arrival_date=segments[0][-1]["arrival_airport"]["date"],
        return_departure_date=segments[1][0]["departure_airport"]["date"],
    )


def normalize_hotel(property_data: dict, trip: TripParameters) -> Hotel:
    """Use the tax-inclusive full-stay quote; never substitute a nightly/base-only price."""
    if property_data.get("type") != "hotel" or trip.rooms != 1:
        raise ValueError("Only single-room hotel quotes are supported")
    return Hotel(
        id=property_data["property_token"],
        name=property_data["name"],
        total=Decimal(str(property_data["total_price"]["extracted_price"])),
        currency=trip.currency,
        stars=property_data.get("extracted_hotel_class"),
        check_in=trip.hotel_check_in,
        check_out=trip.hotel_check_out,
        rooms=1,
        adults=trip.adults,
        room="Room type must be confirmed with the supplier",
        link=property_data.get("link"),
    )


class SearchAPIProvider:
    source = "SearchAPI (Google Flights and Google Hotels)"

    def __init__(self, client: httpx.Client | None = None, api_key: str | None = None):
        self.api_key = api_key or os.getenv("SEARCHAPI_API_KEY")
        self.client = client or httpx.Client(timeout=60)
        self.warnings = [
            "Search examines return options for up to 3 outbound flights and one page of hotels; prices must be rechecked before booking.",
            "Quoted totals exclude meals, transfers and extras not included by the supplier.",
        ]

    def close(self):
        self.client.close()

    def _get(self, engine: str, params: dict) -> dict:
        if not self.api_key:
            raise SearchError("Set SEARCHAPI_API_KEY in .env for live flight and hotel searches.")
        for attempt in range(3):
            response = self.client.get(
                "https://www.searchapi.io/api/v1/search",
                params={"engine": engine, **params},
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            # Don't retry quota/authentication failures; only transient server failures.
            if response.status_code >= 500 and attempt < 2:
                time.sleep(2**attempt)
                continue
            if response.is_error:
                hint = " Check your API key." if response.status_code in {401, 403} else ""
                if response.status_code == 429:
                    hint = " Check your SearchAPI quota or rate limit."
                raise SearchError(f"SearchAPI {engine} failed (HTTP {response.status_code}).{hint}")
            data = response.json()
            if not isinstance(data, dict):
                raise SearchError(f"SearchAPI {engine} returned an invalid response.")
            if data.get("error"):
                raise SearchError(
                    f"SearchAPI {engine} reported a search error. Check inputs and account quota."
                )
            # When returned, ensure the provider honored the requested currency.
            actual_currency = data.get("search_parameters", {}).get("currency")
            if actual_currency and params.get("currency") and actual_currency != params["currency"]:
                raise SearchError("SearchAPI returned a different currency than requested.")
            return data
        raise SearchError("SearchAPI request failed after retries.")

    def resolve(self, location: str, field: str) -> tuple[str, str]:
        location = location.strip()
        data = self._get(
            "google_flights_location_search",
            {
                "q": location,
                "search_type": "departure" if field == "origin" else "arrival",
            },
        )
        candidates = []
        matches = []
        for item in data.get("locations", []):
            if item.get("type") == "city" and item.get("kgmid"):
                name = item.get("full_name") or item.get("short_name")
                candidate = {"name": name, "id": item["kgmid"]}
                candidates.append(candidate)
                if location.casefold() in {
                    str(name).casefold(),
                    str(item.get("short_name")).casefold(),
                }:
                    matches.append((item["kgmid"], name))
            airports = item.get("airports", [])
            if item.get("type") == "airport":
                airports = [item, *airports]
            for airport in airports:
                if airport.get("airport_code") == location.upper():
                    city = airport.get("city") or airport.get("short_name")
                    if city:
                        matches.append((airport["airport_code"], city))
        matches = list(dict.fromkeys(matches))
        if len(matches) == 1:
            return matches[0]
        raise LocationError(
            field, f"Specify an unambiguous {field} city or airport IATA code.", candidates
        )

    def flights(
        self, trip: TripParameters, origin: str, destination: str, *, sort_by: str = "top_flights"
    ) -> list[Flight]:
        params = {
            "departure_id": origin,
            "arrival_id": destination,
            "flight_type": "round_trip",
            "sort_by": sort_by,
            "outbound_date": trip.departure_date.isoformat(),
            "return_date": trip.return_date.isoformat(),
            "adults": trip.adults,
            "currency": trip.currency,
            "travel_class": CABINS[trip.cabin],
            "stops": ["nonstop", "one_stop_or_fewer", "two_stops_or_fewer"][trip.max_stops],
            "hl": "en",
        }
        data = self._get("google_flights", params)
        options = data.get("best_flights", []) + data.get("other_flights", [])
        outbound_options = []
        seen = set()
        for option in options:
            token = option.get("departure_token")
            if token and token not in seen:
                outbound_options.append(option)
                seen.add(token)
        results = []
        for outbound in outbound_options[:3]:
            returns = self._get(
                "google_flights", {**params, "departure_token": outbound["departure_token"]}
            )
            inbound_options = returns.get("best_flights", []) + returns.get("other_flights", [])
            for inbound in inbound_options:
                try:
                    results.append(normalize_round_trip(outbound, inbound, trip))
                except (KeyError, ValueError, TypeError, IndexError, ValidationError):
                    self.warnings.append(
                        "Skipped an incomplete or invalid round-trip flight quote."
                    )
        return results

    def hotels(self, trip: TripParameters, city: str, *, sort_by: str = "relevance") -> list[Hotel]:
        if trip.rooms != 1 or trip.adults > 6:
            raise SearchError("SearchAPI hotel search supports one room with up to 6 adults.")
        params = {
            "q": f"Hotels in {city}",
            "check_in_date": trip.hotel_check_in.isoformat(),
            "check_out_date": trip.hotel_check_out.isoformat(),
            "adults": trip.adults,
            "currency": trip.currency,
            "property_type": "hotel",
            "sort_by": sort_by,
            "hl": "en",
        }
        if trip.min_hotel_stars:
            params["hotel_class"] = ",".join(str(x) for x in range(trip.min_hotel_stars, 6))
        data = self._get("google_hotels", params)
        results = []
        for property_data in data.get("properties", []):
            try:
                results.append(normalize_hotel(property_data, trip))
            except (KeyError, ValueError, TypeError, ValidationError):
                self.warnings.append(
                    "Skipped a hotel without a valid full-stay price or hotel details."
                )
        return results
