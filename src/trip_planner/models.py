from datetime import date
from decimal import Decimal
from typing import Literal

import pycountry
from pydantic import BaseModel, ConfigDict, Field, field_validator

COUNTRY_NAMES = frozenset(
    name.casefold()
    for country in pycountry.countries
    for attribute in ("name", "official_name", "common_name")
    if (name := getattr(country, attribute, None))
)


class TripParameters(BaseModel):
    """Unknown required parameters stay null rather than being invented."""

    model_config = ConfigDict(extra="forbid")
    origin: str | None = Field(default=None, description="Departure city or IATA code")
    destination: str | None = Field(default=None, description="Destination city, not just country")
    departure_date: date | None = None
    return_date: date | None = None
    hotel_check_in: date | None = Field(default=None, description="Local hotel arrival date")
    hotel_check_out: date | None = Field(default=None, description="Local hotel departure date")
    adults: int | None = Field(default=None, ge=1, le=9)
    rooms: int | None = Field(default=None, ge=1, le=9)
    budget: Decimal | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description="Total flights plus hotels budget for all travelers",
    )
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    comfort_weight: float = Field(
        default=0.5, ge=0, le=1, description="0 prioritizes price, 1 prioritizes comfort"
    )
    cabin: Literal["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"] = "ECONOMY"
    max_stops: int = Field(default=2, ge=0, le=2, description="Maximum stops per direction")
    min_hotel_stars: int = Field(default=0, ge=0, le=5)
    unsupported_reason: str | None = Field(
        default=None, description="Reason the request requires unsupported trip features"
    )

    @field_validator("origin", "destination")
    @classmethod
    def require_city(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return None if value.casefold() in COUNTRY_NAMES else value

    def issues(self, today: date) -> tuple[list[str], list[str]]:
        required = [
            "origin",
            "destination",
            "departure_date",
            "return_date",
            "hotel_check_in",
            "hotel_check_out",
            "adults",
            "rooms",
            "budget",
            "currency",
        ]
        missing = [
            key
            for key in required
            if getattr(self, key) is None
            or (isinstance(getattr(self, key), str) and not getattr(self, key).strip())
        ]
        invalid = []
        if self.unsupported_reason:
            invalid.append(self.unsupported_reason)
        if self.departure_date and self.departure_date < today:
            invalid.append("departure_date must be today or later")
        if self.return_date and self.departure_date and self.return_date <= self.departure_date:
            invalid.append("return_date must be after departure_date")
        if (
            self.hotel_check_in
            and self.departure_date
            and self.hotel_check_in < self.departure_date
        ):
            invalid.append("hotel_check_in must be on or after departure_date")
        if (
            self.hotel_check_out
            and self.hotel_check_in
            and self.hotel_check_out <= self.hotel_check_in
        ):
            invalid.append("hotel_check_out must be after hotel_check_in")
        if self.hotel_check_out and self.return_date and self.hotel_check_out > self.return_date:
            invalid.append("hotel_check_out must be on or before return_date")
        if (
            self.origin
            and self.destination
            and self.origin.strip().casefold() == self.destination.strip().casefold()
        ):
            invalid.append("origin and destination must differ")
        if self.rooms is not None and self.rooms != 1:
            invalid.append(
                "SearchAPI hotel search currently supports one room; multi-room availability cannot be verified"
            )
        if self.adults is not None and self.adults > 6:
            invalid.append("SearchAPI hotel search supports up to 6 adults in one room")
        return missing, invalid


class Flight(BaseModel):
    id: str
    total: Decimal = Field(gt=0, allow_inf_nan=False)
    currency: str
    stops: int = Field(ge=0)  # Maximum in either direction.
    duration_minutes: int = Field(gt=0)  # Total round-trip itinerary duration.
    cabin: str
    itineraries: list[dict] = Field(default_factory=list)
    arrival_date: date | None = None
    return_departure_date: date | None = None


class Hotel(BaseModel):
    id: str
    name: str
    total: Decimal = Field(gt=0, allow_inf_nan=False)  # Entire stay, all requested rooms.
    currency: str
    stars: int | None = Field(default=None, ge=1, le=5)
    link: str | None = None
    room: str = "Unspecified"
    check_in: date
    check_out: date
    rooms: int = Field(ge=1)
    adults: int = Field(ge=1)
