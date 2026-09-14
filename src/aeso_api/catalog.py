"""Documented AESO endpoint contracts used by the download commands."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatedEndpoint:
    """Configuration for an AESO API operation queried by date range."""

    name: str
    path: str
    output_directory: str
    maximum_days: int
    date_format: str = "%Y-%m-%d"
    start_parameter: str = "startDate"
    end_parameter: str | None = "endDate"
    earliest_date: str | None = None
    delayed_days: int = 0
    end_date_inclusive: bool = True


ACTUAL_FORECAST = DatedEndpoint(
    name="Actual Forecast Report",
    path="/actualforecast-api/v1/load/albertaInternalLoad",
    output_directory="actual_forecast",
    maximum_days=366,
    earliest_date="2000-01-01",
)

GENERATION_CAPACITY = DatedEndpoint(
    name="AIES Generation Capacity",
    path="/aiesgencapacity-api/v1/AIESGenCapacity",
    output_directory="generation_capacity",
    # A 31-calendar-day range crossing the autumn clock change can contain
    # more than 31 * 24 returned hours and is rejected. Use 30 days safely.
    maximum_days=30,
    earliest_date="2011-01-01",
    # Observed responses exclude the supplied endDate despite the portal text.
    end_date_inclusive=False,
)

LOAD_OUTAGE_FORECAST = DatedEndpoint(
    name="Load Outage Forecast",
    path="/loadoutageforecast-api/v1/loadOutageReport",
    output_directory="load_outage_forecast",
    # AESO documents no request limit. Monthly chunks keep requests bounded.
    maximum_days=31,
    earliest_date="2013-09-22",
)

INTERTIE_OUTAGES = DatedEndpoint(
    name="Intertie Outage Report",
    path="/itc/v1/outage",
    output_directory="intertie_outages",
    # The endpoint permits 13 months. Annual chunks avoid month arithmetic.
    maximum_days=366,
    date_format="%Y%m%d",
    earliest_date="2020-11-09",
)

POOL_PRICE = DatedEndpoint(
    name="Pool Price Report",
    path="/poolprice-api/v1.1/price/poolPrice",
    output_directory="pool_price",
    maximum_days=366,
    earliest_date="2000-01-01",
)

SYSTEM_MARGINAL_PRICE = DatedEndpoint(
    name="System Marginal Price Report",
    path="/systemmarginalprice-api/v1.1/price/systemMarginalPrice",
    output_directory="system_marginal_price",
    # AESO describes a 183-day/4,392-hour limit, but some inclusive
    # 183-calendar-day requests are rejected. Use 182 days defensively.
    maximum_days=182,
    earliest_date="2000-01-01",
)

METERED_VOLUME = DatedEndpoint(
    name="Metered Volume Report",
    path="/meteredvolume-api/v1/meteredvolume/details",
    output_directory="metered_volume",
    maximum_days=16,
    earliest_date="2000-01-01",
)

ENERGY_MERIT_ORDER = DatedEndpoint(
    name="Energy Merit Order Report",
    path="/energymeritorder-api/v1/meritOrder/energy",
    output_directory="energy_merit_order",
    maximum_days=1,
    end_parameter=None,
    earliest_date="2009-09-01",
    delayed_days=60,
)

OPERATING_RESERVE_OFFER_CONTROL = DatedEndpoint(
    name="Operating Reserve Offer Control Report",
    path=(
        "/operatingreserveoffercontrol-api/v1/"
        "operatingReserveOfferControl"
    ),
    output_directory="operating_reserve_offer_control",
    maximum_days=1,
    end_parameter=None,
    earliest_date="2012-10-04",
    delayed_days=60,
)

UNIT_COMMITMENT = DatedEndpoint(
    name="Unit Commitment Data",
    path="/unitcommitmentdata-api/v2/unitCommitment",
    output_directory="unit_commitment",
    maximum_days=366,
    earliest_date="2024-07-01",
)


REFERENCE_ENDPOINTS = {
    "asset-list": (
        "/assetlist-api/v1/assetlist",
        "asset_list",
    ),
    "pool-participants": (
        "/PoolParticipant-api/v1/poolparticipantlist",
        "pool_participants",
    ),
}


ENDPOINTS = {
    "actual-forecast": ACTUAL_FORECAST,
    "generation-capacity": GENERATION_CAPACITY,
    "load-outages": LOAD_OUTAGE_FORECAST,
    "intertie-outages": INTERTIE_OUTAGES,
    "pool-price": POOL_PRICE,
    "smp": SYSTEM_MARGINAL_PRICE,
    "metered-volume": METERED_VOLUME,
    "merit-order": ENERGY_MERIT_ORDER,
    "operating-reserve-offers": OPERATING_RESERVE_OFFER_CONTROL,
    "unit-commitment": UNIT_COMMITMENT,
}
