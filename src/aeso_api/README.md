# AESO API downloaders

This package downloads raw responses from documented public AESO APIs. It does
not preprocess or merge the responses into the canonical master dataset.

## Authentication

Create an AESO developer subscription, then expose the key only in the shell
that launches the command:

```bash
export AESO_API_KEY="your-key"
```

Alternatively, put it in the ignored project-root `.env` file. Spaces around
the equals sign and matching single or double quotes are accepted:

```text
AESO_API_KEY = "your-key"
```

Never put the key in a notebook, source file, `.env` committed to Git, command
argument, raw response, or provenance sidecar.

## Core forecasting downloads

Run these from the project root after installing the project in editable mode.
The downloader automatically divides requests into each API's documented safe
date window, writes raw JSON atomically, skips valid existing files, retries
transient server failures, and writes a provenance sidecar with every response.

```bash
aeso-download range actual-forecast \
  --start-date 2015-01-01 --end-date 2026-06-30

aeso-download range generation-capacity \
  --start-date 2015-01-01 --end-date 2026-06-30

aeso-download range load-outages \
  --start-date 2015-01-01 --end-date 2026-06-30

aeso-download range intertie-outages \
  --start-date 2020-11-09 --end-date 2026-06-30

aeso-download range pool-price \
  --start-date 2015-01-01 --end-date 2026-06-30

aeso-download range smp \
  --start-date 2015-01-01 --end-date 2026-06-30

aeso-download snapshot asset-list
```

Raw files are stored below `data/raw/aeso_api/<dataset>/`. Date-range files
are named for their intended inclusive coverage. The generation-capacity
client compensates for that endpoint's observed exclusive-end behavior. Each
`.json` response has a corresponding
`.metadata.json` sidecar containing retrieval time, request parameters, byte
count, content type, and SHA-256 checksum. API keys are never recorded.

## Preprocessing

After all seven core downloads complete, normalize and audit them with:

```bash
aeso-api-preprocess
```

This writes separate Parquet products below
`data/preprocessing/aeso_api/` and audit tables below
`data/audits/preprocessing/aeso_api/`. It does not alter or merge into the
canonical master dataset. Existing actual AIL, pool price, and outage fields
are compared in the overlap audit rather than overwritten.

| Product | Grain | Primary role |
| --- | --- | --- |
| `aeso_actual_forecast_hourly.parquet` | Hour | Load forecast and signed forecast error |
| `aeso_generation_capacity_by_fuel_hourly.parquet` | Hour × fuel/sub-fuel | MC, AC, operational outage, and mothball outage |
| `aeso_generation_capacity_system_hourly.parquet` | Hour | System totals derived from the fuel-level report |
| `aeso_load_outage_forecast_hourly.parquet` | Hour | Forecast large-load outage volume |
| `aeso_intertie_outage_events.parquet` | Event × affected line | Normalized intertie outage intervals |
| `aeso_intertie_outage_hourly.parquet` | Active event-hour | Intertie outage flags and counts |
| `aeso_pool_price_forecast_hourly.parquet` | Hour | AESO pool-price forecast plus actual-price audit field |
| `aeso_smp_intervals.parquet` | Dispatch interval | Realized system marginal price intervals |
| `aeso_smp_hourly.parquet` | Hour | Duration-weighted SMP diagnostics |
| `aeso_asset_list_snapshot.parquet` | Snapshot × asset/participant | Current reference metadata |

## Optional delayed research data

These reports are valuable for retrospective explanation. Merit-order and
operating-reserve offer-control enforce their documented 60-day disclosure
delay. Unit Commitment preserves its explicit issued time and is available
from 1 July 2024; the supplied API contract does not specify a 60-day delay.

```bash
aeso-download range merit-order \
  --start-date 2025-01-01 --end-date 2025-01-31

aeso-download range operating-reserve-offers \
  --start-date 2025-01-01 --end-date 2025-01-31

aeso-download range unit-commitment \
  --start-date 2024-07-01 --end-date 2025-12-31
```

Latest-only reference metadata are selected separately. Existing valid local
snapshots are reused unless `--overwrite` is supplied:

```bash
aeso-download snapshot asset-list
aeso-download snapshot pool-participants
```

Metered volume is optional because it overlaps existing realized generation
data and can be very large. Unfiltered requests are divided into 16-day chunks:

```bash
python -m aeso_api.download_metered_volume \
  --start-date 2025-01-01 --end-date 2025-01-16
```

## Timing boundary

The downloaded historical generation-capacity, outage, and forecast records do
not expose a submission or revision timestamp in the documented response. They
must remain classified as timing-dependent until testing establishes whether a
historical query reproduces the value originally available at that date or a
later revised value.

The pool-price endpoint also contains a published `forecast_pool_price` field.
That field must receive the same issue-time and revision audit before it is used
as an ex-ante model feature.

## Deferred operations

Two operations were intentionally not implemented because their complete
contracts were not supplied:

- Current Supply Demand v2;
- Intertie Capability Report within Intertie Public Reports v1.

Current Supply Demand v1 is also omitted because v2 supersedes it. Pool
Participant data are collected only as latest reference metadata, not as a
historical forecasting feature. Its corporate-contact field should not be
propagated into analytical products.
