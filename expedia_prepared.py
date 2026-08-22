#!/usr/bin/env python3
"""Rebuild expedia_prepared.parquet from the Expedia train.csv file.

The transformation is executed by DuckDB so the 4 GB CSV is processed in a
streaming/parallel pipeline rather than materialized as a pandas DataFrame.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import duckdb


SOURCE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("date_time", "TIMESTAMP"),
    ("site_name", "SMALLINT"),
    ("posa_continent", "SMALLINT"),
    ("user_location_country", "SMALLINT"),
    ("user_location_region", "INTEGER"),
    ("user_location_city", "INTEGER"),
    ("orig_destination_distance", "DOUBLE"),
    ("user_id", "BIGINT"),
    ("is_mobile", "SMALLINT"),
    ("is_package", "SMALLINT"),
    ("channel", "SMALLINT"),
    ("srch_ci", "DATE"),
    ("srch_co", "DATE"),
    ("srch_adults_cnt", "INTEGER"),
    ("srch_children_cnt", "INTEGER"),
    ("srch_rm_cnt", "INTEGER"),
    ("srch_destination_id", "INTEGER"),
    ("srch_destination_type_id", "SMALLINT"),
    ("is_booking", "SMALLINT"),
    ("cnt", "BIGINT"),
    ("hotel_continent", "SMALLINT"),
    ("hotel_country", "SMALLINT"),
    ("hotel_market", "INTEGER"),
    ("hotel_cluster", "SMALLINT"),
)


def sql_string(value: Path | str) -> str:
    """Return a safely quoted DuckDB string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def transformation_query(source: Path) -> str:
    csv_columns = ", ".join(f"'{name}': 'VARCHAR'" for name, _ in SOURCE_COLUMNS)
    typed_columns = ",\n            ".join(
        f"TRY_CAST({name} AS {data_type}) AS {name}"
        for name, data_type in SOURCE_COLUMNS
    )
    parse_failures = "\n                OR ".join(
        f"({name} IS NOT NULL AND TRY_CAST({name} AS {data_type}) IS NULL)"
        for name, data_type in SOURCE_COLUMNS
    )

    return f"""
        WITH raw AS (
            SELECT *
            FROM read_csv(
                {sql_string(source)},
                header = true,
                auto_detect = false,
                columns = {{{csv_columns}}},
                delim = ',',
                quote = '',
                escape = '',
                new_line = '\\n',
                nullstr = '',
                strict_mode = true,
                parallel = true
            )
        ),
        typed AS (
            SELECT
                {typed_columns},
                (
                    {parse_failures}
                ) AS has_unparsed_source_value
            FROM raw
        ),
        features AS (
            SELECT
                *,
                CAST(date_time AS DATE) AS search_date,
                CAST(date_trunc('month', date_time) AS DATE) AS search_month,
                CAST(srch_ci - CAST(date_time AS DATE) AS INTEGER) AS lead_time_days,
                CAST(srch_co - srch_ci AS INTEGER) AS stay_nights,
                CAST(srch_adults_cnt + srch_children_cnt AS INTEGER) AS party_size
            FROM typed
        ),
        quality AS (
            SELECT
                *,
                date_time IS NULL AS invalid_search_datetime,
                srch_ci IS NULL OR srch_co IS NULL OR stay_nights <= 0
                    AS invalid_stay_dates,
                lead_time_days IS NULL OR lead_time_days < 0
                    AS invalid_lead_time,
                srch_adults_cnt IS NULL
                    OR srch_children_cnt IS NULL
                    OR srch_adults_cnt <= 0
                    OR srch_children_cnt < 0
                    AS invalid_party,
                srch_rm_cnt IS NULL OR srch_rm_cnt <= 0 AS invalid_rooms,
                is_booking IS NULL OR is_booking NOT IN (0, 1)
                    AS invalid_booking_flag,
                cnt IS NULL OR cnt <= 0 AS invalid_cnt
            FROM features
        ),
        prepared AS (
            SELECT
                date_time,
                site_name,
                posa_continent,
                user_location_country,
                user_location_region,
                user_location_city,
                orig_destination_distance,
                user_id,
                is_mobile,
                is_package,
                channel,
                srch_ci,
                srch_co,
                srch_adults_cnt,
                srch_children_cnt,
                srch_rm_cnt,
                srch_destination_id,
                srch_destination_type_id,
                is_booking,
                cnt,
                hotel_continent,
                hotel_country,
                hotel_market,
                hotel_cluster,
                search_date,
                search_month,
                lead_time_days,
                stay_nights,
                party_size,
                cnt AS event_weight,
                CAST(is_booking AS BIGINT) * cnt AS booking_weight,
                COALESCE(srch_children_cnt > 0, false) AS is_family,
                COALESCE(lead_time_days BETWEEN 0 AND 7, false) AS is_urgent,
                COALESCE(stay_nights >= 7, false) AS is_long_stay,
                CASE
                    WHEN lead_time_days IS NULL OR lead_time_days < 0 THEN 'invalid'
                    WHEN lead_time_days = 0 THEN 'same_day'
                    WHEN lead_time_days <= 7 THEN '1_7'
                    WHEN lead_time_days <= 30 THEN '8_30'
                    WHEN lead_time_days <= 90 THEN '31_90'
                    ELSE '91_plus'
                END AS lead_time_bucket,
                CASE
                    WHEN stay_nights IS NULL OR stay_nights <= 0 THEN 'invalid'
                    WHEN stay_nights = 1 THEN '1'
                    WHEN stay_nights <= 3 THEN '2_3'
                    WHEN stay_nights <= 6 THEN '4_6'
                    ELSE '7_plus'
                END AS stay_bucket,
                CASE
                    WHEN invalid_party THEN 'unknown'
                    WHEN srch_children_cnt > 0 THEN 'family'
                    WHEN srch_adults_cnt = 1 THEN 'solo'
                    WHEN srch_adults_cnt = 2 THEN 'couple'
                    ELSE 'group'
                END AS party_type,
                invalid_search_datetime,
                invalid_stay_dates,
                invalid_lead_time,
                invalid_party,
                invalid_rooms,
                invalid_booking_flag,
                invalid_cnt,
                orig_destination_distance IS NULL AS distance_missing,
                has_unparsed_source_value,
                NOT (
                    invalid_search_datetime
                    OR invalid_stay_dates
                    OR invalid_lead_time
                    OR invalid_party
                    OR invalid_rooms
                    OR invalid_booking_flag
                    OR invalid_cnt
                    OR has_unparsed_source_value
                ) AS is_valid_for_analysis
            FROM quality
        )
        SELECT * FROM prepared
    """


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parquet_schema(connection: duckdb.DuckDBPyConnection, path: Path) -> list[tuple[str, str]]:
    rows = connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet({sql_string(path)})"
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def verify_parquets(
    connection: duckdb.DuckDBPyConnection, actual: Path, expected: Path
) -> None:
    actual_schema = parquet_schema(connection, actual)
    expected_schema = parquet_schema(connection, expected)
    if actual_schema != expected_schema:
        raise RuntimeError(
            "Parquet schemas differ.\n"
            f"Actual:   {actual_schema}\n"
            f"Expected: {expected_schema}"
        )

    actual_rows = connection.execute(
        f"SELECT count(*) FROM read_parquet({sql_string(actual)})"
    ).fetchone()[0]
    expected_rows = connection.execute(
        f"SELECT count(*) FROM read_parquet({sql_string(expected)})"
    ).fetchone()[0]
    if actual_rows != expected_rows:
        raise RuntimeError(
            f"Row counts differ: actual={actual_rows}, expected={expected_rows}"
        )

    # EXCEPT ALL compares full rows and preserves duplicate multiplicities. Running
    # it in both directions proves exact equality of the two unordered row multisets.
    for left, right, direction in (
        (actual, expected, "actual - expected"),
        (expected, actual, "expected - actual"),
    ):
        mismatch = connection.execute(
            f"""
            SELECT count(*)
            FROM (
                SELECT * FROM read_parquet({sql_string(left)})
                EXCEPT ALL
                SELECT * FROM read_parquet({sql_string(right)})
            )
            """
        ).fetchone()[0]
        if mismatch:
            raise RuntimeError(f"Data differ ({direction}): {mismatch} unmatched rows")

    actual_hash = file_sha256(actual)
    expected_hash = file_sha256(expected)
    print(f"Verified schema: {len(actual_schema)} columns")
    print(f"Verified rows:   {actual_rows:,}")
    print("Verified values: exact equality, including duplicate multiplicities")
    print(f"Actual SHA-256:  {actual_hash}")
    print(f"Expected SHA-256:{expected_hash}")
    if actual_hash != expected_hash:
        print(
            "Physical hashes differ only because parallel DuckDB output does not "
            "guarantee the same row-group ordering; schemas and row values are exact."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("train.csv"))
    parser.add_argument("--output", type=Path, default=Path("expedia_prepared.parquet"))
    parser.add_argument(
        "--verify-against",
        type=Path,
        help="After rebuilding, exactly compare schema and all rows with this Parquet",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacing an existing output file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    reference = args.verify_against.resolve() if args.verify_against else None

    if not source.is_file():
        raise FileNotFoundError(f"Input CSV not found: {source}")
    if output.exists() and not args.force:
        raise FileExistsError(
            f"Output already exists: {output}. Use a different --output or --force."
        )
    if source == output:
        raise ValueError("Input and output paths must differ")
    if reference is not None and not reference.is_file():
        raise FileNotFoundError(f"Reference Parquet not found: {reference}")
    if reference is not None and reference == output:
        raise ValueError("Output and verification reference paths must differ")

    output.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET preserve_insertion_order = false")
    connection.execute("SET enable_progress_bar = true")

    started = time.monotonic()
    print(f"DuckDB {duckdb.__version__}")
    print(f"Reading: {source}")
    print(f"Writing: {output}")
    query = transformation_query(source)
    connection.execute(
        f"COPY ({query}) TO {sql_string(output)} "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    elapsed = time.monotonic() - started
    print(f"Rebuilt in {elapsed:.1f} seconds ({output.stat().st_size:,} bytes)")

    if reference is not None:
        print(f"Verifying against: {reference}")
        verify_parquets(connection, output, reference)

    connection.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
