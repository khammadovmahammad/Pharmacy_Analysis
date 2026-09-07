from pathlib import Path
from time import perf_counter

import duckdb


# ============================================================
# CONFIGURATION
# ============================================================

RAW_FILE = Path("data/raw/(Copy) loyalty_satislar.csv")
DB_FILE = Path("data/pharmacy.duckdb")


# ============================================================
# CHECK FILE
# ============================================================

if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"Raw data file was not found:\n{RAW_FILE.resolve()}"
    )

DB_FILE.parent.mkdir(parents=True, exist_ok=True)


# Convert Windows paths to a DuckDB-friendly format
raw_file_path = str(RAW_FILE.resolve()).replace("\\", "/")
db_file_path = str(DB_FILE.resolve()).replace("\\", "/")


print("=" * 70)
print("PHARMACY ANALYTICS - DATABASE BUILD")
print("=" * 70)

print(f"\nRaw file:")
print(raw_file_path)

print(f"\nDatabase:")
print(db_file_path)


# ============================================================
# CONNECT TO DUCKDB
# ============================================================

con = duckdb.connect(db_file_path)


# ============================================================
# IMPORT AND CLEAN DATA
# ============================================================

print("\nReading CSV and creating transactions table...")

start_time = perf_counter()


con.execute(
    f"""
    CREATE OR REPLACE TABLE transactions AS

    SELECT

        -- Unique transaction/check ID
        NULLIF(TRIM(cek_id), '') AS cek_id,

        -- Pharmacy branch
        NULLIF(TRIM(aptek_id), '') AS aptek_id,

        -- Transaction timestamp
        TRY_CAST(
            NULLIF(TRIM(tarix_saat), '')
            AS TIMESTAMP
        ) AS tarix_saat,

        -- Gross transaction amount before discount
        TRY_CAST(
            REPLACE(
                NULLIF(TRIM(meblegh), ''),
                ',',
                '.'
            )
            AS DECIMAL(18, 2)
        ) AS meblegh,

        -- Discount amount
        TRY_CAST(
            REPLACE(
                NULLIF(TRIM(endirim_meblegi), ''),
                ',',
                '.'
            )
            AS DECIMAL(18, 2)
        ) AS endirim_meblegi,

        -- Customer identifier
        NULLIF(TRIM(musteri_acari), '') AS musteri_acari,

        -- 0 = completed transaction
        -- 1 = returned transaction
        TRY_CAST(
            NULLIF(TRIM(qaytarma), '')
            AS INTEGER
        ) AS qaytarma,

        -- Source column is called reseptli_var
        -- We rename it internally to resepti_var
        TRY_CAST(
            NULLIF(TRIM(reseptli_var), '')
            AS INTEGER
        ) AS resepti_var

    FROM read_csv(
        '{raw_file_path}',
        header = true,
        delim = '|',
        all_varchar = true
    );
    """
)


elapsed = perf_counter() - start_time

print(f"Import completed.")
print(f"Elapsed time: {elapsed:.2f} seconds")


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("DATA VALIDATION")
print("=" * 70)


validation = con.execute(
    """
    SELECT

        COUNT(*) AS total_rows,

        COUNT(DISTINCT cek_id)
            AS unique_cek_ids,

        SUM(
            CASE
                WHEN cek_id IS NULL THEN 1
                ELSE 0
            END
        ) AS null_cek_id,

        SUM(
            CASE
                WHEN aptek_id IS NULL THEN 1
                ELSE 0
            END
        ) AS null_aptek_id,

        SUM(
            CASE
                WHEN tarix_saat IS NULL THEN 1
                ELSE 0
            END
        ) AS invalid_or_null_date,

        SUM(
            CASE
                WHEN meblegh IS NULL THEN 1
                ELSE 0
            END
        ) AS invalid_or_null_meblegh,

        SUM(
            CASE
                WHEN endirim_meblegi IS NULL THEN 1
                ELSE 0
            END
        ) AS invalid_or_null_discount,

        SUM(
            CASE
                WHEN musteri_acari IS NULL THEN 1
                ELSE 0
            END
        ) AS null_customer_key,

        SUM(
            CASE
                WHEN qaytarma IS NULL
                     OR qaytarma NOT IN (0, 1)
                THEN 1
                ELSE 0
            END
        ) AS invalid_qaytarma,

        SUM(
            CASE
                WHEN resepti_var IS NULL
                     OR resepti_var NOT IN (0, 1)
                THEN 1
                ELSE 0
            END
        ) AS invalid_resepti_var

    FROM transactions;
    """
).fetchdf()


print("\nBasic validation:")
print(validation.to_string(index=False))


# ============================================================
# FINANCIAL VALIDATION
# ============================================================

financial_validation = con.execute(
    """
    SELECT

        SUM(
            CASE
                WHEN meblegh < 0
                THEN 1
                ELSE 0
            END
        ) AS negative_meblegh,

        SUM(
            CASE
                WHEN endirim_meblegi < 0
                THEN 1
                ELSE 0
            END
        ) AS negative_discount,

        SUM(
            CASE
                WHEN endirim_meblegi > meblegh
                THEN 1
                ELSE 0
            END
        ) AS discount_greater_than_meblegh,

        SUM(
            CASE
                WHEN meblegh = 0
                THEN 1
                ELSE 0
            END
        ) AS zero_meblegh

    FROM transactions;
    """
).fetchdf()


print("\nFinancial validation:")
print(financial_validation.to_string(index=False))


# ============================================================
# CHECK RETURN VALUES
# ============================================================

returns = con.execute(
    """
    SELECT
        qaytarma,
        COUNT(*) AS transaction_count
    FROM transactions
    GROUP BY qaytarma
    ORDER BY qaytarma;
    """
).fetchdf()


print("\nQaytarma distribution:")
print(returns.to_string(index=False))


# ============================================================
# CHECK PRESCRIPTION VALUES
# ============================================================

prescriptions = con.execute(
    """
    SELECT
        resepti_var,
        COUNT(*) AS transaction_count
    FROM transactions
    GROUP BY resepti_var
    ORDER BY resepti_var;
    """
).fetchdf()


print("\nResepti_var distribution:")
print(prescriptions.to_string(index=False))


# ============================================================
# DATE RANGE
# ============================================================

date_range = con.execute(
    """
    SELECT
        MIN(tarix_saat) AS first_transaction,
        MAX(tarix_saat) AS last_transaction
    FROM transactions;
    """
).fetchdf()


print("\nTransaction date range:")
print(date_range.to_string(index=False))


# ============================================================
# BRANCH COUNT
# ============================================================

branch_count = con.execute(
    """
    SELECT
        COUNT(DISTINCT aptek_id) AS number_of_branches
    FROM transactions;
    """
).fetchdf()


print("\nBranches:")
print(branch_count.to_string(index=False))


# ============================================================
# CUSTOMER COUNT
# ============================================================

customer_count = con.execute(
    """
    SELECT
        COUNT(DISTINCT musteri_acari)
            AS total_unique_customers
    FROM transactions
    WHERE musteri_acari IS NOT NULL;
    """
).fetchdf()


print("\nCustomers:")
print(customer_count.to_string(index=False))


# ============================================================
# CHECK CEK_ID UNIQUENESS
# ============================================================

duplicate_check = con.execute(
    """
    SELECT
        COUNT(*) - COUNT(DISTINCT cek_id)
            AS duplicate_cek_id_rows
    FROM transactions;
    """
).fetchdf()


print("\ncek_id uniqueness:")
print(duplicate_check.to_string(index=False))


# ============================================================
# SAMPLE DATA
# ============================================================

sample = con.execute(
    """
    SELECT *
    FROM transactions
    LIMIT 10;
    """
).fetchdf()


print("\nSample rows:")
print(sample.to_string(index=False))


# ============================================================
# CLOSE CONNECTION
# ============================================================
# ============================================================
# EXPORT DISCOUNT ANOMALIES
# ============================================================

EXPORT_DIR = Path("data/exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

exports = [
    ("negative_discount", "endirim_meblegi < 0"),
    ("positive_discount", "endirim_meblegi > 0"),
]

print("\n" + "=" * 70)
print("DISCOUNT EXPORTS")
print("=" * 70)

for name, condition in exports:

    out_path = str(
        (EXPORT_DIR / f"{name}.csv").resolve()
    ).replace("\\", "/")

    con.execute(
        f"""
        COPY (
            SELECT *
            FROM transactions
            WHERE {condition}
            ORDER BY tarix_saat
        )
        TO '{out_path}'
        (FORMAT CSV, HEADER true, DELIMITER '|');
        """
    )

    row_count = con.execute(
        f"SELECT COUNT(*) FROM transactions WHERE {condition};"
    ).fetchone()[0]

    print(f"\n{name}: {row_count:,} rows")
    print(out_path)
con.close()


print("\n" + "=" * 70)
print("PHASE 1 COMPLETED")
print("=" * 70)