from pathlib import Path
from time import perf_counter

import duckdb


# ============================================================
# CONFIGURATION
# ============================================================

DB_FILE = Path("data/pharmacy.duckdb")

if not DB_FILE.exists():
    raise FileNotFoundError(
        f"Database not found:\n{DB_FILE.resolve()}"
    )


con = duckdb.connect(str(DB_FILE))

print("=" * 80)
print("PHARMACY ANALYTICS - BUILD ANALYTICS TABLES")
print("=" * 80)

start_time = perf_counter()


# ============================================================
# 1. CREATE CLEAN TRANSACTION TABLE
#
# Rules:
#
# - One cek_id = one transaction.
# - If cek_id is duplicated, keep only one row.
# - qaytarma = 1 does not contribute to sales.
# - discount = ABS(endirim_meblegi)
# - Financial anomalies:
#       meblegh <= 0
#       OR discount > meblegh
# - Financial anomalies do not contribute to sales.
# ============================================================

print("\nCreating clean_transactions table...")

con.execute("""
CREATE OR REPLACE TABLE clean_transactions AS

WITH ranked_transactions AS (

    SELECT

        *,

        ROW_NUMBER() OVER (
            PARTITION BY cek_id

            ORDER BY
                TRY_CAST(aptek_id AS DOUBLE),
                tarix_saat
        ) AS cek_rank

    FROM transactions
),

deduplicated AS (

    SELECT
        cek_id,
        aptek_id,
        tarix_saat,
        meblegh,
        endirim_meblegi,
        musteri_acari,
        qaytarma,
        resepti_var

    FROM ranked_transactions

    WHERE cek_rank = 1
),

base AS (

    SELECT

        cek_id,

        -- Convert values such as 9273.0 → 9273
        CASE
            WHEN RIGHT(aptek_id, 2) = '.0'
            THEN LEFT(
                aptek_id,
                LENGTH(aptek_id) - 2
            )

            ELSE aptek_id
        END AS aptek_id,

        tarix_saat,

        -- Month used for monthly reporting
        CAST(
            DATE_TRUNC(
                'month',
                tarix_saat
            )
            AS DATE
        ) AS month,

        meblegh,

        endirim_meblegi,

        -- Discount is always treated as positive magnitude
        ABS(endirim_meblegi)
            AS discount_amount,

        musteri_acari,

        qaytarma,

        resepti_var,

        -- Financial anomaly
        CASE
            WHEN
                qaytarma = 0

                AND (
                    meblegh <= 0
                    OR ABS(endirim_meblegi) > meblegh
                )

            THEN 1

            ELSE 0
        END AS financial_anomaly

    FROM deduplicated
)

SELECT

    *,

    -- --------------------------------------------------------
    -- Can this transaction participate in sales calculations?
    -- --------------------------------------------------------

    CASE
        WHEN
            qaytarma = 0
            AND financial_anomaly = 0

        THEN 1

        ELSE 0
    END AS usable_for_sales,


    -- --------------------------------------------------------
    -- Gross amount
    -- --------------------------------------------------------

    CASE
        WHEN
            qaytarma = 0
            AND financial_anomaly = 0

        THEN meblegh

        ELSE 0
    END AS valid_gross_amount,


    -- --------------------------------------------------------
    -- Discount
    -- --------------------------------------------------------

    CASE
        WHEN
            qaytarma = 0
            AND financial_anomaly = 0

        THEN discount_amount

        ELSE 0
    END AS valid_discount_amount,


    -- --------------------------------------------------------
    -- Actual amount paid
    --
    -- net = gross - absolute discount
    -- --------------------------------------------------------

    CASE
        WHEN
            qaytarma = 0
            AND financial_anomaly = 0

        THEN meblegh - discount_amount

        ELSE 0
    END AS net_amount

FROM base;
""")


# ============================================================
# 2. CUSTOMER SEGMENTATION
#
# Pension logic:
#
# Customer Discount Rate =
# total discount / total gross amount
#
# > 50%  = Pension
# <= 50% = Non-Pension
#
# Calculation uses the customer's full valid history.
# ============================================================

print("Creating customer_segments table...")

con.execute("""
CREATE OR REPLACE TABLE customer_segments AS

WITH customer_totals AS (

    SELECT

        musteri_acari,

        COUNT(*) AS transaction_count,

        SUM(valid_gross_amount)
            AS total_gross_amount,

        SUM(valid_discount_amount)
            AS total_discount_amount,

        SUM(net_amount)
            AS total_net_amount

    FROM clean_transactions

    WHERE usable_for_sales = 1

    GROUP BY musteri_acari
)

SELECT

    musteri_acari,

    transaction_count,

    total_gross_amount,

    total_discount_amount,

    total_net_amount,


    -- Raw rate: 0.60 means 60%
    total_discount_amount
        / NULLIF(total_gross_amount, 0)
        AS discount_rate,


    -- Percentage version for reporting
    ROUND(
        100.0
        * total_discount_amount
        / NULLIF(total_gross_amount, 0),
        2
    ) AS discount_percentage,


    CASE

        WHEN
            total_discount_amount
            / NULLIF(total_gross_amount, 0)
            > 0.50

        THEN 'Pension'

        ELSE 'Non-Pension'

    END AS customer_type

FROM customer_totals;
""")


# ============================================================
# 3. MONTHLY BRANCH METRICS
# ============================================================

print("Creating monthly_branch_metrics table...")

con.execute("""
CREATE OR REPLACE TABLE monthly_branch_metrics AS

WITH joined AS (

    SELECT

        t.*,

        c.customer_type

    FROM clean_transactions t

    LEFT JOIN customer_segments c
        ON t.musteri_acari = c.musteri_acari
),

base_metrics AS (

    SELECT

        month,
        aptek_id,


        -- ====================================================
        -- TRANSACTIONS
        -- ====================================================

        SUM(
            CASE
                WHEN usable_for_sales = 1
                THEN 1
                ELSE 0
            END
        ) AS sales_transactions,


        SUM(
            CASE
                WHEN qaytarma = 1
                THEN 1
                ELSE 0
            END
        ) AS returned_transactions,


        SUM(
            CASE
                WHEN financial_anomaly = 1
                THEN 1
                ELSE 0
            END
        ) AS excluded_financial_anomalies,


        -- ====================================================
        -- SALES
        -- ====================================================

        SUM(valid_gross_amount)
            AS gross_sales,

        SUM(valid_discount_amount)
            AS discount_amount,

        SUM(net_amount)
            AS net_sales,


        -- ====================================================
        -- PRESCRIPTION SALES
        -- ====================================================

        SUM(
            CASE
                WHEN
                    usable_for_sales = 1
                    AND resepti_var = 1

                THEN net_amount

                ELSE 0
            END
        ) AS prescription_volume,


        SUM(
            CASE
                WHEN
                    usable_for_sales = 1
                    AND resepti_var = 0

                THEN net_amount

                ELSE 0
            END
        ) AS non_prescription_volume,


        -- ====================================================
        -- UNIQUE CUSTOMERS
        -- ====================================================

        COUNT(
            DISTINCT CASE

                WHEN usable_for_sales = 1
                THEN musteri_acari

            END
        ) AS unique_customers,


        -- ====================================================
        -- PENSION / NON-PENSION CUSTOMER COUNTS
        -- ====================================================

        COUNT(
            DISTINCT CASE

                WHEN
                    usable_for_sales = 1
                    AND customer_type = 'Pension'

                THEN musteri_acari

            END
        ) AS pension_customers,


        COUNT(
            DISTINCT CASE

                WHEN
                    usable_for_sales = 1
                    AND customer_type = 'Non-Pension'

                THEN musteri_acari

            END
        ) AS non_pension_customers,


        -- ====================================================
        -- PENSION / NON-PENSION SALES VOLUME
        -- ====================================================

        SUM(
            CASE

                WHEN
                    usable_for_sales = 1
                    AND customer_type = 'Pension'

                THEN net_amount

                ELSE 0

            END
        ) AS pension_volume,


        SUM(
            CASE

                WHEN
                    usable_for_sales = 1
                    AND customer_type = 'Non-Pension'

                THEN net_amount

                ELSE 0

            END
        ) AS non_pension_volume


    FROM joined

    GROUP BY
        month,
        aptek_id
)

SELECT

    month,

    aptek_id,


    -- ========================================================
    -- CORE VALUES
    -- ========================================================

    sales_transactions,

    returned_transactions,

    excluded_financial_anomalies,

    gross_sales,

    discount_amount,

    net_sales,

    prescription_volume,

    non_prescription_volume,

    unique_customers,

    pension_customers,

    non_pension_customers,

    pension_volume,

    non_pension_volume,


    -- ========================================================
    -- TRANSACTIONS PER CUSTOMER
    -- ========================================================

    ROUND(

        sales_transactions * 1.0
        / NULLIF(unique_customers, 0),

        2

    ) AS transactions_per_customer,


    -- ========================================================
    -- AVERAGE CHECK VALUE
    -- ========================================================

    ROUND(

        net_sales
        / NULLIF(sales_transactions, 0),

        2

    ) AS average_check,


    -- ========================================================
    -- DISCOUNT RATE
    -- ========================================================

    ROUND(

        100.0
        * discount_amount
        / NULLIF(gross_sales, 0),

        2

    ) AS discount_rate_pct,


    -- ========================================================
    -- PRESCRIPTION SHARE
    -- ========================================================

    ROUND(

        100.0
        * prescription_volume
        / NULLIF(net_sales, 0),

        2

    ) AS prescription_volume_pct,


    ROUND(

        100.0
        * non_prescription_volume
        / NULLIF(net_sales, 0),

        2

    ) AS non_prescription_volume_pct,


    -- ========================================================
    -- PENSION CUSTOMER SHARE
    -- ========================================================

    ROUND(

        100.0
        * pension_customers
        / NULLIF(unique_customers, 0),

        2

    ) AS pension_customer_pct,


    ROUND(

        100.0
        * non_pension_customers
        / NULLIF(unique_customers, 0),

        2

    ) AS non_pension_customer_pct,


    -- ========================================================
    -- PENSION SALES SHARE
    -- ========================================================

    ROUND(

        100.0
        * pension_volume
        / NULLIF(net_sales, 0),

        2

    ) AS pension_volume_pct,


    ROUND(

        100.0
        * non_pension_volume
        / NULLIF(net_sales, 0),

        2

    ) AS non_pension_volume_pct,


    -- ========================================================
    -- RETURN RATE
    -- ========================================================

    ROUND(

        100.0
        * returned_transactions

        / NULLIF(
            sales_transactions
            + returned_transactions,
            0
        ),

        2

    ) AS return_rate_pct


FROM base_metrics

ORDER BY
    month,
    aptek_id;
""")


# ============================================================
# 4. VALIDATE DEDUPLICATION
# ============================================================

print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)


deduplication_summary = con.execute("""
SELECT

    (
        SELECT COUNT(*)
        FROM transactions
    ) AS original_rows,

    (
        SELECT COUNT(DISTINCT cek_id)
        FROM transactions
    ) AS original_unique_cek_ids,

    (
        SELECT COUNT(*)
        FROM clean_transactions
    ) AS cleaned_rows,

    (
        SELECT COUNT(DISTINCT cek_id)
        FROM clean_transactions
    ) AS cleaned_unique_cek_ids,

    (
        SELECT COUNT(*)
        FROM transactions
    )
    -
    (
        SELECT COUNT(*)
        FROM clean_transactions
    ) AS duplicate_rows_removed;
""").fetchdf()


print("\nDeduplication:")
print(
    deduplication_summary.to_string(
        index=False
    )
)


# ============================================================
# 5. CLEAN TRANSACTION SUMMARY
# ============================================================

clean_summary = con.execute("""
SELECT

    COUNT(*) AS cleaned_rows,

    COUNT(DISTINCT cek_id)
        AS unique_cek_ids,

    SUM(
        CASE
            WHEN financial_anomaly = 1
            THEN 1
            ELSE 0
        END
    ) AS financial_anomaly_rows,

    SUM(
        CASE
            WHEN usable_for_sales = 1
            THEN 1
            ELSE 0
        END
    ) AS valid_sales_transactions,

    SUM(
        CASE
            WHEN qaytarma = 1
            THEN 1
            ELSE 0
        END
    ) AS returned_transactions,

    SUM(valid_gross_amount)
        AS gross_sales,

    SUM(valid_discount_amount)
        AS total_discount,

    SUM(net_amount)
        AS net_sales

FROM clean_transactions;
""").fetchdf()


print("\nClean transaction summary:")
print(
    clean_summary.to_string(
        index=False
    )
)


# ============================================================
# 6. CUSTOMER SEGMENTATION SUMMARY
# ============================================================

customer_summary = con.execute("""
SELECT

    COUNT(*) AS classified_customers,

    SUM(
        CASE
            WHEN customer_type = 'Pension'
            THEN 1
            ELSE 0
        END
    ) AS pension_customers,

    SUM(
        CASE
            WHEN customer_type = 'Non-Pension'
            THEN 1
            ELSE 0
        END
    ) AS non_pension_customers,

    ROUND(
        100.0
        * SUM(
            CASE
                WHEN customer_type = 'Pension'
                THEN 1
                ELSE 0
            END
        )
        / NULLIF(COUNT(*), 0),
        2
    ) AS pension_customer_pct,

    ROUND(
        AVG(discount_percentage),
        2
    ) AS average_customer_discount_pct

FROM customer_segments;
""").fetchdf()


print("\nCustomer segmentation:")
print(
    customer_summary.to_string(
        index=False
    )
)


# ============================================================
# 7. MONTHLY BRANCH TABLE SUMMARY
# ============================================================

monthly_summary = con.execute("""
SELECT

    COUNT(*) AS branch_month_records,

    COUNT(DISTINCT aptek_id)
        AS branches,

    COUNT(DISTINCT month)
        AS months,

    MIN(month)
        AS first_month,

    MAX(month)
        AS last_month,

    SUM(sales_transactions)
        AS total_sales_transactions,

    ROUND(
        SUM(net_sales),
        2
    ) AS total_net_sales

FROM monthly_branch_metrics;
""").fetchdf()


print("\nMonthly branch table:")
print(
    monthly_summary.to_string(
        index=False
    )
)


# ============================================================
# 8. SAMPLE MONTHLY BRANCH RESULTS
# ============================================================

sample = con.execute("""
SELECT

    month,

    aptek_id,

    sales_transactions,

    unique_customers,

    gross_sales,

    discount_amount,

    net_sales,

    average_check,

    prescription_volume_pct,

    pension_customers,

    non_pension_customers,

    pension_volume_pct,

    return_rate_pct

FROM monthly_branch_metrics

ORDER BY
    month DESC,
    net_sales DESC

LIMIT 20;
""").fetchdf()


print("\nSample monthly branch metrics:")
print(
    sample.to_string(
        index=False
    )
)


# ============================================================
# 9. CHECK ACCOUNTING RELATIONSHIP
#
# gross - discount should equal net sales
# ============================================================

accounting_check = con.execute("""
SELECT

    ROUND(
        SUM(valid_gross_amount),
        2
    ) AS gross_sales,

    ROUND(
        SUM(valid_discount_amount),
        2
    ) AS discount_amount,

    ROUND(
        SUM(net_amount),
        2
    ) AS net_sales,

    ROUND(
        SUM(valid_gross_amount)
        - SUM(valid_discount_amount)
        - SUM(net_amount),
        2
    ) AS difference

FROM clean_transactions;
""").fetchdf()


print("\nAccounting check:")
print(
    accounting_check.to_string(
        index=False
    )
)


# ============================================================
# FINISH
# ============================================================

elapsed = perf_counter() - start_time

print("\n" + "=" * 80)
print("ANALYTICS TABLES CREATED SUCCESSFULLY")
print(f"Elapsed time: {elapsed:.2f} seconds")
print("=" * 80)


con.close()