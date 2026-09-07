from pathlib import Path
import duckdb


DB_FILE = Path("data/pharmacy.duckdb")

if not DB_FILE.exists():
    raise FileNotFoundError(
        f"Database not found: {DB_FILE.resolve()}"
    )


con = duckdb.connect(str(DB_FILE))


def show(title, query):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    result = con.execute(query).fetchdf()
    print(result.to_string(index=False))


# ============================================================
# 1. DISCOUNT SIGN DISTRIBUTION
# ============================================================

show(
    "DISCOUNT SIGN DISTRIBUTION - SUCCESSFUL TRANSACTIONS",
    """
    SELECT

        COUNT(*) AS successful_transactions,

        SUM(
            CASE
                WHEN endirim_meblegi < 0 THEN 1
                ELSE 0
            END
        ) AS negative_discount,

        SUM(
            CASE
                WHEN endirim_meblegi = 0 THEN 1
                ELSE 0
            END
        ) AS zero_discount,

        SUM(
            CASE
                WHEN endirim_meblegi > 0 THEN 1
                ELSE 0
            END
        ) AS positive_discount

    FROM transactions
    WHERE qaytarma = 0;
    """
)


# ============================================================
# 2. TEST BOTH POSSIBLE NET-SALES FORMULAS
# ============================================================

show(
    "NET SALES FORMULA TEST",
    """
    SELECT

        SUM(meblegh) AS gross_amount,

        SUM(endirim_meblegi) AS signed_discount_amount,

        SUM(meblegh + endirim_meblegi)
            AS net_if_discount_is_negative,

        SUM(meblegh - endirim_meblegi)
            AS net_if_discount_is_positive

    FROM transactions

    WHERE qaytarma = 0;
    """
)


# ============================================================
# 3. CHECK WHETHER GROSS + DISCOUNT BECOMES NEGATIVE
# ============================================================

show(
    "INVALID NET-AMOUNT CHECK",
    """
    SELECT

        SUM(
            CASE
                WHEN meblegh + endirim_meblegi < 0
                THEN 1
                ELSE 0
            END
        ) AS negative_net_amount_rows,

        SUM(
            CASE
                WHEN meblegh + endirim_meblegi = 0
                THEN 1
                ELSE 0
            END
        ) AS zero_net_amount_rows

    FROM transactions

    WHERE qaytarma = 0;
    """
)


# ============================================================
# 4. SAMPLE NEGATIVE DISCOUNTS
# ============================================================

show(
    "SAMPLE NEGATIVE DISCOUNTS",
    """
    SELECT

        cek_id,
        meblegh,
        endirim_meblegi,

        meblegh + endirim_meblegi
            AS calculated_net_amount,

        ROUND(
            ABS(endirim_meblegi) /
            NULLIF(meblegh, 0) * 100,
            2
        ) AS discount_percentage

    FROM transactions

    WHERE
        qaytarma = 0
        AND endirim_meblegi < 0

    LIMIT 20;
    """
)


# ============================================================
# 5. SAMPLE POSITIVE DISCOUNTS
# ============================================================

show(
    "SAMPLE POSITIVE DISCOUNTS",
    """
    SELECT

        cek_id,
        meblegh,
        endirim_meblegi,
        meblegh + endirim_meblegi
            AS calculated_net_amount

    FROM transactions

    WHERE
        qaytarma = 0
        AND endirim_meblegi > 0

    LIMIT 20;
    """
)


# ============================================================
# 6. DUPLICATE CEK_ID STATISTICS
# ============================================================

show(
    "DUPLICATE CEK_ID SUMMARY",
    """
    WITH duplicate_ids AS (

        SELECT
            cek_id,
            COUNT(*) AS row_count

        FROM transactions

        GROUP BY cek_id

        HAVING COUNT(*) > 1
    )

    SELECT

        COUNT(*) AS duplicated_cek_ids,

        SUM(row_count) AS rows_in_duplicate_groups,

        SUM(row_count - 1) AS excess_rows,

        MAX(row_count) AS maximum_rows_for_one_cek

    FROM duplicate_ids;
    """
)


# ============================================================
# 7. CHECK IF DUPLICATES ARE EXACT COPIES
# ============================================================

show(
    "EXACT DUPLICATE ROW SUMMARY",
    """
    WITH duplicate_rows AS (

        SELECT

            cek_id,
            aptek_id,
            tarix_saat,
            meblegh,
            endirim_meblegi,
            musteri_acari,
            qaytarma,
            resepti_var,

            COUNT(*) AS row_count

        FROM transactions

        GROUP BY ALL

        HAVING COUNT(*) > 1
    )

    SELECT

        COUNT(*) AS exact_duplicate_groups,

        SUM(row_count) AS rows_in_exact_duplicate_groups,

        SUM(row_count - 1) AS removable_exact_duplicate_rows

    FROM duplicate_rows;
    """
)


# ============================================================
# 8. SAMPLE DUPLICATED CEK_ID VALUES
# ============================================================

show(
    "SAMPLE DUPLICATED CEK_ID ROWS",
    """
    SELECT t.*

    FROM transactions t

    INNER JOIN (

        SELECT cek_id

        FROM transactions

        GROUP BY cek_id

        HAVING COUNT(*) > 1

        LIMIT 20

    ) d

    ON t.cek_id = d.cek_id

    ORDER BY t.cek_id, t.tarix_saat;
    """
)


# ============================================================
# 9. NEGATIVE GROSS AMOUNT ROWS
# ============================================================

show(
    "NEGATIVE MEBLEGH ROWS",
    """
    SELECT *

    FROM transactions

    WHERE meblegh < 0

    ORDER BY meblegh;
    """
)


# ============================================================
# 10. ZERO GROSS AMOUNT ROWS
# ============================================================

show(
    "SAMPLE ZERO MEBLEGH ROWS",
    """
    SELECT *

    FROM transactions

    WHERE meblegh = 0

    LIMIT 30;
    """
)


# ============================================================
# 11. DISCOUNT MAGNITUDE CHECK
# ============================================================

show(
    "DISCOUNT MAGNITUDE CHECK",
    """
    SELECT

        SUM(
            CASE
                WHEN
                    meblegh > 0
                    AND ABS(endirim_meblegi) > meblegh
                THEN 1
                ELSE 0
            END
        ) AS discount_larger_than_gross,

        SUM(
            CASE
                WHEN
                    meblegh > 0
                    AND ABS(endirim_meblegi) = meblegh
                THEN 1
                ELSE 0
            END
        ) AS exactly_100_percent_discount,

        SUM(
            CASE
                WHEN
                    meblegh > 0
                    AND ABS(endirim_meblegi) / meblegh > 0.50
                THEN 1
                ELSE 0
            END
        ) AS transactions_above_50_percent_discount

    FROM transactions

    WHERE qaytarma = 0;
    """
)

# ============================================================
# 12. EXPORT: DISCOUNT LARGER THAN GROSS
# ============================================================

EXPORT_DIR = Path("data/exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

out_path = str(
    (EXPORT_DIR / "discount_exceeds_gross.csv").resolve()
).replace("\\", "/")

anomaly_filter = """
    qaytarma = 0
    AND meblegh > 0
    AND ABS(endirim_meblegi) > meblegh
"""

con.execute(
    f"""
    COPY (
        SELECT
            cek_id,
            aptek_id,
            tarix_saat,
            CAST(tarix_saat AS DATE) AS tarix,
            musteri_acari,
            meblegh,
            endirim_meblegi,

            CASE
                WHEN endirim_meblegi < 0 THEN 'negative'
                ELSE 'positive'
            END AS discount_sign,

            meblegh + endirim_meblegi AS net_if_signed,
            meblegh - endirim_meblegi AS net_if_positive,

            ROUND(
                ABS(endirim_meblegi) / meblegh * 100,
                2
            ) AS discount_percentage,

            ABS(endirim_meblegi) - meblegh AS excess_amount,

            qaytarma,
            resepti_var

        FROM transactions
        WHERE {anomaly_filter}
        ORDER BY ABS(endirim_meblegi) / meblegh DESC
    )
    TO '{out_path}'
    (FORMAT CSV, HEADER true, DELIMITER '|');
    """
)

exported = con.execute(
    f"SELECT COUNT(*) FROM transactions WHERE {anomaly_filter};"
).fetchone()[0]

print(f"\nExported {exported:,} rows")
print(out_path)

con.close()

print("\n" + "=" * 80)
print("INVESTIGATION COMPLETED")
print("=" * 80)