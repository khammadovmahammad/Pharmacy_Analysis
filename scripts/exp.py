import duckdb

con = duckdb.connect("data/pharmacy.duckdb")

result = con.execute("""
WITH raw_branches AS (
    SELECT DISTINCT aptek_id
    FROM transactions
),

clean_branches AS (
    SELECT DISTINCT aptek_id
    FROM clean_transactions
)

SELECT
    r.aptek_id AS missing_branch

FROM raw_branches r

LEFT JOIN clean_branches c
    ON
    CASE
        WHEN RIGHT(r.aptek_id, 2) = '.0'
        THEN LEFT(r.aptek_id, LENGTH(r.aptek_id) - 2)
        ELSE r.aptek_id
    END = c.aptek_id

WHERE c.aptek_id IS NULL;
""").fetchdf()

print(result)
con.close()