from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

from analytics.loan_analytics import (CONFIG, available_months,
                                      branch_summary, pilot_ranking,
                                      load_customer_features)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pharmacy Analytics Dashboard",
    page_icon="💊",
    layout="wide",
)


# ============================================================
# DATABASE LOCATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_FILE = PROJECT_ROOT / "data" / "pharmacy.duckdb"


if not DB_FILE.exists():
    st.error(
        f"DuckDB database was not found:\n{DB_FILE}"
    )
    st.stop()


# ============================================================
# DATABASE QUERY HELPER
# ============================================================

def run_query(query, params=None):
    """
    Open DuckDB in read-only mode,
    execute a query,
    return a Pandas DataFrame,
    and close the connection.
    """

    con = duckdb.connect(
        str(DB_FILE),
        read_only=True
    )

    try:
        if params is None:
            result = con.execute(query).fetchdf()
        else:
            result = con.execute(
                query,
                params
            ).fetchdf()

    finally:
        con.close()

    return result


# ============================================================
# LOAD FILTER OPTIONS
# ============================================================

@st.cache_data
def load_months():

    query = """
    SELECT DISTINCT month

    FROM clean_transactions

    ORDER BY month;
    """

    df = run_query(query)

    df["month"] = pd.to_datetime(
        df["month"]
    )

    return df


@st.cache_data
def load_branches():

    query = """
    SELECT DISTINCT aptek_id

    FROM clean_transactions

    WHERE aptek_id IS NOT NULL;
    """

    df = run_query(query)

    # Sort numerically instead of alphabetically
    df["sort_value"] = pd.to_numeric(
        df["aptek_id"],
        errors="coerce"
    )

    df = df.sort_values(
        "sort_value"
    )

    return df["aptek_id"].astype(str).tolist()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_overall_metrics(
    start_month,
    end_month,
    branch
):

    # Convert YYYY-MM to dates
    start_date = f"{start_month}-01"

    end_date = (
        pd.Period(
            end_month,
            freq="M"
        )
        .end_time
        .date()
        .isoformat()
    )


    # --------------------------------------------------------
    # ALL BRANCHES
    # --------------------------------------------------------

    if branch == "All Branches":

        query = """
        SELECT

            SUM(net_amount)
                AS net_sales,

            SUM(valid_gross_amount)
                AS gross_sales,

            SUM(valid_discount_amount)
                AS discount_amount,

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

            COUNT(
                DISTINCT CASE
                    WHEN usable_for_sales = 1
                    THEN musteri_acari
                END
            ) AS unique_customers

        FROM clean_transactions

        WHERE
            tarix_saat >= ?
            AND tarix_saat < CAST(? AS DATE) + INTERVAL 1 DAY;
        """

        params = [
            start_date,
            end_date
        ]


    # --------------------------------------------------------
    # SPECIFIC BRANCH
    # --------------------------------------------------------

    else:

        query = """
        SELECT

            SUM(net_amount)
                AS net_sales,

            SUM(valid_gross_amount)
                AS gross_sales,

            SUM(valid_discount_amount)
                AS discount_amount,

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

            COUNT(
                DISTINCT CASE
                    WHEN usable_for_sales = 1
                    THEN musteri_acari
                END
            ) AS unique_customers

        FROM clean_transactions

        WHERE
            tarix_saat >= ?
            AND tarix_saat < CAST(? AS DATE) + INTERVAL 1 DAY
            AND aptek_id = ?;
        """

        params = [
            start_date,
            end_date,
            branch
        ]


    result = run_query(
        query,
        params
    )


    # --------------------------------------------------------
    # CALCULATED KPIs
    # --------------------------------------------------------

    row = result.iloc[0]


    net_sales = (
        float(row["net_sales"])
        if pd.notna(row["net_sales"])
        else 0
    )

    gross_sales = (
        float(row["gross_sales"])
        if pd.notna(row["gross_sales"])
        else 0
    )

    discount_amount = (
        float(row["discount_amount"])
        if pd.notna(row["discount_amount"])
        else 0
    )

    sales_transactions = int(
        row["sales_transactions"]
        if pd.notna(row["sales_transactions"])
        else 0
    )

    returned_transactions = int(
        row["returned_transactions"]
        if pd.notna(row["returned_transactions"])
        else 0
    )

    unique_customers = int(
        row["unique_customers"]
        if pd.notna(row["unique_customers"])
        else 0
    )


    # Average check
    average_check = (
        net_sales / sales_transactions
        if sales_transactions > 0
        else 0
    )


    # Discount rate
    discount_rate = (
        discount_amount
        / gross_sales
        * 100
        if gross_sales > 0
        else 0
    )


    # Return rate
    transaction_denominator = (
        sales_transactions
        + returned_transactions
    )

    return_rate = (
        returned_transactions
        / transaction_denominator
        * 100
        if transaction_denominator > 0
        else 0
    )


    return {
        "net_sales": net_sales,
        "gross_sales": gross_sales,
        "discount_amount": discount_amount,
        "sales_transactions": sales_transactions,
        "returned_transactions": returned_transactions,
        "unique_customers": unique_customers,
        "average_check": average_check,
        "discount_rate": discount_rate,
        "return_rate": return_rate,
    }


# ============================================================
# MONTHLY TREND DATA
# ============================================================

@st.cache_data
def load_monthly_trend(
    start_month,
    end_month,
    branch
):

    start_date = f"{start_month}-01"

    end_date = (
        pd.Period(
            end_month,
            freq="M"
        )
        .end_time
        .date()
        .isoformat()
    )

    # --------------------------------------------------------
    # ALL BRANCHES
    # --------------------------------------------------------

    if branch == "All Branches":

        query = """
        SELECT

            month,

            SUM(net_amount)
                AS net_sales,

            SUM(valid_gross_amount)
                AS gross_sales,

            SUM(valid_discount_amount)
                AS discount_amount,

            SUM(
                CASE
                    WHEN usable_for_sales = 1
                    THEN 1
                    ELSE 0
                END
            ) AS sales_transactions,

            COUNT(
                DISTINCT CASE
                    WHEN usable_for_sales = 1
                    THEN musteri_acari
                END
            ) AS unique_customers

        FROM clean_transactions

        WHERE
            tarix_saat >= ?
            AND tarix_saat < CAST(? AS DATE) + INTERVAL 1 DAY

        GROUP BY month

        ORDER BY month;
        """

        params = [
            start_date,
            end_date
        ]

    # --------------------------------------------------------
    # SPECIFIC BRANCH
    # --------------------------------------------------------

    else:

        query = """
        SELECT

            month,

            SUM(net_amount)
                AS net_sales,

            SUM(valid_gross_amount)
                AS gross_sales,

            SUM(valid_discount_amount)
                AS discount_amount,

            SUM(
                CASE
                    WHEN usable_for_sales = 1
                    THEN 1
                    ELSE 0
                END
            ) AS sales_transactions,

            COUNT(
                DISTINCT CASE
                    WHEN usable_for_sales = 1
                    THEN musteri_acari
                END
            ) AS unique_customers

        FROM clean_transactions

        WHERE
            tarix_saat >= ?
            AND tarix_saat < CAST(? AS DATE) + INTERVAL 1 DAY
            AND aptek_id = ?

        GROUP BY month

        ORDER BY month;
        """

        params = [
            start_date,
            end_date,
            branch
        ]

    df = run_query(
        query,
        params
    )

    df["month"] = pd.to_datetime(
        df["month"]
    )

    df["average_check"] = (
        df["net_sales"]
        / df["sales_transactions"]
    )

    return df


# ============================================================
# PRESCRIPTION ANALYSIS
# ============================================================

@st.cache_data
def load_prescription_analysis(
    start_month,
    end_month,
    branch
):

    start_date = f"{start_month}-01"

    end_date = (
        pd.Period(
            end_month,
            freq="M"
        )
        .end_time
        .date()
        .isoformat()
    )

    # --------------------------------------------------------
    # ALL BRANCHES
    # --------------------------------------------------------

    if branch == "All Branches":

        overall_query = """
        SELECT

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

            SUM(
                CASE
                    WHEN
                        usable_for_sales = 1
                        AND resepti_var = 1
                    THEN 1
                    ELSE 0
                END
            ) AS prescription_transactions,

            SUM(
                CASE
                    WHEN
                        usable_for_sales = 1
                        AND resepti_var = 0
                    THEN 1
                    ELSE 0
                END
            ) AS non_prescription_transactions,

            COUNT(
                DISTINCT CASE
                    WHEN
                        usable_for_sales = 1
                        AND resepti_var = 1
                    THEN musteri_acari
                END
            ) AS prescription_customers,

            COUNT(
                DISTINCT CASE
                    WHEN
                        usable_for_sales = 1
                        AND resepti_var = 0
                    THEN musteri_acari
                END
            ) AS non_prescription_customers

        FROM clean_transactions

        WHERE
            tarix_saat >= ?
            AND tarix_saat < CAST(? AS DATE) + INTERVAL 1 DAY;
        """

        overall_params = [
            start_date,
            end_date
        ]

        monthly_query = """
        SELECT

            month,

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

            SUM(
                CASE
                    WHEN
                        usable_for_sales = 1
                        AND resepti_var = 1
                    THEN 1
                    ELSE 0
                END
            ) AS prescription_transactions,

            SUM(
                CASE
                    WHEN
                        usable_for_sales = 1
                        AND resepti_var = 0
                    THEN 1
                    ELSE 0
                END
            ) AS non_prescription_transactions

        FROM clean_transactions

        WHERE
            tarix_saat >= ?
            AND tarix_saat < CAST(? AS DATE) + INTERVAL 1 DAY

        GROUP BY month

        ORDER BY month;
        """

        monthly_params = [
            start_date,
            end_date
        ]

    # --------------------------------------------------------
    # SPECIFIC BRANCH
    # --------------------------------------------------------

    else:

        overall_query = """
        SELECT

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

            SUM(
                CASE
                    WHEN
                        usable_for_sales = 1
                        AND resepti_var = 1
                    THEN 1
                    ELSE 0
                END
            ) AS prescription_transactions,

            SUM(
                CASE
                    WHEN
                        usable_for_sales = 1
                        AND resepti_var = 0
                    THEN 1
                    ELSE 0
                END
            ) AS non_prescription_transactions,

            COUNT(
                DISTINCT CASE
                    WHEN
                        usable_for_sales = 1
                        AND resepti_var = 1
                    THEN musteri_acari
                END
            ) AS prescription_customers,

            COUNT(
                DISTINCT CASE
                    WHEN
                        usable_for_sales = 1
                        AND resepti_var = 0
                    THEN musteri_acari
                END
            ) AS non_prescription_customers

        FROM clean_transactions

        WHERE
            tarix_saat >= ?
            AND tarix_saat < CAST(? AS DATE) + INTERVAL 1 DAY
            AND aptek_id = ?;
        """

        overall_params = [
            start_date,
            end_date,
            branch
        ]

        monthly_query = """
        SELECT

            month,

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

            SUM(
                CASE
                    WHEN
                        usable_for_sales = 1
                        AND resepti_var = 1
                    THEN 1
                    ELSE 0
                END
            ) AS prescription_transactions,

            SUM(
                CASE
                    WHEN
                        usable_for_sales = 1
                        AND resepti_var = 0
                    THEN 1
                    ELSE 0
                END
            ) AS non_prescription_transactions

        FROM clean_transactions

        WHERE
            tarix_saat >= ?
            AND tarix_saat < CAST(? AS DATE) + INTERVAL 1 DAY
            AND aptek_id = ?

        GROUP BY month

        ORDER BY month;
        """

        monthly_params = [
            start_date,
            end_date,
            branch
        ]

    overall_df = run_query(
        overall_query,
        overall_params
    )

    monthly_df = run_query(
        monthly_query,
        monthly_params
    )

    monthly_df["month"] = pd.to_datetime(
        monthly_df["month"]
    )

    row = overall_df.iloc[0]

    prescription_volume = (
        float(row["prescription_volume"])
        if pd.notna(row["prescription_volume"])
        else 0
    )

    non_prescription_volume = (
        float(row["non_prescription_volume"])
        if pd.notna(row["non_prescription_volume"])
        else 0
    )

    prescription_transactions = int(
        row["prescription_transactions"]
        if pd.notna(row["prescription_transactions"])
        else 0
    )

    non_prescription_transactions = int(
        row["non_prescription_transactions"]
        if pd.notna(row["non_prescription_transactions"])
        else 0
    )

    prescription_customers = int(
        row["prescription_customers"]
        if pd.notna(row["prescription_customers"])
        else 0
    )

    non_prescription_customers = int(
        row["non_prescription_customers"]
        if pd.notna(row["non_prescription_customers"])
        else 0
    )

    total_volume = (
        prescription_volume
        + non_prescription_volume
    )

    prescription_pct = (
        prescription_volume
        / total_volume
        * 100
        if total_volume > 0
        else 0
    )

    non_prescription_pct = (
        non_prescription_volume
        / total_volume
        * 100
        if total_volume > 0
        else 0
    )

    overall = {
        "prescription_volume":
            prescription_volume,

        "non_prescription_volume":
            non_prescription_volume,

        "prescription_transactions":
            prescription_transactions,

        "non_prescription_transactions":
            non_prescription_transactions,

        "prescription_customers":
            prescription_customers,

        "non_prescription_customers":
            non_prescription_customers,

        "prescription_pct":
            prescription_pct,

        "non_prescription_pct":
            non_prescription_pct,
    }

    return overall, monthly_df

# ============================================================
# PENSION / NON-PENSION ANALYSIS
# ============================================================

@st.cache_data
def load_pension_analysis(
    start_month,
    end_month,
    branch
):

    start_date = f"{start_month}-01"

    end_date = (
        pd.Period(
            end_month,
            freq="M"
        )
        .end_time
        .date()
        .isoformat()
    )

    # --------------------------------------------------------
    # OPTIONAL BRANCH FILTER
    # --------------------------------------------------------

    if branch == "All Branches":

        branch_condition = ""

        params = [
            start_date,
            end_date
        ]

    else:

        branch_condition = """
            AND t.aptek_id = ?
        """

        params = [
            start_date,
            end_date,
            branch
        ]

    # --------------------------------------------------------
    # OVERALL PENSION METRICS
    # --------------------------------------------------------

    overall_query = f"""
    SELECT

        -- ====================================================
        -- CUSTOMER COUNTS
        -- ====================================================

        COUNT(
            DISTINCT CASE
                WHEN c.customer_type = 'Pension'
                THEN t.musteri_acari
            END
        ) AS pension_customers,

        COUNT(
            DISTINCT CASE
                WHEN c.customer_type = 'Non-Pension'
                THEN t.musteri_acari
            END
        ) AS non_pension_customers,


        -- ====================================================
        -- SALES VOLUME
        -- ====================================================

        SUM(
            CASE
                WHEN c.customer_type = 'Pension'
                THEN t.net_amount
                ELSE 0
            END
        ) AS pension_volume,

        SUM(
            CASE
                WHEN c.customer_type = 'Non-Pension'
                THEN t.net_amount
                ELSE 0
            END
        ) AS non_pension_volume,


        -- ====================================================
        -- TRANSACTION COUNTS
        -- ====================================================

        SUM(
            CASE
                WHEN c.customer_type = 'Pension'
                THEN 1
                ELSE 0
            END
        ) AS pension_transactions,

        SUM(
            CASE
                WHEN c.customer_type = 'Non-Pension'
                THEN 1
                ELSE 0
            END
        ) AS non_pension_transactions


    FROM clean_transactions t

    INNER JOIN customer_segments c
        ON t.musteri_acari = c.musteri_acari

    WHERE

        t.usable_for_sales = 1

        AND t.tarix_saat >= ?

        AND t.tarix_saat
            < CAST(? AS DATE) + INTERVAL 1 DAY

        {branch_condition};
    """

    # --------------------------------------------------------
    # MONTHLY PENSION METRICS
    # --------------------------------------------------------

    monthly_query = f"""
    SELECT

        t.month,

        -- Pension sales
        SUM(
            CASE
                WHEN c.customer_type = 'Pension'
                THEN t.net_amount
                ELSE 0
            END
        ) AS pension_volume,

        -- Non-pension sales
        SUM(
            CASE
                WHEN c.customer_type = 'Non-Pension'
                THEN t.net_amount
                ELSE 0
            END
        ) AS non_pension_volume,

        -- Pension customers
        COUNT(
            DISTINCT CASE
                WHEN c.customer_type = 'Pension'
                THEN t.musteri_acari
            END
        ) AS pension_customers,

        -- Non-pension customers
        COUNT(
            DISTINCT CASE
                WHEN c.customer_type = 'Non-Pension'
                THEN t.musteri_acari
            END
        ) AS non_pension_customers,

        -- Pension transactions
        SUM(
            CASE
                WHEN c.customer_type = 'Pension'
                THEN 1
                ELSE 0
            END
        ) AS pension_transactions,

        -- Non-pension transactions
        SUM(
            CASE
                WHEN c.customer_type = 'Non-Pension'
                THEN 1
                ELSE 0
            END
        ) AS non_pension_transactions


    FROM clean_transactions t

    INNER JOIN customer_segments c
        ON t.musteri_acari = c.musteri_acari

    WHERE

        t.usable_for_sales = 1

        AND t.tarix_saat >= ?

        AND t.tarix_saat
            < CAST(? AS DATE) + INTERVAL 1 DAY

        {branch_condition}

    GROUP BY t.month

    ORDER BY t.month;
    """

    overall_df = run_query(
        overall_query,
        params
    )

    monthly_df = run_query(
        monthly_query,
        params
    )

    monthly_df["month"] = pd.to_datetime(
        monthly_df["month"]
    )

    row = overall_df.iloc[0]

    # --------------------------------------------------------
    # SAFE VALUES
    # --------------------------------------------------------

    pension_customers = int(
        row["pension_customers"]
        if pd.notna(row["pension_customers"])
        else 0
    )

    non_pension_customers = int(
        row["non_pension_customers"]
        if pd.notna(row["non_pension_customers"])
        else 0
    )

    pension_volume = float(
        row["pension_volume"]
        if pd.notna(row["pension_volume"])
        else 0
    )

    non_pension_volume = float(
        row["non_pension_volume"]
        if pd.notna(row["non_pension_volume"])
        else 0
    )

    pension_transactions = int(
        row["pension_transactions"]
        if pd.notna(row["pension_transactions"])
        else 0
    )

    non_pension_transactions = int(
        row["non_pension_transactions"]
        if pd.notna(row["non_pension_transactions"])
        else 0
    )

    # --------------------------------------------------------
    # TOTALS
    # --------------------------------------------------------

    total_customers = (
        pension_customers
        + non_pension_customers
    )

    total_volume = (
        pension_volume
        + non_pension_volume
    )

    # --------------------------------------------------------
    # CUSTOMER SHARES
    # --------------------------------------------------------

    pension_customer_pct = (
        pension_customers
        / total_customers
        * 100
        if total_customers > 0
        else 0
    )

    non_pension_customer_pct = (
        non_pension_customers
        / total_customers
        * 100
        if total_customers > 0
        else 0
    )

    # --------------------------------------------------------
    # SALES SHARES
    # --------------------------------------------------------

    pension_volume_pct = (
        pension_volume
        / total_volume
        * 100
        if total_volume > 0
        else 0
    )

    non_pension_volume_pct = (
        non_pension_volume
        / total_volume
        * 100
        if total_volume > 0
        else 0
    )

    # --------------------------------------------------------
    # AVERAGE CHECK
    # --------------------------------------------------------

    pension_average_check = (
        pension_volume
        / pension_transactions
        if pension_transactions > 0
        else 0
    )

    non_pension_average_check = (
        non_pension_volume
        / non_pension_transactions
        if non_pension_transactions > 0
        else 0
    )

    # --------------------------------------------------------
    # AVERAGE SPEND PER CUSTOMER
    # --------------------------------------------------------

    pension_spend_per_customer = (
        pension_volume
        / pension_customers
        if pension_customers > 0
        else 0
    )

    non_pension_spend_per_customer = (
        non_pension_volume
        / non_pension_customers
        if non_pension_customers > 0
        else 0
    )

    overall = {

        "pension_customers":
            pension_customers,

        "non_pension_customers":
            non_pension_customers,

        "pension_volume":
            pension_volume,

        "non_pension_volume":
            non_pension_volume,

        "pension_transactions":
            pension_transactions,

        "non_pension_transactions":
            non_pension_transactions,

        "pension_customer_pct":
            pension_customer_pct,

        "non_pension_customer_pct":
            non_pension_customer_pct,

        "pension_volume_pct":
            pension_volume_pct,

        "non_pension_volume_pct":
            non_pension_volume_pct,

        "pension_average_check":
            pension_average_check,

        "non_pension_average_check":
            non_pension_average_check,

        "pension_spend_per_customer":
            pension_spend_per_customer,

        "non_pension_spend_per_customer":
            non_pension_spend_per_customer,
    }

    return overall, monthly_df
# ============================================================
# BRANCH RANKING
# ============================================================

@st.cache_data
def load_branch_ranking(
    start_month,
    end_month
):

    start_date = f"{start_month}-01"

    end_date = (
        pd.Period(
            end_month,
            freq="M"
        )
        .end_time
        .date()
        .isoformat()
    )

    query = """
    SELECT

        aptek_id,

        SUM(net_amount)
            AS net_sales,

        SUM(valid_gross_amount)
            AS gross_sales,

        SUM(valid_discount_amount)
            AS discount_amount,

        SUM(
            CASE
                WHEN usable_for_sales = 1
                THEN 1
                ELSE 0
            END
        ) AS sales_transactions,

        COUNT(
            DISTINCT CASE
                WHEN usable_for_sales = 1
                THEN musteri_acari
            END
        ) AS unique_customers,

        SUM(
            CASE
                WHEN qaytarma = 1
                THEN 1
                ELSE 0
            END
        ) AS returned_transactions

    FROM clean_transactions

    WHERE
        tarix_saat >= ?
        AND tarix_saat < CAST(? AS DATE) + INTERVAL 1 DAY

    GROUP BY aptek_id

    ORDER BY net_sales DESC;
    """

    df = run_query(
        query,
        [
            start_date,
            end_date
        ]
    )

    # ========================================================
    # CALCULATED METRICS
    # ========================================================

    df["average_check"] = (
        df["net_sales"]
        / df["sales_transactions"]
    )

    df["discount_rate"] = (
        df["discount_amount"]
        / df["gross_sales"]
        * 100
    )

    df["return_rate"] = (
        df["returned_transactions"]
        /
        (
            df["sales_transactions"]
            + df["returned_transactions"]
        )
        * 100
    )

    # Ranking starts from 1
    df.insert(
        0,
        "rank",
        range(1, len(df) + 1)
    )

    return df

# ============================================================
# HEADER
# ============================================================

st.title("💊 Pharmacy Analytics Dashboard")

st.caption(
    "Branch performance, customer activity and sales analytics"
)


# ============================================================
# LOAD FILTER VALUES
# ============================================================

months_df = load_months()

branches = load_branches()


month_options = (
    months_df["month"]
    .dt.strftime("%Y-%m")
    .tolist()
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Filters")


selected_branch = st.sidebar.selectbox(
    "Branch",
    ["All Branches"] + branches
)


start_month = st.sidebar.selectbox(
    "Start Month",
    month_options,
    index=0
)


end_month = st.sidebar.selectbox(
    "End Month",
    month_options,
    index=len(month_options) - 1
)


# ============================================================
# VALIDATE MONTH RANGE
# ============================================================

if start_month > end_month:

    st.error(
        "Start Month cannot be later than End Month."
    )

    st.stop()


# ============================================================
# LOAD FILTERED DATA
# ============================================================

metrics = load_overall_metrics(
    start_month,
    end_month,
    selected_branch
)


monthly_df = load_monthly_trend(
    start_month,
    end_month,
    selected_branch
)

prescription_metrics, prescription_monthly_df = (
    load_prescription_analysis(
        start_month,
        end_month,
        selected_branch
    )
)

pension_metrics, pension_monthly_df = (
    load_pension_analysis(
        start_month,
        end_month,
        selected_branch
    )
)

branch_ranking_df = load_branch_ranking(
    start_month,
    end_month
)

# ============================================================
# FILTER SUMMARY
# ============================================================

if selected_branch == "All Branches":

    st.subheader(
        f"Company Overview — "
        f"{start_month} to {end_month}"
    )

else:

    st.subheader(
        f"Branch {selected_branch} — "
        f"{start_month} to {end_month}"
    )


# ============================================================
# KPI ROW 1
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        label="Net Sales",
        value=f"{metrics['net_sales']:,.2f} AZN"
    )


with col2:

    st.metric(
        label="Sales Transactions",
        value=f"{metrics['sales_transactions']:,}"
    )


with col3:

    st.metric(
        label="Unique Customers",
        value=f"{metrics['unique_customers']:,}"
    )


with col4:

    st.metric(
        label="Average Check",
        value=f"{metrics['average_check']:,.2f} AZN"
    )


# ============================================================
# KPI ROW 2
# ============================================================

col5, col6, col7, col8 = st.columns(4)


with col5:

    st.metric(
        label="Gross Sales",
        value=f"{metrics['gross_sales']:,.2f} AZN"
    )


with col6:

    st.metric(
        label="Discount Amount",
        value=f"{metrics['discount_amount']:,.2f} AZN"
    )


with col7:

    st.metric(
        label="Discount Rate",
        value=f"{metrics['discount_rate']:.2f}%"
    )


with col8:

    st.metric(
        label="Return Rate",
        value=f"{metrics['return_rate']:.2f}%"
    )


# ============================================================
# SPACING
# ============================================================

st.divider()


# ============================================================
# MONTHLY NET SALES TREND
# ============================================================

st.subheader("Monthly Net Sales")


fig_sales = px.line(
    monthly_df,
    x="month",
    y="net_sales",
    markers=True,
    labels={
        "month": "Month",
        "net_sales": "Net Sales (AZN)"
    }
)


fig_sales.update_layout(
    xaxis_title="Month",
    yaxis_title="Net Sales (AZN)",
    hovermode="x unified"
)


st.plotly_chart(
    fig_sales,
    use_container_width=True
)
# ============================================================================
# PASTE INTO app.py
#
# 1) Put this import next to the existing imports at the top of app.py:
#
#        from analytics.loan_analytics import (CONFIG, available_months,
#                                              branch_summary, pilot_ranking,
#                                              load_customer_features)
#
# 2) Paste the block below immediately AFTER the existing KPI row in
#    app.py (after the "Monthly Net Sales" section header is fine too).
#    Nothing above it is modified, so all current KPIs stay exactly as they are.
# ============================================================================

st.markdown("---")
st.subheader("Loan Pilot Overview")

_months = available_months()
_as_of = _months[-1]

with st.spinner("Loading customer model..."):
    _f = load_customer_features(_as_of)
    _b = branch_summary(_f, _as_of)
    _ranked = pilot_ranking(_b)

_seg = _f["segment"].value_counts()
_elig = int(_f["eligible"].sum())
_top = _ranked.iloc[0] if len(_ranked) else None

_c = st.columns(6)
_c[0].metric("Total Customers", f"{len(_f):,}")
_c[1].metric("Frequent Customers", f"{int(_seg.get('Frequent', 0)):,}",
             f"{100*_seg.get('Frequent', 0)/len(_f):.1f}% of base")
_c[2].metric("Regular Customers", f"{int(_seg.get('Regular', 0)):,}",
             f"{100*_seg.get('Regular', 0)/len(_f):.1f}% of base")
_c[3].metric("Loan-Eligible Customers", f"{_elig:,}",
             f"{100*_elig/len(_f):.1f}% of base")
_c[4].metric("Branches", f"{_f['primary_branch'].nunique():,}")
_c[5].metric("Top Pilot Branch", str(_top["branch"]) if _top is not None else "-",
             f"score {_top['pilot_score']:.0f}/100" if _top is not None else None)

_c2 = st.columns(4)
_c2[0].metric("Median Basket (all customers)", f"{_f['avg_basket'].median():.2f}")
_c2[1].metric("Median Monthly Spend", f"{_f['monthly_spend'].median():.2f}")
_c2[2].metric("Eligible Median Monthly Spend",
              f"{_f.loc[_f['eligible'], 'monthly_spend'].median():.2f}")
_c2[3].metric("Active Customers (last month)",
              f"{int((_f['recency_months'] == 0).sum()):,}")

st.caption(
    f"Segments and scores use the {CONFIG['WINDOW_MONTHS']} months ending {_as_of}. "
    "Full detail on the Customer Segmentation, Customer Scoring and Pilot Branch "
    "Selection pages."
)


# ============================================================
# TRANSACTIONS + CUSTOMERS
# ============================================================

chart_col1, chart_col2 = st.columns(2)


# ------------------------------------------------------------
# MONTHLY TRANSACTIONS
# ------------------------------------------------------------

with chart_col1:

    st.subheader(
        "Monthly Transactions"
    )

    fig_transactions = px.bar(
        monthly_df,
        x="month",
        y="sales_transactions",
        labels={
            "month": "Month",
            "sales_transactions":
                "Transactions"
        }
    )

    fig_transactions.update_layout(
        xaxis_title="Month",
        yaxis_title="Transactions"
    )

    st.plotly_chart(
        fig_transactions,
        use_container_width=True
    )


# ------------------------------------------------------------
# MONTHLY UNIQUE CUSTOMERS
# ------------------------------------------------------------

with chart_col2:

    st.subheader(
        "Monthly Unique Customers"
    )

    fig_customers = px.line(
        monthly_df,
        x="month",
        y="unique_customers",
        markers=True,
        labels={
            "month": "Month",
            "unique_customers":
                "Unique Customers"
        }
    )

    fig_customers.update_layout(
        xaxis_title="Month",
        yaxis_title="Unique Customers"
    )

    st.plotly_chart(
        fig_customers,
        use_container_width=True
    )


# ============================================================
# AVERAGE CHECK TREND
# ============================================================

st.subheader(
    "Average Check by Month"
)


fig_average_check = px.line(
    monthly_df,
    x="month",
    y="average_check",
    markers=True,
    labels={
        "month": "Month",
        "average_check":
            "Average Check (AZN)"
    }
)


fig_average_check.update_layout(
    xaxis_title="Month",
    yaxis_title="Average Check (AZN)"
)


st.plotly_chart(
    fig_average_check,
    use_container_width=True
)

# ============================================================
# PRESCRIPTION ANALYSIS
# ============================================================

st.divider()

st.header("Prescription Analysis")

st.caption(
    "Comparison of prescription and non-prescription sales."
)


# ============================================================
# PRESCRIPTION KPI CARDS
# ============================================================

p1, p2, p3, p4 = st.columns(4)


with p1:

    st.metric(
        label="Prescription Sales",
        value=(
            f"{prescription_metrics['prescription_volume']:,.2f} AZN"
        )
    )


with p2:

    st.metric(
        label="Non-Prescription Sales",
        value=(
            f"{prescription_metrics['non_prescription_volume']:,.2f} AZN"
        )
    )


with p3:

    st.metric(
        label="Prescription Share",
        value=(
            f"{prescription_metrics['prescription_pct']:.2f}%"
        )
    )


with p4:

    st.metric(
        label="Non-Prescription Share",
        value=(
            f"{prescription_metrics['non_prescription_pct']:.2f}%"
        )
    )


# ============================================================
# SECOND PRESCRIPTION KPI ROW
# ============================================================

p5, p6, p7, p8 = st.columns(4)


with p5:

    st.metric(
        label="Prescription Transactions",
        value=f"{prescription_metrics['prescription_transactions']:,}"
    )


with p6:

    st.metric(
        label="Non-Prescription Transactions",
        value=f"{prescription_metrics['non_prescription_transactions']:,}"
    )


with p7:

    st.metric(
        label="Prescription Customers",
        value=f"{prescription_metrics['prescription_customers']:,}"
    )


with p8:

    st.metric(
        label="Non-Prescription Customers",
        value=f"{prescription_metrics['non_prescription_customers']:,}"
    )


# ============================================================
# PRESCRIPTION VOLUME SPLIT
# ============================================================

prescription_split_df = pd.DataFrame({
    "Type": [
        "Prescription",
        "Non-Prescription"
    ],

    "Net Sales": [
        prescription_metrics[
            "prescription_volume"
        ],

        prescription_metrics[
            "non_prescription_volume"
        ]
    ]
})


prescription_chart_col1, prescription_chart_col2 = (
    st.columns(2)
)


# ------------------------------------------------------------
# PIE CHART
# ------------------------------------------------------------

with prescription_chart_col1:

    st.subheader(
        "Sales Volume Distribution"
    )

    fig_prescription_pie = px.pie(
        prescription_split_df,
        names="Type",
        values="Net Sales",
        hole=0.45
    )

    fig_prescription_pie.update_traces(
        textposition="inside",
        textinfo="percent+label"
    )

    st.plotly_chart(
        fig_prescription_pie,
        use_container_width=True
    )


# ------------------------------------------------------------
# TRANSACTION COMPARISON
# ------------------------------------------------------------

with prescription_chart_col2:

    st.subheader(
        "Transaction Distribution"
    )

    transaction_split_df = pd.DataFrame({
        "Type": [
            "Prescription",
            "Non-Prescription"
        ],

        "Transactions": [
            prescription_metrics[
                "prescription_transactions"
            ],

            prescription_metrics[
                "non_prescription_transactions"
            ]
        ]
    })


    fig_transaction_split = px.bar(
        transaction_split_df,
        x="Type",
        y="Transactions",
        text_auto=","
    )


    fig_transaction_split.update_layout(
        xaxis_title="",
        yaxis_title="Transactions"
    )


    st.plotly_chart(
        fig_transaction_split,
        use_container_width=True
    )


# ============================================================
# MONTHLY PRESCRIPTION VS NON-PRESCRIPTION
# ============================================================

st.subheader(
    "Monthly Prescription vs Non-Prescription Sales"
)


prescription_long_df = (
    prescription_monthly_df[
        [
            "month",
            "prescription_volume",
            "non_prescription_volume"
        ]
    ]
    .melt(
        id_vars="month",

        value_vars=[
            "prescription_volume",
            "non_prescription_volume"
        ],

        var_name="Type",

        value_name="Net Sales"
    )
)


prescription_long_df["Type"] = (
    prescription_long_df["Type"]
    .replace(
        {
            "prescription_volume":
                "Prescription",

            "non_prescription_volume":
                "Non-Prescription"
        }
    )
)


fig_monthly_prescription = px.bar(
    prescription_long_df,

    x="month",

    y="Net Sales",

    color="Type",

    barmode="stack",

    labels={
        "month": "Month",
        "Net Sales": "Net Sales (AZN)"
    }
)


fig_monthly_prescription.update_layout(
    hovermode="x unified",
    xaxis_title="Month",
    yaxis_title="Net Sales (AZN)"
)


st.plotly_chart(
    fig_monthly_prescription,
    use_container_width=True
)
# ============================================================
# PENSION / NON-PENSION ANALYSIS
# ============================================================

st.divider()

st.header(
    "Pension vs Non-Pension Analysis"
)

st.caption(
    "Customers with an overall valid discount rate above 50% "
    "are classified as Pension customers."
)


# ============================================================
# CUSTOMER KPI ROW
# ============================================================

pen1, pen2, pen3, pen4 = st.columns(4)


with pen1:

    st.metric(
        label="Pension Customers",
        value=f"{pension_metrics['pension_customers']:,}"
    )


with pen2:

    st.metric(
        label="Non-Pension Customers",
        value=f"{pension_metrics['non_pension_customers']:,}"
    )


with pen3:

    st.metric(
        label="Pension Customer Share",
        value=f"{pension_metrics['pension_customer_pct']:.2f}%"
    )


with pen4:

    st.metric(
        label="Non-Pension Customer Share",
        value=f"{pension_metrics['non_pension_customer_pct']:.2f}%"
    )


# ============================================================
# SALES KPI ROW
# ============================================================

pen5, pen6, pen7, pen8 = st.columns(4)


with pen5:

    st.metric(
        label="Pension Sales",
        value=f"{pension_metrics['pension_volume']:,.2f} AZN"
    )


with pen6:

    st.metric(
        label="Non-Pension Sales",
        value=f"{pension_metrics['non_pension_volume']:,.2f} AZN"
    )


with pen7:

    st.metric(
        label="Pension Sales Share",
        value=f"{pension_metrics['pension_volume_pct']:.2f}%"
    )


with pen8:

    st.metric(
        label="Non-Pension Sales Share",
        value=f"{pension_metrics['non_pension_volume_pct']:.2f}%"
    )


# ============================================================
# AVERAGE KPI ROW
# ============================================================

pen9, pen10, pen11, pen12 = st.columns(4)


with pen9:

    st.metric(
        label="Pension Average Check",
        value=f"{pension_metrics['pension_average_check']:,.2f} AZN"
    )


with pen10:

    st.metric(
        label="Non-Pension Average Check",
        value=f"{pension_metrics['non_pension_average_check']:,.2f} AZN"
    )


with pen11:

    st.metric(
        label="Pension Spend / Customer",
        value=f"{pension_metrics['pension_spend_per_customer']:,.2f} AZN"
    )


with pen12:

    st.metric(
        label="Non-Pension Spend / Customer",
        value=f"{pension_metrics['non_pension_spend_per_customer']:,.2f} AZN"
    )


# ============================================================
# CUSTOMER + SALES DISTRIBUTION CHARTS
# ============================================================

pension_chart_col1, pension_chart_col2 = (
    st.columns(2)
)


# ------------------------------------------------------------
# CUSTOMER DISTRIBUTION
# ------------------------------------------------------------

with pension_chart_col1:

    st.subheader(
        "Customer Distribution"
    )

    pension_customer_df = pd.DataFrame({

        "Type": [
            "Pension",
            "Non-Pension"
        ],

        "Customers": [
            pension_metrics[
                "pension_customers"
            ],

            pension_metrics[
                "non_pension_customers"
            ]
        ]
    })

    fig_pension_customers = px.pie(
        pension_customer_df,
        names="Type",
        values="Customers",
        hole=0.45
    )

    fig_pension_customers.update_traces(
        textposition="inside",
        textinfo="percent+label"
    )

    st.plotly_chart(
        fig_pension_customers,
        use_container_width=True
    )


# ------------------------------------------------------------
# SALES DISTRIBUTION
# ------------------------------------------------------------

with pension_chart_col2:

    st.subheader(
        "Sales Volume Distribution"
    )

    pension_sales_df = pd.DataFrame({

        "Type": [
            "Pension",
            "Non-Pension"
        ],

        "Net Sales": [
            pension_metrics[
                "pension_volume"
            ],

            pension_metrics[
                "non_pension_volume"
            ]
        ]
    })

    fig_pension_sales = px.pie(
        pension_sales_df,
        names="Type",
        values="Net Sales",
        hole=0.45
    )

    fig_pension_sales.update_traces(
        textposition="inside",
        textinfo="percent+label"
    )

    st.plotly_chart(
        fig_pension_sales,
        use_container_width=True
    )


# ============================================================
# MONTHLY PENSION VS NON-PENSION SALES
# ============================================================

st.subheader(
    "Monthly Pension vs Non-Pension Sales"
)

pension_sales_long_df = (
    pension_monthly_df[
        [
            "month",
            "pension_volume",
            "non_pension_volume"
        ]
    ]
    .melt(
        id_vars="month",

        value_vars=[
            "pension_volume",
            "non_pension_volume"
        ],

        var_name="Type",

        value_name="Net Sales"
    )
)

pension_sales_long_df["Type"] = (
    pension_sales_long_df["Type"]
    .replace(
        {
            "pension_volume":
                "Pension",

            "non_pension_volume":
                "Non-Pension"
        }
    )
)

fig_monthly_pension_sales = px.bar(
    pension_sales_long_df,

    x="month",

    y="Net Sales",

    color="Type",

    barmode="stack",

    labels={
        "month": "Month",
        "Net Sales": "Net Sales (AZN)"
    }
)

fig_monthly_pension_sales.update_layout(
    hovermode="x unified",
    xaxis_title="Month",
    yaxis_title="Net Sales (AZN)"
)

st.plotly_chart(
    fig_monthly_pension_sales,
    use_container_width=True
)


# ============================================================
# MONTHLY PENSION CUSTOMER COUNTS
# ============================================================

st.subheader(
    "Monthly Pension vs Non-Pension Customers"
)

pension_customer_long_df = (
    pension_monthly_df[
        [
            "month",
            "pension_customers",
            "non_pension_customers"
        ]
    ]
    .melt(
        id_vars="month",

        value_vars=[
            "pension_customers",
            "non_pension_customers"
        ],

        var_name="Type",

        value_name="Customers"
    )
)

pension_customer_long_df["Type"] = (
    pension_customer_long_df["Type"]
    .replace(
        {
            "pension_customers":
                "Pension",

            "non_pension_customers":
                "Non-Pension"
        }
    )
)

fig_monthly_pension_customers = px.line(
    pension_customer_long_df,

    x="month",

    y="Customers",

    color="Type",

    markers=True,

    labels={
        "month": "Month",
        "Customers": "Unique Customers"
    }
)

fig_monthly_pension_customers.update_layout(
    hovermode="x unified",
    xaxis_title="Month",
    yaxis_title="Unique Customers"
)

st.plotly_chart(
    fig_monthly_pension_customers,
    use_container_width=True
)

# ============================================================
# BRANCH RANKING
# ============================================================

st.divider()

st.header("Branch Ranking")

st.caption(
    f"All branches ranked by Net Sales "
    f"from {start_month} to {end_month}."
)


# ============================================================
# TOP BRANCH KPIs
# ============================================================

if not branch_ranking_df.empty:

    top_branch = branch_ranking_df.iloc[0]

    rank1, rank2, rank3, rank4 = st.columns(4)


    with rank1:

        st.metric(
            label="Top Branch",
            value=str(top_branch["aptek_id"])
        )


    with rank2:

        st.metric(
            label="Top Branch Net Sales",
            value=f"{top_branch['net_sales']:,.2f} AZN"
        )


    with rank3:

        st.metric(
            label="Top Branch Transactions",
            value=f"{int(top_branch['sales_transactions']):,}"
        )


    with rank4:

        st.metric(
            label="Top Branch Customers",
            value=f"{int(top_branch['unique_customers']):,}"
        )


# ============================================================
# TOP 20 BRANCH CHART
# ============================================================

st.subheader("Top 20 Branches by Net Sales")

top_20_branches = (
    branch_ranking_df
    .head(20)
    .copy()
)


# Reverse so highest branch appears at top
top_20_branches = (
    top_20_branches
    .sort_values(
        "net_sales",
        ascending=True
    )
)


fig_branch_ranking = px.bar(
    top_20_branches,

    x="net_sales",

    y="aptek_id",

    orientation="h",

    text="net_sales",

    labels={
        "net_sales": "Net Sales (AZN)",
        "aptek_id": "Branch"
    }
)


fig_branch_ranking.update_traces(
    texttemplate="%{text:,.0f}",
    textposition="outside"
)


fig_branch_ranking.update_layout(
    xaxis_title="Net Sales (AZN)",
    yaxis_title="Branch",
    height=700
)


st.plotly_chart(
    fig_branch_ranking,
    use_container_width=True
)


# ============================================================
# COMPLETE BRANCH RANKING TABLE
# ============================================================

st.subheader("Complete Branch Ranking")


branch_display_df = (
    branch_ranking_df.copy()
)


branch_display_df = branch_display_df.rename(
    columns={
        "rank": "Rank",

        "aptek_id":
            "Branch",

        "net_sales":
            "Net Sales",

        "gross_sales":
            "Gross Sales",

        "discount_amount":
            "Discount",

        "sales_transactions":
            "Transactions",

        "unique_customers":
            "Unique Customers",

        "average_check":
            "Average Check",

        "discount_rate":
            "Discount Rate (%)",

        "returned_transactions":
            "Returns",

        "return_rate":
            "Return Rate (%)"
    }
)


# ============================================================
# ROUND DISPLAY VALUES
# ============================================================

branch_display_df[
    "Net Sales"
] = branch_display_df[
    "Net Sales"
].round(2)


branch_display_df[
    "Gross Sales"
] = branch_display_df[
    "Gross Sales"
].round(2)


branch_display_df[
    "Discount"
] = branch_display_df[
    "Discount"
].round(2)


branch_display_df[
    "Average Check"
] = branch_display_df[
    "Average Check"
].round(2)


branch_display_df[
    "Discount Rate (%)"
] = branch_display_df[
    "Discount Rate (%)"
].round(2)


branch_display_df[
    "Return Rate (%)"
] = branch_display_df[
    "Return Rate (%)"
].round(2)


# ============================================================
# DISPLAY ALL BRANCHES
# ============================================================

st.dataframe(
    branch_display_df,

    use_container_width=True,

    hide_index=True,

    height=700
)

# ============================================================
# MONTHLY DATA TABLE
# ============================================================

with st.expander(
    "View Monthly Data"
):

    display_df = monthly_df.copy()

    display_df["month"] = (
        display_df["month"]
        .dt.strftime("%Y-%m")
    )

    display_df = display_df.rename(
        columns={
            "month": "Month",
            "net_sales": "Net Sales",
            "gross_sales": "Gross Sales",
            "discount_amount":
                "Discount Amount",
            "sales_transactions":
                "Transactions",
            "unique_customers":
                "Unique Customers",
            "average_check":
                "Average Check"
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )