# Deploying the executive credit simulation

`executive_app.py` is a standalone, one-page version of the Pilot Credit Simulation for
executives. It is deliberately separate from the analyst dashboard:

|  | Analyst dashboard (`dashboard/app.py`) | Executive app (`executive_app.py`) |
|---|---|---|
| Data | `data/pharmacy.duckdb`, 1.6 GB | `dashboard/executive_data/*.parquet`, ~1 MB |
| Cold start | ~70 seconds | ~1 second |
| Pages | 9, including analyst tools | 1 |
| Customer identifiers | present | **none** |
| Database | DuckDB required | **none** |
| Where it runs | internal machine only | anywhere |

The extract holds seven behavioural columns — `primary_branch`, `segment`, `score`,
`monthly_spend`, `avg_basket`, `n_tx`, `tenure_months` — and no customer key. It is verified
to produce **numerically identical** simulation results to the live model.

---

## 1. Build the extract

```powershell
.venv\Scripts\python.exe scripts\06_build_executive_extract.py
```

Takes about 70 seconds (it runs the full model once) and writes:

```
dashboard/executive_data/customers.parquet   ~1,017 KB, 59,195 rows
dashboard/executive_data/branches.parquet    ~8 KB, 100 rows
dashboard/executive_data/meta.json           as-of month, row counts, build time
```

The script asserts that no identifier column reached the output and fails loudly if one did.
**Rerun this after every data refresh**, then commit the files — the deployed app reads them
straight from the repo.

## 2. Set the password

Locally, copy the example and fill it in:

```powershell
copy .streamlit\secrets.toml.example .streamlit\secrets.toml
```

```toml
app_password = "your-strong-shared-password"
```

`.streamlit/secrets.toml` is gitignored and must never be committed. On a host, paste the same
line into its Secrets box instead.

## 3. Run it locally to check

```powershell
.venv\Scripts\python.exe -m streamlit run executive_app.py
```

You should get a password prompt, then the simulation.

---

## Option A — Streamlit Community Cloud (recommended for "any browser")

Free, needs no server, and gives an `https://<name>.streamlit.app` link that opens anywhere
with no VPN.

1. Commit and push, including `dashboard/executive_data/` and `requirements.txt`:
   ```powershell
   git add requirements.txt DEPLOY.md executive_app.py dashboard .gitignore .streamlit/secrets.toml.example
   git commit -m "Add standalone executive credit simulation app"
   git push
   ```
2. Go to **share.streamlit.io** and sign in with the GitHub account that owns
   `khammadovmahammad/Pharmacy_Analysis` (the repo is private — Community Cloud can deploy
   private repos for their owner).
3. **New app** → pick the repo and branch → set **Main file path** to `executive_app.py`.
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   app_password = "your-strong-shared-password"
   ```
5. Deploy. First build takes a few minutes while dependencies install.
6. Optional but recommended: in **Settings → Sharing**, restrict viewers to specific email
   addresses. That sits on top of the password, so a leaked password alone is not enough.

**Updating the figures later:** rerun step 1, commit the refreshed Parquet, push. The app
redeploys itself within a minute and the executives' link never changes.

**Note on sleeping:** free apps sleep after inactivity and take ~30 seconds to wake. Open the
link a few minutes before any meeting.

## Option B — Docker (internal server or Azure)

Use this if the business decides the data should not sit with a third party.

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY executive_app.py .
COPY dashboard/analytics/credit_sim.py dashboard/analytics/credit_sim_ui.py dashboard/analytics/
COPY dashboard/executive_data/ dashboard/executive_data/
RUN touch dashboard/analytics/__init__.py
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
ENTRYPOINT ["streamlit", "run", "executive_app.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t pilot-credit-sim .
docker run -d -p 8501:8501 \
  -e STREAMLIT_SERVER_HEADLESS=true \
  -v /etc/pilot-secrets/secrets.toml:/app/.streamlit/secrets.toml:ro \
  pilot-credit-sim
```

Note the image copies only `credit_sim.py` and `credit_sim_ui.py` — not `loan_analytics.py`,
not the database. Put nginx or your load balancer in front for HTTPS.

For **Azure Container Apps**, push that image to ACR and deploy it; set `app_password` as a
container-app secret and map it into `.streamlit/secrets.toml`, or mount it from Key Vault.

---

## Security position

**What the app holds:** branch-level customer quality, the credit strategy, and expected
profitability per branch. Commercially sensitive, but **not personal data** — the extract has
no customer identifier, so nothing in it maps back to an individual.

**What the shared password gives you:** a gate that keeps casual visitors out.

**What it does not give you:** any record of who opened the model, or instant revocation when
someone leaves. The password can also be forwarded. For a board-level pilot document that is
usually acceptable; if it stops being acceptable, swap `check_password()` in
`executive_app.py` for `st.login()` with Microsoft Entra ID — your Streamlit version (1.61.1)
supports it natively, and the link stays the same.

**Never commit** `.streamlit/secrets.toml`, `data/`, or the DuckDB file. `.gitignore` already
blocks all three; the single deliberate exception is
`!dashboard/executive_data/*.parquet`.

---

## Verification performed

- Extract equals the live model **exactly** (0.00e+00 relative difference) across all nine
  headline metrics, on three separate configurations.
- Both apps run clean through `streamlit.testing.v1.AppTest`, agreeing on all seven headline
  KPIs.
- The password gate blocks with no password and with a wrong one; no KPI renders before login.
- `credit_sim.py` and `credit_sim_ui.py` import and run a full simulation with `duckdb` import
  deliberately blocked — proving the executive app has no database dependency.
- Engine regression suites still pass: 24 base checks, 18 re-borrow-gap checks.
