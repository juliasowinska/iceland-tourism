import streamlit as st
import numpy as np
import pandas as pd
import joblib
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick

st.set_page_config(page_title="Iceland Tourism – What-if", layout="wide")
st.title("Iceland Tourism – What-if scenarios (driver-level)")

COL = {
    "history": "tab:blue",       
    "fitted":  "lightskyblue",       
    "baseline": "#ffc99d",      # niebieski
    "scenario": "darkorange",      # pomarańczowy
    "grid": "#cfcfcf"
}

# =========================
# 1) LOAD ARTEFAKTÓW (cache)
# =========================
@st.cache_resource
def load_models():
    ols_overnight = joblib.load("ols_overnight.joblib")
    ols_cpi       = joblib.load("ols_cpi.joblib")
    ols_emp       = joblib.load("ols_emp.joblib")
    ols_turnover  = joblib.load("ols_turnover.joblib")

    resid_sarima_overnight = joblib.load("resid_sarima_overnight.joblib")
    resid_sarima_cpi       = joblib.load("resid_sarima_cpi.joblib")
    resid_sarima_emp       = joblib.load("resid_sarima_emp.joblib")
    resid_sarima_turnover  = joblib.load("resid_sarima_turnover.joblib")

    return (ols_overnight, ols_cpi, ols_emp, ols_turnover,
            resid_sarima_overnight, resid_sarima_cpi, resid_sarima_emp, resid_sarima_turnover)

@st.cache_data
def load_data():
    df_clean = pd.read_parquet("df_clean.parquet")
    low_base = pd.read_parquet("low_base.parquet")
    # upewniamy się, że indeksy są datami
    df_clean.index = pd.to_datetime(df_clean.index)
    low_base.index = pd.to_datetime(low_base.index)
    return df_clean.sort_index(), low_base.sort_index()

(df_clean, low_base) = load_data()
(ols_overnight, ols_cpi, ols_emp, ols_turnover,
 resid_sarima_overnight, resid_sarima_cpi, resid_sarima_emp, resid_sarima_turnover) = load_models()

def compute_turnover_fitted(df_clean):
    # OLS fitted na historii (na tych samych zmiennych, co w modelu)
    exog_t = ols_turnover.model.exog_names
    vars_t = [v for v in exog_t if v != "const"]

    X_hist = sm.add_constant(df_clean[vars_t].copy())
    X_hist = X_hist[exog_t]

    reg_fitted = ols_turnover.predict(X_hist)

    # SARIMA fitted reszt na historii
    resid_fitted = resid_sarima_turnover.get_prediction(
        start=df_clean.index[0],
        end=df_clean.index[-1],
        dynamic=False
    ).predicted_mean

    resid_fitted = resid_fitted.reindex(df_clean.index)

    fitted = (reg_fitted + resid_fitted).rename("turnover_fitted")
    return fitted

# =========================
# 2) FUNKCJE FORECAST (te same co w notebooku, tylko używają wczytanych modeli)
# =========================
def forecast_main_from_low(low_df):
    main_df = pd.DataFrame(index=low_df.index)

    # OVERNIGHT
    exog_ov = ols_overnight.model.exog_names
    vars_ov = [v for v in exog_ov if v != "const"]
    X_ov = sm.add_constant(low_df[vars_ov].copy())
    X_ov = X_ov[exog_ov]
    reg_ov = ols_overnight.predict(X_ov)
    resid_ov = resid_sarima_overnight.forecast(steps=len(low_df))
    resid_ov.index = low_df.index
    main_df["overnight_stays"] = reg_ov + resid_ov

    # CPI
    exog_cpi = ols_cpi.model.exog_names
    vars_cpi = [v for v in exog_cpi if v != "const"]
    X_cpi = sm.add_constant(low_df[vars_cpi].copy())
    X_cpi = X_cpi[exog_cpi]
    reg_cpi = ols_cpi.predict(X_cpi)
    resid_cpi = resid_sarima_cpi.forecast(steps=len(low_df))
    resid_cpi.index = low_df.index
    main_df["cpi_accommodation"] = reg_cpi + resid_cpi

    # EMP
    exog_emp = ols_emp.model.exog_names
    vars_emp = [v for v in exog_emp if v != "const"]
    X_emp = sm.add_constant(low_df[vars_emp].copy())
    X_emp = X_emp[exog_emp]
    reg_emp = ols_emp.predict(X_emp)
    resid_emp = resid_sarima_emp.forecast(steps=len(low_df))
    resid_emp.index = low_df.index
    main_df["empoyment_tourism"] = reg_emp + resid_emp

    return main_df

def forecast_turnover_from_main(main_df):
    exog_t = ols_turnover.model.exog_names
    vars_t = [v for v in exog_t if v != "const"]
    X_t = sm.add_constant(main_df[vars_t].copy())
    X_t = X_t[exog_t]
    reg_turn = ols_turnover.predict(X_t)
    resid_t = resid_sarima_turnover.forecast(steps=len(main_df))
    resid_t.index = main_df.index
    return (reg_turn + resid_t).rename("turnover_forecast")

turnover_fitted = compute_turnover_fitted(df_clean)

# =========================
# 3) Scenario multiplier path m(t)
# =========================
def build_multiplier_path(index, m_target: float, profile: str,
                          K_ramp: int = 6, K_up: int = 3, H_hold: int = 3, K_down: int = 3) -> pd.Series:
    n = len(index)
    t = np.arange(n, dtype=float)

    if profile == "Constant":
        w = np.ones(n, dtype=float)

    elif profile == "Ramp-up":
        K = max(int(K_ramp), 1)
        w = np.clip(t / K, 0.0, 1.0)

    elif profile == "Temporary":
        Ku = max(int(K_up), 1)
        H  = max(int(H_hold), 0)
        Kd = max(int(K_down), 1)

        w = np.zeros(n, dtype=float)
        # up
        up_end = min(Ku, n)
        if up_end > 0:
            w[:up_end] = np.linspace(0.0, 1.0, up_end, endpoint=True)

        # hold
        hold_start = Ku
        hold_end = min(Ku + H, n)
        if hold_end > hold_start:
            w[hold_start:hold_end] = 1.0

        # down
        down_start = Ku + H
        down_end = min(Ku + H + Kd, n)
        if down_end > down_start:
            w[down_start:down_end] = np.linspace(1.0, 0.0, down_end - down_start, endpoint=True)

    else:
        # fallback
        w = np.ones(n, dtype=float)

    m = 1.0 + (float(m_target) - 1.0) * w
    return pd.Series(m, index=index, name="m_path")

# Sensowne zakresy mnożników (min, max, default, step)
DRIVER_RANGES = {
    "passengers":     (0.70, 1.30, 1.00, 0.01),
    "occupancy":      (0.85, 1.15, 1.00, 0.01),
    "length_of_stay": (0.85, 1.15, 1.00, 0.01),
    "rental_cars":    (0.70, 1.30, 1.00, 0.01),
    "cpi_iceland":    (0.95, 1.05, 1.00, 0.005),
    "cpi_global":_

# =========================
# 4) UI: scenario settings + driver multipliers
# =========================
st.sidebar.header("Scenario settings")

profile = st.sidebar.selectbox(
    "Multiplier profile m(t)",
    ["Constant", "Ramp-up", "Temporary"],
    index=1  # domyślnie Ramp-up wygląda najbardziej naturalnie
)

K_ramp = 6
K_up = 3
H_hold = 3
K_down = 3

if profile == "Ramp-up":
    K_ramp = st.sidebar.slider("Ramp length (periods)", 1, 24, 6, 1)
elif profile == "Temporary":
    cA, cB = st.sidebar.columns(2)
    with cA:
        K_up = st.sidebar.slider("Up (periods)", 1, 24, 3, 1)
        H_hold = st.sidebar.slider("Hold (periods)", 0, 24, 3, 1)
    with cB:
        K_down = st.sidebar.slider("Down (periods)", 1, 24, 3, 1)

show_mpath = st.sidebar.checkbox("Show m(t) preview", value=False)

st.sidebar.header("Driver multipliers (target level vs baseline)")

pass_mult = driver_slider("passengers", "passengers (target x)")
occ_mult  = driver_slider("occupancy", "occupancy (target x)")
los_mult  = driver_slider("length_of_stay", "length_of_stay (target x)")
rent_mult = driver_slider("rental_cars", "rental_cars (target x)")
cpiisl_mult = driver_slider("cpi_iceland", "CPI Iceland (target x)")
cpiglobal_mult = driver_slider("cpi_global", "CPI global (target x)")
usa_mult = driver_slider("USA", "USA index (target x)")
trends_mult = driver_slider("google_trends", "Google trends (target x)")
unemp_mult = driver_slider("unemployment", "Unemployment (target x)")


# =========================
# 5) baseline + scenario
# =========================
main_base = forecast_main_from_low(low_base)
fc_base = forecast_turnover_from_main(main_base)

low_scen = low_base.copy()
for col, mult in {
    "passengers": pass_mult, "occupancy": occ_mult, "length_of_stay": los_mult,
    "rental_cars": rent_mult, "cpi_iceland": cpiisl_mult, "cpi_global": cpiglobal_mult,
    "USA": usa_mult, "google_trends": trends_mult, "unemployment": unemp_mult
}.items():
    if col in low_scen.columns:
        m_path = build_multiplier_path(
            low_base.index, m_target=mult, profile=profile,
            K_ramp=K_ramp, K_up=K_up, H_hold=H_hold, K_down=K_down
        )
        # kluczowa zmiana: scenariusz = baseline * m(t)
        low_scen[col] = low_base[col] * m_path

main_scen = forecast_main_from_low(low_scen)
fc_scen = forecast_turnover_from_main(main_scen)

# =========================
# 6) wykresy
# =========================
def format_year_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for lab in ax.get_xticklabels():
        lab.set_rotation(45)
        lab.set_ha("right")

def format_y(ax):
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))

SMALL = (5.5, 3.2)
WIDE  = (14, 3.6)  

if show_mpath:
    st.subheader("Preview: multiplier path m(t)")
    fig, ax = plt.subplots(figsize=SMALL)
    # podgląd dla jednego przykładowego drivera (np. passengers)
    m_preview = build_multiplier_path(low_base.index, m_target=pass_mult, profile=profile,
                                      K_ramp=K_ramp, K_up=K_up, H_hold=H_hold, K_down=K_down)
    ax.plot(m_preview.index, m_preview.values, linewidth=2)
    ax.set_title("m(t) for passengers")
    ax.grid(alpha=0.3)
    format_year_axis(ax)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.2f}"))
    st.pyplot(fig, clear_figure=True)

c1, c2, c3 = st.columns(3)

with c1:
    fig, ax = plt.subplots(figsize=SMALL)
    ax.plot(main_base.index, main_base["overnight_stays"], "--",
        label="baseline", color=COL["baseline"], linewidth=2, alpha=0.9)
    ax.plot(main_scen.index, main_scen["overnight_stays"],
        label="scenario", color=COL["scenario"], linewidth=2)
    ax.set_title("overnight_stays")
    ax.grid(alpha=0.3)
    format_year_axis(ax); format_y(ax)
    ax.legend(fontsize=8)
    st.pyplot(fig, clear_figure=True)

with c2:
    fig, ax = plt.subplots(figsize=SMALL)
    ax.plot(main_base.index, main_base["cpi_accommodation"], "--",
        label="baseline", color=COL["baseline"], linewidth=2, alpha=0.9)
    ax.plot(main_scen.index, main_scen["cpi_accommodation"],
        label="scenario", color=COL["scenario"], linewidth=2)
    ax.set_title("cpi_accommodation")
    ax.grid(alpha=0.3)
    format_year_axis(ax); format_y(ax)
    ax.legend(fontsize=8)
    st.pyplot(fig, clear_figure=True)

with c3:
    fig, ax = plt.subplots(figsize=SMALL)
    ax.plot(main_base.index, main_base["empoyment_tourism"], "--",
        label="baseline", color=COL["baseline"], linewidth=2, alpha=0.9)
    ax.plot(main_scen.index, main_scen["empoyment_tourism"],
        label="scenario", color=COL["scenario"], linewidth=2)
    ax.set_title("employment_tourism")
    ax.grid(alpha=0.3)
    format_year_axis(ax); format_y(ax)
    ax.legend(fontsize=8)
    st.pyplot(fig, clear_figure=True)

fig, ax = plt.subplots(figsize=WIDE)
ax.plot(df_clean.index, df_clean["turnover"],
        label="History", color=COL["history"], linewidth=2)
ax.plot(turnover_fitted.index, turnover_fitted.values,
        label="Baseline (history) – fitted", color=COL["fitted"], linewidth=2, alpha=0.9)
ax.plot(fc_base.index, fc_base.values,
        label="Baseline forecast", color=COL["baseline"], linewidth=2, marker="o")
ax.plot(fc_scen.index, fc_scen.values,
        label="Scenario forecast", color=COL["scenario"], linewidth=2, marker="o")
ax.axvline(df_clean.index.max(), color="gray", linestyle="--", alpha=0.7)
ax.set_title("Turnover – history, fitted, baseline vs scenario")
ax.grid(alpha=0.3)
format_year_axis(ax); format_y(ax)
ax.legend()
st.pyplot(fig, clear_figure=True)


with st.expander("Tables"):
    st.subheader("Scenario drivers (low-level)")
    st.dataframe(low_scen)
    st.subheader("Scenario main components")
    st.dataframe(main_scen)
    st.subheader("Scenario turnover forecast")
    st.dataframe(fc_scen.to_frame())
