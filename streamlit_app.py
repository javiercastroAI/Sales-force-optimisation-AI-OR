from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from optimai_pipeline import (
    Config,
    build_route_compare_table,
    compare_route_map,
    generate_pilot_aida_scripts,
    has_openai_api_key,
    main,
    route_divergence,
    run_field_pilot_experiment,
    viz_anomalies,
    viz_anomaly_map,
    viz_nba_hist,
    viz_roc,
    viz_route,
    viz_segmentation,
    viz_territories,
)


st.set_page_config(
    page_title="OptimAI Field Pilot (Research Grade)",
    layout="wide",
)

st.title("OptimAI Field Sales Pilot")
st.caption(
    "Research-grade evaluation of algorithmic management vs current heuristic management for face-to-face sales."
)
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f4fbf8 0%, #f8fafc 100%);
    }
    div[data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 700;
    }
    div[data-testid="stMetricDelta"] {
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

QA_FAST = os.getenv("OPTIMAI_QA_FAST", "0") == "1"

DEFAULTS = {
    "n_customers": 1800 if QA_FAST else 3600,
    "n_territories": 8 if QA_FAST else 20,
    "daily_visit_quota": 8 if QA_FAST else 10,
    "nba_lambda_km": 4.0 if QA_FAST else 5.0,
    "pilot_days": 20 if QA_FAST else 40,
    "monte_carlo_runs": 120 if QA_FAST else 300,
    "bootstrap_iterations": 800 if QA_FAST else 2000,
    "permutation_iterations": 1500 if QA_FAST else 5000,
}

PLOT_THEME = {
    "figure.facecolor": "#f8faf8",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#1f2937",
    "axes.labelcolor": "#1f2937",
    "xtick.color": "#374151",
    "ytick.color": "#374151",
    "grid.color": "#d1d5db",
    "grid.linestyle": "--",
    "grid.alpha": 0.55,
    "axes.grid": True,
    "font.size": 10,
    "axes.titleweight": "bold",
}


KPI_LABELS = {
    "sales": "Sales",
    "efficiency": "Efficiency",
    "customer_satisfaction": "Customer Satisfaction",
}


KPI_UNITS = {
    "sales": "EUR",
    "efficiency": "EUR/hour",
    "customer_satisfaction": "CSAT",
}


def render_notebook_visual(plot_fn, *args) -> None:
    real_show = plt.show
    plt.show = lambda *a, **k: None
    try:
        with plt.rc_context(PLOT_THEME):
            plot_fn(*args)
            fig = plt.gcf()
            fig.patch.set_facecolor("#f5f9f7")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    finally:
        plt.show = real_show


def render_styled_plot(fig) -> None:
    fig.patch.set_facecolor("#f5f9f7")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


with st.sidebar:
    st.header("Pipeline")
    n_customers = st.slider(
        "Synthetic accounts", min_value=1000, max_value=12000, value=DEFAULTS["n_customers"], step=100
    )
    n_territories = st.slider(
        "Total reps/territories", min_value=2, max_value=40, value=DEFAULTS["n_territories"], step=1
    )
    daily_visit_quota = st.slider(
        "Daily visits per rep", min_value=6, max_value=16, value=DEFAULTS["daily_visit_quota"], step=1
    )
    nba_lambda_km = st.slider(
        "NBA route reward (km)", min_value=0.0, max_value=15.0, value=DEFAULTS["nba_lambda_km"], step=0.5
    )
    random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
    barcelona_mode = st.selectbox("Synthetic geography", options=["gaussian", "uniform"], index=0)

    st.header("Pilot Design")
    pilot_salespersons = st.number_input("Pilot salespersons", min_value=2, max_value=2, value=2, step=1)
    pilot_days = st.slider("Pilot days", min_value=20, max_value=120, value=DEFAULTS["pilot_days"], step=5)
    monte_carlo_runs = st.slider(
        "Monte Carlo runs", min_value=100, max_value=1200, value=DEFAULTS["monte_carlo_runs"], step=100
    )

    st.header("Inference")
    bootstrap_iterations = st.slider(
        "Bootstrap iterations", min_value=500, max_value=10000, value=DEFAULTS["bootstrap_iterations"], step=500
    )
    permutation_iterations = st.slider(
        "Permutation iterations", min_value=1000, max_value=10000, value=DEFAULTS["permutation_iterations"], step=500
    )
    run_clicked = st.button("Run Research-Grade Pilot", type="primary")


if "research_run" not in st.session_state:
    st.session_state.research_run = None
if "aida_scripts_df" not in st.session_state:
    st.session_state.aida_scripts_df = None
if "aida_scripts_meta" not in st.session_state:
    st.session_state.aida_scripts_meta = {}

if run_clicked or st.session_state.research_run is None:
    with st.spinner("Running pipeline and pilot simulation..."):
        cfg = Config(
            n_territories=n_territories,
            daily_visit_quota=daily_visit_quota,
            nba_lambda_km=nba_lambda_km,
            random_state=int(random_state),
            channel_daily_capacity={"field": int(n_territories * daily_visit_quota)},
            pilot_salespersons=int(pilot_salespersons),
            pilot_days=pilot_days,
            monte_carlo_runs=monte_carlo_runs,
            bootstrap_iterations=bootstrap_iterations,
            permutation_iterations=permutation_iterations,
        )
        artifacts = main(
            cfg=cfg,
            n_customers=n_customers,
            barcelona_mode=barcelona_mode,
            show_initial_visuals=False,
        )
        pilot = run_field_pilot_experiment(
            artifacts=artifacts,
            cfg=cfg,
            n_salespersons=int(pilot_salespersons),
            n_runs=monte_carlo_runs,
            n_days=pilot_days,
            random_state=int(random_state),
        )
        st.session_state.research_run = {
            "cfg": cfg,
            "artifacts": artifacts,
            "pilot": pilot,
        }
        st.session_state.aida_scripts_df = None
        st.session_state.aida_scripts_meta = {}


payload = st.session_state.research_run
cfg = payload["cfg"]
artifacts = payload["artifacts"]
pilot = payload["pilot"]
df = artifacts.df

if pilot.kpi_summary.empty or pilot.stats_summary.empty:
    st.error("Pilot experiment returned no measurable units. Increase accounts or territories.")
    st.stop()


impact = pilot.kpi_summary.merge(pilot.stats_summary, on="kpi", how="left")
impact = impact.sort_values("kpi")

st.subheader("Experiment Design")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Pilot Salespersons", f"{len(pilot.selected_reps)}")
col_b.metric("Monte Carlo Units", f"{len(pilot.unit_metrics):,}")
col_c.metric("Pilot Days", f"{cfg.pilot_days}")
col_d.metric("Visits per Rep/Day", f"{cfg.daily_visit_quota}")
st.write(
    "Paired evaluation per rep-day: `heuristic` (before/current management) vs `algorithmic` (after/treatment). "
    "Inference uses bootstrap CIs, paired permutation tests, Wilcoxon, sign test, and FDR correction."
)


st.subheader("KPI Delta vs Heuristic Baseline")
metric_cols = st.columns(3)
for i, kpi in enumerate(["sales", "efficiency", "customer_satisfaction"]):
    row = impact[impact["kpi"] == kpi]
    if row.empty:
        continue
    r = row.iloc[0]
    label = KPI_LABELS.get(kpi, kpi)
    value = f"{r['algorithmic_mean']:.2f} {KPI_UNITS[kpi]}"
    delta = f"{r['delta_pct']:+.2f}%"
    metric_cols[i].metric(label=label, value=value, delta=delta)
    metric_cols[i].caption(
        f"Delta CI95: [{r['delta_ci_low']:.3f}, {r['delta_ci_high']:.3f}] | "
        f"Perm p={r['permutation_pvalue']:.4f} | FDR q={r['fdr_qvalue']:.4f}"
    )


view = impact.copy()
view["heuristic_mean"] = view["heuristic_mean"].round(4)
view["algorithmic_mean"] = view["algorithmic_mean"].round(4)
view["delta_abs"] = view["delta_abs"].round(4)
view["delta_pct"] = view["delta_pct"].round(3)
view["delta_ci_low"] = view["delta_ci_low"].round(4)
view["delta_ci_high"] = view["delta_ci_high"].round(4)
view["ttest_pvalue"] = view["ttest_pvalue"].round(6)
view["permutation_pvalue"] = view["permutation_pvalue"].round(6)
view["wilcoxon_pvalue"] = view["wilcoxon_pvalue"].round(6)
view["sign_test_pvalue"] = view["sign_test_pvalue"].round(6)
view["fdr_qvalue"] = view["fdr_qvalue"].round(6)
view["effect_size_cohens_d"] = view["effect_size_cohens_d"].round(4)
view["win_probability"] = view["win_probability"].round(4)

st.dataframe(view, use_container_width=True, hide_index=True)


st.subheader("Per-Rep Before/After Delta")
rep_strategy = (
    pilot.unit_metrics
    .groupby(["rep_id", "strategy"], as_index=False)
    [["sales_eur", "sales_per_hour_eur", "avg_post_visit_csat", "route_km"]]
    .mean()
)
rep_pivot = rep_strategy.pivot(index="rep_id", columns="strategy")
rep_table = pd.DataFrame({
    "rep_id": rep_pivot.index.astype(int),
    "sales_delta": rep_pivot[("sales_eur", "algorithmic")] - rep_pivot[("sales_eur", "heuristic")],
    "efficiency_delta": rep_pivot[("sales_per_hour_eur", "algorithmic")] - rep_pivot[("sales_per_hour_eur", "heuristic")],
    "csat_delta": rep_pivot[("avg_post_visit_csat", "algorithmic")] - rep_pivot[("avg_post_visit_csat", "heuristic")],
    "route_km_delta": rep_pivot[("route_km", "algorithmic")] - rep_pivot[("route_km", "heuristic")],
})
rep_table.index.name = None
rep_table = rep_table.sort_values("rep_id", ignore_index=True)
st.dataframe(rep_table.round(4), use_container_width=True, hide_index=True)


st.subheader("Distribution Diagnostics")
plot_left, plot_right = st.columns(2)


def paired_delta_series(metric_col: str) -> np.ndarray:
    paired = (
        pilot.unit_metrics
        .pivot_table(
            index=["run_id", "day", "rep_id"],
            columns="strategy",
            values=metric_col,
            aggfunc="mean",
        )
        .dropna()
    )
    return (paired["algorithmic"] - paired["heuristic"]).to_numpy(dtype=float)


with plot_left:
    with plt.rc_context(PLOT_THEME):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(paired_delta_series("sales_eur"), bins=40, alpha=0.9, color="#006d77")
        ax.axvline(0.0, color="#111827", linewidth=1)
        ax.set_title("Sales Delta Distribution (Algorithmic - Heuristic)")
        ax.set_xlabel("Sales Delta (EUR)")
        ax.set_ylabel("Frequency")
        render_styled_plot(fig)

    with plt.rc_context(PLOT_THEME):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(paired_delta_series("avg_post_visit_csat"), bins=35, alpha=0.9, color="#2a9d8f")
        ax.axvline(0.0, color="#111827", linewidth=1)
        ax.set_title("CSAT Delta Distribution")
        ax.set_xlabel("CSAT Delta")
        ax.set_ylabel("Frequency")
        render_styled_plot(fig)

with plot_right:
    with plt.rc_context(PLOT_THEME):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(paired_delta_series("sales_per_hour_eur"), bins=40, alpha=0.9, color="#ee9b00")
        ax.axvline(0.0, color="#111827", linewidth=1)
        ax.set_title("Efficiency Delta Distribution")
        ax.set_xlabel("Sales per Hour Delta (EUR/hour)")
        ax.set_ylabel("Frequency")
        render_styled_plot(fig)

    strategy_route = (
        pilot.unit_metrics
        .groupby("strategy", as_index=False)["route_km"]
        .mean()
        .sort_values("strategy")
    )
    with plt.rc_context(PLOT_THEME):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(strategy_route["strategy"], strategy_route["route_km"], color=["#9ca3af", "#0a9396"])
        ax.set_title("Average Route Distance by Strategy")
        ax.set_ylabel("Route KM")
        render_styled_plot(fig)


st.subheader("Route Comparison for Pilot Reps")
rep_choice = st.selectbox("Pilot rep", options=pilot.selected_reps, index=0)
rep_plan = pilot.route_plan[pilot.route_plan["rep_id"] == rep_choice]
heur_order = rep_plan[rep_plan["strategy"] == "heuristic"].sort_values("stop")["row_index"].astype(int).tolist()
algo_order = rep_plan[rep_plan["strategy"] == "algorithmic"].sort_values("stop")["row_index"].astype(int).tolist()
pool = list(dict.fromkeys(algo_order + heur_order))

st.write(f"Route divergence score: `{route_divergence(algo_order, heur_order)}`")
if algo_order and heur_order:
    compare_tbl = build_route_compare_table(df, algo_order, heur_order).rename(
        columns={"dist_stop": "heuristic_stop"}
    )
    st.dataframe(compare_tbl, use_container_width=True, hide_index=True)

    fmap = compare_route_map(
        df,
        rep_choice,
        algo_order,
        heur_order,
        pool,
        primary_label="Algorithmic route",
        secondary_label="Heuristic route",
    )
    if fmap is not None:
        components.html(fmap._repr_html_(), height=680, scrolling=True)
    else:
        st.info("Folium map unavailable. Install with `python3 -m pip install folium`.")
else:
    st.warning("No valid routes found for this rep under one of the strategies.")


st.subheader("LLM AIDA Scripts for Pilot Visits")
st.caption(
    "Generates a customer-specific AIDA visit script for each selected pilot visit "
    "using customer profile + NBA outputs."
)
st.caption(
    "Mode logic: `recovery_then_sales` resolves dissatisfaction/risk before AIDA; "
    "`pure_sales_industry` is direct commercial script with industry language."
)

aida_col1, aida_col2, aida_col3 = st.columns([1.2, 1.0, 1.4])
with aida_col1:
    aida_model = st.text_input("OpenAI model", value="gpt-4o-mini")
with aida_col2:
    max_visits_for_scripts = st.slider(
        "Visits per rep to script",
        min_value=1,
        max_value=max(1, int(cfg.daily_visit_quota)),
        value=min(6, int(cfg.daily_visit_quota)),
        step=1,
    )
with aida_col3:
    api_key_present = has_openai_api_key()
    if api_key_present:
        st.success("OpenAI API key detected from .env/env.")
    else:
        st.warning("OpenAI API key not detected in .env/env.")

generate_scripts_clicked = st.button(
    "Generate AIDA Scripts (Pilot Reps)",
    disabled=not api_key_present,
)

if generate_scripts_clicked:
    with st.spinner("Generating AIDA scripts with OpenAI..."):
        scripts_df = generate_pilot_aida_scripts(
            artifacts=artifacts,
            pilot=pilot,
            strategy="algorithmic",
            model=aida_model,
            max_visits_per_rep=max_visits_for_scripts,
        )
    st.session_state.aida_scripts_df = scripts_df
    st.session_state.aida_scripts_meta = {
        "model": aida_model,
        "strategy": "algorithmic",
        "max_visits_per_rep": max_visits_for_scripts,
    }

scripts_df = st.session_state.aida_scripts_df
if isinstance(scripts_df, pd.DataFrame) and not scripts_df.empty:
    meta = st.session_state.aida_scripts_meta
    st.caption(
        f"Model: `{meta.get('model', 'n/a')}` | Strategy: `{meta.get('strategy', 'algorithmic')}` | "
        f"Visits/rep: `{meta.get('max_visits_per_rep', 'n/a')}`"
    )
    summary_cols = [
        "rep_id",
        "stop",
        "client_id",
        "industry",
        "engagement_mode",
        "nba_score",
        "predicted_potential",
        "churn_prob",
        "recommended_channel",
        "error",
    ]
    summary_cols = [c for c in summary_cols if c in scripts_df.columns]
    st.dataframe(
        scripts_df[summary_cols].sort_values(["rep_id", "stop"]).round(4),
        use_container_width=True,
        hide_index=True,
    )

    rep_ids = sorted(scripts_df["rep_id"].dropna().astype(int).unique().tolist())
    script_tabs = st.tabs([f"Rep {r}" for r in rep_ids])
    for tab, rid in zip(script_tabs, rep_ids):
        rep_scripts = scripts_df[scripts_df["rep_id"] == rid].sort_values("stop")
        with tab:
            for visit in rep_scripts.itertuples(index=False):
                title = (
                    f"Stop {int(visit.stop)} | Client {int(visit.client_id)} | "
                    f"NBA {float(visit.nba_score):.2f} | Mode {visit.engagement_mode}"
                )
                with st.expander(title, expanded=(int(visit.stop) == 1)):
                    context_col, nba_col = st.columns([2.2, 1.2])
                    with context_col:
                        st.markdown("**Visit Context**")
                        st.code(str(visit.context_snapshot), language="text")
                    with nba_col:
                        st.markdown("**NBA Snapshot**")
                        st.write(f"- `industry`: {visit.industry}")
                        st.write(f"- `engagement_mode`: {visit.engagement_mode}")
                        st.write(f"- `predicted_potential`: EUR {float(visit.predicted_potential):.0f}")
                        st.write(f"- `churn_prob`: {float(visit.churn_prob):.2f}")
                        st.write(f"- `recommended_channel`: {visit.recommended_channel}")

                    st.markdown("**AIDA Script**")
                    if str(visit.error).strip():
                        st.error(str(visit.error))
                    else:
                        st.markdown(str(visit.aida_script))


st.subheader("Notebook Visual Suite (All Algorithm Steps)")
st.caption(
    "Original notebook diagnostics rendered in-app for each pipeline stage, with enhanced styling."
)

viz_tabs = st.tabs([
    "Segmentation",
    "Territories",
    "Churn ROC",
    "NBA Distribution",
    "Anomaly Score",
    "Anomaly Map",
    "Route Execution",
])

with viz_tabs[0]:
    render_notebook_visual(viz_segmentation, df)
    st.caption("Value vs potential segmentation output from model scoring and bucketization.")

with viz_tabs[1]:
    render_notebook_visual(viz_territories, df)
    st.caption("KMeans territory assignment and geographic spread.")

with viz_tabs[2]:
    render_notebook_visual(viz_roc, df)
    st.caption("Churn classifier ROC curve.")

with viz_tabs[3]:
    render_notebook_visual(viz_nba_hist, df)
    st.caption("NBA score distribution used for prioritization.")

with viz_tabs[4]:
    render_notebook_visual(viz_anomalies, df)
    st.caption("Isolation Forest anomaly scores against NBA scores.")

with viz_tabs[5]:
    render_notebook_visual(viz_anomaly_map, df)
    st.caption("Geospatial location of detected anomalies.")

with viz_tabs[6]:
    if algo_order:
        render_notebook_visual(viz_route, df, rep_choice, algo_order, pool)
        st.caption("Algorithmic face-to-face route for the selected pilot salesperson.")
    else:
        st.info("Route visualization not available for this rep.")


with st.expander("Routing Decision Audit (Algorithmic Greedy Selector)"):
    _, _, decisions = artifacts.router.optimise_for_rep(
        df[df.rep_id == rep_choice],
        return_decisions=True,
    )
    if decisions:
        st.caption(
            "Per-step candidate ranking with effective cost, travel time, and feasibility. "
            "This mirrors notebook decision snapshots."
        )
        for step_idx, snap in enumerate(decisions, start=1):
            show_cols = [
                "client_id",
                "pure_dist_km",
                "reward_km",
                "effective_cost_km",
                "arrival_min",
                "finish_min",
                "tw_start",
                "tw_end",
                "feasible",
                "chosen_flag",
            ]
            show_cols = [c for c in show_cols if c in snap.columns]
            view_step = snap[show_cols].copy()
            numeric_cols = ["pure_dist_km", "reward_km", "effective_cost_km", "arrival_min", "finish_min"]
            for col in numeric_cols:
                if col in view_step.columns:
                    view_step[col] = view_step[col].round(2)
            st.markdown(f"**Step {step_idx}**")
            st.dataframe(view_step, use_container_width=True, hide_index=True)
    else:
        st.info("No route decision snapshots available for this rep.")


with st.expander("Pipeline Monitoring and Quality"):
    if artifacts.validator_issues:
        st.warning("Quality gate issues: " + " | ".join(artifacts.validator_issues))
    st.dataframe(artifacts.channel_tracking, use_container_width=True, hide_index=True)
    st.dataframe(artifacts.channel_summary, use_container_width=True, hide_index=True)
    st.dataframe(artifacts.monitoring, use_container_width=True, hide_index=True)
    st.dataframe(artifacts.execution_trace, use_container_width=True, hide_index=True)
