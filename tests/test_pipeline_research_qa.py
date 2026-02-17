from __future__ import annotations

import pandas as pd

from optimai_pipeline import (
    Config,
    build_route_compare_table,
    generate_pilot_aida_scripts,
    main,
    route_divergence,
    run_field_pilot_experiment,
)


REQUIRED_ENRICHED_COLUMNS = {
    "budget_fit_score",
    "authority_access_score",
    "need_intensity_score",
    "timeline_urgency_score",
    "relationship_strength",
    "meeting_accept_rate",
    "competitor_pressure_index",
    "implementation_risk_score",
    "value_realization_score",
    "account_complexity_index",
    "strategic_fit_score",
    "product_adoption_index",
    "buying_committee_size",
    "decision_cycle_days",
    "nps_score",
}


def _qa_cfg(seed: int = 123) -> Config:
    return Config(
        random_state=seed,
        n_territories=8,
        daily_visit_quota=8,
        channel_daily_capacity={"field": 64},
        pilot_salespersons=2,
        pilot_days=12,
        monte_carlo_runs=40,
        bootstrap_iterations=600,
        permutation_iterations=2000,
    )


def _build_artifacts_and_pilot(seed: int = 123):
    cfg = _qa_cfg(seed)
    artifacts = main(cfg=cfg, n_customers=700, show_initial_visuals=False)
    pilot = run_field_pilot_experiment(
        artifacts=artifacts,
        cfg=cfg,
        n_salespersons=2,
        n_runs=cfg.monte_carlo_runs,
        n_days=cfg.pilot_days,
        random_state=seed,
    )
    return cfg, artifacts, pilot


def test_pipeline_field_only_and_enriched_schema():
    cfg, artifacts, _ = _build_artifacts_and_pilot()
    df = artifacts.df

    assert cfg.channels == ("field",)
    assert set(df["recommended_channel"].unique()) == {"field"}
    assert REQUIRED_ENRICHED_COLUMNS.issubset(df.columns)


def test_quality_gate_and_missingness_constraints_hold():
    cfg, artifacts, _ = _build_artifacts_and_pilot()
    df = artifacts.df

    assert artifacts.validator_issues == []
    amount_nan_ratio = float(df["amount"].isna().mean())
    assert amount_nan_ratio <= cfg.max_allowed_nan_ratio


def test_pilot_outputs_complete_and_research_grade_stats_present():
    _, _, pilot = _build_artifacts_and_pilot()

    assert len(pilot.selected_reps) == 2
    assert not pilot.route_plan.empty
    assert not pilot.unit_metrics.empty

    expected_kpis = {"sales", "efficiency", "customer_satisfaction"}
    assert set(pilot.kpi_summary["kpi"].tolist()) == expected_kpis
    assert set(pilot.stats_summary["kpi"].tolist()) == expected_kpis

    expected_stat_cols = {
        "delta_ci_low",
        "delta_ci_high",
        "ttest_pvalue",
        "permutation_pvalue",
        "wilcoxon_pvalue",
        "sign_test_pvalue",
        "effect_size_cohens_d",
        "fdr_qvalue",
        "significant_fdr_95",
    }
    assert expected_stat_cols.issubset(pilot.stats_summary.columns)


def test_algorithmic_beats_heuristic_on_target_kpis_with_significance():
    _, _, pilot = _build_artifacts_and_pilot()
    joined = pilot.kpi_summary.merge(pilot.stats_summary, on="kpi", how="left")

    # Core business objective: improve all three KPIs vs current heuristic baseline.
    assert (joined["delta_abs"] > 0).all()
    assert (joined["delta_pct"] > 0).all()

    # Research-grade confidence: FDR-controlled significance for all KPIs.
    assert joined["significant_fdr_95"].all()


def test_route_comparison_helpers_are_consistent():
    _, _, pilot = _build_artifacts_and_pilot()
    route_plan = pilot.route_plan
    rep_id = int(pilot.selected_reps[0])

    algo_order = (
        route_plan[(route_plan["rep_id"] == rep_id) & (route_plan["strategy"] == "algorithmic")]
        .sort_values("stop")["row_index"]
        .astype(int)
        .tolist()
    )
    heur_order = (
        route_plan[(route_plan["rep_id"] == rep_id) & (route_plan["strategy"] == "heuristic")]
        .sort_values("stop")["row_index"]
        .astype(int)
        .tolist()
    )
    assert len(algo_order) > 0
    assert len(heur_order) > 0

    # Build synthetic df lookup from route_plan records to exercise the helper.
    # We only need a subset of fields used by build_route_compare_table.
    lookup_cols = ["row_index", "client_id", "nba_score", "strategy"]
    lookup = (
        route_plan[lookup_cols]
        .drop_duplicates(subset=["row_index"])
        .set_index("row_index")
        .rename(columns={"strategy": "recommended_channel"})
    )
    table = build_route_compare_table(lookup, algo_order, heur_order)

    assert len(table) == len(algo_order)
    assert set(table.columns) == {"nba_stop", "dist_stop", "client_id", "nba_score", "channel"}
    assert route_divergence(algo_order, heur_order) >= 0


def test_reproducibility_of_kpi_and_stats_for_fixed_seed():
    _, _, pilot_a = _build_artifacts_and_pilot(seed=321)
    _, _, pilot_b = _build_artifacts_and_pilot(seed=321)

    a_kpi = pilot_a.kpi_summary.sort_values("kpi").reset_index(drop=True)
    b_kpi = pilot_b.kpi_summary.sort_values("kpi").reset_index(drop=True)
    pd.testing.assert_frame_equal(a_kpi, b_kpi)

    stats_cols = [
        "kpi",
        "delta_ci_low",
        "delta_ci_high",
        "permutation_pvalue",
        "fdr_qvalue",
        "significant_fdr_95",
    ]
    a_stats = pilot_a.stats_summary[stats_cols].sort_values("kpi").reset_index(drop=True)
    b_stats = pilot_b.stats_summary[stats_cols].sort_values("kpi").reset_index(drop=True)
    pd.testing.assert_frame_equal(a_stats, b_stats)


def test_pilot_aida_scripts_generated_for_pilot_reps_only():
    _, artifacts, pilot = _build_artifacts_and_pilot(seed=222)
    alg_plan = (
        pilot.route_plan[pilot.route_plan["strategy"] == "algorithmic"]
        .sort_values(["rep_id", "stop"])
        .reset_index(drop=True)
    )
    first_idx = int(alg_plan.loc[0, "row_index"])
    artifacts.df.loc[first_idx, "csat_score"] = 4.0
    artifacts.df.loc[first_idx, "recent_issue_flag"] = 1
    artifacts.df.loc[first_idx, "churn_prob"] = 0.60
    prompts = []

    def fake_llm(prompt: str) -> str:
        prompts.append(prompt)
        assert "NBA score" in prompt
        assert "Do NOT mention these terms" in prompt
        return (
            "Attention:\n"
            "- Open with a relevant business insight.\n"
            "Interest:\n"
            "- Link NBA signals to their current priorities.\n"
            "Desire:\n"
            "- Quantify expected business impact.\n"
            "Action:\n"
            "- Propose concrete next step and owner.\n"
            "Compliance: Confirm consent and meeting note policy."
        )

    scripts = generate_pilot_aida_scripts(
        artifacts=artifacts,
        pilot=pilot,
        strategy="algorithmic",
        max_visits_per_rep=2,
        llm_callable=fake_llm,
    )
    assert not scripts.empty
    assert set(scripts["rep_id"]).issubset(set(pilot.selected_reps))
    assert set(scripts["strategy"]) == {"algorithmic"}
    assert scripts.groupby("rep_id")["stop"].count().max() <= 2
    assert scripts["aida_script"].str.contains("Attention:", regex=False).all()
    assert scripts["engagement_mode"].isin(
        {"recovery_then_sales", "pure_sales_industry", "consultative_growth"}
    ).all()
    assert "recovery_then_sales" in set(scripts["engagement_mode"])
    assert (scripts["error"].fillna("") == "").all()
    assert len(prompts) == len(scripts)
