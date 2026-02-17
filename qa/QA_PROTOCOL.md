# Research-Grade QA Protocol

## Scope
- `optimai_pipeline.py`
- `streamlit_app.py`

## QA Dimensions
1. Build integrity: syntax compilation for all executable entry points.
2. Pipeline integrity: deterministic field-only workflow and enriched synthetic schema.
3. Experimental integrity: before/after paired evaluation and robust statistics.
4. Reproducibility: fixed-seed deterministic KPI and inference outputs.
5. App integrity: Streamlit app executes and renders without runtime exceptions.

## Statistical Acceptance Criteria
- KPI family: `sales`, `efficiency`, `customer_satisfaction`
- For fixed QA seed/config:
  - positive delta vs heuristic for all three KPIs
  - FDR-controlled significance (`q < 0.05`) for all three KPIs

## Execution
Run:

```bash
bash qa/run_research_grade_qa.sh
```

Artifacts:
- `qa/pytest-report.xml`
- `qa/last_qa_run.log`
