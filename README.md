# Sales-force-optimisation-AI-OR
End-to-end pipeline for sales force optimisation. Sales potential, churn, anomalies, segmentation, Next Best Action and routing
# OptimAI – Enterprise Sales Force & Omnichannel Automation Pipeline

**Author:** Javier Castro (dnAI)  
**Date:** July 2025  
**License** This project is released under the MIT license. See the LICENSE file for full details
---

## Overview

**OptimAI** is an end-to-end **enterprise sales force and omnichannel automation pipeline** designed to orchestrate data, predictive models, routing logic, and operational constraints into a single, coherent decision system.

The pipeline transforms raw customer, interaction, and geospatial data into **Next Best Actions (NBA)** across field sales, contact centres, and digital channels, balancing commercial impact, customer experience, and operational capacity.

OptimAI is not a demo or point solution: it is a **system-level reference implementation** for production-grade commercial AI.

---

## Key Capabilities

### Sales Intelligence & Prediction
- Gradient Boosting models for **churn prediction** and **future value / potential estimation**
- Isolation Forest–based **anomaly detection** to flag outliers and risk accounts
- Enterprise-ready training, validation, and monitoring hooks

### Segmentation & NBA Scoring
- Quantile-based value and potential segmentation
- Composite **Next Best Action (NBA)** score combining:
  - Churn risk
  - Predicted potential
  - Recency
  - Digital engagement
  - Support friction
- Transparent weighting and explainable scoring logic

### Omnichannel Allocation
- Intelligent channel recommendation across:
  - Field sales
  - Outbound contact centre
  - Inbound contact centre
  - Assisted e-commerce
  - Self-service e-commerce
- Capacity-aware allocation with SLA constraints
- Preference-aware scoring and confidence estimation

### Field Sales Routing
- Territory design using geospatial clustering
- Representative-to-territory assignment via optimal matching
- **NBA-aware routing** using greedy TSP heuristics (with OR-Tools optional)
- Time windows, service duration, and travel-time constraints
- Optional interactive **Folium maps** for route inspection

### Contact Centre Scheduling
- Daily scheduling by channel and priority
- Backlog and SLA monitoring
- Channel-level demand vs capacity summaries

### Sales Preparation Automation
- Automated **pre-call / pre-visit preparation scripts**
- Scripts aligned with:
  - NBA rationale
  - Customer context
  - Risk signals
  - Compliance and SLA reminders

### Enterprise Guardrails
- Data quality validation gates
- Execution tracing and auditability
- KPI monitoring (AUC, RMSE, throughput, backlog)
- Structured logging and observability hooks

---

## Synthetic Geography & Data

To enable realistic experimentation without sensitive data, OptimAI includes:
- A **Barcelona-like synthetic geography**
- Configurable land masks and optional polygon constraints
- Synthetic customers, reps, territories, and interaction histories

This allows full pipeline execution while remaining data-safe.

---

## Architecture Summary

At a high level, the pipeline follows this sequence:

1. Synthetic data generation and validation  
2. Predictive model training and scoring  
3. Segmentation and NBA computation  
4. Territory design and anomaly detection  
5. Omnichannel recommendation and prioritisation  
6. Field routing and contact-centre scheduling  
7. Script generation, monitoring, and execution tracing  

All stages are configurable and designed to be independently extensible.

---

## Technologies Used

- **Python 3.10+**
- **Pandas / NumPy**
- **Scikit-learn** (Gradient Boosting, Isolation Forest, clustering)
- **SciPy** (optimisation)
- **Matplotlib** (analysis visualisations)
- **Folium** (interactive geospatial maps)
- **OR-Tools** (optional routing optimisation)
- **Shapely** (optional geospatial constraints)
- **Jupyter Notebook** (execution and inspection)

---

## Running the Pipeline

The main orchestration entry point is the `main()` function.

Example:

```python
cfg = Config(n_territories=20)
artifacts = main(cfg=cfg, n_customers=3200, show_initial_visuals=True)
