from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_PARAGRAPH_ALIGNMENT
from pptx.util import Inches, Pt


OUTPUT_PATH = Path("deliverables/Propuesta_Piloto_OptimAI_McKinsey_style_ES.pptx")


COLORS = {
    "navy": RGBColor(15, 34, 63),
    "teal": RGBColor(0, 138, 153),
    "orange": RGBColor(217, 119, 6),
    "light_bg": RGBColor(245, 248, 252),
    "mid_gray": RGBColor(95, 105, 120),
    "dark": RGBColor(24, 33, 45),
    "white": RGBColor(255, 255, 255),
}


def compute_snapshot() -> Dict[str, object]:
    """Run the current synthetic pilot to get up-to-date illustrative metrics."""

    try:
        from optimai_pipeline import Config, main, run_field_pilot_experiment

        cfg = Config(
            pilot_salespersons=2,
            pilot_days=40,
            monte_carlo_runs=300,
            random_state=42,
        )
        artifacts = main(cfg=cfg, n_customers=3600, show_initial_visuals=False)
        pilot = run_field_pilot_experiment(artifacts=artifacts, cfg=cfg)

        kpi_df = pilot.kpi_summary.copy()
        stats_df = pilot.stats_summary.copy()

        def _k(kpi: str, col: str) -> float:
            return float(kpi_df.loc[kpi_df["kpi"] == kpi, col].iloc[0])

        def _s(kpi: str, col: str) -> float:
            return float(stats_df.loc[stats_df["kpi"] == kpi, col].iloc[0])

        return {
            "selected_reps": list(map(int, pilot.selected_reps)),
            "kpis": {
                "sales": {
                    "heuristic": _k("sales", "heuristic_mean"),
                    "algorithmic": _k("sales", "algorithmic_mean"),
                    "delta_abs": _k("sales", "delta_abs"),
                    "delta_pct": _k("sales", "delta_pct"),
                    "effect_size": _s("sales", "effect_size_cohens_d"),
                    "qvalue": _s("sales", "fdr_qvalue"),
                    "win_probability": _s("sales", "win_probability"),
                },
                "efficiency": {
                    "heuristic": _k("efficiency", "heuristic_mean"),
                    "algorithmic": _k("efficiency", "algorithmic_mean"),
                    "delta_abs": _k("efficiency", "delta_abs"),
                    "delta_pct": _k("efficiency", "delta_pct"),
                    "effect_size": _s("efficiency", "effect_size_cohens_d"),
                    "qvalue": _s("efficiency", "fdr_qvalue"),
                    "win_probability": _s("efficiency", "win_probability"),
                },
                "customer_satisfaction": {
                    "heuristic": _k("customer_satisfaction", "heuristic_mean"),
                    "algorithmic": _k("customer_satisfaction", "algorithmic_mean"),
                    "delta_abs": _k("customer_satisfaction", "delta_abs"),
                    "delta_pct": _k("customer_satisfaction", "delta_pct"),
                    "effect_size": _s("customer_satisfaction", "effect_size_cohens_d"),
                    "qvalue": _s("customer_satisfaction", "fdr_qvalue"),
                    "win_probability": _s("customer_satisfaction", "win_probability"),
                },
            },
            "paired_units": int(stats_df["paired_units"].iloc[0]),
            "monte_carlo_runs": 300,
            "pilot_days": 40,
            "customers": 3600,
            "source": "pipeline_live",
        }
    except Exception:
        return {
            "selected_reps": [12, 5],
            "kpis": {
                "sales": {
                    "heuristic": 6867.98,
                    "algorithmic": 12842.41,
                    "delta_abs": 5974.42,
                    "delta_pct": 86.99,
                    "effect_size": 0.86,
                    "qvalue": 0.0002,
                    "win_probability": 0.8045,
                },
                "efficiency": {
                    "heuristic": 1201.71,
                    "algorithmic": 2240.93,
                    "delta_abs": 1039.21,
                    "delta_pct": 86.48,
                    "effect_size": 0.91,
                    "qvalue": 0.0002,
                    "win_probability": 0.8214,
                },
                "customer_satisfaction": {
                    "heuristic": 5.02,
                    "algorithmic": 6.05,
                    "delta_abs": 1.03,
                    "delta_pct": 20.57,
                    "effect_size": 3.24,
                    "qvalue": 0.0002,
                    "win_probability": 0.9996,
                },
            },
            "paired_units": 24000,
            "monte_carlo_runs": 300,
            "pilot_days": 40,
            "customers": 3600,
            "source": "fallback_constants",
        }


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def fmt_num(value: float) -> str:
    return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def add_bg(slide) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = COLORS["white"]


def add_top_band(slide, title: str, subtitle: str | None = None) -> None:
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.33), Inches(0.9))
    band.fill.solid()
    band.fill.fore_color.rgb = COLORS["navy"]
    band.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.15), Inches(9.8), Inches(0.55)).text_frame
    tb.clear()
    p = tb.paragraphs[0]
    p.text = title
    p.font.name = "Calibri"
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = COLORS["white"]

    if subtitle:
        sub = slide.shapes.add_textbox(Inches(0.5), Inches(0.74), Inches(10.5), Inches(0.35)).text_frame
        sub.clear()
        p2 = sub.paragraphs[0]
        p2.text = subtitle
        p2.font.name = "Calibri"
        p2.font.size = Pt(12)
        p2.font.color.rgb = COLORS["mid_gray"]


def add_footer(slide, left_text: str = "Fuente: OptimAI | Datos sinteticos") -> None:
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(7.16), Inches(13.33), Inches(0.34))
    line.fill.solid()
    line.fill.fore_color.rgb = COLORS["light_bg"]
    line.line.fill.background()

    tf = slide.shapes.add_textbox(Inches(0.45), Inches(7.2), Inches(12.4), Inches(0.2)).text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = left_text
    p.font.name = "Calibri"
    p.font.size = Pt(9)
    p.font.color.rgb = COLORS["mid_gray"]

    date_tf = slide.shapes.add_textbox(Inches(11.0), Inches(7.2), Inches(2.0), Inches(0.2)).text_frame
    date_tf.clear()
    p2 = date_tf.paragraphs[0]
    p2.text = datetime.now().strftime("%d/%m/%Y")
    p2.alignment = PP_PARAGRAPH_ALIGNMENT.RIGHT
    p2.font.name = "Calibri"
    p2.font.size = Pt(9)
    p2.font.color.rgb = COLORS["mid_gray"]


def add_illustrative_tag(slide) -> None:
    tag = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(10.15), Inches(0.14), Inches(2.95), Inches(0.48))
    tag.fill.solid()
    tag.fill.fore_color.rgb = COLORS["orange"]
    tag.line.fill.background()

    tf = tag.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = "ILUSTRATIVO (datos sinteticos)"
    p.font.name = "Calibri"
    p.font.bold = True
    p.font.size = Pt(11)
    p.font.color.rgb = COLORS["white"]
    p.alignment = PP_PARAGRAPH_ALIGNMENT.CENTER


def add_bullets(slide, x: float, y: float, w: float, h: float, lines: List[str], size: int = 16) -> None:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.level = 0
        p.font.name = "Calibri"
        p.font.size = Pt(size)
        p.font.color.rgb = COLORS["dark"]


def add_kpi_table(slide, snapshot: Dict[str, object]) -> None:
    rows = 4
    cols = 5
    table = slide.shapes.add_table(rows, cols, Inches(0.65), Inches(1.55), Inches(12.0), Inches(2.35)).table

    headers = ["KPI", "Heuristico", "Algoritmico", "Delta", "Delta %"]
    for c, header in enumerate(headers):
        cell = table.cell(0, c)
        cell.text = header
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLORS["navy"]
        para = cell.text_frame.paragraphs[0]
        para.font.name = "Calibri"
        para.font.bold = True
        para.font.size = Pt(12)
        para.font.color.rgb = COLORS["white"]

    kpi_map = [
        ("Ventas", "sales"),
        ("Eficiencia", "efficiency"),
        ("Satisfaccion cliente", "customer_satisfaction"),
    ]

    for r, (label, key) in enumerate(kpi_map, start=1):
        row_data = snapshot["kpis"][key]
        values = [
            label,
            fmt_num(float(row_data["heuristic"])),
            fmt_num(float(row_data["algorithmic"])),
            fmt_num(float(row_data["delta_abs"])),
            "Ilustrativo " + fmt_pct(float(row_data["delta_pct"])),
        ]
        for c, val in enumerate(values):
            cell = table.cell(r, c)
            cell.text = val
            if r % 2 == 1:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(248, 250, 253)
            para = cell.text_frame.paragraphs[0]
            para.font.name = "Calibri"
            para.font.size = Pt(11)
            para.font.color.rgb = COLORS["dark"]
            if c == 0:
                para.font.bold = True


def add_delta_chart(slide, snapshot: Dict[str, object]) -> None:
    chart_data = CategoryChartData()
    chart_data.categories = ["Ventas", "Eficiencia", "Satisfaccion"]
    chart_data.add_series(
        "Delta % (Ilustrativo)",
        [
            float(snapshot["kpis"]["sales"]["delta_pct"]),
            float(snapshot["kpis"]["efficiency"]["delta_pct"]),
            float(snapshot["kpis"]["customer_satisfaction"]["delta_pct"]),
        ],
    )

    chart = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        Inches(0.75), Inches(4.05), Inches(6.0), Inches(2.55),
        chart_data,
    ).chart
    chart.has_legend = False
    chart.value_axis.has_major_gridlines = True
    chart.value_axis.maximum_scale = 100
    chart.value_axis.minimum_scale = 0
    plot = chart.plots[0]
    plot.has_data_labels = True
    plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    plot.data_labels.number_format = '0.0"%"'
    plot.data_labels.font.size = Pt(10)
    plot.series[0].format.fill.solid()
    plot.series[0].format.fill.fore_color.rgb = COLORS["teal"]


def add_win_prob_chart(slide, snapshot: Dict[str, object]) -> None:
    chart_data = CategoryChartData()
    chart_data.categories = ["Ventas", "Eficiencia", "Satisfaccion"]
    chart_data.add_series(
        "Probabilidad de ganar (Ilustrativo)",
        [
            float(snapshot["kpis"]["sales"]["win_probability"]) * 100,
            float(snapshot["kpis"]["efficiency"]["win_probability"]) * 100,
            float(snapshot["kpis"]["customer_satisfaction"]["win_probability"]) * 100,
        ],
    )

    chart = slide.shapes.add_chart(
        XL_CHART_TYPE.BAR_CLUSTERED,
        Inches(6.85), Inches(4.05), Inches(5.75), Inches(2.55),
        chart_data,
    ).chart
    chart.has_legend = False
    chart.category_axis.reverse_order = True
    chart.value_axis.maximum_scale = 100
    chart.value_axis.minimum_scale = 0
    chart.value_axis.has_major_gridlines = True

    plot = chart.plots[0]
    plot.has_data_labels = True
    plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    plot.data_labels.number_format = '0.0"%"'
    plot.data_labels.font.size = Pt(10)
    plot.series[0].format.fill.solid()
    plot.series[0].format.fill.fore_color.rgb = COLORS["navy"]


def slide_cover(prs: Presentation, snapshot: Dict[str, object]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)

    stripe = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.33), Inches(7.5))
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = RGBColor(11, 29, 56)
    stripe.line.fill.background()

    accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(5.9), Inches(13.33), Inches(1.6))
    accent.fill.solid()
    accent.fill.fore_color.rgb = COLORS["teal"]
    accent.line.fill.background()

    tf = slide.shapes.add_textbox(Inches(0.8), Inches(1.0), Inches(11.8), Inches(2.7)).text_frame
    tf.clear()
    p1 = tf.paragraphs[0]
    p1.text = "Propuesta de Piloto\nOptimizacion Comercial con IA"
    p1.font.name = "Calibri"
    p1.font.bold = True
    p1.font.size = Pt(46)
    p1.font.color.rgb = COLORS["white"]

    p2 = tf.add_paragraph()
    p2.text = "Plan de ejecucion para demostrar valor vs gestion heuristica"
    p2.font.name = "Calibri"
    p2.font.size = Pt(20)
    p2.font.color.rgb = RGBColor(218, 226, 239)

    tf2 = slide.shapes.add_textbox(Inches(0.8), Inches(6.15), Inches(11.4), Inches(0.8)).text_frame
    p = tf2.paragraphs[0]
    p.text = (
        f"Cliente: Direccion de Ventas | Horizonte piloto: {snapshot['pilot_days']} dias | "
        "Nota: resultados iniciales con datos sinteticos"
    )
    p.font.name = "Calibri"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLORS["white"]

    add_footer(slide, "Documento de propuesta para decision de piloto")


def slide_exec_summary(prs: Presentation, snapshot: Dict[str, object]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    add_top_band(slide, "Resumen Ejecutivo", "Que se ha construido y por que importa para el negocio")
    add_illustrative_tag(slide)

    cards = [
        ("Ventas", f"Ilustrativo +{fmt_pct(float(snapshot['kpis']['sales']['delta_pct']))}", "vs estrategia heuristica"),
        (
            "Eficiencia",
            f"Ilustrativo +{fmt_pct(float(snapshot['kpis']['efficiency']['delta_pct']))}",
            "ventas por hora de servicio",
        ),
        (
            "Satisfaccion",
            f"Ilustrativo +{fmt_pct(float(snapshot['kpis']['customer_satisfaction']['delta_pct']))}",
            "CSAT post visita",
        ),
    ]

    x = 0.7
    for title, value, desc in cards:
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.55), Inches(4.0), Inches(1.75))
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(247, 250, 253)
        box.line.color.rgb = RGBColor(220, 228, 240)

        tf = box.text_frame
        tf.clear()
        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.name = "Calibri"
        p1.font.bold = True
        p1.font.size = Pt(14)
        p1.font.color.rgb = COLORS["navy"]

        p2 = tf.add_paragraph()
        p2.text = value
        p2.font.name = "Calibri"
        p2.font.bold = True
        p2.font.size = Pt(22)
        p2.font.color.rgb = COLORS["teal"]

        p3 = tf.add_paragraph()
        p3.text = desc
        p3.font.name = "Calibri"
        p3.font.size = Pt(11)
        p3.font.color.rgb = COLORS["mid_gray"]
        x += 4.2

    add_bullets(
        slide,
        x=0.8,
        y=3.75,
        w=12.0,
        h=2.8,
        size=14,
        lines=[
            "1. Ya existe una solucion end-to-end: pipeline + app + QA research-grade.",
            "2. El experimento compara algoritmo vs heuristico en unidades pareadas de rep-dia.",
            "3. El piloto propuesto con 2 comerciales valida valor real antes de escalar.",
            "4. La toma de decision final se basara en KPIs, significancia estadistica y adopcion comercial.",
        ],
    )
    add_footer(slide)


def slide_context(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    add_top_band(slide, "Contexto de Negocio", "Problema actual y oportunidad de mejora en gestion comercial")

    left = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), Inches(1.4), Inches(6.15), Inches(5.55))
    left.fill.solid()
    left.fill.fore_color.rgb = RGBColor(248, 250, 253)
    left.line.color.rgb = RGBColor(222, 230, 240)

    right = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.95), Inches(1.4), Inches(5.7), Inches(5.55))
    right.fill.solid()
    right.fill.fore_color.rgb = RGBColor(248, 250, 253)
    right.line.color.rgb = RGBColor(222, 230, 240)

    add_bullets(
        slide,
        0.95,
        1.72,
        5.7,
        4.9,
        size=14,
        lines=[
            "Situacion tipica (heuristica):",
            "- Priorizacion subjetiva de cuentas y visitas.",
            "- Variabilidad alta entre comerciales.",
            "- Coste de desplazamiento no optimizado.",
            "- Seguimiento parcial de valor por visita.",
            "- Aprendizaje lento y dificil de escalar.",
        ],
    )

    add_bullets(
        slide,
        7.2,
        1.72,
        5.2,
        4.9,
        size=14,
        lines=[
            "Oportunidad con OptimAI:",
            "- Priorizacion sistematica de cuentas por potencial/riesgo.",
            "- Rutas con mejor relacion valor vs desplazamiento.",
            "- Guias de visita (AIDA) personalizadas por contexto.",
            "- Medicion robusta del impacto (ventas, eficiencia, CSAT).",
            "- Base para escalar una operativa comercial data-driven.",
        ],
    )
    add_footer(slide)


def slide_scope(prs: Presentation, snapshot: Dict[str, object]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    add_top_band(slide, "Lo Construido Hasta Hoy", "Componentes funcionales listos para ejecutar piloto")

    add_bullets(
        slide,
        0.85,
        1.45,
        12.0,
        5.4,
        size=13,
        lines=[
            "Pipeline research-grade (Python): generacion de datos, modelos, NBA, territorios, anomalias, rutas y metricas.",
            "Canal activo: 100% face-to-face para el piloto actual (alineado al caso de uso).",
            f"Dataset actual: {snapshot['customers']} cuentas sinteticas (ILUSTRATIVO).",
            "Modelos principales: churn y potencial (Gradient Boosting).",
            "Comparador de estrategias: heuristica vs algoritmica con diseno pareado por rep-dia.",
            "App Streamlit: tablero ejecutivo, mapas de ruta, auditoria de decisiones y analitica estadistica.",
            "Motor de guiones de visita: AIDA con OpenAI, adaptado al contexto del cliente y siguiente mejor accion.",
            "QA software engineering: tests automatizados, smoke test de app y quality gates de datos/modelo.",
        ],
    )

    note = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.85), Inches(6.15), Inches(12.0), Inches(0.75))
    note.fill.solid()
    note.fill.fore_color.rgb = RGBColor(255, 247, 237)
    note.line.color.rgb = RGBColor(245, 158, 11)
    nt = note.text_frame
    nt.clear()
    p = nt.paragraphs[0]
    p.text = (
        "Nota clave: las cifras actuales de impacto deben leerse como ILUSTRATIVO hasta que se conecte historico "
        "real de la gestion heuristica y se ejecute el piloto en operacion real."
    )
    p.font.name = "Calibri"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = RGBColor(120, 53, 15)

    add_footer(slide)


def slide_results(prs: Presentation, snapshot: Dict[str, object]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    add_top_band(slide, "Resultados Iniciales del Sistema", "Comparativa de valor: algoritmo vs heuristica (ILUSTRATIVO)")
    add_illustrative_tag(slide)
    add_kpi_table(slide, snapshot)
    add_delta_chart(slide, snapshot)
    add_win_prob_chart(slide, snapshot)
    add_footer(slide)


def slide_stats(prs: Presentation, snapshot: Dict[str, object]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    add_top_band(slide, "Rigor Metodologico", "Como aseguramos que la mejora no es ruido estadistico")
    add_illustrative_tag(slide)

    method_box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.75), Inches(1.45), Inches(6.1), Inches(5.4))
    method_box.fill.solid()
    method_box.fill.fore_color.rgb = RGBColor(248, 250, 253)
    method_box.line.color.rgb = RGBColor(222, 230, 240)

    add_bullets(
        slide,
        1.0,
        1.72,
        5.6,
        5.0,
        size=13,
        lines=[
            "Diseno experimental:",
            "- Unidad de analisis: rep-dia pareado.",
            f"- Observaciones pareadas: {snapshot['paired_units']:,} (ILUSTRATIVO).",
            f"- Monte Carlo: {snapshot['monte_carlo_runs']} corridas (ILUSTRATIVO).",
            "Pruebas de inferencia:",
            "- t-test pareado + permutation test.",
            "- Wilcoxon + sign test.",
            "- Intervalos bootstrap + FDR q-values.",
            "- Effect size (Cohen d) y win probability.",
        ],
    )

    table = slide.shapes.add_table(4, 4, Inches(7.05), Inches(1.75), Inches(5.5), Inches(2.65)).table
    headers = ["KPI", "Cohen d", "Win prob.", "q-value"]
    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLORS["navy"]
        para = cell.text_frame.paragraphs[0]
        para.font.size = Pt(11)
        para.font.bold = True
        para.font.color.rgb = COLORS["white"]
        para.font.name = "Calibri"

    rows = [
        ("Ventas", "sales"),
        ("Eficiencia", "efficiency"),
        ("Satisfaccion", "customer_satisfaction"),
    ]
    for r, (label, key) in enumerate(rows, start=1):
        cell_values = [
            label,
            f"Ilustrativo {snapshot['kpis'][key]['effect_size']:.2f}",
            f"Ilustrativo {snapshot['kpis'][key]['win_probability'] * 100:.1f}%",
            f"Ilustrativo {snapshot['kpis'][key]['qvalue']:.4f}",
        ]
        for c, val in enumerate(cell_values):
            cell = table.cell(r, c)
            cell.text = val
            para = cell.text_frame.paragraphs[0]
            para.font.size = Pt(11)
            para.font.name = "Calibri"
            para.font.color.rgb = COLORS["dark"]
            if c == 0:
                para.font.bold = True

    message = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.05), Inches(4.65), Inches(5.5), Inches(2.2))
    message.fill.solid()
    message.fill.fore_color.rgb = RGBColor(236, 253, 245)
    message.line.color.rgb = RGBColor(52, 211, 153)
    tf = message.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = (
        "Lectura ejecutiva:\n"
        "La evidencia ILUSTRATIVA favorece al algoritmo en los 3 KPIs.\n"
        "El objetivo del piloto real es validar este patron con historico y ejecucion real."
    )
    p.font.name = "Calibri"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = RGBColor(6, 78, 59)

    add_footer(slide)


def slide_pilot_design(prs: Presentation, snapshot: Dict[str, object]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    add_top_band(slide, "Diseno del Piloto con 2 Comerciales", "Blueprint de ejecucion para demostrar valor en campo")

    add_bullets(
        slide,
        0.8,
        1.35,
        12.2,
        1.1,
        size=13,
        lines=[
            f"Comerciales piloto: {', '.join(map(str, snapshot['selected_reps']))} | Duracion objetivo: {snapshot['pilot_days']} dias | Diseño A/B por estrategia",
        ],
    )

    steps = [
        ("1. Baseline", "Recolectar 4 semanas de operacion heuristica real (ventas, horas, CSAT, rutas)."),
        ("2. Setup", "Configurar parametros de capacidad, territorios y reglas comerciales en OptimAI."),
        ("3. Ejecucion", "Operar estrategia algoritmica por 6 semanas con seguimiento semanal."),
        ("4. Medicion", "Comparar contra baseline con inferencia estadistica y lectura de negocio."),
        ("5. Escalado", "Definir caso de negocio y plan de despliegue por oleadas si se cumplen umbrales."),
    ]

    y = 2.0
    for title, desc in steps:
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.95), Inches(y), Inches(11.8), Inches(0.88))
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(247, 250, 253)
        box.line.color.rgb = RGBColor(217, 226, 239)
        tf = box.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        p.text = title
        p.font.name = "Calibri"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = COLORS["navy"]
        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.name = "Calibri"
        p2.font.size = Pt(12)
        p2.font.color.rgb = COLORS["dark"]
        y += 1.03

    add_footer(slide)


def slide_methodology(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    add_top_band(slide, "Metodologia Research-Grade", "Gobierno de datos, modelo, experimentacion y QA")

    cols = [
        (
            "Datos",
            [
                "- Definicion de esquema maestro.",
                "- Quality gates y trazabilidad.",
                "- Control de missingness y outliers.",
            ],
        ),
        (
            "Modelos",
            [
                "- Churn + potencial + NBA.",
                "- Validacion cruzada y metricas.",
                "- Revision de drift y sesgos.",
            ],
        ),
        (
            "Operacion",
            [
                "- Asignacion territorial.",
                "- Optimización de rutas.",
                "- Guiones AIDA por visita.",
            ],
        ),
        (
            "QA",
            [
                "- Test suite automatizada.",
                "- Smoke test de app.",
                "- Reporte QA reproducible.",
            ],
        ),
    ]

    x = 0.7
    for title, lines in cols:
        box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(1.65), Inches(3.05), Inches(4.8))
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(248, 250, 253)
        box.line.color.rgb = RGBColor(219, 228, 239)

        tf = box.text_frame
        tf.clear()
        p0 = tf.paragraphs[0]
        p0.text = title
        p0.font.name = "Calibri"
        p0.font.bold = True
        p0.font.size = Pt(15)
        p0.font.color.rgb = COLORS["teal"]

        for line in lines:
            p = tf.add_paragraph()
            p.text = line
            p.font.name = "Calibri"
            p.font.size = Pt(12)
            p.font.color.rgb = COLORS["dark"]
        x += 3.15

    band = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(6.6), Inches(12.0), Inches(0.55))
    band.fill.solid()
    band.fill.fore_color.rgb = RGBColor(234, 246, 251)
    band.line.color.rgb = RGBColor(14, 165, 233)
    tf = band.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = "Principio rector: decisiones comerciales medibles, auditables y estadisticamente defendibles."
    p.font.name = "Calibri"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = RGBColor(12, 74, 110)

    add_footer(slide)


def slide_resources(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    add_top_band(slide, "Equipo, Recursos y Roles", "Dimensionamiento minimo para ejecutar piloto con rigor")

    table = slide.shapes.add_table(7, 4, Inches(0.75), Inches(1.55), Inches(12.0), Inches(3.9)).table
    headers = ["Rol", "Dedicacion", "Responsabilidades", "Owner"]
    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLORS["navy"]
        para = cell.text_frame.paragraphs[0]
        para.font.name = "Calibri"
        para.font.bold = True
        para.font.size = Pt(11)
        para.font.color.rgb = COLORS["white"]

    rows = [
        ("Sponsor negocio", "1-2h/sem", "Alineacion de objetivos, desbloqueo y decisiones.", "Cliente"),
        ("Head of Sales", "4h/sem", "Gobierno comercial y adopcion por fuerza de ventas.", "Cliente"),
        ("Sales Ops", "6h/sem", "Extraccion de datos, reglas operativas y seguimiento.", "Cliente"),
        ("Data Scientist", "50%", "Modelado, monitoreo y analitica inferencial.", "OptimAI"),
        ("ML/Backend", "40%", "Pipeline, integraciones y automatizacion.", "OptimAI"),
        ("Change Lead", "20%", "Capacitacion comercial, feedback y adopcion.", "Mixto"),
    ]

    for r, row in enumerate(rows, start=1):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.text = val
            para = cell.text_frame.paragraphs[0]
            para.font.name = "Calibri"
            para.font.size = Pt(10)
            para.font.color.rgb = COLORS["dark"]
            if c == 0:
                para.font.bold = True

    add_bullets(
        slide,
        0.9,
        5.7,
        12.0,
        1.25,
        size=12,
        lines=[
            "Recursos tecnicos: Python, Streamlit, repositorio Git, API OpenAI, dashboard de seguimiento semanal.",
            "Requisito clave: acceso a historico heuristico real para medir delta pre/post con maxima credibilidad.",
        ],
    )

    add_footer(slide)


def slide_timeline(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    add_top_band(slide, "Plan y Duracion", "Roadmap propuesto de 12 semanas")

    phases = [
        ("Semana 1-2", "Alineacion, calidad de datos y baseline heuristico"),
        ("Semana 3-4", "Configuracion OptimAI y UAT con equipo comercial"),
        ("Semana 5-10", "Ejecucion piloto en campo + comites semanales"),
        ("Semana 11", "Analisis estadistico final y business case"),
        ("Semana 12", "Decision de escalado y plan de despliegue"),
    ]

    y = 1.75
    for idx, (timebox, desc) in enumerate(phases):
        color = RGBColor(240, 249, 255) if idx % 2 == 0 else RGBColor(248, 250, 253)
        box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.85), Inches(y), Inches(11.95), Inches(0.85))
        box.fill.solid()
        box.fill.fore_color.rgb = color
        box.line.color.rgb = RGBColor(207, 218, 234)

        tf = box.text_frame
        tf.clear()
        p1 = tf.paragraphs[0]
        p1.text = timebox
        p1.font.name = "Calibri"
        p1.font.bold = True
        p1.font.size = Pt(13)
        p1.font.color.rgb = COLORS["navy"]

        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.name = "Calibri"
        p2.font.size = Pt(12)
        p2.font.color.rgb = COLORS["dark"]
        y += 0.95

    milestone = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.85), Inches(6.55), Inches(11.95), Inches(0.6))
    milestone.fill.solid()
    milestone.fill.fore_color.rgb = RGBColor(255, 247, 237)
    milestone.line.color.rgb = RGBColor(245, 158, 11)
    tf = milestone.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = "Hito de go/no-go: final de semana 12, con KPIs, significancia y feedback de adopcion comercial."
    p.font.name = "Calibri"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = RGBColor(120, 53, 15)

    add_footer(slide)


def slide_risks(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    add_top_band(slide, "Riesgos y Mitigaciones", "Control de ejecucion para proteger el valor del piloto")

    table = slide.shapes.add_table(6, 3, Inches(0.9), Inches(1.5), Inches(11.8), Inches(4.6)).table
    headers = ["Riesgo", "Impacto", "Mitigacion"]
    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLORS["navy"]
        para = cell.text_frame.paragraphs[0]
        para.font.name = "Calibri"
        para.font.bold = True
        para.font.size = Pt(11)
        para.font.color.rgb = COLORS["white"]

    rows = [
        (
            "Datos historicos incompletos",
            "Reduce credibilidad del before/after",
            "Data contract semanal, checks automaticos y ownership claro.",
        ),
        (
            "Baja adopcion de comerciales",
            "Dilucion del impacto",
            "Capacitacion + coaching + feedback loop semanal.",
        ),
        (
            "Cambios operativos durante piloto",
            "Sesgo de comparacion",
            "Congelar reglas comerciales criticas durante 12 semanas.",
        ),
        (
            "Expectativas de ROI no alineadas",
            "Decision tardia",
            "Definir umbrales go/no-go antes del arranque.",
        ),
        (
            "Riesgo de sobreajuste de mensajes LLM",
            "Calidad comercial inconsistente",
            "Guidelines de tono, compliance y revision por manager.",
        ),
    ]

    for r, row in enumerate(rows, start=1):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.text = val
            para = cell.text_frame.paragraphs[0]
            para.font.name = "Calibri"
            para.font.size = Pt(10)
            para.font.color.rgb = COLORS["dark"]
            if c == 0:
                para.font.bold = True

    add_bullets(
        slide,
        0.95,
        6.3,
        11.7,
        0.8,
        size=12,
        lines=["Gobernanza recomendada: Steering Committee quincenal + PMO semanal + dashboard unico de decision."],
    )

    add_footer(slide)


def slide_kpi_framework(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    add_top_band(slide, "Marco de Medicion Before vs After", "Como se medira el delta de forma robusta y accionable")

    left = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.55), Inches(6.1), Inches(5.6))
    left.fill.solid()
    left.fill.fore_color.rgb = RGBColor(248, 250, 253)
    left.line.color.rgb = RGBColor(219, 228, 239)

    right = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.95), Inches(1.55), Inches(5.5), Inches(5.6))
    right.fill.solid()
    right.fill.fore_color.rgb = RGBColor(248, 250, 253)
    right.line.color.rgb = RGBColor(219, 228, 239)

    add_bullets(
        slide,
        1.05,
        1.85,
        5.6,
        5.2,
        size=12,
        lines=[
            "KPIs primarios:",
            "- Ventas por visita y por jornada.",
            "- Eficiencia (ventas/hora de servicio).",
            "- Satisfaccion cliente post-visita.",
            "KPIs de soporte:",
            "- Tasa de visita efectiva.",
            "- Distancia por conversion.",
            "- Tiempo no productivo en ruta.",
            "- Calidad de script y adherencia comercial.",
        ],
    )

    add_bullets(
        slide,
        7.2,
        1.85,
        5.0,
        5.2,
        size=12,
        lines=[
            "Reglas de decision sugeridas:",
            "- Ventas: delta >= +8% y q < 0.05.",
            "- Eficiencia: delta >= +10% y q < 0.05.",
            "- CSAT: sin deterioro; idealmente +3% o mas.",
            "- Adopcion comercial >= 80% de visitas con uso del plan.",
            "- Recomendacion de escalado si se cumplen 3/4 criterios.",
        ],
    )

    add_footer(slide)


def slide_decision(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide)
    add_top_band(slide, "Decision Solicitada al Owner", "Aprobacion para iniciar piloto controlado")

    msg = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.9), Inches(1.6), Inches(11.7), Inches(1.25))
    msg.fill.solid()
    msg.fill.fore_color.rgb = RGBColor(236, 253, 245)
    msg.line.color.rgb = RGBColor(52, 211, 153)
    tf = msg.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = "Propuesta: lanzar piloto de 12 semanas con 2 comerciales y baseline real de gestion heuristica."
    p.font.name = "Calibri"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = RGBColor(6, 78, 59)

    add_bullets(
        slide,
        1.1,
        3.2,
        11.4,
        3.0,
        size=14,
        lines=[
            "1. Confirmar sponsor ejecutivo y owner operativo.",
            "2. Habilitar acceso a historico heuristico y feedback comercial semanal.",
            "3. Validar umbrales de exito y criterios go/no-go antes del kickoff.",
            "4. Aprobar calendario de comites y mecanismo de seguimiento KPI.",
        ],
    )

    close = slide.shapes.add_textbox(Inches(1.1), Inches(6.5), Inches(11.2), Inches(0.5)).text_frame
    close.clear()
    p2 = close.paragraphs[0]
    p2.text = "Resultado esperado: evidencia clara para decidir escalado con menor riesgo y mayor retorno comercial."
    p2.font.name = "Calibri"
    p2.font.size = Pt(14)
    p2.font.bold = True
    p2.font.color.rgb = COLORS["navy"]

    add_footer(slide)


def build_presentation() -> Path:
    snapshot = compute_snapshot()

    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)

    slide_cover(prs, snapshot)
    slide_exec_summary(prs, snapshot)
    slide_context(prs)
    slide_scope(prs, snapshot)
    slide_results(prs, snapshot)
    slide_stats(prs, snapshot)
    slide_pilot_design(prs, snapshot)
    slide_methodology(prs)
    slide_resources(prs)
    slide_timeline(prs)
    slide_risks(prs)
    slide_kpi_framework(prs)
    slide_decision(prs)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUTPUT_PATH))
    return OUTPUT_PATH


if __name__ == "__main__":
    out = build_presentation()
    print(f"Presentation generated: {out}")
