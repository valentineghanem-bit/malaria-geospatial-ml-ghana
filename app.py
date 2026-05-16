#!/usr/bin/env python3
"""
Malaria Geospatial ML — Ghana 260 Districts
Interactive Dash dashboard: incidence mapping, ITN analysis, ML risk scores.
Run: python app.py  →  http://127.0.0.1:8050
"""
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

DATA = os.path.join(os.path.dirname(__file__), "data", "processed",
                    "Ghana_Malaria_260District_MasterDataset.csv")
df = pd.read_csv(DATA)

OUTCOMES = {
    "incidence":     "Malaria Incidence (per 1,000)",
    "parasitemia":   "Parasitaemia Prevalence (%)",
    "xgb_risk_score":"XGBoost Risk Score",
    "u5mr":          "Under-5 Mortality Rate",
}
PREDICTORS = {
    "itn_coverage":  "ITN Coverage (%)",
    "water_access":  "Water Access (%)",
    "sanitation":    "Sanitation Coverage (%)",
    "pop_density":   "Population Density",
}
SHAP_VALS = {
    "itn_coverage": 0.41, "parasitemia": 0.38, "water_access": 0.27,
    "sanitation": 0.19, "pop_density": 0.11,
}
RISK_COLORS = {"Very High": "#c0392b", "High": "#e74c3c", "Moderate": "#f39c12",
               "Low": "#27ae60", "Very Low": "#2ecc71"}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
                title="Malaria Geospatial ML — Ghana")
server = app.server

def kpi(label, value, color="warning"):
    return dbc.Card(dbc.CardBody([
        html.P(label, className="text-muted mb-1", style={"fontSize": "0.73rem"}),
        html.H5(value, className=f"text-{color} mb-0 fw-bold"),
    ]), className="mb-2 h-100")

app.layout = dbc.Container(fluid=True, style={"backgroundColor": "#0d1117", "minHeight": "100vh"}, children=[
    dbc.Row(dbc.Col(html.H4(
        "Geospatial Clustering & ML Prediction of Malaria Burden — Ghana 260 Districts",
        className="text-center text-light py-3"))),

    dbc.Row([
        dbc.Col(kpi("Global Moran's I", "0.672 (p<0.001)", "success"), md=2),
        dbc.Col(kpi("Bivariate HH Clusters", "36 (ITN deficit × incidence)", "danger"), md=3),
        dbc.Col(kpi("Priority-1 Hotspots (Gi*)", "38 districts (99.9% CI)", "warning"), md=3),
        dbc.Col(kpi("XGBoost AUC", "0.923 (LODO-CV)", "info"), md=2),
        dbc.Col(kpi("Top SHAP Predictor", "ITN Coverage |SHAP|=0.41", "secondary"), md=2),
    ], className="mb-3"),

    dbc.Tabs([
        # ── Spatial Distribution ─────────────────────────────────────────
        dbc.Tab(label="Spatial Distribution", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Outcome:", className="text-light mt-3"),
                    dcc.Dropdown(id="mal-metric",
                                 options=[{"label": v, "value": k} for k, v in OUTCOMES.items()],
                                 value="incidence", clearable=False, style={"color": "#000"}),
                ], md=4),
                dbc.Col([
                    html.Label("Ecological zone:", className="text-light mt-3"),
                    dcc.Dropdown(id="eco-filter",
                                 options=[{"label": "All", "value": "all"}] + [
                                     {"label": z, "value": z}
                                     for z in sorted(df.ecological_zone.dropna().unique())],
                                 value="all", clearable=False, style={"color": "#000"}),
                ], md=4),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="mal-map"), md=8),
                dbc.Col(dcc.Graph(id="hotspot-pie"), md=4),
            ]),
        ]),

        # ── ML Risk Scores ────────────────────────────────────────────────
        dbc.Tab(label="ML Risk Scores", children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id="risk-dist"), md=6),
                dbc.Col(dcc.Graph(id="shap-bar"), md=6),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Predictor:", className="text-light mt-3"),
                    dcc.Dropdown(id="pred-col",
                                 options=[{"label": v, "value": k} for k, v in PREDICTORS.items()],
                                 value="itn_coverage", clearable=False, style={"color": "#000"}),
                ], md=4),
            ]),
            dbc.Row([dbc.Col(dcc.Graph(id="pred-scatter"), md=10)]),
        ]),

        # ── District Table ───────────────────────────────────────────────
        dbc.Tab(label="District Explorer", children=[
            dbc.Row(dbc.Col([
                html.Label("Risk tier:", className="text-light mt-3"),
                dcc.Dropdown(id="risk-filter",
                             options=[{"label": "All", "value": "all"}] + [
                                 {"label": t, "value": t}
                                 for t in ["Very High", "High", "Moderate", "Low", "Very Low"]
                                 if t in df.get("xgb_risk_tier", pd.Series()).values],
                             value="all", clearable=False, style={"color": "#000"}),
            ], md=4)),
            dbc.Row(dbc.Col(dash_table.DataTable(
                id="mal-table",
                columns=[
                    {"name": "District", "id": "district"},
                    {"name": "Region", "id": "region"},
                    {"name": "Incidence", "id": "incidence"},
                    {"name": "Parasitaemia %", "id": "parasitemia"},
                    {"name": "ITN Coverage %", "id": "itn_coverage"},
                    {"name": "XGB Risk Score", "id": "xgb_risk_score"},
                    {"name": "Risk Tier", "id": "xgb_risk_tier"},
                    {"name": "Gi* Class", "id": "gi_classification"},
                ],
                page_size=20, sort_action="native", filter_action="native",
                style_table={"overflowX": "auto"},
                style_header={"backgroundColor": "#1f2937", "color": "white"},
                style_data={"backgroundColor": "#111827", "color": "white"},
                style_data_conditional=[
                    {"if": {"filter_query": '{xgb_risk_tier} = "Very High"'},
                     "color": "#e74c3c", "fontWeight": "bold"},
                    {"if": {"filter_query": '{gi_classification} contains "Priority-1"'},
                     "backgroundColor": "#2c1810"},
                ],
            ))),
        ]),
    ]),
])


@app.callback(Output("mal-map", "figure"), Output("hotspot-pie", "figure"),
              Input("mal-metric", "value"), Input("eco-filter", "value"))
def update_map(metric, zone):
    d = df if zone == "all" else df[df.ecological_zone == zone]
    label = OUTCOMES[metric]
    fig = px.scatter(d, x="district_id", y=metric, color=metric,
                     hover_name="district",
                     hover_data={"region": True, "xgb_risk_tier": True,
                                 "gi_classification": True, "district_id": False},
                     color_continuous_scale="YlOrRd", size_max=12,
                     title=f"{label} — All Districts")
    fig.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                      font_color="white", margin=dict(t=40, b=10))

    if "gi_classification" in df.columns:
        gi_counts = df.gi_classification.value_counts()
        fig_pie = px.pie(values=gi_counts.values, names=gi_counts.index,
                         title="Gi* Hotspot/Coldspot Classification",
                         color_discrete_sequence=px.colors.sequential.RdBu)
    else:
        fig_pie = go.Figure()
    fig_pie.update_layout(paper_bgcolor="#161b22", font_color="white", margin=dict(t=40))
    return fig, fig_pie


@app.callback(Output("risk-dist", "figure"), Output("shap-bar", "figure"),
              Output("pred-scatter", "figure"), Input("pred-col", "value"))
def update_ml(pred):
    fig_risk = px.histogram(df, x="xgb_risk_score", color="xgb_risk_tier",
                             nbins=30, title="XGBoost Risk Score Distribution",
                             color_discrete_map=RISK_COLORS,
                             labels={"xgb_risk_score": "Risk Score"})
    fig_risk.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                           font_color="white", margin=dict(t=40))

    shap_df = pd.DataFrame({
        "feature": [PREDICTORS.get(k, k) for k in SHAP_VALS],
        "importance": list(SHAP_VALS.values()),
    }).sort_values("importance")
    fig_shap = px.bar(shap_df, x="importance", y="feature", orientation="h",
                      title="SHAP Feature Importance (XGBoost)",
                      color="importance", color_continuous_scale="Reds")
    fig_shap.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                           font_color="white", showlegend=False, margin=dict(t=40))

    fig_pred = px.scatter(df, x=pred, y="xgb_risk_score", color="ecological_zone",
                          hover_name="district", trendline="ols",
                          title=f"{PREDICTORS.get(pred, pred)} vs XGBoost Risk Score",
                          labels={pred: PREDICTORS.get(pred, pred),
                                  "xgb_risk_score": "Risk Score"})
    fig_pred.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                           font_color="white", margin=dict(t=40))
    return fig_risk, fig_shap, fig_pred


@app.callback(Output("mal-table", "data"), Input("risk-filter", "value"))
def update_table(tier):
    d = df if tier == "all" else df[df.xgb_risk_tier == tier]
    cols = ["district", "region", "incidence", "parasitemia",
            "itn_coverage", "xgb_risk_score", "xgb_risk_tier", "gi_classification"]
    return d[[c for c in cols if c in d.columns]].round(3).to_dict("records")


if __name__ == "__main__":
    print("Dashboard: http://127.0.0.1:8050")
    app.run(debug=False, port=8050)
