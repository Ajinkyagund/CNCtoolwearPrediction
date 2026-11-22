"""
Streamlit dashboard for CNC tool wear + RUL prediction.
Run (after `python train.py`):  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, confusion_matrix, roc_curve

from cncwear.metrics import format_mean_std

st.set_page_config(page_title="CNC Tool Wear & RUL", page_icon="🛠️", layout="wide")

MODELS, OUT = Path("models"), Path("outputs")
if not (MODELS / "meta.json").exists():
    st.error("No trained models found. Run `python train.py` first.")
    st.stop()

COLORS = {"Healthy": "#16a34a", "Moderate": "#f59e0b", "Worn": "#dc2626"}
ICON = {"Healthy": "🟢", "Moderate": "🟡", "Worn": "🔴"}
ACCENT, GREY, LIGHT = "#2563eb", "#94a3b8", "#cbd5e1"

# ------------------------------------------------------------------ styling
st.markdown(
    """
<style>
.block-container {padding-top: 1.4rem; max-width: 1280px;}
#MainMenu, footer {visibility: hidden;}
.hero {padding: 1.5rem 1.8rem; border-radius: 18px; margin-bottom: 1.1rem; color: #fff;
       background: linear-gradient(120deg, #1e3a8a 0%, #2563eb 55%, #38bdf8 100%);}
.hero h1 {color: #fff; font-size: 1.75rem; margin: 0 0 .3rem 0; padding: 0;}
.hero p {margin: 0; opacity: .93; font-size: 1rem;}
.card {border: 1px solid rgba(100,116,139,.28); border-radius: 14px; padding: .95rem 1.15rem;
       background: rgba(100,116,139,.07); height: 100%;}
.card .label {font-size: .74rem; opacity: .7; text-transform: uppercase; letter-spacing: .05em;}
.card .value {font-size: 1.75rem; font-weight: 700; line-height: 1.25;}
.card .sub {font-size: .78rem; opacity: .65;}
.badge {display: inline-block; padding: .4rem 1.1rem; border-radius: 999px; font-weight: 700;
        color: #fff; font-size: 1.15rem;}
.step {border-left: 4px solid #2563eb; padding: .3rem .9rem; margin-bottom: .6rem;}
.step b {display: block;}
</style>
""",
    unsafe_allow_html=True,
)


def card(label, value, sub="", color=None):
    style = f' style="color:{color}"' if color else ""
    return (f'<div class="card"><div class="label">{label}</div>'
            f'<div class="value"{style}>{value}</div><div class="sub">{sub}</div></div>')


def tidy(fig, height=360, title=None):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50 if title else 15, b=10),
                      title=dict(text=title, x=0.01) if title else None,
                      legend=dict(orientation="h", y=-0.2))
    return fig


# ------------------------------------------------------------------ assets
def _opt(name):
    p = OUT / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource
def load_assets():
    meta = json.loads((MODELS / "meta.json").read_text())
    a = {
        "meta": meta,
        "rul_model": joblib.load(MODELS / "rul_model.joblib"),
        "wear_model": joblib.load(MODELS / "wear_model.joblib"),
        "data": pd.read_csv(OUT / "processed_features.csv"),
        "background": _opt("background.csv"),
        "eda": _opt("stage1_eda_summary.csv"),
        "split": _opt("stage2_split_summary.csv"),
        "per_class": _opt("stage7_wear_per_class.csv"),
        "per_tool": _opt("stage7_per_tool_errors.csv"),
    }
    for k in ("rul", "wear"):
        a[f"s4_{k}"] = _opt(f"stage4_{k}_baselines.csv")
        a[f"s5_{k}"] = _opt(f"stage5_{k}_nested.csv")
        a[f"bp_{k}"] = _opt(f"stage5_{k}_best_params.csv")
        a[f"abl_{k}"] = _opt(f"stage6_{k}_ablation.csv")
        a[f"test_{k}"] = _opt(f"stage7_{k}_test_metrics.csv")
        shap = _opt(f"stage8_shap_{k}.csv")
        a[f"shap_{k}"] = shap.set_index(shap.columns[0])["mean_abs_shap"] if shap is not None else None
    imp = _opt("stage8_feature_importance.csv")
    a["importance"] = imp.set_index(imp.columns[0])["importance"] if imp is not None else None
    return a


A = load_assets()
meta, rul_model, wear_model, data = A["meta"], A["rul_model"], A["wear_model"], A["data"].copy()
S = meta["summary"]
FEATS, NAMES = meta["features"], meta["class_names"]
HL, WL = meta["healthy_limit"], meta["worn_limit"]
MAX_RUL = max(float(data["RUL"].max()), 1.0)
data["pred_wear_name"] = data["pred_wear"].map(dict(enumerate(NAMES)))
test = data[data["split"] == "test"].reset_index(drop=True)


def predict(frame):
    frame = frame[FEATS]
    rul = np.clip(rul_model.predict(frame), 0, None)
    cls = wear_model.predict(frame).astype(int)
    try:
        conf = wear_model.predict_proba(frame).max(axis=1)
    except Exception:
        conf = np.full(len(frame), np.nan)
    return rul, cls, conf


def advice(state, rul):
    if state == "Worn":
        return "error", "Replace the tool now. Wear is past the limit, so surface finish and part accuracy are at risk."
    if state == "Moderate" or rul <= 2:
        return "warning", "Plan a tool change soon. Wear is progressing and little useful life is left."
    return "success", "Tool is in good condition. Keep machining and keep monitoring."


def mean_std_bar(summary, metric, lower, title):
    d = summary.sort_values(f"{metric}_mean", ascending=lower).reset_index(drop=True)
    colors = [ACCENT if i == 0 else (LIGHT if m.startswith("Baseline") else GREY) for i, m in enumerate(d["Model"])]
    fig = go.Figure(go.Bar(y=d["Model"], x=d[f"{metric}_mean"], orientation="h", marker_color=colors,
                           error_x=dict(type="data", array=d[f"{metric}_std"].fillna(0), color="#475569")))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title=f"{metric} (mean ± std over folds), {'lower' if lower else 'higher'} is better")
    return tidy(fig, 60 * len(d) + 120, title)


def ablation_bar(table, title):
    d = table[table["Variant"] != "All features (full model)"].copy()
    d["Component removed"] = d["Variant"].str.replace("Without ", "", regex=False)
    d = d.sort_values("Performance drop")
    metric = table["Primary metric"].iloc[0]
    fig = go.Figure(go.Bar(y=d["Component removed"], x=d["Performance drop"], orientation="h",
                           marker_color=[ACCENT if v > 0 else GREY for v in d["Performance drop"]]))
    fig.update_xaxes(title=f"Performance drop in {metric} when removed (positive = component helps)")
    return tidy(fig, 50 * len(d) + 120, title)


def importance_bar(series, title, xlabel):
    top = series.sort_values(ascending=False).head(15)[::-1]
    fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color=ACCENT))
    fig.update_xaxes(title=xlabel)
    return tidy(fig, 440, title)


# ------------------------------------------------------------------ sidebar + header
with st.sidebar:
    st.markdown("### 🛠️ CNC Wear AI")
    st.caption("AI/ML Mini Project · Practical 07")
    st.markdown("**Selected models** (chosen by nested CV)")
    st.write(f"RUL: `{S['models']['rul']['name']}`")
    st.write(f"Wear state: `{S['models']['wear']['name']}`")
    st.markdown("**Wear states (flank wear VB)**")
    st.write(f"🟢 Healthy: below {HL} mm")
    st.write(f"🟡 Moderate: {HL} to {WL} mm")
    st.write(f"🔴 Worn: {WL} mm or more")
    st.divider()
    st.caption(f"Train: {S['train_tools']} tools · Test: {S['test_tools']} tools (held out, evaluated once). "
               "Everything except the Test tab uses training tools only.")

st.markdown(
    '<div class="hero"><h1>AI-Based CNC Tool Wear &amp; Remaining Useful Life Prediction</h1>'
    "<p>Machine-learning predictive maintenance: classify tool wear and estimate how many runs are left "
    "before the cutter must be replaced.</p></div>",
    unsafe_allow_html=True,
)
if S.get("quick_mode"):
    st.warning("These results come from a QUICK smoke-test run. Run `python train.py` (without --quick) "
               "before reporting any number.")

tabs = st.tabs(["🏠 Overview", "🔮 Live prediction", "⏪ Tool life replay", "🧪 Validation (stages 4-6)",
                "🎯 Test results (stage 7)", "🧠 Explainability", "📁 Batch"])

# ================================================================== Overview
with tabs[0]:
    r, w = S["models"]["rul"], S["models"]["wear"]
    k = st.columns(5)
    k[0].markdown(card("Tools (train / test)", f"{S['train_tools']} / {S['test_tools']}",
                       f"{S['train_runs'] + S['test_runs']} machining runs"), unsafe_allow_html=True)
    k[1].markdown(card("Features", f"{len(FEATS)}", "from 6 sensor channels + process settings"),
                  unsafe_allow_html=True)
    k[2].markdown(card("Test RUL error (MAE)", f"{r['test']['MAE']:.2f} runs", f"R² = {r['test']['R2']:.2f}",
                       ACCENT), unsafe_allow_html=True)
    k[3].markdown(card("Test wear-state F1", f"{w['test']['F1']:.2f}", f"accuracy {w['test']['Accuracy']:.0%}",
                       ACCENT), unsafe_allow_html=True)
    k[4].markdown(card("Gain over best baseline", f"{r['improvement_vs_best_baseline']:+.0%}",
                       "RUL error, nested CV", ACCENT), unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns([1.6, 1])
    with c1:
        fig = go.Figure()
        top = float(data["VB"].max()) * 1.05
        for lo, hi, name in [(0, HL, "Healthy"), (HL, WL, "Moderate"), (WL, top, "Worn")]:
            fig.add_hrect(y0=lo, y1=hi, fillcolor=COLORS[name], opacity=0.09, line_width=0)
        for case, g in data.groupby("case"):
            is_test = g["split"].iloc[0] == "test"
            fig.add_trace(go.Scatter(
                x=g["run"], y=g["VB"], mode="lines", showlegend=False,
                line=dict(width=2.6 if is_test else 1.4, color="#ea580c" if is_test else "rgba(37,99,235,.5)"),
                hovertemplate=f"Tool {case} ({'test' if is_test else 'train'})<br>run %{{x}}<br>VB %{{y:.2f}} mm<extra></extra>"))
        fig.update_xaxes(title="Run number")
        fig.update_yaxes(title="Flank wear VB (mm)", range=[0, top])
        st.plotly_chart(tidy(fig, 380, "Wear curves: blue = training tools, orange = held-out test tools"))
    with c2:
        counts = data["wear_state"].value_counts().reindex(NAMES)
        fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.55,
                               marker=dict(colors=[COLORS[n] for n in counts.index]), sort=False))
        st.plotly_chart(tidy(fig, 380, "Wear-state distribution"))

    st.markdown("#### The 8-stage pipeline")
    stages = [("1 · Data & EDA", "Missing values, class balance, correlations."),
              ("2 · Split", "Whole tools held out as the test set, stratified by wear class."),
              ("3 · Preprocess", "Imputer + scaler inside a Pipeline: no leakage."),
              ("4 · Baselines", "Simple rules vs 6 models, repeated CV, mean ± std."),
              ("5 · Tuning", "Grid search inside nested cross-validation, fixed seed."),
              ("6 · Ablation", "Remove one sensor group at a time and measure the drop."),
              ("7 · Test", "Evaluate once on the held-out tools: confusion matrix, ROC."),
              ("8 · Deliver", "SHAP explanations, repo, README, this demo app.")]
    for row in (stages[:4], stages[4:]):
        cols = st.columns(4)
        for col, (t, d) in zip(cols, row):
            col.markdown(f'<div class="step"><b>{t}</b>{d}</div>', unsafe_allow_html=True)

# ================================================================== Live prediction
with tabs[1]:
    controls = (["run"] if "run" in FEATS else []) + meta["top_features"]
    st.session_state.setdefault("base_vals", dict(meta["defaults"]))
    for f in controls:
        st.session_state.setdefault(f"s_{f}", float(meta["defaults"][f]))

    def set_preset(k):
        sub = data[(data["split"] == "train") & (data["wear_class"] == k)]
        if sub.empty:
            return
        st.session_state["base_vals"] = {f: float(sub[f].median()) for f in FEATS}
        for f in controls:
            lo, hi = float(meta["min"][f]), float(meta["max"][f])
            st.session_state[f"s_{f}"] = float(np.clip(sub[f].median(), lo, hi))

    st.markdown("#### Describe the tool's current condition")
    p = st.columns([1, 1, 1, 2.2])
    p[0].button("🟢 Fresh tool", on_click=set_preset, args=(0,), use_container_width=True)
    p[1].button("🟡 Mid-life tool", on_click=set_preset, args=(1,), use_container_width=True)
    p[2].button("🔴 Worn tool", on_click=set_preset, args=(2,), use_container_width=True)
    p[3].caption("Presets load typical sensor values for each wear state. Then adjust the sliders yourself.")

    left, right = st.columns([1.15, 1])
    with left:
        with st.container(border=True):
            st.markdown("**Most influential signals**")
            sc = st.columns(2)
            for i, f in enumerate(controls):
                lo, hi = float(meta["min"][f]), float(meta["max"][f])
                if hi <= lo:
                    hi = lo + 1.0
                with sc[i % 2]:
                    st.slider(f, lo, hi, step=max((hi - lo) / 200, 1e-6), format="%.4g", key=f"s_{f}")

    vals = dict(st.session_state["base_vals"])
    vals.update({f: st.session_state[f"s_{f}"] for f in controls})
    row_df = pd.DataFrame([vals])[FEATS]
    rul, cls, conf = predict(row_df)
    state = NAMES[cls[0]]

    with right:
        with st.container(border=True):
            st.markdown("**Prediction**")
            st.markdown(f'<span class="badge" style="background:{COLORS[state]}">{ICON[state]} {state}</span>',
                        unsafe_allow_html=True)
            st.caption("" if np.isnan(conf[0]) else f"Classifier confidence: {conf[0]:.0%}")
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=float(rul[0]), number=dict(suffix=" runs", valueformat=".1f"),
                title=dict(text="Remaining useful life"),
                gauge=dict(axis=dict(range=[0, MAX_RUL]), bar=dict(color=ACCENT),
                           steps=[dict(range=[0, MAX_RUL * 0.2], color="rgba(220,38,38,.28)"),
                                  dict(range=[MAX_RUL * 0.2, MAX_RUL * 0.5], color="rgba(245,158,11,.28)"),
                                  dict(range=[MAX_RUL * 0.5, MAX_RUL], color="rgba(22,163,74,.28)")])))
            gauge.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=0))
            st.plotly_chart(gauge)
            level, text = advice(state, rul[0])
            getattr(st, level)(text)

    with st.expander("🔍 Why this prediction? (SHAP explanation)"):
        st.caption("Each bar shows how much a feature pushed this prediction up or down compared with a typical "
                   "training run. It takes a few seconds.")
        if A["background"] is None:
            st.info("Background data not found (outputs/background.csv). Re-run `python train.py`.")
        elif st.button("Explain this prediction"):
            try:
                from cncwear.explain import local_shap

                bg = A["background"][FEATS]
                with st.spinner("Computing SHAP values..."):
                    s_rul, base_rul = local_shap(rul_model, bg, row_df, "reg")
                    s_wear, base_wear = local_shap(wear_model, bg, row_df, "clf")
                e1, e2 = st.columns(2)
                for col, s, base, ttl, up in [
                        (e1, s_rul, base_rul, "Remaining useful life (runs)", "more life"),
                        (e2, s_wear, base_wear, "Probability of Worn", "higher wear risk")]:
                    top = s.reindex(s.abs().sort_values(ascending=False).index).head(8)[::-1]
                    fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                                           marker_color=[ACCENT if v > 0 else "#ef4444" for v in top.values]))
                    fig.update_xaxes(title=f"Blue pushes towards {up}")
                    col.plotly_chart(tidy(fig, 340, f"{ttl}: typical {base:.2f}"))
            except Exception as err:
                st.warning(f"SHAP explanation unavailable ({type(err).__name__}). "
                           "Install it with `pip install shap`.")

# ================================================================== Replay
with tabs[2]:
    st.markdown("#### Replay one tool from its first run to end of life")
    st.caption("Held-out TEST tools were never seen by the model. For training tools the prediction comes from "
               "cross-validation (the model was trained on the other tools only).")
    kind_of = data.groupby("case")["split"].first()
    order = sorted(kind_of.index, key=lambda c: (kind_of[c] != "test", c))
    case = st.selectbox("Choose a tool", order, format_func=lambda c: f"Tool {c} · " + (
        "held-out TEST tool" if kind_of[c] == "test" else "training tool (cross-validated prediction)"))
    g = data[data["case"] == case].sort_values("run").reset_index(drop=True)
    runs = [int(x) for x in g["run"]]
    sel = st.select_slider("Current run", options=runs, value=runs[len(runs) // 2]) if len(runs) > 1 else runs[0]
    row = g[g["run"] == sel].iloc[0]

    a = st.columns(4)
    a[0].markdown(card("Measured wear VB", f"{row['VB']:.2f} mm", f"run {sel}"), unsafe_allow_html=True)
    a[1].markdown(card("Actual state", f"{ICON[row['wear_state']]} {row['wear_state']}", "from VB"),
                  unsafe_allow_html=True)
    ok = row["pred_wear_name"] == row["wear_state"]
    a[2].markdown(card("Predicted state", f"{ICON[row['pred_wear_name']]} {row['pred_wear_name']}",
                       "✔ correct" if ok else "✘ differs from actual", COLORS[row["pred_wear_name"]]),
                  unsafe_allow_html=True)
    a[3].markdown(card("RUL actual vs predicted", f"{row['RUL']:.0f} vs {max(row['pred_rul'], 0):.1f}", "runs",
                       ACCENT), unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=g["run"], y=g["RUL"], name="Actual", mode="lines+markers",
                                 line=dict(color="#64748b", dash="dot")))
        fig.add_trace(go.Scatter(x=g["run"], y=g["pred_rul"].clip(lower=0), name="Predicted",
                                 mode="lines+markers", line=dict(color=ACCENT)))
        fig.add_vline(x=sel, line_width=1, line_dash="dash", line_color=GREY)
        fig.update_xaxes(title="Run")
        fig.update_yaxes(title="RUL (runs)")
        st.plotly_chart(tidy(fig, 340, "Remaining useful life"))
    with c2:
        fig = go.Figure()
        for lo, hi, name in [(0, HL, "Healthy"), (HL, WL, "Moderate"), (WL, max(float(g["VB"].max()), WL) * 1.1, "Worn")]:
            fig.add_hrect(y0=lo, y1=hi, fillcolor=COLORS[name], opacity=0.09, line_width=0)
        fig.add_trace(go.Scatter(x=g["run"], y=g["VB"], mode="lines", line=dict(color=GREY), showlegend=False))
        for name in NAMES:
            sub = g[g["pred_wear_name"] == name]
            if len(sub):
                fig.add_trace(go.Scatter(x=sub["run"], y=sub["VB"], mode="markers", name=f"Predicted {name}",
                                         marker=dict(size=11, color=COLORS[name])))
        fig.add_vline(x=sel, line_width=1, line_dash="dash", line_color=GREY)
        fig.update_xaxes(title="Run")
        fig.update_yaxes(title="Flank wear VB (mm)")
        st.plotly_chart(tidy(fig, 340, "Wear and predicted state"))

    with st.expander("Show data table"):
        t = g[["run", "VB", "RUL", "pred_rul", "wear_state", "pred_wear_name"]].copy()
        t["pred_rul"] = t["pred_rul"].round(1)
        st.dataframe(t.rename(columns={"RUL": "Actual RUL", "pred_rul": "Predicted RUL",
                                       "wear_state": "Actual state", "pred_wear_name": "Predicted state"}),
                     hide_index=True)

# ================================================================== Validation
with tabs[3]:
    st.markdown("#### Validation on the training tools (stages 4 to 6)")
    st.caption("Repeated stratified group cross-validation: every validation fold contains tools the model did not "
               "train on. Values are mean ± std over folds.")
    task = st.radio("Task", ["RUL regression", "Wear-state classification"], horizontal=True)
    key, kind, pm, lower = (("rul", "reg", "RMSE", True) if task == "RUL regression"
                            else ("wear", "clf", "F1", False))

    st.markdown("##### Stage 4 · Baselines vs models (default parameters)")
    if A[f"s4_{key}"] is not None:
        c1, c2 = st.columns([1, 1.1])
        c1.dataframe(format_mean_std(A[f"s4_{key}"], kind), hide_index=True)
        c2.plotly_chart(mean_std_bar(A[f"s4_{key}"], pm, lower, "Baselines (light) vs models"))

    st.markdown("##### Stage 5 · Tuned models, nested cross-validation")
    if A[f"s5_{key}"] is not None:
        c1, c2 = st.columns([1, 1.1])
        c1.dataframe(format_mean_std(A[f"s5_{key}"], kind), hide_index=True)
        c2.plotly_chart(mean_std_bar(A[f"s5_{key}"], pm, lower, "Tuned models (best in blue)"))
        with st.expander("Best hyper-parameters (grid search on the training tools)"):
            st.dataframe(A[f"bp_{key}"], hide_index=True)

    st.markdown("##### Stage 6 · Ablation: remove one component at a time")
    if A[f"abl_{key}"] is not None:
        c1, c2 = st.columns([1, 1.1])
        ab = A[f"abl_{key}"]
        show = format_mean_std(ab.rename(columns={"Variant": "Model"}), kind).rename(columns={"Model": "Variant"})
        show["Performance drop"] = ab["Performance drop"].round(3)
        c1.dataframe(show, hide_index=True)
        c2.plotly_chart(ablation_bar(ab, f"Which sensor group matters? ({S['models'][key]['name']})"))

# ================================================================== Test results
with tabs[4]:
    st.markdown("#### Held-out test tools (stage 7): evaluated once")
    st.info("The test tools were set aside before any modelling. They were used once, after the models and "
            "hyper-parameters were chosen on the training tools. Because only a few tools are held out, treat "
            "these numbers as indicative, not exact.")
    if A["split"] is not None:
        with st.expander("How the data was split (by tool, stratified by wear class)"):
            st.dataframe(A["split"], hide_index=True)

    a, b = st.columns([1, 1.5])
    with a:
        st.markdown("**RUL regression**")
        if A["test_rul"] is not None:
            st.dataframe(A["test_rul"].round(3), hide_index=True)
    with b:
        st.markdown("**Wear-state classification**")
        if A["test_wear"] is not None:
            st.dataframe(A["test_wear"].round(3), hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        cm = confusion_matrix(test["wear_class"], test["pred_wear"].astype(int), labels=[0, 1, 2])
        fig = px.imshow(cm, x=NAMES, y=NAMES, text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="Actual", color="Runs"))
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(tidy(fig, 380, f"Confusion matrix ({S['models']['wear']['name']})"))
    with c2:
        fig = go.Figure()
        for i, n in enumerate(NAMES):
            pos = (test["wear_class"] == i).to_numpy()
            col = f"p_{n}"
            if col in test and test[col].notna().all() and 0 < pos.sum() < len(pos):
                fpr, tpr, _ = roc_curve(pos, test[col])
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{n} (AUC {auc(fpr, tpr):.2f})",
                                         line=dict(color=COLORS[n], width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                                 line=dict(color=GREY, dash="dash")))
        fig.update_xaxes(title="False positive rate")
        fig.update_yaxes(title="True positive rate")
        st.plotly_chart(tidy(fig, 380, "ROC curves (one-vs-rest)"))

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(test, x="RUL", y=test["pred_rul"].clip(lower=0), color="wear_state",
                         color_discrete_map=COLORS, category_orders={"wear_state": NAMES}, hover_data=["case", "run"],
                         labels={"RUL": "Actual RUL (runs)", "y": "Predicted RUL (runs)", "wear_state": ""})
        lim = [0, max(test["RUL"].max(), test["pred_rul"].max()) * 1.05 + 0.5]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="#64748b", dash="dash"),
                                 showlegend=False))
        st.plotly_chart(tidy(fig, 380, f"Predicted vs actual RUL ({S['models']['rul']['name']})"))
    with c2:
        st.markdown("**Error analysis per test tool**")
        if A["per_tool"] is not None:
            st.dataframe(A["per_tool"].round(3), hide_index=True)
        if A["per_class"] is not None:
            st.markdown("**Per-class results**")
            st.dataframe(A["per_class"].round(3), hide_index=True)

# ================================================================== Explainability
with tabs[5]:
    st.markdown("#### What drives wear and RUL?")
    c1, c2 = st.columns(2)
    with c1:
        if A["shap_rul"] is not None:
            st.plotly_chart(importance_bar(A["shap_rul"], "SHAP: drivers of the RUL prediction", "Mean |SHAP value|"))
        elif A["importance"] is not None:
            st.plotly_chart(importance_bar(A["importance"], "Top features for RUL", "Importance"))
    with c2:
        if A["shap_wear"] is not None:
            st.plotly_chart(importance_bar(A["shap_wear"], "SHAP: drivers of the 'Worn' probability",
                                           "Mean |SHAP value|"))
        else:
            corr = data[FEATS + ["VB"]].corr()["VB"].drop("VB").dropna()
            corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(15)[::-1]
            fig = go.Figure(go.Bar(x=corr.values, y=corr.index, orientation="h", marker_color=ACCENT))
            st.plotly_chart(tidy(fig, 440, "Features most correlated with wear"))

    if A["importance"] is not None and A["shap_rul"] is not None:
        with st.expander("Random Forest importance (a second opinion)"):
            st.plotly_chart(importance_bar(A["importance"], "Random Forest feature importance (RUL)", "Importance"))

    st.markdown("**Explore a signal**")
    rank = A["shap_rul"] if A["shap_rul"] is not None else A["importance"]
    feat = st.selectbox("Feature", list(rank.index[:30]) if rank is not None else FEATS[:30])
    fig = px.scatter(data, x=feat, y="VB", color="wear_state", symbol="split", color_discrete_map=COLORS,
                     category_orders={"wear_state": NAMES}, hover_data=["case", "run"],
                     labels={"VB": "Flank wear VB (mm)", "wear_state": "", "split": ""})
    st.plotly_chart(tidy(fig, 380, f"{feat} vs flank wear"))
    st.caption("A feature is useful when the three colours separate along the x-axis.")

# ================================================================== Batch
with tabs[6]:
    st.markdown("#### Predict for many runs at once")
    st.write("Upload a CSV with the training feature columns. Missing columns are filled with typical values.")
    st.download_button("⬇️ Download CSV template", pd.DataFrame([meta["defaults"]]).to_csv(index=False),
                       "template.csv", "text/csv")
    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        df_in = pd.read_csv(up)
        missing = [f for f in FEATS if f not in df_in.columns]
        if missing:
            st.info(f"{len(missing)} feature columns not in your file; typical values were used for them.")
        for f in missing:
            df_in[f] = meta["defaults"][f]
        rul, cls, conf = predict(df_in)
        res = df_in.copy()
        res.insert(0, "Predicted wear state", [NAMES[i] for i in cls])
        res.insert(0, "Predicted RUL (runs)", rul.round(1))
        st.dataframe(res)
        st.download_button("⬇️ Download predictions", res.to_csv(index=False), "predictions.csv", "text/csv")
