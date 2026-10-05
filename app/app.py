"""Streamlit demo: predict a patient's risk of physical-health decline on chemo.

Run from the repo root:
    streamlit run app/app.py
"""
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from utils import (BINARY_INPUTS, INTEGER_INPUTS, LAB_INPUTS, ROOT, config, label,
                   load_app_inputs, load_bundle, predict_risk, risk_band, sweep,
                   training_defaults)

st.set_page_config(page_title="Chemo PRO Decline Risk", page_icon="🩺", layout="wide")

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"


@st.cache_resource
def get_bundle():
    return load_bundle()


@st.cache_data
def get_defaults():
    return training_defaults()


@st.cache_data
def get_app_inputs():
    return load_app_inputs()


try:
    bundle = get_bundle()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

defaults = get_defaults()
spec = get_app_inputs()
key_inputs = {k["column"]: k for k in spec["key_inputs"]}
threshold = bundle["threshold"]

PRESETS = {
    "Typical patient (training medians)": {},
    "Example: patient who declined (P1741)": spec["example_patients"]["largest_gph_drop_patient"]["features"],
    "Example: patient who improved (P1640)": spec["example_patients"]["largest_gph_improvement_patient"]["features"],
}


def apply_preset():
    """Copy the chosen preset into the input widgets."""
    base = dict(defaults, **PRESETS[st.session_state.preset])
    st.session_state.base_values = base
    # New widget keys make the what-if sliders restart at the new patient's values.
    st.session_state.patient_version = st.session_state.get("patient_version", 0) + 1
    for col in key_inputs:
        v = base.get(col)
        if col in LAB_INPUTS:
            missing = v is None or (isinstance(v, float) and np.isnan(v))
            st.session_state[f"miss_{col}"] = missing
            v = defaults[col] if missing else v
        if col in BINARY_INPUTS:
            v = "Yes" if int(v) == 1 else "No"
        elif col in INTEGER_INPUTS:
            v = int(v)
        elif key_inputs[col]["type"] == "numeric":
            v = float(v)
        st.session_state[f"in_{col}"] = v


if "base_values" not in st.session_state:
    st.session_state.preset = list(PRESETS)[0]
    apply_preset()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Patient information")
    st.caption("Only information available **before** the first chemotherapy dose.")
    st.selectbox("Load a patient", list(PRESETS), key="preset", on_change=apply_preset)

    values = dict(st.session_state.base_values)
    for col, meta in key_inputs.items():
        k = f"in_{col}"
        if meta["type"] == "categorical":
            values[col] = st.selectbox(label(col), meta["allowed_values"], key=k)
        elif col in BINARY_INPUTS:
            values[col] = 1 if st.radio(label(col), ["No", "Yes"], key=k, horizontal=True) == "Yes" else 0
        elif col in INTEGER_INPUTS:
            values[col] = st.slider(label(col), int(meta["min"]), int(meta["max"]), key=k)
        else:
            lo, hi = float(meta["min"]), float(meta["max"])
            if col in LAB_INPUTS:
                missing = st.checkbox(f"{label(col)}: not measured", key=f"miss_{col}")
                if missing:
                    values[col] = np.nan
                    continue
            values[col] = st.slider(label(col), lo, hi, key=k, step=0.1 if hi - lo > 5 else 0.01)
    st.caption("The other 40 inputs (vitals, other labs, medications...) are kept at the "
               "loaded patient's values, or typical values for the default patient.")

# ---------------------------------------------------------------- header
st.title("Predicting physical-health decline on chemotherapy")
st.write(f"**Task:** {bundle['target_description']} (PROMIS Global Physical Health). "
         f"**Model:** {bundle['model_name']}, cross-validated AUROC {bundle['cv_auroc']:.3f}. "
         "Trained on a synthetic dataset modelled on Ostberg & Peterson (CS229, 2020).")

tab_pred, tab_perf, tab_about = st.tabs(["Prediction & what-if", "Model performance", "How it works"])

# ---------------------------------------------------------------- prediction
with tab_pred:
    risk = predict_risk(bundle, values)
    band, verdict = risk_band(risk, threshold)
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted risk of decline", f"{risk:.0%}")
    c2.metric("Risk band", band, help=f"High = at or above the model's decision threshold "
                                     f"({threshold:.0%}), chosen to maximise F1 on training data.")
    c3.metric("Average patient in our data", f"{0.204:.0%}", help="Share of patients with a 10+ point drop.")
    (st.error if band == "High" else st.warning if band == "Moderate" else st.success)(verdict)
    st.progress(min(risk, 1.0))

    st.divider()
    st.subheader("What-if risk simulator")
    st.write("Change one or two inputs and see how the predicted risk responds, "
             "with everything else about this patient held fixed.")

    numeric_keys = [c for c, m in key_inputs.items() if m["type"] == "numeric" and c not in BINARY_INPUTS]
    w1, w2 = st.columns(2)
    changes = {}
    for col_ui, idx, default_feat in [(w1, 1, "albumin_g_dl"), (w2, 2, "ecog_status")]:
        with col_ui:
            feat = st.selectbox(f"Input {idx}", ["(none)"] + numeric_keys,
                                index=numeric_keys.index(default_feat) + 1 if idx == 1 else 0,
                                format_func=lambda c: c if c == "(none)" else label(c), key=f"wf{idx}")
            if feat == "(none)" or feat in changes:
                continue
            m = key_inputs[feat]
            cur = values[feat] if not pd.isna(values[feat]) else defaults[feat]
            if feat in INTEGER_INPUTS:
                changes[feat] = st.slider(f"New {label(feat)}", int(m["min"]), int(m["max"]),
                                          int(cur), key=f"wv{idx}_{feat}_{st.session_state.patient_version}")
            else:
                changes[feat] = st.slider(f"New {label(feat)}", float(m["min"]), float(m["max"]),
                                          float(cur), key=f"wv{idx}_{feat}_{st.session_state.patient_version}")

    new_values = dict(values, **changes)
    new_risk = predict_risk(bundle, new_values)
    d1, d2 = st.columns(2)
    d1.metric("Current risk", f"{risk:.0%}")
    d2.metric("Risk after changes", f"{new_risk:.0%}", delta=(f"{(new_risk - risk) * 100:+.1f} pts" if abs(new_risk - risk) >= 0.0005 else None),
              delta_color="inverse")

    if changes:
        feat = next(iter(changes))
        m = key_inputs[feat]
        grid = (np.arange(int(m["min"]), int(m["max"]) + 1) if feat in INTEGER_INPUTS
                else np.linspace(m["min"], m["max"], 60))
        curves = [sweep(bundle, values, feat, grid).assign(scenario="Current patient")]
        if len(changes) > 1:
            curves.append(sweep(bundle, new_values, feat, grid).assign(
                scenario=f"With {label(list(changes)[1])} = {list(changes.values())[1]:g}"))
        df = pd.concat(curves)
        scen = df["scenario"].unique().tolist()
        y_max = min(1.0, max(df["risk"].max(), new_risk, threshold) * 1.25 + 0.05)
        line = alt.Chart(df).mark_line(strokeWidth=2.5).encode(
            x=alt.X("value:Q", title=label(feat)),
            y=alt.Y("risk:Q", title="Predicted risk", axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0, y_max])),
            color=alt.Color("scenario:N", scale=alt.Scale(domain=scen, range=[BLUE, ORANGE][:len(scen)]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=[alt.Tooltip("value:Q", title=label(feat), format=".2f"),
                     alt.Tooltip("risk:Q", title="Risk", format=".1%"), "scenario:N"])
        rule = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(strokeDash=[4, 4], color=GREY).encode(
            y="t:Q")
        x_cur = values[feat] if not pd.isna(values[feat]) else defaults[feat]

        def dot(x, y, color, name):
            return alt.Chart(pd.DataFrame({"value": [x], "risk": [y], "point": [name]})).mark_point(
                size=120, filled=True, color=color, stroke="white", strokeWidth=2, opacity=1).encode(
                x="value:Q", y="risk:Q", tooltip=["point:N", alt.Tooltip("risk:Q", title="Risk", format=".1%")])

        layers = line + rule + dot(x_cur, risk, "#0b0b0b", "Now")
        if new_risk != risk or changes[feat] != x_cur:
            layers = layers + dot(changes[feat], new_risk, ORANGE, "What-if")
        st.altair_chart(layers.properties(height=320), width="stretch")
        st.caption(f"Curve: risk across the full range of {label(feat)}. Dashed line: decision "
                   f"threshold ({threshold:.0%}). Black dot: current value; orange dot: what-if value.")

# ---------------------------------------------------------------- performance
with tab_perf:
    results = config.RESULTS_DIR
    target = bundle["target"]
    table_path = results / f"model_comparison_{target}.csv"
    if table_path.exists():
        st.subheader("All models on the held-out test set (360 patients)")
        st.dataframe(pd.read_csv(table_path).style.format(precision=3), hide_index=True,
                     width="stretch")
        st.caption("\\* = final model, chosen by 5-fold cross-validated AUROC on the training set "
                   "(the test set was never used for any choice).")
        summary_path = results / f"test_summary_{target}.json"
        if summary_path.exists():
            s = json.load(open(summary_path))
            lo, hi = s["test_auroc_ci"]
            st.info(f"Final model test AUROC **{s['test_auroc']:.3f}** (bootstrap 95% CI {lo:.3f} - {hi:.3f}). "
                    "For reference, the paper's best model reached 0.771 on real Stanford data.")
        figs = config.FIGURES_DIR
        f1, f2 = st.columns(2)
        for col_ui, name, cap in [
            (f1, f"11_roc_curves_{target}.png", "ROC curves"),
            (f2, f"12_confusion_matrix_{target}.png", "Confusion matrix (final model)"),
            (f1, f"13_feature_importance_{target}.png", "Which inputs matter"),
            (f2, f"14_ablation_{target}.png", "Ablation by feature group"),
        ]:
            if (figs / name).exists():
                col_ui.image(str(figs / name), caption=cap, width="stretch")
    else:
        st.warning("Run `python -m src.evaluate` to generate the results.")

# ---------------------------------------------------------------- about
with tab_about:
    st.markdown(f"""
**What the model predicts.** Whether a patient's PROMIS Global Physical Health (GPH) score
falls by 10 or more points (one standard deviation) between the pre-treatment survey and the
on-chemotherapy survey.

**What goes in.** 52 pieces of information known before the first chemo dose: demographics,
cancer type and stage, ECOG status, comorbidities, medications, recent hospital use, vitals,
labs, and the baseline PROMIS physical (GPH) and mental (GMH) scores. The on-chemo score is
never an input, because it is what defines the answer.

**How a prediction is made.** The app puts your inputs into one row with the same columns as
the dataset. The saved pipeline fills any missing lab with the training median (and flags that
it was missing), standardises the numbers, one-hot encodes the categories, then the
{bundle['model_name']} outputs a probability. That probability is compared with the threshold
({threshold:.0%}) chosen on the training data.

**Limitations.** The data are synthetic, generated to resemble the paper's Stanford STARR
cohort, so the results show the method works rather than giving clinical evidence. A risk
score is a screening aid for extra support, not a diagnosis.
""")
