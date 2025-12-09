# app_streamlit_corrected.py
"""
Professional Breast Cancer Predictor MVP
- Modern, medical-grade UI/UX design
- Improved spacing, typography, and color scheme
- Enhanced user experience with better information hierarchy
- Normalizes risk labels to only: High Risk / Medium Risk / Low Risk
"""

import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import plotly.express as px
import plotly.graph_objects as go
import traceback
from datetime import datetime

st.set_page_config(
    page_title="Breast Cancer Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------
# Professional Medical UI CSS
# -------------------------
st.markdown(
    """
    <style>
    /* Base styling */
    .main { background-color: #f8fafc; }
    .stApp { background-color: #f8fafc; }

    /* Professional header */
    .main-header { 
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    .header-title { 
        font-size: 2.25rem; 
        color: white; 
        font-weight: 700; 
        font-family: 'Inter', sans-serif;
        margin-bottom: 0.5rem;
    }
    .header-subtitle { 
        color: #e0e7ff; 
        font-size: 1.1rem; 
        font-weight: 400;
        line-height: 1.5;
        max-width: 600px;
    }

    /* Enhanced cards */
    .professional-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .professional-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
    }

    /* Section headers */
    .section-title {
        font-size: 1.5rem;
        color: #1e293b;
        font-weight: 600;
        margin-bottom: 1.25rem;
        font-family: 'Inter', sans-serif;
        border-left: 4px solid #3b82f6;
        padding-left: 1rem;
    }

    /* Risk badges */
    .risk-badge {
        padding: 0.5rem 1rem;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        font-size: 0.875rem;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    .risk-badge::before {
        content: "●";
        font-size: 0.75rem;
    }

    /* Interpretation sections */
    .interpretation-header {
        color: #1e293b;
        font-size: 1.125rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f1f5f9;
    }

    .recommendation-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #bae6fd;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    /* Button enhancements */
    .stButton button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border: none;
        transition: all 0.2s ease;
    }

    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* File uploader enhancement */
    .uploadedFile {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 1.5rem;
        background: #f8fafc;
    }

    /* Metric displays */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.875rem;
    }

    /* Custom spacing */
    .section-spacing {
        margin: 2rem 0;
    }

    /* Progress indicators */
    .progress-container {
        background: #f1f5f9;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 1rem 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Try to import real backend; otherwise demo stub
# -------------------------
try:
    import predictor_backend as pb  # your real backend file

    has_real_backend = True
except Exception as e:
    pb = None
    has_real_backend = False
    import_error_msg = str(e)


def demo_predict_from_excel(path):
    """Demo stub: reads input file and returns fake predictions + features_df"""
    df = pd.read_excel(path)
    n = len(df)
    rng = np.random.default_rng(12345)
    scores = rng.random(n) * 100 + 50  # More realistic score range
    # simple 3-group risk
    labels = pd.qcut(scores, q=3, labels=['Low Risk', 'Medium Risk', 'High Risk'])
    results = []
    for i, s in enumerate(scores):
        results.append({
            "index": i,
            "original_index": int(df.index[i]) if hasattr(df, "index") else i,
            "ensemble_score": float(s),
            "risk_group": str(labels[i]),
            "model_scores": {"demo_model": float(s)},
            "timestamp": datetime.now().isoformat()
        })
    return results, df


def demo_generate_report(results, features_df, out_docx_path, include_visualizations=True):
    """Minimal demo report generator using python-docx"""
    try:
        from docx import Document
    except Exception:
        raise ImportError("Install python-docx to generate reports: pip install python-docx")
    doc = Document()
    doc.add_heading("Breast Cancer Clinical Risk Assessment Report", level=1)
    doc.add_paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}")
    doc.add_paragraph(f"Number of patients assessed: {len(results)}")
    doc.add_paragraph("This report contains AI-powered breast cancer risk assessments and clinical recommendations.")
    doc.save(out_docx_path)
    return out_docx_path


# Bind functions
if has_real_backend and hasattr(pb, "predict_breast_cancer_from_excel") and hasattr(pb,
                                                                                    "generate_breast_cancer_clinical_report"):
    predict_from_excel = pb.predict_breast_cancer_from_excel
    generate_report = pb.generate_breast_cancer_clinical_report
else:
    predict_from_excel = demo_predict_from_excel
    generate_report = demo_generate_report
    if has_real_backend:
        # imported but missing required functions
        missing = []
        if not hasattr(pb, "predict_breast_cancer_from_excel"):
            missing.append("predict_breast_cancer_from_excel")
        if not hasattr(pb, "generate_breast_cancer_clinical_report"):
            missing.append("generate_breast_cancer_clinical_report")
        import_error_msg = f"predictor_backend imported but missing: {missing}"
    else:
        import_error_msg = import_error_msg if 'import_error_msg' in globals() else "predictor_backend not found; using demo backend."

# -------------------------
# Normalization helper: map any incoming risk label to one of three canonical labels
# -------------------------
def normalize_risk_label(raw_label):
    """
    Map any incoming label to exactly one of:
      - 'High Risk'
      - 'Medium Risk'
      - 'Low Risk'
    Uses substring matching (case-insensitive).
    """
    if raw_label is None:
        return "Low Risk"
    s = str(raw_label).strip().lower()
    # prioritize 'high'
    if "high" in s:
        return "High Risk"
    if "medium" in s or "mod" in s:
        return "Medium Risk"
    # fallback -> low
    return "Low Risk"

# Professional Risk color mapping (only 3 canonical labels)
RISK_COLOR_MAP = {
    "High Risk": "#ef4444",
    "Medium Risk": "#f59e0b",
    "Low Risk": "#10b981"
}
DEFAULT_RISK_COLOR = "#6b7280"


def risk_color(label):
    return RISK_COLOR_MAP.get(str(label), DEFAULT_RISK_COLOR)


# -------------------------
# Clinical Interpretation Content (unchanged, uses canonical keys)
# -------------------------
CLINICAL_INTERPRETATIONS = {
    "High Risk": {
        "title": "HIGH RISK PROFILE - URGENT ONCOLOGICAL EVALUATION RECOMMENDED",
        "color": "#ef4444",
        "icon": "⚠️",
        "interpretation_points": [
            "Breast cancer risk score: {score:.1f} (Significantly Elevated)",
            "Clinical features suggest high probability of malignancy",
            "Immediate comprehensive breast evaluation required",
            "High likelihood of aggressive tumor characteristics",
            "Consideration for biopsy and multidisciplinary review"
        ],
        "recommendation_title": "IMMEDIATE ACTIONS REQUIRED:",
        "management_points": [
            "Urgent referral to breast specialist/surgical oncology",
            "Diagnostic mammogram with tomosynthesis if not already performed",
            "Targeted breast ultrasound for characterization",
            "Core needle biopsy for pathological confirmation",
            "Multidisciplinary tumor board review",
            "MRI breast if high suspicion despite negative other imaging",
            "Genetic counseling referral if appropriate"
        ]
    },
    "Medium Risk": {
        "title": "MODERATE RISK PROFILE - ENHANCED SURVEILLANCE INDICATED",
        "color": "#f59e0b",
        "icon": "📊",
        "interpretation_points": [
            "Breast cancer risk score: {score:.1f} (Moderately Elevated)",
            "Some concerning features present requiring follow-up",
            "Short-interval imaging follow-up recommended",
            "Consider additional diagnostic modalities",
            "Close clinical monitoring essential"
        ],
        "recommendation_title": "RECOMMENDED MANAGEMENT:",
        "management_points": [
            "Short-term follow-up imaging in 6 months",
            "Consider diagnostic mammogram versus screening mammogram",
            "Targeted ultrasound for any suspicious areas",
            "Clinical breast examination in 3-6 months",
            "Consider second opinion from breast specialist",
            "Document stability on subsequent imaging",
            "Patient education on breast awareness"
        ]
    },
    "Low Risk": {
        "title": "LOW RISK PROFILE - ROUTINE SCREENING ADEQUATE",
        "color": "#10b981",
        "icon": "✅",
        "interpretation_points": [
            "Breast cancer risk score: {score:.1f} (Favorable Profile)",
            "Clinical features consistent with benign characteristics",
            "Continue with age-appropriate screening guidelines",
            "Standard follow-up intervals appropriate",
            "Reassurance and routine surveillance recommended"
        ],
        "recommendation_title": "STANDARD MANAGEMENT:",
        "management_points": [
            "Continue routine screening mammography per guidelines",
            "Annual clinical breast examination",
            "Regular self-breast awareness",
            "General breast health education",
            "Maintain healthy lifestyle factors",
            "Age-appropriate cancer screening adherence"
        ]
    }
}


def get_clinical_interpretation(risk_group, score):
    """Get clinical interpretation based on risk group"""
    # assume risk_group is one of canonical keys: 'High Risk', 'Medium Risk', 'Low Risk'
    interpretation_key = risk_group if risk_group in CLINICAL_INTERPRETATIONS else "Low Risk"

    base_interpretation = CLINICAL_INTERPRETATIONS.get(interpretation_key, CLINICAL_INTERPRETATIONS["Low Risk"])

    # Format points with actual score
    formatted_interpretation = base_interpretation.copy()
    formatted_interpretation["interpretation_points"] = [
        point.format(score=score) for point in base_interpretation["interpretation_points"]
    ]

    return formatted_interpretation, interpretation_key


# -------------------------
# Professional Header
# -------------------------
st.markdown(
    """
    <div class="main-header">
        <div style="max-width:1200px; margin:0 auto; padding:0 2rem;">
            <div class="header-title">Breast Cancer Risk Assessment Platform</div>
            <div class="header-subtitle">
                AI-powered clinical decision support for breast cancer risk stratification, 
                featuring ensemble model predictions, patient-level analytics, and evidence-based clinical recommendations.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Main Content Container
# -------------------------
with st.container():
    # System Status Bar
    if not has_real_backend:
        st.markdown(
            f'<div style="background: #fef3c7; border: 1px solid #f59e0b; border-radius: 10px; padding: 1rem; margin: 1rem 0;">'
            f'<div style="display: flex; align-items: center; gap: 0.5rem; color: #92400e; font-weight: 600;">'
            #f'🔬 DEMO MODE ACTIVE • {import_error_msg}'
            f'</div></div>',
            unsafe_allow_html=True
        )

    # Upload Section
    st.markdown('<div class="professional-card">', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="section-title">Patient Data Upload</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="color: #64748b; margin-bottom: 1.5rem; line-height: 1.6;">'
            'Upload standardized patient data in Excel format for AI-powered breast cancer risk assessment. '
            'The system will analyze clinical features and provide risk stratification with clinical recommendations.'
            '</div>',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            '<div class="metric-card" style="text-align: center;">'
            '<div style="font-size: 0.875rem; color: #64748b; margin-bottom: 0.5rem;">System Status</div>'
            '<div style="font-size: 1.125rem; font-weight: 600; color: #10b981;">● Operational</div>'
            '</div>',
            unsafe_allow_html=True
        )

    uploaded_file = st.file_uploader(
        "Upload Patient Data Excel File",
        type=["xlsx"],
        help="Upload Excel file with patient clinical features for risk assessment"
    )

    # File requirements
    with st.expander("📋 Data Requirements & Format Specifications"):
        st.markdown("""
        **Required Data Format:**
        - Excel file (.xlsx) format
        - One row per patient
        - Columns should match trained feature set
        - No missing values in critical fields

        **Supported Features:**
        - Clinical demographics
        - Imaging characteristics  
        - Histopathological markers
        - Genetic risk factors
        - Lifestyle and family history
        """)

    run_analysis = st.button(
        "🚀 Run Comprehensive Risk Analysis",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # Process uploaded file
    results = None
    features_df = None
    if uploaded_file is not None and run_analysis:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
        try:
            tmp.write(uploaded_file.read())
            tmp.flush()
        finally:
            tmp.close()

        with st.spinner("🔄 Running comprehensive risk analysis..."):
            progress_bar = st.progress(0)
            for i in range(100):
                # Simulate progress for demo
                progress_bar.progress(i + 1)

            try:
                results, features_df = predict_from_excel(tmp.name)

                # Normalize incoming risk labels to canonical 3 labels
                for r in results:
                    raw = r.get("risk_group", None)
                    r["risk_group"] = normalize_risk_label(raw)

            except Exception as e:
                st.error("❌ Analysis failed. Please check your data format and try again.")
                st.exception(traceback.format_exc())
                results = None
                features_df = None
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

        if results is not None:
            st.session_state["bc_results"] = results
            st.session_state["bc_features"] = features_df
            st.success(f"✅ Analysis complete! Processed {len(results)} patients successfully.")

# -------------------------
# Results Display
# -------------------------
if "bc_results" in st.session_state and st.session_state["bc_results"] is not None:
    results = st.session_state["bc_results"]
    features_df = st.session_state["bc_features"]

    # Results Overview
    st.markdown('<div class="professional-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analysis Results Overview</div>', unsafe_allow_html=True)

    # Summary Metrics
    risk_counts = pd.DataFrame(results)["risk_group"].value_counts()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Patients", len(results))
    with col2:
        high_risk = risk_counts.get("High Risk", 0)
        st.metric("High Risk Cases", high_risk, delta=f"{high_risk / len(results) * 100:.1f}%")
    with col3:
        medium_risk = risk_counts.get("Medium Risk", 0)
        st.metric("Medium Risk Cases", medium_risk, delta=f"{medium_risk / len(results) * 100:.1f}%")
    with col4:
        low_risk = risk_counts.get("Low Risk", 0)
        st.metric("Low Risk Cases", low_risk, delta=f"{low_risk / len(results) * 100:.1f}%")

    # Detailed Results Table
    st.markdown("#### Patient Risk Assessment Details")
    df_display = pd.DataFrame([{
        "Patient ID": f"PAT_{int(r.get('index', 0)) + 1:03d}",
        "Risk Score": f"{r.get('ensemble_score', 0):.1f}",
        "Risk Category": r.get("risk_group", "")
    } for r in results])

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Risk Category": st.column_config.TextColumn(
                width="medium",
                help="AI-determined risk stratification"
            )
        }
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Analytics Visualization
    st.markdown('<div class="professional-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Population Analytics</div>', unsafe_allow_html=True)

    r_df = pd.DataFrame(results)
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        if "ensemble_score" in r_df.columns:
            fig = px.histogram(
                r_df,
                y="ensemble_score",
                nbins=min(20, max(4, len(r_df) // 2)),
                title="Risk Score Distribution",
                color_discrete_sequence=["#3b82f6"]
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=50, b=10),
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Count of Patients",
                yaxis_title="Ensemble Risk Score",
                bargap = 0.15
            )
            st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        if "risk_group" in r_df.columns:
            pie = r_df["risk_group"].value_counts().reset_index()
            pie.columns = ["risk_group", "count"]
            fig2 = px.pie(
                pie,
                names="risk_group",
                values="count",
                title="Risk Category Distribution",
                hole=0.4,
                color="risk_group",
                color_discrete_map=RISK_COLOR_MAP
            )
            fig2.update_layout(
                margin=dict(l=10, r=10, t=50, b=10),
                height=350,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Patient Analysis Section
# -------------------------
if "bc_results" in st.session_state and "bc_features" in st.session_state:
    st.markdown('<div class="professional-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Individual Patient Analysis</div>', unsafe_allow_html=True)

    results = st.session_state["bc_results"]
    features_df = st.session_state["bc_features"]

    # Ensure features_df is a DataFrame
    if not isinstance(features_df, pd.DataFrame):
        try:
            features_df = pd.DataFrame(features_df)
        except Exception:
            st.error("Feature data is not in tabular form.")
            features_df = None

    # Patient Selector
    col1, col2 = st.columns([1, 2])
    with col1:
        try:
            idx_list = [int(r["index"]) for r in results]
            patient_names = [f"PAT_{idx + 1:03d}" for idx in idx_list]
            sel_patient = st.selectbox("Select Patient", patient_names)
            patient_pos = int(sel_patient.split("_")[1]) - 1
        except Exception:
            patient_pos = 0

    with col2:
        if patient_pos < len(results):
            patient_result = results[patient_pos]
            risk_group = patient_result.get("risk_group", "N/A")
            color = risk_color(risk_group)
            st.markdown(
                f'<div style="display: flex; align-items: center; gap: 1rem; justify-content: flex-end;">'
                f'<div style="font-size: 0.875rem; color: #64748b;">Current Selection:</div>'
                f'<span class="risk-badge" style="background:{color}">{risk_group}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    # Patient Details
    if features_df is not None and patient_pos < len(results):
        patient_result = results[patient_pos]
        patient_features = features_df.iloc[patient_pos] if hasattr(features_df, 'iloc') else None

        if patient_result is not None and patient_features is not None:
            # Patient Summary
            score = patient_result.get("ensemble_score", None)
            risk_group = patient_result.get("risk_group", "N/A")

            st.markdown("#### Patient Risk Profile")
            summary_col1, summary_col2 = st.columns(2)

            with summary_col1:
                st.metric("Risk Score", f"{score:.1f}" if score is not None else "N/A")
            with summary_col2:
                st.metric("Risk Category", risk_group)

            # Feature Analysis
            st.markdown("#### Clinical factors")
            try:
                numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
            except Exception:
                numeric_cols = []
                for c in features_df.columns:
                    try:
                        pd.to_numeric(features_df[c])
                        numeric_cols.append(c)
                    except Exception:
                        continue

            if len(numeric_cols) > 0:
                patient_numeric = patient_features[numeric_cols].astype(float)
                means = features_df[numeric_cols].mean()
                stds = features_df[numeric_cols].std().replace(0, np.nan).fillna(0)
                diffs = (patient_numeric - means).abs()
                top_feats = diffs.sort_values(ascending=False).head(8).index.tolist()

                # Feature comparison chart
                feat_names = top_feats[::-1]
                patient_vals = patient_numeric[top_feats].values[::-1]
                means_vals = means[top_feats].values[::-1]
                stds_vals = stds[top_feats].values[::-1]
                lowers = [m - s for m, s in zip(means_vals, stds_vals)]
                uppers = [m + s for m, s in zip(means_vals, stds_vals)]

                fig = go.Figure()
                # Add Normal Range bars; only the first bar will create the legend entry for "Normal Range"
                for i, (low, high) in enumerate(zip(lowers, uppers)):
                    show_leg = True if i == 0 else False
                    fig.add_trace(go.Bar(
                        x=[high - low],
                        y=[feat_names[i]],
                        base=[low],
                        orientation='h',
                        marker=dict(color='rgba(203,213,225,0.6)'),
                        showlegend=show_leg,
                        name='Normal Range' if show_leg else None,
                        hovertemplate="Normal range width: %{x:.2f} (low=%{base:.2f}, high=%{x:.2f})<extra></extra>",
                    ))

                # Add patient values as scatter with clear legend
                fig.add_trace(go.Scatter(
                    x=patient_vals,
                    y=feat_names,
                    mode='markers',
                    marker=dict(color='#ef4444', size=14, line=dict(width=2, color='white')),
                    name='Patient Value',
                    hovertemplate='%{x:.2f} (patient)<extra></extra>'
                ))

                fig.update_layout(
                    height=400,
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title="Feature Value",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Clinical Interpretation Section
# -------------------------
if "bc_results" in st.session_state and "bc_features" in st.session_state:
    st.markdown('<div class="professional-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Clinical Interpretation & Recommendations</div>', unsafe_allow_html=True)

    if 'patient_result' in locals() and patient_result is not None:
        score = patient_result.get("ensemble_score", 0)
        risk_group = patient_result.get("risk_group", "Low Risk")

        interpretation, interpretation_key = get_clinical_interpretation(risk_group, score)
        interpretation_color = interpretation["color"]
        interpretation_icon = interpretation["icon"]

        # Risk Banner
        st.markdown(
            f'<div style="background: {interpretation_color}15; border-left: 4px solid {interpretation_color}; '
            f'padding: 1.5rem; border-radius: 0 12px 12px 0; margin: 1rem 0;">'
            f'<div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">'
            f'<div style="font-size: 1.5rem;">{interpretation_icon}</div>'
            f'<div style="font-size: 1.25rem; font-weight: 600; color: {interpretation_color};">{interpretation["title"]}</div>'
            f'</div>'
            f'<div style="color: #475569; line-height: 1.6;">'
            f'{"<br>• ".join([""] + interpretation["interpretation_points"])}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Recommendations
        st.markdown(
            f'<div class="recommendation-card">'
            f'<div style="font-size: 1.125rem; font-weight: 600; color: #1e293b; margin-bottom: 1rem;">{interpretation["recommendation_title"]}</div>'
            f'<div style="color: #475569; line-height: 1.8;">'
            f'{"<br>• ".join([""] + interpretation["management_points"])}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )
