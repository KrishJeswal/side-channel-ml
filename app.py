import os
import sys
import numpy as np
import streamlit as st
import joblib
import h5py

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.components.data_transformation import DataTransformationConfig, DataTransformation
from src.utils import compute_snr, compute_ge, AES_SBOX

CORRECT_KEY_BYTE = 0xe0
TARGET_BYTE      = 2
N_TRACES_RANGE   = [10, 25, 50, 100, 200, 500]

MODEL_LABELS = {
    "mlp":                 "MLP (Neural Network)",
    "xgboost":             "XGBoost",
    "random_forest":       "Random Forest",
    "decision_tree":       "Decision Tree",
    "logistic_regression": "Logistic Regression",
    "svm":                 "SVM (RBF kernel)",
}
STRATEGY_LABELS = {
    "pca":   "PCA (Principal Component Analysis)",
    "snr":   "SNR-based POI Selection",
    "anova": "ANOVA F-test POI Selection",
}

st.set_page_config(
    page_title="CipherTrace",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

.stApp { background: #0a0e1a; color: #e2e8f0; }

div[data-baseweb="select"],
div[data-baseweb="select"] > div,
div[data-testid="stSelectbox"] [role="button"],
div[data-testid="stSelectbox"] input { cursor: pointer !important; }

div[data-testid="stSlider"] > div,
div[data-testid="stSlider"] [role="slider"] { cursor: pointer !important; }

.stButton button {
    background: linear-gradient(135deg, #0369a1, #0284c7) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    width: 100%;
    cursor: pointer !important;
}
.stButton button:hover { background: linear-gradient(135deg, #0284c7, #0369a1) !important; }

.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    overflow: hidden;
}
.main-header h1 { font-family: 'Sora', sans-serif; font-weight: 700; font-size: 2rem; color: #f1f5f9; margin: 0 0 0.4rem 0; letter-spacing: -0.02em; }
.main-header p  { color: #94a3b8; font-size: 0.95rem; margin: 0; font-weight: 300; }
.accent { color: #38bdf8; }

.metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    height: 150px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.metric-card .label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; margin-bottom: 0.4rem; }
.metric-card .value { font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #38bdf8; margin: 0.2rem 0; }
.metric-card .sub   { font-size: 0.75rem; color: #475569; margin-top: 0.2rem; }

.result-badge { display: inline-block; padding: 0.3rem 0.9rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.badge-success { background: #064e3b; color: #34d399; border: 1px solid #065f46; }
.badge-warn    { background: #451a03; color: #fb923c; border: 1px solid #7c2d12; }
.badge-info    { background: #0c4a6e; color: #38bdf8; border: 1px solid #0369a1; }

.section-title { font-size: 1.2rem; font-weight: 600; color: #cbd5e1; text-transform: uppercase; letter-spacing: 0.06em; margin: 1.5rem 0 0.8rem 0; border-left: 3px solid #38bdf8; padding-left: 0.8rem; }

.info-box { background: #0c1930; border: 1px solid #1e3a5f; border-radius: 8px; padding: 1rem 1.2rem; font-size: 0.95rem; color: #93c5fd; margin: 0.8rem 0; font-family: 'JetBrains Mono', monospace; line-height: 1.7; }

.extraction-card { background: #0f172a; border-radius: 12px; padding: 2.5rem; text-align: center; margin: 1.5rem 0; box-shadow: 0 10px 25px rgba(0,0,0,0.2); }
.extraction-card h3           { margin: 0; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 0.1em; color: #94a3b8; }
.extraction-card .rank-number { font-family: 'JetBrains Mono', monospace; font-size: 4rem; font-weight: 700; margin: 0.5rem 0; }
.extraction-card .rank-text   { font-size: 1.3rem; font-weight: 600; margin-bottom: 1rem; }
.extraction-card .desc        { color: #cbd5e1; font-size: 1rem; line-height: 1.6; max-width: 800px; margin: 0 auto; }

hr { border-color: #1e293b; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_attack_data():
    path = os.path.join("artifacts", "raw", "ASCAD_processed.h5")
    with h5py.File(path, "r") as f:
        X_prof  = f["Profiling_traces/traces"][:].astype(np.float32)
        pt_prof = f["Profiling_traces/metadata/plaintext"][:]
        X_atk   = f["Attack_traces/traces"][:].astype(np.float32)
        pt_atk  = f["Attack_traces/metadata/plaintext"][:]
    y_prof = np.array([
        bin(AES_SBOX[int(pt[TARGET_BYTE]) ^ CORRECT_KEY_BYTE]).count('1')
        for pt in pt_prof
    ], dtype=np.int32)
    snr = compute_snr(X_prof, y_prof)
    return X_prof, y_prof, pt_prof, X_atk, pt_atk, snr


def load_model_and_transformer(model_name, strategy, k):
    model_path       = os.path.join("artifacts", "models", f"{model_name}_{strategy}_k{k}.joblib")
    transformer_key  = f"pca_n{k}" if strategy == "pca" else f"{strategy}_k{k}"
    transformer_path = os.path.join("artifacts", "models", f"transformer_{transformer_key}.joblib")

    if not os.path.exists(model_path):
        return None, None, f"Model file not found: {model_path}"
    if not os.path.exists(transformer_path):
        return None, None, f"Transformer file not found: {transformer_path}"

    return joblib.load(model_path), joblib.load(transformer_path), None


def get_trace_count_ge(ge_values, n):
    """Safely retrieve GE at a specific trace count."""
    return ge_values[N_TRACES_RANGE.index(n)] if n in N_TRACES_RANGE else ge_values[min(3, len(ge_values) - 1)]


st.markdown("""
<div class="main-header">
  <h1>CipherTrace: Side-Channel Attack <span class="accent">Demo</span></h1>
  <p>ML-based profiling attack on AES-128 · ASCAD Dataset · Krish Jeswal, RVCE</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Attack Configuration")
    st.markdown("""
    **Configure your attack variables here:**
    * **Classifier**: The ML algorithm used to predict the key byte.
    * **POI Strategy**: How the algorithm selects Points of Interest from raw traces.
    * **k**: Number of features extracted from raw power samples.
    * **GE experiments**: Attack iterations averaged to calculate Guessing Entropy.
    """)

    model_name = st.selectbox(
        "Classifier",
        options=list(MODEL_LABELS.keys()),
        format_func=lambda x: MODEL_LABELS[x],
        index=0
    )

    strategy = st.selectbox(
        "POI Strategy",
        options=list(STRATEGY_LABELS.keys()),
        format_func=lambda x: STRATEGY_LABELS[x],
        index=0
    )

    k = st.select_slider(
        "k (POIs / PCA components)",
        options=[20, 50, 100, 200],
        value=100
    )

    n_experiments = st.slider(
        "GE experiments", min_value=10, max_value=200, value=50, step=10
    )

    st.markdown("---")
    st.markdown("### Quick Reference")
    st.markdown("""
    <div class="info-box">
    <strong>Best HW Config:</strong><br>
    MLP + PCA + k=50 → GE=16.36<br><br>
    <strong>Best Overall (Identity):</strong><br>
    MLP + PCA + k=100 → GE=0.46<br>
    (Identity model, not available here)<br><br>
    <strong>Benchmarks:</strong><br>
    Random Guessing ≈ 127 GE<br>
    Successful Attack: &lt; 5 GE
    </div>
    """, unsafe_allow_html=True)

    run_attack = st.button("▶ Run Attack", use_container_width=True)

with st.spinner("Loading ASCAD dataset..."):
    try:
        X_prof, y_prof, pt_prof, X_atk, pt_atk, snr = load_attack_data()
        data_ok = True
    except Exception as e:
        st.error(f"Failed to load ASCAD_processed.h5: {e}")
        data_ok = False

tab1, tab2, tab3 = st.tabs(["Run Attack", "Dataset Guide", "Technical About"])

with tab1:
    if not data_ok:
        st.warning("Dataset not loaded. Ensure ASCAD_processed.h5 is at `artifacts/raw/`.")
        st.stop()

    st.markdown("""
    ### What Happens When You Run an Attack?
    When you click **Run Attack**, the application executes a four-step profiling attack:

    1. **Feature Extraction**: Raw power measurements are transformed using your chosen strategy down to `k` features, isolating the most leaky time samples.
    2. **ML Prediction**: The selected classifier outputs a probability score for each of the 9 Hamming Weight (HW) classes.
    3. **Log-Likelihood Accumulation**: All 256 possible key byte hypotheses are scored by accumulating log-probabilities across multiple traces.
    4. **Key Ranking (Guessing Entropy)**: Hypotheses are sorted by score. GE = average rank of the correct key across experiments.
    """)

    if strategy == "anova":
        st.warning("⚠️ ANOVA is known to perform poorly on masked ASCAD — GE typically worsens with more traces. Select PCA or SNR for a meaningful attack.")

    if run_attack:
        model, transformer, err = load_model_and_transformer(model_name, strategy, k)

        if err:
            st.error(
                f"⚠️ {err}\n\n"
                f"Run the training pipeline first:\n"
                f"`python -m src.pipeline.train_pipeline --strategy {strategy} --k {k}`"
            )
        else:
            with st.spinner(f"Running GE evaluation over {n_experiments} experiments..."):
                X_atk_t  = transformer.transform(X_atk)
                probs    = model.predict_proba(X_atk_t)

                ge_values = compute_ge(
                    probs, pt_atk, CORRECT_KEY_BYTE,
                    n_traces_range=N_TRACES_RANGE,
                    n_experiments=n_experiments
                )

                hw_pred   = int(np.argmax(probs[0]))
                ntd       = next((n for n, ge in zip(N_TRACES_RANGE, ge_values) if ge < 1), None)

            st.markdown("---")
            st.markdown('<div class="section-title">Attack Results</div>', unsafe_allow_html=True)

            ge_500    = ge_values[-1]
            ge_at_100 = get_trace_count_ge(ge_values, 100)

            badge_class = "badge-success" if ge_500 < 5  else ("badge-warn" if ge_500 < 50 else "badge-info")
            badge_text  = "KEY BROKEN"    if ge_500 < 5  else ("PARTIAL SUCCESS" if ge_500 < 50 else "FAILED TO BREAK")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="label">GE @ 500 traces</div>
                  <div class="value">{ge_500:.1f}</div>
                  <div class="sub"><span class="result-badge {badge_class}">{badge_text}</span></div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="label">GE @ 100 traces</div>
                  <div class="value">{ge_at_100:.1f}</div>
                  <div class="sub">rank of correct key</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                ntd_display = str(ntd) if ntd else ">500"
                st.markdown(f"""
                <div class="metric-card">
                  <div class="label">NtD (Traces to GE&lt;1)</div>
                  <div class="value">{ntd_display}</div>
                  <div class="sub">number of traces to disclose</div>
                </div>""", unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="label">Predicted HW (Trace 0)</div>
                  <div class="value">{hw_pred}</div>
                  <div class="sub">Hamming Weight class</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""                        
            **Understanding Your Results:**
            * **GE @ 500 traces**: Average rank of the correct key out of 256 after 500 traces. Lower is better.
            * **GE @ 100 traces**: Rank after only 100 traces — shows how fast the model learns.
            * **NtD**: Exact trace count where GE drops below 1 (correct key locked in as rank #1).
            """)

            correct_key_rank = max(1, int(round(ge_500)) + 1)

            if correct_key_rank == 1:
                card_border = "border: 2px solid #34d399;"
                card_color  = "#34d399"
                rank_title  = "Key Successfully Extracted!"
                rank_desc   = "The model perfectly identified the secret byte (0xe0) from 256 possibilities. No brute-forcing required."
            elif correct_key_rank <= 50:
                card_border = "border: 2px solid #fb923c;"
                card_color  = "#fb923c"
                rank_title  = "Partial Success — Search Space Narrowed"
                rank_desc   = "The model significantly narrowed the search space. A reduced brute-force attack from this rank would succeed rapidly."
            else:
                card_border = "border: 2px solid #ef4444;"
                card_color  = "#ef4444"
                rank_title  = "Attack Failed"
                rank_desc   = "The model failed to learn the leakage patterns from the hardware masking defense."

            st.markdown('<div class="section-title">Attack Performance & Analysis</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="extraction-card" style="{card_border}">
                <h3>Final Key Prediction Rank</h3>
                <div class="rank-number" style="color: {card_color};">#{correct_key_rank}</div>
                <div class="rank-text" style="color: {card_color};">{rank_title}</div>
                <div class="desc">
                    Out of 256 possible byte hypotheses, your configuration placed the correct secret key (0xe0)
                    at <b>Rank {correct_key_rank}</b>.<br><br>{rank_desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

            model_feedback = {
                "mlp":                 "You chose **MLP (Neural Network)**, which produces continuous, well-calibrated sigmoid outputs perfectly suited for log-likelihood accumulation.",
                "xgboost":             "You chose **XGBoost**. Tree models produce poorly calibrated, overconfident probabilities that ruin log-likelihood accumulation. Post-hoc calibration made GE worse (119.53 → 135.67) in experiments.",
                "random_forest":       "You chose **Random Forest**. Despite having the highest raw classification accuracy, its overconfident probabilities hurt GE significantly.",
                "decision_tree":       "You chose **Decision Tree**. It had the best Macro F1 (0.1094) in benchmarks but one of the worst GE values — demonstrating that F1 does not predict attack success.",
                "logistic_regression": "You chose **Logistic Regression** — a linear baseline. It performs consistently poorly across all strategies on this masked dataset.",
                "svm":                 "You chose **SVM (RBF kernel)**. Competitive on accuracy but probability calibration issues hurt GE performance.",
            }.get(model_name, "")

            strategy_feedback = {
                "pca":   "Your choice of **PCA** is the best strategy for masked ASCAD. It allows MLP to exploit non-linear distributed leakage patterns invisible to univariate metrics.",
                "snr":   "Your choice of **SNR** captures first-order univariate leakage. On masked ASCAD the max SNR is only 0.0032, and SNR misses the feature interactions that PCA exploits.",
                "anova": "Your choice of **ANOVA** is actively harmful on masked ASCAD. It selects samples that are statistically significant but uncorrelated with the HW leakage model.",
            }.get(strategy, "")

            st.markdown(f"""
### 🔍 Technical Breakdown

**1. Classifier Choice:** {model_feedback}

**2. Strategy Choice:** {strategy_feedback}

**3. Benchmark Comparison:** Best HW result: **MLP + PCA + k=50 → GE=16.36** at 500 traces.
Best overall (Identity model, 256 classes): **GE=0.46** — key fully broken at 500 traces.
            """)

    else:
        st.info("Configure the attack parameters in the sidebar and click **▶ Run Attack** to begin.")

with tab2:
    if not data_ok:
        st.warning("Dataset not loaded. Ensure ASCAD_processed.h5 is at `artifacts/raw/`.")
        st.stop()

    st.markdown("""
    <div class="section-title">Dataset Characteristics & Target Device</div>

    The ASCAD dataset was sourced from the **French National Cybersecurity Agency (ANSSI)**. It captures an
    **ATMega8515 microcontroller executing masked AES-128 encryption**.

    * **Profiling traces**: 50,000 × 700 samples (training)
    * **Attack traces**: 10,000 × 700 samples (testing)
    * **Dtype**: float32 · Value range: [-66.00, 47.00]
    * **Target byte index**: 2 · Correct key byte: 0xe0 = 224

    **Hardware Defense — First-Order Boolean Masking:**
    The device randomly splits internal variables to hide first-order leakages.
    Maximum SNR across all 700 samples is only **0.0032** (peak at sample 567).
    Mean traces per class show no visible divergence — the masking works against standard analysis.

    **Feature Overlap:** SNR and ANOVA agree on only 13/50 POIs.
    On masked ASCAD, ANOVA selects statistically significant samples uncorrelated with the HW model — actively harmful.

    <div class="section-title">HW Class Distribution</div>

    | HW Class | Count | Proportion | Note |
    |---|---|---|---|
    | 0 | 166 | 0.3% | Extremely rare |
    | 1 | 1,568 | 3.1% | |
    | 2 | 5,500 | 11.0% | |
    | 3 | 10,848 | 21.7% | |
    | 4 | 13,747 | 27.5% | Most common |
    | 5 | 11,023 | 22.0% | |
    | 6 | 5,416 | 10.8% | |
    | 7 | 1,532 | 3.1% | |
    | 8 | 200 | 0.4% | Extremely rare |

    Distribution matches Binomial B(8, 0.5) — confirms correct label generation.
    Class imbalance is inherent and expected, which is why **Macro F1** is used as the primary metric over accuracy.
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("""
    <div class="section-title">1. What is a Profiling Side-Channel Attack?</div>

    A **profiling side-channel attack** exploits unintended physical information leaked by a cryptographic device.
    When a chip runs AES, its power consumption fluctuates depending on the data it processes internally.
    By capturing these fluctuations and labelling them, ML models can reconstruct the secret key from unknown
    traces — without ever attacking the mathematics of AES.

    <div class="section-title">2. Novel Research Contributions</div>

    **A. PCA Outperforms Domain-Specific Feature Selection**
    PCA consistently beats SNR and ANOVA on masked ASCAD across all classifiers.
    PCA k=100 is the optimal dimensionality — Macro F1 drops at k=200, suggesting noise beyond 100 components.

    **B. SHAP Leakage Localization**
    SHAP analysis on the best MLP model shows only 1–2/10 overlap with SNR's top-10 leakage points.
    This is the finding: SNR identifies first-order univariate leakage; SHAP reveals the MLP exploits
    distributed, non-linear patterns that SNR cannot detect. This explains the MLP's advantage mechanistically.

    **C. Macro F1 Does Not Predict GE**
    Decision Tree achieved the best Macro F1 (0.1094) but one of the worst GE values.
    MLP had mid-range F1 but dramatically outperformed all models on GE.
    Tree models produce overconfident probabilities that ruin log-likelihood accumulation.
    Post-hoc calibration made XGBoost GE worse (119.53 → 135.67) — the gap is architectural, not fixable.

    **D. Identity Model Breaks the Key — HW Does Not**

    | Leakage Model | Classes | GE@100 | GE@500 | NtD |
    |---|---|---|---|---|
    | Hamming Weight | 9 | 51.34 | 12.03 | Not reached |
    | Identity ★ | 256 | 9.81 | 0.46 | 500 traces |

    Conventional SCA wisdom favours HW because fewer classes simplifies classification.
    On masked ASCAD with MLP+PCA, Identity is dramatically superior — 256-class probability outputs
    preserve fine-grained distinctions that HW's 9-class binning destroys during log-likelihood accumulation.

    <div class="section-title">3. Project Information</div>

    * **Author**: Krish Jeswal
    * **Institute**: RVCE Bengaluru | ETE Batch 2024–2028
    * **GitHub**: KrishJeswal
    """, unsafe_allow_html=True)