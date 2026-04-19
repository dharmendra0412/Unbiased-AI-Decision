"""
AI Bias Detector & Fairness Analyzer  |  python -m streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from fairlearn.metrics import (
    demographic_parity_ratio, equalized_odds_ratio,
    MetricFrame, selection_rate, false_positive_rate, false_negative_rate,
)
try:
    import google.generativeai as genai
    GEMINI_OK = True
except ImportError:
    GEMINI_OK = False

st.set_page_config(page_title="AI Bias Detector", page_icon="⚖️", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
*{font-family:'Inter',sans-serif;}
.hero{background:linear-gradient(135deg,#1a1f2e,#0f3460);border-radius:16px;
      padding:2rem 2.5rem;margin-bottom:1.5rem;border:1px solid #2d3748;}
.hero h1{color:#e2e8f0;font-size:2.2rem;margin:0;}
.hero p{color:#94a3b8;margin-top:.4rem;}
.box{background:#1e2433;border-radius:12px;padding:1.2rem 1.5rem;
     border-left:5px solid #7c3aed;margin-bottom:1rem;}
.box h3{color:#e2e8f0;margin:0 0 .4rem 0;font-size:1rem;}
.box p{color:#94a3b8;margin:0;font-size:.9rem;line-height:1.6;}
.note{background:#16213e;border-radius:10px;padding:1rem 1.3rem;
      color:#cbd5e1;font-size:.88rem;line-height:1.7;margin:.5rem 0;
      border:1px solid #2d3748;}
.note b{color:#a78bfa;}
.good{background:#0f2d1a;border-left:4px solid #22c55e;border-radius:8px;
      padding:.8rem 1rem;color:#86efac;font-size:.9rem;}
.warn{background:#2d2200;border-left:4px solid #f59e0b;border-radius:8px;
      padding:.8rem 1rem;color:#fcd34d;font-size:.9rem;}
.bad{background:#2d1a1a;border-left:4px solid #ef4444;border-radius:8px;
     padding:.8rem 1rem;color:#fca5a5;font-size:.9rem;}
.kcard{background:#1e2433;border-radius:12px;padding:1rem 1.2rem;
       text-align:center;border:1px solid #2d3748;}
.kcard .lbl{font-size:.72rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em;}
.kcard .val{font-size:1.9rem;font-weight:700;margin:.2rem 0;}
.kcard .sub{font-size:.75rem;color:#94a3b8;}
.tip{background:#1e1e2e;border-radius:8px;padding:.7rem 1rem;
     color:#a78bfa;font-size:.85rem;margin-top:.4rem;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚖️ AI Bias Detector")
    st.caption("v2.0 · Fairlearn + Gemini AI")
    st.divider()
    st.markdown("**🤖 Google Gemini AI (Optional)**")
    st.caption("Paste your API key for an AI-written bias report.")
    gemini_key = st.text_input("Gemini API Key", type="password",
                               placeholder="AIza...")
    if gemini_key:
        st.success("✅ Gemini connected!")
    else:
        st.info("Free key: [aistudio.google.com](https://aistudio.google.com)")
    st.divider()
    st.markdown("**ℹ️ Score Guide**")
    st.caption("1.0 = perfectly fair\n0.8–1.0 = acceptable\nBelow 0.8 = biased 🚨")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>⚖️ AI Bias Detector & Fairness Analyzer</h1>
  <p>Upload a dataset → the tool trains an AI → checks if it treats everyone fairly. No coding needed!</p>
</div>""", unsafe_allow_html=True)

with st.expander("🎓 What is AI Bias? — Click to learn before you start"):
    st.markdown("""
    **AI Bias** = when an AI makes unfair decisions against certain groups of people.  
    Example: A hiring AI trained on old data where men were hired more often will keep rejecting women — even equally qualified ones.

    | Where AI decides | Bias example |
    |---|---|
    | 🏢 Hiring | AI rejects women more often |
    | 🏦 Bank Loans | AI rejects people from certain zip codes |
    | 🏥 Healthcare | AI suggests fewer treatments for minorities |
    | 🎓 Education | AI scores essays lower for non-native speakers |

    This tool detects these patterns and tells you how to fix them.
    """)

st.divider()

# ── Step 1: Load Data ─────────────────────────────────────────────────────────
st.markdown("## 📁 Step 1 — Load Your Data")
st.markdown('<div class="note">A <b>dataset</b> is a spreadsheet (CSV file). Each row = one person. '
            'Each column = one piece of info (Age, Score, Hired, etc.)</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
uploaded = c1.file_uploader("Upload your CSV", type=["csv"])
with c2:
    st.markdown("<br>", unsafe_allow_html=True)
    use_demo = st.checkbox("✅ Use built-in demo (hiring bias)", value=(uploaded is None))

@st.cache_data
def make_demo():
    np.random.seed(42)
    N = 500
    g = np.random.choice(["Male","Female"], N, p=[0.55,0.45])
    e = np.clip(np.random.normal(5,3,N).astype(int),0,20)
    s = np.clip(np.random.normal(70,15,N),10,100).astype(int)
    edu = np.random.choice(["High School","Bachelor's","Master's","PhD"],N,p=[0.15,.50,.25,.10])
    p = 0.30*np.ones(N)
    p[g=="Male"]+=0.27; p[e>=5]+=0.10; p[s>=75]+=0.10
    p[edu=="Master's"]+=0.05; p[edu=="PhD"]+=0.08
    hired = (np.random.rand(N)<np.clip(p,0,1)).astype(int)
    return pd.DataFrame({"Gender":g,"Years_Experience":e,"Coding_Score":s,"Education":edu,"Hired":hired})

if uploaded:
    df = pd.read_csv(uploaded); use_demo = False
elif use_demo:
    df = make_demo()
    st.success("✅ Demo loaded — a fake hiring dataset with engineered gender bias.")
else:
    st.warning("Upload a CSV or enable the demo above."); st.stop()

with st.expander("👀 Preview data"):
    st.dataframe(df.head(), hide_index=True)
    st.caption(f"{len(df)} rows · {len(df.columns)} columns")

st.divider()

# ── Step 2: Configure ─────────────────────────────────────────────────────────
st.markdown("## ⚙️ Step 2 — Tell the Tool What to Check")
st.markdown('<div class="note">'
            '<b>Target</b> = what the AI decides (e.g. Hired). '
            '<b>Sensitive</b> = the group to audit (e.g. Gender). '
            '<b>Features</b> = what info the AI uses to decide.</div>', unsafe_allow_html=True)

cols = df.columns.tolist()
a1, a2 = st.columns(2)
tgt = a1.selectbox("🎯 Target column (what AI decides)", cols,
                   index=cols.index("Hired") if "Hired" in cols else len(cols)-1)
sen = a2.selectbox("👥 Sensitive column (group to audit)", cols,
                   index=cols.index("Gender") if "Gender" in cols else 0)
feats = st.multiselect("📊 Feature columns (AI inputs)",
                       [c for c in cols if c not in [tgt,sen]],
                       default=[c for c in cols if c not in [tgt,sen]])

# Warn if sensitive column is continuous
IS_NUM = pd.api.types.is_numeric_dtype(df[sen])
N_UNQ  = df[sen].nunique()
DO_BIN = IS_NUM and N_UNQ > 6
if DO_BIN:
    st.markdown(f'<div class="warn">⚠️ <b>{sen}</b> is a number with {N_UNQ} unique values. '
                'The tool will auto-group it into <b>Low / Medium / High</b> for fair analysis.</div>',
                unsafe_allow_html=True)

st.markdown(f'<div class="tip">💡 You are asking: "Does the AI treat <b>{sen}</b> '
            f'groups differently when deciding <b>{tgt}</b>?"</div>', unsafe_allow_html=True)
st.divider()

# ── Step 3: Analyze ───────────────────────────────────────────────────────────
st.markdown("## 🔍 Step 3 — Run the Fairness Check")
if st.button("🚀 Check for Bias Now!", type="primary"):
    if not feats:
        st.error("Please select at least one feature column."); st.stop()

    with st.spinner("Training model and checking fairness…"):
        X = df[feats].copy()
        for c in X.select_dtypes("object").columns:
            X[c] = LabelEncoder().fit_transform(X[c].astype(str))

        # Encode target
        raw_y = df[tgt]
        if pd.api.types.is_numeric_dtype(raw_y):
            y = raw_y.astype(int)
        else:
            uv = raw_y.dropna().unique()
            if len(uv) != 2:
                st.error(f"❌ '{tgt}' has {len(uv)} values {list(uv)}. Need exactly 2 (e.g. Yes/No).")
                st.stop()
            le2 = LabelEncoder()
            y = pd.Series(le2.fit_transform(raw_y.astype(str)), index=raw_y.index)
            lmap = dict(zip(le2.classes_, le2.transform(le2.classes_)))
            st.info("ℹ️ Target encoded: " + " · ".join(f"**{k}**→{v}" for k,v in lmap.items()))

        # Encode sensitive
        if DO_BIN:
            sens = pd.qcut(df[sen],q=3,labels=["Low","Medium","High"],duplicates="drop").astype(str)
            st.info(f"ℹ️ '{sen}' grouped into: {sorted(sens.unique().tolist())}")
        else:
            sens = df[sen].astype(str)

        X_tr,X_te,y_tr,y_te,s_tr,s_te = train_test_split(
            X,y,sens,test_size=0.3,random_state=42,stratify=y)
        model = LogisticRegression(max_iter=1000,class_weight="balanced")
        model.fit(X_tr,y_tr); preds = model.predict(X_te)
        acc = accuracy_score(y_te,preds)
        dp  = demographic_parity_ratio(y_te,preds,sensitive_features=s_te)
        eo  = equalized_odds_ratio(y_te,preds,sensitive_features=s_te)
        mf  = MetricFrame(
            metrics={"Selection Rate":selection_rate,"FPR":false_positive_rate,"FNR":false_negative_rate},
            y_true=y_te,y_pred=preds,sensitive_features=s_te)
        gdf = mf.by_group.reset_index()
        gdf.columns = [sen]+list(gdf.columns[1:])
        fi_vals = np.abs(model.coef_[0]) if hasattr(model,"coef_") else None

        # ── Save everything so results survive when Gemini button is clicked ──
        st.session_state["results"] = dict(
            acc=acc, dp=dp, eo=eo, gdf=gdf, sen=sen, tgt=tgt,
            feats=feats, fi_vals=fi_vals
        )

# ── Display results from session_state (persists across all button clicks) ───
if "results" in st.session_state:
    r    = st.session_state["results"]
    acc  = r["acc"]; dp = r["dp"]; eo = r["eo"]; gdf = r["gdf"]
    sen  = r["sen"]; tgt = r["tgt"]; feats = r["feats"]; fi_vals = r["fi_vals"]

    st.divider()
    st.markdown("## 📊 Step 4 — Your Results")

    # Top verdict banner
    if dp < 0.8:
        st.markdown(f'<div class="bad">🚨 <b>Bias Found!</b> The AI treats <b>{sen}</b> groups very differently.</div>',
                    unsafe_allow_html=True)
    elif dp < 0.9:
        st.markdown(f'<div class="warn">⚠️ <b>Mild Bias.</b> Small but measurable difference between <b>{sen}</b> groups.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="good">✅ <b>Looks Fair!</b> The AI treats <b>{sen}</b> groups roughly equally.</div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI cards
    dp_col = "#22c55e" if dp>=0.9 else ("#f59e0b" if dp>=0.8 else "#ef4444")
    eo_col = "#22c55e" if eo>=0.9 else ("#f59e0b" if eo>=0.8 else "#ef4444")
    k1,k2,k3 = st.columns(3)
    k1.markdown(f'<div class="kcard"><div class="lbl">Model Accuracy</div>'
                f'<div class="val" style="color:#06b6d4">{acc:.0%}</div>'
                f'<div class="sub">How often AI was correct</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kcard"><div class="lbl">Demographic Parity</div>'
                f'<div class="val" style="color:{dp_col}">{dp:.2f}</div>'
                f'<div class="sub">1.0=fair · below 0.8=biased</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kcard"><div class="lbl">Equalized Odds</div>'
                f'<div class="val" style="color:{eo_col}">{eo:.2f}</div>'
                f'<div class="sub">1.0=fair · below 0.8=biased</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Explanations
    st.markdown("### 📖 What Do These Scores Mean?")
    e1,e2 = st.columns(2)
    e1.markdown(f'<div class="note"><b>📊 Demographic Parity = {dp:.2f}</b><br><br>'
                f'Are all groups selected at the same rate?<br>'
                f'Example: if 60% of Males are hired, ~60% of Females should be too.<br><br>'
                f'• 1.0 = equal ✅<br>• 0.8–1.0 = acceptable ⚠️<br>• Below 0.8 = biased 🚨</div>',
                unsafe_allow_html=True)
    e2.markdown(f'<div class="note"><b>⚖️ Equalized Odds = {eo:.2f}</b><br><br>'
                f'Does the AI make the same types of mistakes for all groups?<br>'
                f'Example: if it wrongly rejects 10% of qualified Males, it should also only reject ~10% of qualified Females.<br><br>'
                f'• 1.0 = equal mistakes ✅<br>• 0.8–1.0 = minor ⚠️<br>• Below 0.8 = unfair 🚨</div>',
                unsafe_allow_html=True)

    st.divider()

    # Chart: Selection Rates
    st.markdown("### 📈 Selection Rate by Group")
    st.markdown('<div class="note">Shows what % of each group the AI said YES to. '
                'All bars should be roughly the same height for a fair AI.</div>', unsafe_allow_html=True)
    fig,ax = plt.subplots(figsize=(8,4))
    fig.patch.set_facecolor("#1e2433"); ax.set_facecolor("#1e2433")
    grps = gdf[sen].astype(str).tolist()
    rates = gdf["Selection Rate"].tolist()
    clrs = ["#7c3aed","#06b6d4","#f59e0b","#22c55e","#ef4444","#ec4899"]
    bars = ax.bar(grps,rates,color=clrs[:len(grps)],edgecolor="#0e1117",width=0.5)
    ax.axhline(max(rates)*0.8,color="#ef4444",linestyle="--",linewidth=1.5,
               label=f"80% Rule ({max(rates)*0.8:.0%}) — below = biased")
    for b,r in zip(bars,rates):
        ax.text(b.get_x()+b.get_width()/2,r+0.01,f"{r:.0%}",
                ha="center",va="bottom",color="white",fontsize=12,fontweight="bold")
    ax.set_ylim(0,1); ax.set_ylabel("% Selected",color="#94a3b8")
    ax.set_title(f"Selection Rate per {sen} Group",color="#e2e8f0",fontsize=13)
    ax.legend(facecolor="#1e2433",labelcolor="#94a3b8",fontsize=9)
    ax.tick_params(colors="#94a3b8")
    for sp in ax.spines.values(): sp.set_edgecolor("#2d3748")
    st.pyplot(fig)

    # Table
    st.markdown("### 📋 Detailed Numbers by Group")
    st.markdown('<div class="note"><b>Selection Rate</b> = % AI said YES · '
                '<b>FPR</b> = wrongly said YES · <b>FNR</b> = wrongly said NO</div>',
                unsafe_allow_html=True)
    tbl = gdf.copy(); tbl.columns=[sen,"Selection Rate","FPR","FNR"]
    for c in ["Selection Rate","FPR","FNR"]:
        tbl[c]=tbl[c].apply(lambda x:f"{x:.1%}")
    st.dataframe(tbl, hide_index=True)

    st.divider()

    # Feature importance
    st.markdown("### 🧠 Why Does This Bias Exist?")
    w1,w2 = st.columns(2)
    w1.markdown('<div class="box"><h3>📜 Biased Training Data</h3>'
                '<p>The AI learned from old historical decisions that were already unfair. '
                'It copies those patterns without knowing right from wrong.</p></div>',
                unsafe_allow_html=True)
    w2.markdown('<div class="box"><h3>🔗 Proxy Variables</h3>'
                '<p>Other columns secretly carry group information. '
                'E.g. "job title" correlates with gender — the AI learns bias indirectly.</p></div>',
                unsafe_allow_html=True)

    if fi_vals is not None:
        fi = pd.DataFrame({"Feature":feats,"Influence":fi_vals}).sort_values("Influence")
        fig2,ax2 = plt.subplots(figsize=(8,max(3,len(fi)*0.45)))
        fig2.patch.set_facecolor("#1e2433"); ax2.set_facecolor("#1e2433")
        ax2.barh(fi["Feature"],fi["Influence"],color="#7c3aed",edgecolor="#0e1117")
        ax2.set_xlabel("Influence on AI decisions",color="#94a3b8")
        ax2.set_title("Feature Influence (bigger = more impact)",color="#e2e8f0")
        ax2.tick_params(colors="#94a3b8")
        for sp in ax2.spines.values(): sp.set_edgecolor("#2d3748")
        st.pyplot(fig2)

    st.divider()

    # Fixes
    st.markdown("### 💡 How to Fix the Bias")
    f1,f2,f3 = st.columns(3)
    f1.markdown('<div class="box"><h3>🗃️ Fix 1: Clean Your Data</h3>'
                '<p>Balance your dataset — ensure equal examples per group. '
                'Use oversampling (SMOTE) for minority groups or undersampling for majority groups.</p></div>',
                unsafe_allow_html=True)
    f2.markdown('<div class="box"><h3>⚙️ Fix 2: Fairer Algorithm</h3>'
                '<p>Use Fairlearn\'s <code>ExponentiatedGradient</code> — trains with a fairness constraint built in. '
                'The AI learns to be accurate AND fair simultaneously.</p></div>',
                unsafe_allow_html=True)
    f3.markdown('<div class="box"><h3>🎚️ Fix 3: Threshold Tuning</h3>'
                '<p>Use different decision cutoffs per group to equalize selection rates. '
                'Fairlearn\'s <code>ThresholdOptimizer</code> does this automatically.</p></div>',
                unsafe_allow_html=True)

    # Final verdict
    st.divider()
    st.markdown("### ⚖️ Final Verdict — Should You Use This AI?")
    if dp < 0.8:
        st.markdown(f'<div class="bad">🚨 <b>DO NOT deploy yet.</b><br><br>'
                    f'Fairness Score = <b>{dp:.2f}</b> — below the 0.80 legal threshold. '
                    f'The AI significantly discriminates against certain <b>{sen}</b> groups. '
                    f'Apply the fixes above before going live.</div>', unsafe_allow_html=True)
    elif dp < 0.9:
        st.markdown(f'<div class="warn">⚠️ <b>Use with caution.</b><br><br>'
                    f'Fairness Score = <b>{dp:.2f}</b> — legal but showing disparity. '
                    f'Clean your data and recheck.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="good">✅ <b>Looks fair!</b><br><br>'
                    f'Fairness Score = <b>{dp:.2f}</b> — groups are treated equally. '
                    f'Monitor regularly after deployment.</div>', unsafe_allow_html=True)

    # Downloads
    st.markdown("<br>", unsafe_allow_html=True)
    d1,d2 = st.columns(2)
    d1.download_button("⬇️ Download Fairness Report", gdf.to_csv(index=False),
                       "fairness_report.csv","text/csv")
    if use_demo:
        d2.download_button("⬇️ Download Demo Dataset", make_demo().to_csv(index=False),
                           "demo_data.csv","text/csv")

    # Gemini AI section
    st.divider()
    st.markdown("### 🤖 Ask Gemini AI to Explain Your Results")
    if not GEMINI_OK:
        st.warning("Run: `pip install google-generativeai` to enable this feature.")
    elif not gemini_key:
        st.markdown('<div class="tip">🔑 Enter your <b>Gemini API Key</b> in the sidebar to unlock '
                    'AI-written explanations. Free key at '
                    '<a href="https://aistudio.google.com" target="_blank">aistudio.google.com</a></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="note">Click below — Gemini will read your results and write a '
                    '<b>plain-English report</b>: what the bias means, why it exists, '
                    'and exactly how to fix it.</div>', unsafe_allow_html=True)
        if st.button("✨ Generate AI Explanation with Gemini"):
            verdict_str = ("SIGNIFICANT BIAS" if dp<0.8 else ("MILD BIAS" if dp<0.9 else "FAIR"))
            prompt = f"""You are an AI fairness expert explaining results to a non-technical audience.

Fairness Analysis Results:
- Target variable (what AI decides): {tgt}
- Sensitive attribute (group audited): {sen}
- Model accuracy: {acc:.1%}
- Demographic Parity Ratio: {dp:.3f} → {verdict_str}
- Equalized Odds Ratio: {eo:.3f}
- Legal threshold: 0.80

Group breakdown:
{gdf.to_string(index=False)}

Write a clear, friendly report with 4 sections:
1. **What We Found** — 2-3 simple sentences summarizing the bias.
2. **Why This Is a Problem** — real-world harm this could cause.
3. **Why This Bias Probably Exists** — likely root cause in plain English.
4. **What To Do Next** — 3 specific actionable steps.

Use simple language. No jargon. Sound like a trusted expert advisor."""

            with st.spinner("🤖 Gemini is finding available models and writing your report…"):
                genai.configure(api_key=gemini_key)
                txt = None; last_err = None; used_model = None

                # Auto-discover models available to this API key
                try:
                    all_models = [
                        m.name for m in genai.list_models()
                        if "generateContent" in m.supported_generation_methods
                        and "flash" in m.name.lower()  # prefer fast/cheap flash models
                    ]
                    # Also add pro as fallback
                    pro_models = [
                        m.name for m in genai.list_models()
                        if "generateContent" in m.supported_generation_methods
                        and "pro" in m.name.lower()
                    ]
                    MODELS = all_models + pro_models
                except Exception:
                    # If list_models fails, try common names
                    MODELS = ["gemini-2.0-flash","gemini-1.5-flash",
                              "gemini-1.5-pro","gemini-pro"]

                for model_name in MODELS:
                    try:
                        resp = genai.GenerativeModel(model_name).generate_content(prompt)
                        txt  = resp.text
                        used_model = model_name
                        break
                    except Exception as e:
                        last_err = e
                        err_str = str(e)
                        if "429" in err_str or "quota" in err_str.lower() or "404" in err_str:
                            continue  # try next model
                        else:
                            break

                if txt:
                    st.caption(f"✅ Generated using: `{used_model}`")
                    st.markdown('<div style="background:#1a1f2e;border:1px solid #7c3aed;'
                                'border-radius:12px;padding:1.5rem 2rem;margin-top:1rem;">'
                                '<div style="color:#a78bfa;font-size:.8rem;text-transform:uppercase;'
                                'letter-spacing:.1em;margin-bottom:.8rem;">🤖 Gemini AI Report</div>',
                                unsafe_allow_html=True)
                    st.markdown(txt)
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.download_button("⬇️ Download AI Report", txt,
                                       "gemini_report.txt", "text/plain")
                elif last_err and ("429" in str(last_err) or "quota" in str(last_err).lower()):
                    st.markdown('<div class="warn">⚠️ <b>Gemini Free Quota Exhausted</b><br><br>'
                                'All available models hit their daily free limit.<br><br>'
                                '<b>Fix options:</b><br>'
                                '✅ <b>Wait 24 hours</b> for the quota to reset automatically<br>'
                                '✅ <b>New API key:</b> create a new project at '
                                '<a href="https://aistudio.google.com" target="_blank">aistudio.google.com</a>'
                                ' (each new project gets a fresh quota)<br>'
                                '✅ <b>Enable billing</b> at '
                                '<a href="https://console.cloud.google.com" target="_blank">console.cloud.google.com</a>'
                                ' (very cheap — $0.075 per 1M tokens)</div>',
                                unsafe_allow_html=True)
                else:
                    st.error(f"❌ Gemini Error: {last_err}\n\nCheck your API key is correct at aistudio.google.com")

else:
    st.markdown("""
    <div style="text-align:center;padding:3rem;">
        <div style="font-size:3.5rem">⚖️</div>
        <h3 style="color:#94a3b8;">Complete Steps 1 & 2 above, then click the button!</h3>
        <p style="color:#64748b;">The tool will train an AI and check if it treats everyone fairly.</p>
    </div>""", unsafe_allow_html=True)
