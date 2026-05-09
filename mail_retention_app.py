import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from google import genai as google_genai
from datetime import datetime
import re

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mailkeeper · Mail Retention App",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0f1117; color: #e8eaf0; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #1e2535;
}

/* Metric cards */
.metric-card {
    background: #161b27;
    border: 1px solid #1e2535;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: #3b82f6;
}
.metric-card.green::before { background: #22c55e; }
.metric-card.red::before   { background: #ef4444; }
.metric-card.yellow::before { background: #64748b; }
.metric-label { font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; font-family: 'DM Mono', monospace; }
.metric-value { font-size: 28px; font-weight: 600; color: #e8eaf0; }
.metric-delta { font-size: 12px; margin-top: 4px; font-family: 'DM Mono', monospace; }
.delta-pos { color: #22c55e; }
.delta-neg { color: #ef4444; }

/* Alert banners */
.alert-critical { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); border-radius: 8px; padding: 12px 16px; margin: 8px 0; }
.alert-good     { background: rgba(34,197,94,0.1);  border: 1px solid rgba(34,197,94,0.3);  border-radius: 8px; padding: 12px 16px; margin: 8px 0; }

/* AI summary box */
.ai-box {
    background: #0f1117;
    border: 1px solid #8b5cf6;
    border-radius: 12px;
    padding: 0;
    margin-top: 16px;
    overflow: hidden;
}
.ai-box-header {
    background: #8b5cf6;
    padding: 10px 20px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: #ffffff;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-family: 'DM Mono', monospace;
    font-weight: 600;
}
.ai-box-body {
    padding: 20px 24px;
    font-size: 14px;
    line-height: 1.8;
    color: #cbd5e1;
}

/* Significance badges */
.sig-badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; font-family: 'DM Mono', monospace; }
.sig-yes  { background: rgba(34,197,94,0.15);  color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
.sig-no   { background: rgba(107,114,128,0.15); color: #9ca3af; border: 1px solid rgba(107,114,128,0.3); }
.sig-warn { background: rgba(100,116,139,0.15); color: #94a3b8; border: 1px solid rgba(100,116,139,0.3); }

/* Tab styling */
.stTabs [data-baseweb="tab"] { font-family: 'DM Mono', monospace; font-size: 13px; }

/* Multiselect tags — override Streamlit default red/orange */
[data-baseweb="tag"] {
    background-color: #6d28d9 !important;
    border-color: #7c3aed !important;
}
[data-baseweb="tag"] span { color: #ede9fe !important; }
[data-baseweb="tag"] svg  { fill: #ede9fe !important; }

/* Active tab underline color */
.stTabs [aria-selected="true"] { color: #8b5cf6 !important; border-bottom-color: #8b5cf6 !important; }

/* Headers */
h1, h2, h3 { font-family: 'DM Sans', sans-serif; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Load data ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Tesk_Task___Mail_Retention.csv.gz", sep=";", compression="gzip")
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")
    df["is_read"]    = df["read_ts"].notna().astype(int)
    df["is_clicked"] = df["click_ts"].notna().astype(int)
    df["is_credit"]  = df["total_credits"].notna().astype(int)
    return df

df = load_data()

PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#9ca3af", family="DM Mono"),
    xaxis=dict(gridcolor="#1e2535", linecolor="#1e2535"),
    yaxis=dict(gridcolor="#1e2535", linecolor="#1e2535"),
    margin=dict(l=10, r=10, t=40, b=10),
)
COLORS = {"Buyer": "#3b82f6", "Not Buyer": "#8b5cf6"}

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✉️ Mailkeeper")
    st.markdown("<p style='color:#6b7280;font-size:12px;font-family:DM Mono'>Mail Retention Analytics</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("**Date range**")
    min_date, max_date = df["date"].min().date(), df["date"].max().date()
    date_range = st.date_input("", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    st.markdown("**Segment**")
    buyer_filter = st.multiselect("Buyer status", ["Buyer", "Not Buyer"], default=["Buyer", "Not Buyer"])

    st.markdown("**Rule**")
    rules = st.multiselect("Rule", sorted(df["rule"].unique()), default=sorted(df["rule"].unique()))

    st.divider()
    st.markdown("<p style='color:#6b7280;font-size:11px;font-family:DM Mono'>Google AI Studio API Key</p>", unsafe_allow_html=True)
    api_key = st.text_input("", type="password", placeholder="AIza...")

# ── Filter data ──────────────────────────────────────────────────────────────────
if len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start, end = pd.Timestamp(min_date), pd.Timestamp(max_date)

mask = (
    (df["date"] >= start) &
    (df["date"] <= end) &
    (df["buyer"].isin(buyer_filter)) &
    (df["rule"].isin(rules))
)
fdf = df[mask].copy()

# ── AI helper ───────────────────────────────────────────────────────────────────
def call_ai(prompt: str) -> str:
    if not api_key:
        return None   # явно None коли ключа нема
    try:
        client = google_genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"AI error: {e}"

def md_to_html(text: str) -> str:
    # bold **text**
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # italic *text*
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    # нумерований список
    lines = text.split('\n')
    html_lines = []
    for line in lines:
        m = re.match(r'^(\d+)\.\s+(.*)', line)
        if m:
            html_lines.append(f'<li>{m.group(2)}</li>')
        elif line.strip() == '':
            html_lines.append('<br>')
        else:
            html_lines.append(line)
    text = '\n'.join(html_lines)
    # wrap li в ol
    text = re.sub(r'(<li>.*?</li>\n?)+', lambda m: f'<ol>{m.group(0)}</ol>', text, flags=re.DOTALL)
    return text

def render_ai_box(result: str | None, title: str) -> None:
    if result is None:
        st.warning("Enter your Google AI Studio API key in the sidebar to generate AI summaries.")
        return
    html = md_to_html(result)
    st.markdown(f"""
    <div class="ai-box">
        <div class="ai-box-header">✦ &nbsp;{title}</div>
        <div class="ai-box-body">{html}</div>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["Monitoring", "A/B Analysis"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MONITORING
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Monitoring")
    st.markdown(f"<p style='color:#6b7280;font-size:12px;font-family:DM Mono'>{start.date()} → {end.date()} · {len(fdf):,} emails</p>", unsafe_allow_html=True)

    # ── KPI cards ───────────────────────────────────────────────────────────────
    total     = len(fdf)
    open_rate = fdf["is_read"].mean() * 100
    ctr       = fdf["is_clicked"].mean() * 100
    credit_cr = fdf["is_credit"].mean() * 100
    ctor      = (fdf["is_clicked"].sum() / fdf["is_read"].sum() * 100) if fdf["is_read"].sum() > 0 else 0

    # Compare to full dataset baseline
    base_open = df["is_read"].mean() * 100
    base_ctr  = df["is_clicked"].mean() * 100
    base_cr   = df["is_credit"].mean() * 100

    def delta_html(val, base):
        d = val - base
        cls = "delta-pos" if d >= 0 else "delta-neg"
        sign = "+" if d >= 0 else ""
        return f'<span class="{cls}">{sign}{d:.2f}pp vs baseline</span>'

    def card_color(val, base):
        d = val - base
        if d > 1:   return "green"
        if d < -1:  return "red"
        return "yellow"

    col1, col2, col3, col4, col5 = st.columns(5)
    cards = [
        (col1, "Emails Sent", f"{total:,}", "", ""),
        (col2, "Open Rate",  f"{open_rate:.2f}%", card_color(open_rate, base_open), delta_html(open_rate, base_open)),
        (col3, "CTR",        f"{ctr:.2f}%",       card_color(ctr, base_ctr),        delta_html(ctr, base_ctr)),
        (col4, "CTOR",       f"{ctor:.2f}%",      "", ""),
        (col5, "Credit Conv", f"{credit_cr:.2f}%", card_color(credit_cr, base_cr),  delta_html(credit_cr, base_cr)),
    ]
    for col, label, value, color, delta in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card {color}">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-delta">{delta}</div>
            </div>""", unsafe_allow_html=True)

    # ── Alerts ───────────────────────────────────────────────────────────────────
    alerts = []
    if ctr < base_ctr - 1.5:
        alerts.append(("critical", f"🚨 CTR critically low: {ctr:.2f}% (baseline {base_ctr:.2f}%)"))
    if open_rate < base_open - 3:
        alerts.append(("critical", f"🚨 Open Rate critically low: {open_rate:.2f}% (baseline {base_open:.2f}%)"))
    if ctr > base_ctr + 1.5:
        alerts.append(("good", f"✅ CTR above baseline: {ctr:.2f}% (+{ctr-base_ctr:.2f}pp)"))
    if open_rate > base_open + 3:
        alerts.append(("good", f"✅ Open Rate above baseline: {open_rate:.2f}% (+{open_rate-base_open:.2f}pp)"))

    if alerts:
        st.markdown("**Alerts**")
        for atype, msg in alerts:
            css = "alert-critical" if atype == "critical" else "alert-good"
            st.markdown(f'<div class="{css}">{msg}</div>', unsafe_allow_html=True)

    st.divider()

    # ── Trend chart ──────────────────────────────────────────────────────────────
    st.markdown("### Trend")
    metric_opt = st.selectbox("Metric", ["CTR", "Open Rate", "Credit Conversion"], key="trend_metric")
    metric_col = {"CTR": "is_clicked", "Open Rate": "is_read", "Credit Conversion": "is_credit"}[metric_opt]

    daily = (
        fdf.groupby(["date", "buyer"])[metric_col]
        .mean().reset_index()
        .rename(columns={metric_col: "value"})
    )
    daily["value"] *= 100

    fig_trend = go.Figure()
    for buyer, color in COLORS.items():
        sub = daily[daily["buyer"] == buyer]
        if sub.empty:
            continue
        # Trend line (rolling avg)
        fig_trend.add_trace(go.Scatter(
            x=sub["date"], y=sub["value"].rolling(2, min_periods=1).mean(),
            mode="lines", line=dict(color=color, width=1.5, dash="dot"),
            showlegend=False, hoverinfo="skip"
        ))
        fig_trend.add_trace(go.Scatter(
            x=sub["date"], y=sub["value"],
            mode="lines+markers", name=buyer,
            line=dict(color=color, width=2.5),
            marker=dict(size=7, color=color),
            hovertemplate=f"<b>{buyer}</b><br>%{{x|%d %b}}<br>{metric_opt}: %{{y:.2f}}%<extra></extra>"
        ))

    # Critical threshold line
    baseline_val = df[metric_col].mean() * 100
    fig_trend.add_hline(y=baseline_val, line_dash="dash", line_color="#4b5563",
                        annotation_text=f"Baseline {baseline_val:.2f}%",
                        annotation_font_color="#6b7280")

    fig_trend.update_layout(title=f"{metric_opt} over time", **PLOT_THEME,
                            legend=dict(bgcolor="rgba(0,0,0,0)"), height=350)
    st.plotly_chart(fig_trend, use_container_width=True)

    # ── Breakdown charts ─────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### By Rule")
        rule_stats = fdf.groupby("rule").agg(
            open_rate=("is_read", "mean"),
            ctr=("is_clicked", "mean"),
            n=("user_id", "count")
        ).reset_index().sort_values("ctr", ascending=False)
        rule_stats[["open_rate", "ctr"]] *= 100

        fig_rule = go.Figure()
        fig_rule.add_trace(go.Bar(name="Open Rate", x=rule_stats["rule"], y=rule_stats["open_rate"],
                                  marker_color="#3b82f6", opacity=0.85))
        fig_rule.add_trace(go.Bar(name="CTR", x=rule_stats["rule"], y=rule_stats["ctr"],
                                  marker_color="#8b5cf6", opacity=0.85))
        fig_rule.update_layout(barmode="group", title="Open Rate & CTR by Rule",
                               **PLOT_THEME, height=300, legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_rule, use_container_width=True)

    with col_b:
        st.markdown("### By Message Type")
        resp_stats = fdf.groupby("response").agg(
            open_rate=("is_read", "mean"),
            ctr=("is_clicked", "mean"),
        ).reset_index()
        resp_stats[["open_rate", "ctr"]] *= 100
        resp_stats["response"] = resp_stats["response"].str.replace("_", " ")

        fig_resp = go.Figure()
        fig_resp.add_trace(go.Bar(name="Open Rate", x=resp_stats["response"], y=resp_stats["open_rate"],
                                  marker_color="#3b82f6", opacity=0.85))
        fig_resp.add_trace(go.Bar(name="CTR", x=resp_stats["response"], y=resp_stats["ctr"],
                                  marker_color="#22c55e", opacity=0.85))
        fig_resp.update_layout(barmode="group", title="Open Rate & CTR by Message Type",
                               **PLOT_THEME, height=300, legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_resp, use_container_width=True)

    # ── CTR by Rule x Buyer ──────────────────────────────────────────────────────
    st.markdown("### CTR by Rule & Buyer")
    rule_buyer = fdf.groupby(["rule", "buyer"])["is_clicked"].mean().reset_index()
    rule_buyer["is_clicked"] *= 100
    rule_order = fdf.groupby("rule")["is_clicked"].mean().sort_values(ascending=False).index.tolist()
    fig_rb = go.Figure()
    for buyer, color in COLORS.items():
        sub = rule_buyer[rule_buyer["buyer"] == buyer]
        sub = sub.set_index("rule").reindex(rule_order).reset_index()
        fig_rb.add_trace(go.Bar(
            name=buyer, x=sub["rule"], y=sub["is_clicked"],
            marker_color=color,
            text=sub["is_clicked"].round(2).astype(str) + "%",
            textposition="outside", textfont=dict(color="#e8eaf0", size=10),
        ))
    fig_rb.update_layout(
        barmode="group", title="CTR by Rule — Buyer vs Not Buyer",
        **{k: v for k, v in PLOT_THEME.items() if k != "yaxis"},
        height=320, legend=dict(bgcolor="rgba(0,0,0,0)"),
        yaxis=dict(ticksuffix="%", gridcolor="#1e2535", linecolor="#1e2535"),
    )
    st.plotly_chart(fig_rb, use_container_width=True)

    # ── AI Summary ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🤖 AI Summary")

    if st.button("Generate AI Summary", key="ai_mon"):
        summary_data = f"""
Period: {start.date()} to {end.date()}
Total emails: {total:,}
Open Rate: {open_rate:.2f}% (baseline: {base_open:.2f}%)
CTR: {ctr:.2f}% (baseline: {base_ctr:.2f}%)
CTOR: {ctor:.2f}%
Credit Conversion: {credit_cr:.2f}% (baseline: {base_cr:.2f}%)

By buyer:
{fdf.groupby('buyer').agg(open_rate=('is_read','mean'), ctr=('is_clicked','mean'), credit=('is_credit','mean')).mul(100).round(2).to_string()}

By rule (CTR):
{fdf.groupby('rule')['is_clicked'].mean().mul(100).round(2).sort_values(ascending=False).to_string()}

By message type (CTR):
{fdf.groupby('response')['is_clicked'].mean().mul(100).round(2).sort_values(ascending=False).to_string()}
"""
        prompt = f"""You are a senior product analyst at a premium dating app. 
Analyze this email marketing data and write a concise executive summary (5-7 sentences).
Focus on: key performance vs baseline, buyer vs non-buyer differences, which segments perform best, and 2-3 actionable recommendations.
Be specific with numbers. Write in English.

Data:
{summary_data}"""

        with st.spinner("Generating summary..."):
            summary = call_ai(prompt)
        render_ai_box(summary, "Summary generated")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — AB ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## A/B Test Analysis")

    col_sel1, col_sel2, col_sel3 = st.columns(3)
    with col_sel1:
        selected_group = st.selectbox("Select test group", ["group_1", "group_2", "group_3", "group_4"])
    with col_sel2:
        selected_metric = st.selectbox("Metric", ["CTR", "Open Rate", "Credit Conversion"])
    with col_sel3:
        segment = st.selectbox("User segment", ["All", "Buyer", "Not Buyer"])

    metric_col_ab = {"CTR": "is_clicked", "Open Rate": "is_read", "Credit Conversion": "is_credit"}[selected_metric]

    ab_df = fdf.copy()
    if segment != "All":
        ab_df = ab_df[ab_df["buyer"] == segment]

    ab_df = ab_df[ab_df[selected_group].isin(["Test", "Control"])]

    test_data    = ab_df[ab_df[selected_group] == "Test"][metric_col_ab]
    control_data = ab_df[ab_df[selected_group] == "Control"][metric_col_ab]

    n_test    = len(test_data)
    n_control = len(control_data)
    rate_test    = test_data.mean() * 100
    rate_control = control_data.mean() * 100
    delta        = rate_test - rate_control
    rel_lift     = (delta / rate_control * 100) if rate_control > 0 else 0

    # Z-test for proportions
    p_test    = test_data.mean()
    p_control = control_data.mean()
    p_pool    = (test_data.sum() + control_data.sum()) / (n_test + n_control)
    se        = np.sqrt(p_pool * (1 - p_pool) * (1/n_test + 1/n_control))
    z_stat    = (p_test - p_control) / se if se > 0 else 0
    p_value   = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    is_sig    = p_value < 0.05

    # ── KPI cards ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    delta_color = "green" if delta > 0 else "red"

    with c1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Test n</div>
        <div class="metric-value">{n_test:,}</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Control n</div>
        <div class="metric-value">{n_control:,}</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card {delta_color}"><div class="metric-label">Test {selected_metric}</div>
        <div class="metric-value">{rate_test:.2f}%</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Control {selected_metric}</div>
        <div class="metric-value">{rate_control:.2f}%</div></div>""", unsafe_allow_html=True)
    with c5:
        sign = "+" if delta > 0 else ""
        st.markdown(f"""<div class="metric-card {delta_color}"><div class="metric-label">Δ Lift</div>
        <div class="metric-value">{sign}{delta:.2f}pp</div>
        <div class="metric-delta"><span class="{'delta-pos' if delta>0 else 'delta-neg'}">{sign}{rel_lift:.1f}% relative</span></div>
        </div>""", unsafe_allow_html=True)

    # ── Statistical significance ─────────────────────────────────────────────────
    st.divider()
    st.markdown("### Statistical Significance")

    sig_cols = st.columns(4)
    sig_data = [
        ("Z-statistic", f"{z_stat:.3f}"),
        ("P-value", f"{p_value:.4f}"),
        ("Significance (α=0.05)", "✅ Significant" if is_sig else "❌ Not significant"),
        ("Confidence level", f"{(1-p_value)*100:.1f}%" if p_value < 1 else "—"),
    ]
    for col, (label, value) in zip(sig_cols, sig_data):
        with col:
            color = ""
            if "Significant" in value:
                color = "green" if is_sig else "red"
            st.markdown(f"""<div class="metric-card {color}">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="font-size:20px">{value}</div>
            </div>""", unsafe_allow_html=True)

    # ── Charts ───────────────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=["Control", "Test"],
            y=[rate_control, rate_test],
            marker_color=["#4b5563", "#3b82f6" if delta >= 0 else "#ef4444"],
            text=[f"{rate_control:.2f}%", f"{rate_test:.2f}%"],
            textposition="outside",
            textfont=dict(color="#e8eaf0"),
            width=0.5,
        ))
        fig_bar.add_hline(y=rate_control, line_dash="dash", line_color="#6b7280")
        theme_no_yaxis = {k: v for k, v in PLOT_THEME.items() if k != "yaxis"}
        fig_bar.update_layout(title=f"{selected_metric} — Test vs Control<br><sup>{selected_group.upper()} · {segment}</sup>",
                              **theme_no_yaxis, height=350, showlegend=False,
                              yaxis=dict(ticksuffix="%", gridcolor="#1e2535", linecolor="#1e2535"))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        # By buyer breakdown
        buyer_ab = fdf[fdf[selected_group].isin(["Test", "Control"])].groupby(
            ["buyer", selected_group])[metric_col_ab].mean().reset_index()
        buyer_ab[metric_col_ab] *= 100

        fig_buyer = go.Figure()
        for grp, color in [("Control", "#4b5563"), ("Test", "#3b82f6")]:
            sub = buyer_ab[buyer_ab[selected_group] == grp]
            fig_buyer.add_trace(go.Bar(
                name=grp, x=sub["buyer"], y=sub[metric_col_ab],
                marker_color=color,
                text=sub[metric_col_ab].round(2).astype(str) + "%",
                textposition="outside", textfont=dict(color="#e8eaf0"),
            ))
        fig_buyer.update_layout(barmode="group", title=f"{selected_metric} by Buyer × Group",
                                **{k: v for k, v in PLOT_THEME.items() if k != "yaxis"},
                                height=350, legend=dict(bgcolor="rgba(0,0,0,0)"),
                                yaxis=dict(ticksuffix="%", gridcolor="#1e2535", linecolor="#1e2535"))
        st.plotly_chart(fig_buyer, use_container_width=True)

    # ── All groups summary table ─────────────────────────────────────────────────
    st.markdown("### All Groups Summary")
    rows = []
    for g in ["group_1", "group_2", "group_3", "group_4"]:
        gdf = fdf[fdf[g].isin(["Test", "Control"])]
        if segment != "All":
            gdf = gdf[gdf["buyer"] == segment]
        t = gdf[gdf[g] == "Test"][metric_col_ab]
        c = gdf[gdf[g] == "Control"][metric_col_ab]
        if len(t) == 0 or len(c) == 0:
            continue
        rt, rc = t.mean() * 100, c.mean() * 100
        d = rt - rc
        pp = (t.sum() + c.sum()) / (len(t) + len(c))
        se_g = np.sqrt(pp * (1 - pp) * (1/len(t) + 1/len(c)))
        z_g = (t.mean() - c.mean()) / se_g if se_g > 0 else 0
        pv = 2 * (1 - stats.norm.cdf(abs(z_g)))
        rows.append({
            "Group": g.upper(),
            f"Control {selected_metric}": f"{rc:.2f}%",
            f"Test {selected_metric}": f"{rt:.2f}%",
            "Δ (pp)": f"{'+' if d>0 else ''}{d:.2f}",
            "Z-stat": f"{z_g:.2f}",
            "P-value": f"{pv:.4f}",
            "Significant": "✅" if pv < 0.05 else "❌",
            "Winner": "Test" if d > 0 else "Control",
        })

    if rows:
        summary_df = pd.DataFrame(rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ── AI Recommendation ────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🤖 AI Recommendation")

    if st.button("Generate AI Recommendation", key="ai_ab"):
        ab_summary = f"""
Test: {selected_group.upper()}
Metric: {selected_metric}
Segment: {segment}

Test group: n={n_test:,}, {selected_metric}={rate_test:.2f}%
Control group: n={n_control:,}, {selected_metric}={rate_control:.2f}%
Delta: {delta:+.2f}pp ({rel_lift:+.1f}% relative lift)
Z-statistic: {z_stat:.3f}
P-value: {p_value:.4f}
Statistically significant: {is_sig}

All groups summary:
{pd.DataFrame(rows).to_string() if rows else 'N/A'}
"""
        prompt = f"""You are a senior product analyst at a premium dating app with email marketing expertise.
Based on the A/B test results below, provide:
1. Clear interpretation of the test result (2-3 sentences)
2. Whether to roll out, stop, or iterate (with reasoning)
3. Caveats or risks to consider
4. Next recommended test to run

Be specific, data-driven, and concise. Write in English.

Test data:
{ab_summary}"""

        with st.spinner("Generating recommendation..."):
            recommendation = call_ai(prompt)
        render_ai_box(recommendation, "Recommendation generated")
