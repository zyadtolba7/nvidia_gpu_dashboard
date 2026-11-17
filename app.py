from __future__ import annotations

import json
import os
import re
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "nvidia_gpu_manufacturing.xlsx"
TEXT_COLUMNS = [
    "unit_serial", "product_model", "gpu_architecture", "wafer_id", "fab_site",
    "foundry_partner", "substrate_supplier", "production_line", "shift",
    "qa_result", "defect_type",
]
NUMERIC_COLUMNS = [
    "vram_gb", "process_node_nm", "core_clock_mhz", "memory_clock_mhz",
    "power_draw_w", "test_temp_c", "stress_test_hours", "unit_cost_usd",
]
PALETTE = ["#7968F5", "#13AA9B", "#F39A42", "#E84E7A", "#477AF4", "#7FB854"]

st.set_page_config(
    page_title="GPU Manufacturing Story",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data(show_spinner="بنقرأ ملف التصنيع وبنجهّز البيانات...")
def load_and_clean(path: str, modified: float) -> tuple[pd.DataFrame, dict]:
    del modified
    raw = pd.read_excel(path, sheet_name="raw_gpu_manufacturing", dtype=str, keep_default_na=False)
    raw_rows = len(raw)
    raw_duplicates = int(raw.duplicated().sum())
    df = raw.copy()

    for col in TEXT_COLUMNS:
        if col not in df:
            continue
        missing_tokens = {"": pd.NA, "N/A": pd.NA, "n/a": pd.NA, "NA": pd.NA}
        # "None" is the legitimate no-defect label in this dataset.
        if col != "defect_type":
            missing_tokens["None"] = pd.NA
        values = (df[col].astype(str).str.replace("\u00a0", " ", regex=False).str.strip()
                  .replace(missing_tokens))
        # Normalize case-only variants while preserving the most common spelling.
        frequencies = values.dropna().value_counts()
        canonical = {}
        for value, count in frequencies.items():
            key = value.casefold()
            if key not in canonical or count > frequencies.get(canonical[key], 0):
                canonical[key] = value
        df[col] = values.map(lambda value: canonical.get(value.casefold(), value) if pd.notna(value) else pd.NA)

    categories = {
        "product_model": [
            "GeForce RTX 4060", "GeForce RTX 4070", "GeForce RTX 4080", "GeForce RTX 4090",
            "GeForce RTX 5060", "GeForce RTX 5070", "GeForce RTX 5080", "GeForce RTX 5090",
        ],
        "gpu_architecture": ["Ada Lovelace", "Blackwell"],
        "fab_site": ["Taipei", "Hsinchu", "Phoenix", "Seoul"],
        "foundry_partner": ["TSMC", "Samsung"],
        "substrate_supplier": ["Unimicron", "Ibiden", "AT&S", "Nan Ya PCB"],
        "production_line": ["GPU-A1", "GPU-A2", "GPU-B1", "GPU-B2", "GPU-C1"],
        "shift": ["Day", "Evening", "Night"],
        "qa_result": ["Pass", "Fail"],
        "defect_type": ["None", "Memory error", "Thermal failure", "Solder defect", "Power instability", "Fan connector", "Packaging damage"],
    }
    for col, valid_values in categories.items():
        if col in df:
            lookup = {value.casefold(): value for value in valid_values}
            df[col] = df[col].map(lambda value: lookup.get(value.casefold(), value) if pd.notna(value) else pd.NA)

    df["production_date"] = pd.to_datetime(df["production_date"], errors="coerce")
    missing_before_rules = 0
    for col in NUMERIC_COLUMNS:
        if col in df:
            cleaned = (df[col].astype(str).str.replace("\u00a0", "", regex=False).str.strip()
                       .str.replace(r"[$,]", "", regex=True))
            cleaned = cleaned.replace({"": pd.NA, "N/A": pd.NA, "n/a": pd.NA, "NA": pd.NA})
            df[col] = pd.to_numeric(cleaned, errors="coerce")
    missing_before_rules = int(df[NUMERIC_COLUMNS].isna().sum().sum())

    # Business checks: impossible placeholders are retained as missing, not treated as real measurements.
    allowed_vram = {8, 12, 16, 24, 32}
    df.loc[~df["vram_gb"].isin(allowed_vram), "vram_gb"] = pd.NA
    df.loc[~df["process_node_nm"].isin({4, 5}), "process_node_nm"] = pd.NA
    for col, low, high in [
        ("core_clock_mhz", 500, 5000), ("memory_clock_mhz", 1000, 50000),
        ("power_draw_w", 1, 1000), ("test_temp_c", 0, 120),
        ("stress_test_hours", 0, 24), ("unit_cost_usd", 1, 1000),
    ]:
        df.loc[(df[col] < low) | (df[col] > high), col] = pd.NA

    duplicates_removed = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    df["month"] = df["production_date"].dt.to_period("M").dt.to_timestamp()
    df["is_pass"] = df["qa_result"].eq("Pass")
    df["is_fail"] = df["qa_result"].eq("Fail")
    df["has_defect"] = df["is_fail"]

    quality = {
        "raw_rows": raw_rows,
        "clean_rows": len(df),
        "raw_duplicates": raw_duplicates,
        "duplicates_removed": duplicates_removed,
        "missing_numeric": int(df[NUMERIC_COLUMNS].isna().sum().sum()),
        "numeric_missing_added_by_rules": max(0, int(df[NUMERIC_COLUMNS].isna().sum().sum()) - missing_before_rules),
        "missing_dates": int(df["production_date"].isna().sum()),
    }
    return df, quality


def fmt_num(value, decimals=0):
    if pd.isna(value):
        return "—"
    return f"{value:,.{decimals}f}"


def pct(numerator, denominator):
    return float(numerator / denominator * 100) if denominator else 0.0


def build_local_insights(data: pd.DataFrame) -> list[str]:
    if data.empty:
        return ["مفيش سجلات مطابقة للفلاتر الحالية. وسّع نطاق الاختيار عشان نطلع ملاحظات مفيدة."]
    notes = []
    tested = int(data["qa_result"].notna().sum())
    fails = int(data["is_fail"].sum())
    pass_rate = pct(int(data["is_pass"].sum()), tested)
    notes.append(
        f"معدل اجتياز الجودة في الاختيار الحالي <bdi dir='ltr'><strong>{pass_rate:.1f}%</strong></bdi> "
        f"من <bdi dir='ltr'>{tested:,}</bdi> بطاقة لها نتيجة اختبار. راقب تغيّر المعدل حسب الخط والوردية قبل زيادة الإنتاج."
    )
    site = data.dropna(subset=["fab_site", "qa_result"]).groupby("fab_site").agg(
        passed=("is_pass", "sum"), tested=("qa_result", "size"), units=("unit_serial", "nunique")
    )
    site = site[site.tested >= 20]
    if len(site) > 1:
        site["pass_rate"] = site.passed / site.tested * 100
        best, worst = site.pass_rate.idxmax(), site.pass_rate.idxmin()
        if best != worst:
            notes.append(
                f"موقع <bdi dir='ltr'><strong>{escape(str(best))}</strong></bdi> حقق أعلى اجتياز "
                f"(<bdi dir='ltr'>{site.loc[best, 'pass_rate']:.1f}%</bdi>)، بينما <bdi dir='ltr'><strong>{escape(str(worst))}</strong></bdi> الأقل "
                f"(<bdi dir='ltr'>{site.loc[worst, 'pass_rate']:.1f}%</bdi>). قارن ظروف الاختبار والورديات بين الموقعين قبل تعميم أي تغيير."
            )
    defect = data.loc[data.is_fail & data.defect_type.notna(), "defect_type"].value_counts()
    if not defect.empty:
        notes.append(
            f"العيب الأكثر تكرارًا هو <bdi dir='ltr'><strong>{escape(str(defect.index[0]))}</strong></bdi> "
            f"بعدد <bdi dir='ltr'>{int(defect.iloc[0]):,}</bdi> حالة. افحص الخطوط والورديات المرتبطة به لتحديد تجربة تحسين محدودة."
        )
    hot = int((data.test_temp_c > 85).sum())
    if hot:
        notes.append(
            f"<bdi dir='ltr'>{hot:,}</bdi> سجل اختبار تجاوز <bdi dir='ltr'>85°C</bdi>. راجع طريقة التبريد ومعايرة الاختبار، "
            "ثم قارن نتيجة الجودة قبل وبعد الإجراء."
        )
    return notes[:4]


def answer_locally(data: pd.DataFrame, question: str) -> str:
    """Answer common dashboard questions from the current filtered aggregates."""
    q = question.casefold()
    group_column = None
    group_label = None
    for terms, column, label in [
        (("موقع", "مواقع", "fab", "site"), "fab_site", "الموقع"),
        (("خط", "خطوط", "line"), "production_line", "خط الإنتاج"),
        (("وردية", "ورديات", "shift"), "shift", "الوردية"),
        (("موديل", "موديلات", "بطاقة", "model"), "product_model", "الموديل"),
    ]:
        if any(term in q for term in terms):
            group_column, group_label = column, label
            break
    if group_column:
        grouped = data.dropna(subset=[group_column, "qa_result"]).groupby(group_column).agg(
            passed=("is_pass", "sum"), tested=("qa_result", "size"), fails=("is_fail", "sum")
        )
        grouped = grouped[grouped.tested >= 5]
        if not grouped.empty:
            grouped["pass_rate"] = grouped.passed / grouped.tested * 100
            worst, best = grouped.pass_rate.idxmin(), grouped.pass_rate.idxmax()
            return (
                f"الأولوية للمراجعة: {group_label} {worst}.\n"
                f"معدل اجتياز الجودة فيه {grouped.loc[worst, 'pass_rate']:.1f}% من "
                f"{int(grouped.loc[worst, 'tested']):,} اختبارًا.\n"
                f"للمقارنة، {group_label} الأعلى أداءً هو {best} بمعدل "
                f"{grouped.loc[best, 'pass_rate']:.1f}%.\n"
                f"التوصية: راجع ظروف الاختبار في {worst}، ثم قارن معدل الاجتياز قبل المراجعة وبعدها."
            )
        return f"البيانات الحالية لا تحتوي عينات كافية للمقارنة حسب {group_label}. وسّع الفلاتر أو الفترة."
    if any(term in q for term in ("عيب", "عيوب", "defect", "فشل", "fail")):
        counts = data.loc[data.is_fail, "defect_type"].dropna().value_counts()
        if counts.empty:
            return "لا تظهر حالات فشل مسجلة في الاختيار الحالي. تأكد من فلاتر نتيجة الجودة والفترة."
        return (
            f"العيب الأكثر تكرارًا: {counts.index[0]}.\n"
            f"عدد الحالات: {int(counts.iloc[0]):,}.\n"
            "التوصية: قارن ظهور العيب بين خطوط الإنتاج والورديات لتحديد أين تبدأ المراجعة."
        )
    if any(term in q for term in ("حرارة", "حراره", "درجة", "temperature", "thermal")):
        temp = data.dropna(subset=["test_temp_c", "qa_result"]).copy()
        temp["حرارة مرتفعة"] = temp.test_temp_c > 85
        groups = temp.groupby("حرارة مرتفعة").agg(passed=("is_pass", "sum"), tested=("qa_result", "size"))
        if len(groups) == 2:
            high = pct(int(groups.loc[True, "passed"]), int(groups.loc[True, "tested"]))
            low = pct(int(groups.loc[False, "passed"]), int(groups.loc[False, "tested"]))
            return (
                f"فوق 85°C: اجتاز {high:.1f}% من الاختبارات.\n"
                f"عند 85°C أو أقل: اجتاز {low:.1f}%.\n"
                "دي مقارنة وصفية في بيانات تدريبية، ولا تثبت أن الحرارة سببت الفشل."
            )
        return "لا توجد سجلات كافية للمقارنة بين الاختبارات الأعلى من 85°C وباقي الاختبارات."
    if any(term in q for term in ("تكلفة", "تكلفه", "cost", "سعر")):
        costs = data.dropna(subset=["product_model", "unit_cost_usd"]).groupby("product_model").agg(cost=("unit_cost_usd", "mean"), units=("unit_serial", "size"))
        if not costs.empty:
            highest = costs.cost.idxmax()
            lowest = costs.cost.idxmin()
            return (
                f"أعلى متوسط تكلفة: {costs.loc[highest, 'cost']:.2f} USD لموديل {highest}.\n"
                f"أقل متوسط تكلفة: {costs.loc[lowest, 'cost']:.2f} USD لموديل {lowest}.\n"
                "التوصية: قارن التكلفة مع الجودة وحجم العينة قبل تعديل خطة التوريد."
            )
    tested = int(data.qa_result.notna().sum())
    rate = pct(int(data.is_pass.sum()), tested)
    defects = data.loc[data.is_fail, "defect_type"].dropna().value_counts()
    defect_text = f" وأبرز عيب {defects.index[0]} ({int(defects.iloc[0]):,} حالة)" if not defects.empty else ""
    return (
        f"معدل اجتياز الجودة: {rate:.1f}% من {tested:,} بطاقة لها نتيجة.\n"
        f"حالات الفشل: {int(data.is_fail.sum()):,}.{defect_text}\n"
        "اسأل عن موقع أو خط إنتاج أو وردية أو موديل أو عيب لمقارنة محددة."
    )


def rtl_safe_html(text: str) -> str:
    """Format answer lines safely and isolate Latin labels and numeric tokens in RTL Arabic."""
    # Some model responses include HTML break tags as literal text. Convert them to
    # real line boundaries before escaping, then render styled line blocks below.
    text = re.sub(r"\\?\s*<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)
    token = re.compile(r"(?<![\w])(?:[A-Za-z][A-Za-z0-9]*(?:[-_/][A-Za-z0-9]+)*|\d+(?:[,.]\d+)*(?:%|°C|USD)?)", re.UNICODE)
    lines = []
    for line in text.splitlines():
        safe_line = escape(line.strip())
        safe_line = token.sub(lambda match: f"<bdi dir='ltr'>{match.group(0)}</bdi>", safe_line)
        if safe_line:
            lines.append(f"<span class='ai-answer-line'>{safe_line}</span>")
    return "".join(lines)


def ask_openai(data: pd.DataFrame, question: str) -> str | None:
    api_key = ""
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        pass
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None
    try:
        from openai import OpenAI

        known = int(data.qa_result.notna().sum())
        summary = {
            "units_in_scope": int(len(data)),
            "tested_units": known,
            "pass_rate_pct": round(pct(int(data.is_pass.sum()), known), 2),
            "fail_count": int(data.is_fail.sum()),
            "average_unit_cost_usd": round(float(data.unit_cost_usd.mean()), 2) if data.unit_cost_usd.notna().any() else None,
            "average_test_temp_c": round(float(data.test_temp_c.mean()), 2) if data.test_temp_c.notna().any() else None,
            "defects": data.loc[data.is_fail, "defect_type"].value_counts().head(5).to_dict(),
            "pass_rate_by_site": (data.dropna(subset=["fab_site", "qa_result"]).groupby("fab_site").is_pass.mean().mul(100).round(2).to_dict()),
            "pass_rate_by_model": (data.dropna(subset=["product_model", "qa_result"]).groupby("product_model").is_pass.mean().mul(100).round(2).to_dict()),
        }
        response = OpenAI(api_key=api_key).responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            instructions=(
                "أنت مساعد تحليل تصنيع وجودة. أجب بالعربية المصرية المبسطة وباختصار. "
                "رتب الرد في 2 إلى 4 أسطر قصيرة: النتيجة، الدليل الرقمي، ثم التوصية. ضع كل فكرة في سطر مستقل. "
                "اعزل أسماء الموديلات والخطوط والأرقام داخل الجملة ولا تخلط ترتيبها. لا تستخدم HTML أو علامات Markdown. "
                "اعتمد فقط على الملخص الرقمي المرسل، واذكر الدليل والافتراض بوضوح. "
                "لا تعتبر البيانات حقيقية؛ إنها بيانات توضيحية. اقترح تجربة صغيرة قابلة للقياس قبل أي تغيير إنتاجي."
            ),
            input=json.dumps({"question": question, "aggregated_summary": summary}, ensure_ascii=False),
        )
        return response.output_text
    except Exception as exc:
        return f"ملخص الذكاء الاصطناعي غير متاح الآن ({type(exc).__name__}). استخدم الملاحظات المحسوبة من البيانات أسفل الصفحة."


def chart_layout(fig: go.Figure, height: int = 330, dark: bool = False) -> go.Figure:
    fg = "#E6EAF6" if dark else "#252638"
    grid = "rgba(170,180,205,.14)" if dark else "rgba(45,51,72,.10)"
    fig.update_layout(
        height=height,
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Cairo, sans-serif", color=fg),
        margin=dict(l=18, r=18, t=42, b=16),
        hoverlabel=dict(font_size=13, font_family="Cairo"),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right", itemclick=False, itemdoubleclick=False),
        dragmode=False,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=grid, zeroline=False, automargin=True)
    return fig


if not DATA_PATH.exists():
    st.error(f"ملف بيانات التصنيع غير موجود: {DATA_PATH}")
    st.stop()

df, quality = load_and_clean(str(DATA_PATH), DATA_PATH.stat().st_mtime)
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "ai_open" not in st.session_state:
    st.session_state.ai_open = False
if "ai_answer" not in st.session_state:
    st.session_state.ai_answer = ""
dark = st.session_state.dark_mode

bg = "#090D1B" if dark else "#F4F4F8"
panel = "#151A2D" if dark else "#FFFFFF"
fg = "#EEF1FA" if dark else "#1B2033"
muted = "#AAB3CA" if dark else "#626B80"
border = "#2C3450" if dark else "#E5E5EF"
hero = "linear-gradient(135deg, #151A37 0%, #20234C 55%, #292353 100%)" if dark else "linear-gradient(135deg, #FFFFFF 0%, #F2F0FF 58%, #FFF8F5 100%)"
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"], [data-testid="stMarkdownContainer"] {{ font-family:'Cairo',sans-serif; }}
    .stApp {{ background:{bg}; color:{fg}; }}
    [data-testid="stAppViewContainer"] .main .block-container {{ max-width:1450px; padding:1rem 1.25rem 3rem; }}
    h1,h2,h3,p,label {{ color:{fg}; text-align:right; }}
    .topline {{ color:{muted}; font-size:.75rem; letter-spacing:.08em; font-weight:700; text-align:left; }}
    .hero {{ background:{hero}; border:1px solid {border}; border-radius:20px; padding:1.3rem 1.5rem; margin:.45rem 0 .8rem; }}
    .hero h1 {{ font-size:1.85rem; margin:.3rem 0; }}
    .hero p {{ color:{muted}; margin:.2rem 0; }}
    .eyebrow {{ color:#8C7BFF; font-weight:800; font-size:.77rem; letter-spacing:.08em; }}
    .pill {{ display:inline-block; border-radius:999px; padding:.25rem .65rem; margin:.12rem .15rem; border:1px solid {border}; background:{panel}; color:{fg}; font-size:.75rem; }}
    .story {{ border:1px solid {border}; border-radius:15px; padding:1rem 1.1rem; background:{panel}; margin:.45rem 0; }}
    .story p {{ margin:.25rem 0; color:{muted}; line-height:1.95; direction:rtl; text-align:right; unicode-bidi:plaintext; }}
    .ai-answer bdi {{ direction:ltr; unicode-bidi:isolate; }}
    .ai-answer-line {{ display:block; padding:.15rem .75rem .15rem 0; margin:.2rem 0; border-right:3px solid #13AA9B; }}
    .insight {{ border:1px solid {border}; border-right:4px solid #13AA9B; background:{panel}; border-radius:12px; padding:12px 15px; margin:9px 0; color:{fg}; direction:rtl; text-align:right; line-height:1.9; }}
    [data-testid="stMetric"] {{ background:{panel}; border:1px solid {border}; border-radius:15px; padding:14px 16px; direction:rtl; }}
    [data-testid="stMetricLabel"] {{ justify-content:flex-end; color:{muted}; font-size:.78rem; }}
    [data-testid="stMetricValue"] {{ color:{fg}; direction:ltr; unicode-bidi:isolate; text-align:right; }}
    [data-testid="stPlotlyChart"] {{ background:{panel}; border:1px solid {border}; border-radius:15px; padding:6px; }}
    [data-testid="stExpander"] {{ background:{panel}; border:1px solid {border}; border-radius:13px; }}
    div[data-testid="stDataFrame"] {{ direction:ltr; }}
    div[role="radiogroup"] {{ gap:.45rem; }}
    div[role="radiogroup"] label {{ border:1px solid {border}; border-radius:12px; padding:.45rem .75rem; background:{panel}; }}
    @media(max-width:800px) {{
      [data-testid="stAppViewContainer"] .main .block-container {{ padding:.7rem .65rem 2rem; }}
      [data-testid="stHorizontalBlock"] {{ flex-wrap:wrap; gap:.5rem; }}
      .hero {{ padding:1rem; border-radius:16px; }} .hero h1 {{ font-size:1.45rem; }}
      [data-testid="stMetric"] {{ padding:10px 12px; min-height:80px; }}
      [data-testid="stMetricValue"] {{ font-size:1.3rem; }}
      [data-testid="stPlotlyChart"] {{ padding:2px; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def keep_selected(data: pd.DataFrame, column: str, selection: list[str]) -> pd.DataFrame:
    if selection:
        return data[data[column].astype(str).isin(selection)]
    return data


with st.container():
    top_left, top_ai, top_right = st.columns([6, 2, 2])
    with top_left:
        st.markdown("<div class='topline'>BASIRAH · GPU MANUFACTURING STORY · V1</div>", unsafe_allow_html=True)
    with top_ai:
        if st.button("✦ اسأل AI", use_container_width=True, key="ai_toggle"):
            st.session_state.ai_open = not st.session_state.ai_open
    with top_right:
        light_label = "☀️ Light mode" if dark else "🌙 Dark mode"
        if st.button(light_label, use_container_width=True, key="mode_toggle"):
            st.session_state.dark_mode = not dark
            st.rerun()

st.markdown(
    "<div class='hero'><div class='eyebrow'>DECISION QUESTION</div>"
    "<h1>رحلة تصنيع كروت GPU · من خط الإنتاج لنتيجة الجودة</h1>"
    "<p>أين تظهر مشكلات الجودة؟ وما أول إجراء نجرّبه لتحسين نسبة الاجتياز؟</p>"
    "<span class='pill'>🟢 بيانات توضيحية</span><span class='pill'>🧪 ملف تدريبي</span>"
    "<span class='pill'>15,000 سجل خام</span><span class='pill'>لا تمثل بيانات فعلية من NVIDIA</span></div>",
    unsafe_allow_html=True,
)

with st.expander("🎛️ النطاق والفلاتر · كل الرسومات والمؤشرات تتحدث مع الاختيار", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        level = st.selectbox("المستوى", ["كل التصنيع", "موقع التصنيع", "خط الإنتاج", "الوردية", "الموديل"])
    level_map = {
        "كل التصنيع": None, "موقع التصنيع": "fab_site", "خط الإنتاج": "production_line",
        "الوردية": "shift", "الموديل": "product_model",
    }
    level_col = level_map[level]
    with c2:
        if level_col:
            level_options = sorted(df[level_col].dropna().astype(str).unique())
            level_selection = st.multiselect(f"اختار {level}", level_options, key=f"level_{level}")
        else:
            level_selection = []
            st.caption("النطاق الحالي: كل مواقع وخطوط الإنتاج")
    with c3:
        valid_dates = df.production_date.dropna()
        date_value = st.date_input(
            "فترة التصنيع",
            value=(valid_dates.min().date(), valid_dates.max().date()) if not valid_dates.empty else None,
            key="production_dates",
        )
    f1, f2, f3, f4 = st.columns(4)
    filter_defs = [
        ("gpu_architecture", "المعمارية"), ("foundry_partner", "المسبك"),
        ("production_line", "خط الإنتاج"), ("shift", "الوردية"),
        ("product_model", "موديل البطاقة"), ("fab_site", "الموقع"),
        ("qa_result", "نتيجة الجودة"),
    ]
    filter_cols = [f1, f2, f3, f4]
    selections = {}
    for index, (column, label) in enumerate(filter_defs):
        with filter_cols[index % len(filter_cols)]:
            opts = sorted(df[column].dropna().astype(str).unique())
            selections[column] = st.multiselect(label, opts, key=f"filter_{column}")

filtered = df.copy()
if level_col and level_selection:
    filtered = keep_selected(filtered, level_col, level_selection)
for column, selected in selections.items():
    filtered = keep_selected(filtered, column, selected)
if isinstance(date_value, tuple) and len(date_value) == 2:
    start, end = pd.Timestamp(date_value[0]), pd.Timestamp(date_value[1]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    filtered = filtered[filtered.production_date.between(start, end) | filtered.production_date.isna()]
st.caption(f"النطاق الحالي: {len(filtered):,} بطاقة بعد التنظيف · {len(df):,} سجل صالح/محتفَظ به")

if filtered.empty:
    st.warning("مفيش سجلات مطابقة للفلاتر الحالية. غيّر الاختيارات أو وسّع الفترة.")
    st.stop()

if st.session_state.ai_open:
    with st.container(border=True):
        ai_head, ai_close = st.columns([8, 1])
        with ai_head:
            st.markdown("### Basirah Assistant · تحليل الأرقام الحالية")
            st.caption("اسأل عن نتيجة الجودة أو الإنتاج أو التكلفة. الملخص يستخدم السجلات الظاهرة بعد الفلاتر.")
        with ai_close:
            if st.button("إغلاق", key="close_ai"):
                st.session_state.ai_open = False
                st.rerun()
        with st.form("ai_question_form_top"):
            question = st.text_input("سؤالك", placeholder="مثال: أنهي خط إنتاج أراجعه الأول؟")
            submitted = st.form_submit_button("إرسال السؤال", type="primary")
        if submitted and question.strip():
            ai_text = ask_openai(filtered, question.strip())
            if ai_text:
                st.session_state.ai_answer = ai_text
            else:
                st.session_state.ai_answer = "إجابة محسوبة من الأرقام الحالية بعد الفلاتر.\n" + answer_locally(filtered, question.strip())
        if st.session_state.ai_answer:
            st.markdown(f"<div class='story'><b>Basirah Assistant</b><p class='ai-answer'>{rtl_safe_html(st.session_state.ai_answer)}</p></div>", unsafe_allow_html=True)

tested = int(filtered.qa_result.notna().sum())
passed = int(filtered.is_pass.sum())
failed = int(filtered.is_fail.sum())
pass_rate = pct(passed, tested)
fail_rate = pct(failed, tested)
avg_cost = float(filtered.unit_cost_usd.mean()) if filtered.unit_cost_usd.notna().any() else float("nan")
avg_temp = float(filtered.test_temp_c.mean()) if filtered.test_temp_c.notna().any() else float("nan")

st.markdown("### حكاية البيانات في 5 مراحل")
st.markdown(
    "<div class='story'><b>من الملخص إلى القرار</b><p>ابدأ بالصورة العامة، ثم تتبّع النتيجة والإنتاج والعملية والمدخلات، "
    "واختم بإجراء عملي يمكن اختباره. كل مرحلة تعرض سجلات الفلاتر نفسها.</p></div>",
    unsafe_allow_html=True,
)

steps = ["✦ Brief", "1. Outcome", "2. Output", "3. Process", "4. Input", "A. Action"]
step = st.radio("رحلة التحليل", steps, horizontal=True, label_visibility="collapsed", key="story_step")
chart_cfg = {"displaylogo": False, "scrollZoom": False, "doubleClick": False, "displayModeBar": False, "responsive": True}


def kpi_row():
    cols = st.columns(5)
    cols[0].metric("بطاقات ضمن الاختيار", f"{len(filtered):,}")
    cols[1].metric("اجتازت الجودة", f"{pass_rate:.1f}%", help=f"{passed:,} من {tested:,} بطاقة لها نتيجة")
    cols[2].metric("حالات فشل", f"{failed:,}", help=f"معدل الفشل: {fail_rate:.1f}%")
    cols[3].metric("متوسط تكلفة الوحدة", f"${fmt_num(avg_cost, 2)}")
    cols[4].metric("متوسط حرارة الاختبار", f"{fmt_num(avg_temp, 1)}°C")


def pass_gauge(value: float, title: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number={"suffix": "%", "font": {"size": 38}},
        title={"text": title},
        gauge={"axis": {"range": [0, 100], "tickwidth": 1}, "bar": {"color": "#13AA9B"},
               "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
               "steps": [{"range": [0, 70], "color": "rgba(232,78,122,.14)"}, {"range": [70, 90], "color": "rgba(243,154,66,.14)"}, {"range": [90, 100], "color": "rgba(19,170,155,.15)"}]},
    ))
    st.plotly_chart(chart_layout(fig, 260, dark), use_container_width=True, config=chart_cfg)


def render_monthly():
    monthly = filtered.dropna(subset=["month"]).groupby("month").agg(
        units=("unit_serial", "size"), passed=("is_pass", "sum"), failed=("is_fail", "sum"), tested=("qa_result", "count")
    ).reset_index()
    monthly["اجتازت"] = monthly.passed
    monthly["فشلت"] = monthly.failed
    long = monthly.melt(id_vars="month", value_vars=["اجتازت", "فشلت"], var_name="نتيجة الجودة", value_name="عدد البطاقات")
    fig = px.bar(long, x="month", y="عدد البطاقات", color="نتيجة الجودة", barmode="stack", color_discrete_sequence=["#13AA9B", "#E84E7A"], hover_data={"month": "|%Y-%m"})
    fig.update_layout(xaxis_title="شهر الإنتاج", yaxis_title="عدد البطاقات", hovermode="x unified")
    st.plotly_chart(chart_layout(fig, 360, dark), use_container_width=True, config=chart_cfg)


if step == "✦ Brief":
    kpi_row()
    best_defect = filtered.loc[filtered.is_fail, "defect_type"].dropna().value_counts()
    defect_text = f"العيب الأكثر تكرارًا هو {best_defect.index[0]} ({best_defect.iloc[0]:,} حالة)." if not best_defect.empty else "لا توجد حالات فشل مسجلة في هذا الاختيار."
    st.markdown(
        f"<div class='story'><span class='pill'>✦ BRIEF</span><p><b>من كل 100 بطاقة لها نتيجة اختبار، {pass_rate:.1f} اجتازت الجودة.</b> "
        f"{defect_text} ابدأ بمقارنة الموقع والخط والوردية المرتبطة بالعيب قبل اعتماد أي تغيير في الإنتاج.</p></div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns([1, 1.7])
    with left:
        pass_gauge(pass_rate, "نسبة اجتياز الجودة")
    with right:
        st.markdown("#### اتجاه نتيجة الجودة")
        render_monthly()
    st.markdown("#### القصة في خمس خطوات")
    rows = [
        ("نتيجة الجودة", f"اجتازت {pass_rate:.1f}% من البطاقات التي ظهرت لها نتيجة اختبار، وفشلت {failed:,} بطاقة.", "#E84E7A"),
        ("حجم الإنتاج", f"يعرض التحليل {len(filtered):,} سجل إنتاج بعد تطبيق اختياراتك في الفلاتر.", "#F39A42"),
        ("مراحل التصنيع", "قارن خطوط الإنتاج والورديات وحرارة الاختبار لتعرف أين يتركز انخفاض الجودة.", "#13AA9B"),
        ("المواد والموردون", "قارن بيانات المسبك والمورد والعقدة التصنيعية لمعرفة ما إذا كان لها ارتباط بنتائج الجودة.", "#477AF4"),
        ("الخطوة التالية", "اختر سببًا محتملًا واحدًا، وجرّب تحسينه، ثم قارن نسبة اجتياز الجودة قبل التجربة وبعدها.", "#7968F5"),
    ]
    for name, summary, color in rows:
        st.markdown(f"<div class='story' style='border-right:4px solid {color};padding:.65rem 1rem'><b>{name}</b><p>{summary}</p></div>", unsafe_allow_html=True)
elif step == "1. Outcome":
    st.markdown("### النتيجة · هل اجتازت البطاقات اختبارات الجودة؟")
    kpi_row()
    left, right = st.columns(2)
    with left:
        pass_gauge(pass_rate, "نسبة الاجتياز")
    with right:
        status = filtered.qa_result.value_counts(dropna=False).rename_axis("نتيجة الجودة").reset_index(name="عدد البطاقات")
        status["نتيجة الجودة"] = status["نتيجة الجودة"].fillna("غير مسجل")
        fig = px.pie(status, names="نتيجة الجودة", values="عدد البطاقات", hole=.62, color_discrete_sequence=["#13AA9B", "#E84E7A", "#8C93A8"])
        fig.update_traces(textposition="inside", textinfo="percent+label", hovertemplate="%{label}<br>%{value:,} بطاقة<extra></extra>")
        st.plotly_chart(chart_layout(fig, 310, dark), use_container_width=True, config=chart_cfg)
    st.markdown("#### نتيجة الجودة عبر الزمن")
    render_monthly()
elif step == "2. Output":
    st.markdown("### الإنتاج · أين تتركز الكميات؟")
    kpi_row()
    left, right = st.columns(2)
    with left:
        model = filtered.groupby("product_model", as_index=False).agg(units=("unit_serial", "size"), passed=("is_pass", "sum"), failed=("is_fail", "sum"))
        fig = px.bar(model.sort_values("units"), x="units", y="product_model", orientation="h", color="units", color_continuous_scale="Purples", hover_data={"passed": True, "failed": True})
        fig.update_layout(xaxis_title="عدد البطاقات", yaxis_title="الموديل", coloraxis_showscale=False)
        st.plotly_chart(chart_layout(fig, 370, dark), use_container_width=True, config=chart_cfg)
    with right:
        site = filtered.groupby("fab_site", as_index=False).agg(units=("unit_serial", "size"), passed=("is_pass", "sum"), failed=("is_fail", "sum"))
        fig = px.bar(site.sort_values("units"), x="units", y="fab_site", orientation="h", color="failed", color_continuous_scale="OrRd", hover_data={"passed": True, "failed": True})
        fig.update_layout(xaxis_title="عدد البطاقات", yaxis_title="موقع التصنيع", coloraxis_colorbar_title="فشل الجودة")
        st.plotly_chart(chart_layout(fig, 370, dark), use_container_width=True, config=chart_cfg)
    st.markdown("#### الإنتاج والنتيجة عبر الزمن")
    render_monthly()
elif step == "3. Process":
    st.markdown("### العملية · أي خط أو وردية يستحق المراجعة؟")
    kpi_row()
    left, right = st.columns(2)
    with left:
        line = filtered.dropna(subset=["production_line", "qa_result"]).groupby("production_line").agg(passed=("is_pass", "sum"), tested=("qa_result", "size"), failed=("is_fail", "sum")).reset_index()
        line["اجتياز الجودة %"] = line.passed / line.tested * 100
        fig = px.bar(line.sort_values("اجتياز الجودة %"), x="اجتياز الجودة %", y="production_line", orientation="h", color="اجتياز الجودة %", color_continuous_scale="Tealgrn", hover_data={"tested": True, "failed": True})
        fig.update_layout(xaxis_title="نسبة الاجتياز", yaxis_title="خط الإنتاج", coloraxis_showscale=False, xaxis_ticksuffix="%")
        st.plotly_chart(chart_layout(fig, 360, dark), use_container_width=True, config=chart_cfg)
    with right:
        shift = filtered.dropna(subset=["shift", "qa_result"]).groupby("shift").agg(passed=("is_pass", "sum"), tested=("qa_result", "size"), failed=("is_fail", "sum")).reset_index()
        shift["اجتياز الجودة %"] = shift.passed / shift.tested * 100
        fig = px.bar(shift, x="shift", y="اجتياز الجودة %", color="shift", color_discrete_sequence=PALETTE, hover_data={"tested": True, "failed": True})
        fig.update_layout(xaxis_title="الوردية", yaxis_title="نسبة الاجتياز", yaxis_ticksuffix="%", showlegend=False)
        st.plotly_chart(chart_layout(fig, 360, dark), use_container_width=True, config=chart_cfg)
    st.markdown("#### حرارة الاختبار مقابل مدته")
    scatter = filtered.dropna(subset=["test_temp_c", "stress_test_hours"])
    fig = px.scatter(scatter, x="test_temp_c", y="stress_test_hours", color="qa_result", symbol="qa_result", hover_data=["unit_serial", "product_model", "production_line", "fab_site"], color_discrete_map={"Pass": "#13AA9B", "Fail": "#E84E7A"}, opacity=.68)
    fig.update_layout(xaxis_title="حرارة الاختبار °C", yaxis_title="مدة اختبار الضغط بالساعات")
    st.plotly_chart(chart_layout(fig, 390, dark), use_container_width=True, config=chart_cfg)
elif step == "4. Input":
    st.markdown("### المدخلات · التكلفة والمورد والمسبك")
    kpi_row()
    left, right = st.columns(2)
    with left:
        costs = filtered.dropna(subset=["product_model", "unit_cost_usd"]).groupby("product_model", as_index=False).agg(cost=("unit_cost_usd", "mean"), units=("unit_serial", "size"))
        fig = px.bar(costs.sort_values("cost"), x="cost", y="product_model", orientation="h", color="cost", color_continuous_scale="Purples", hover_data={"units": True})
        fig.update_layout(xaxis_title="متوسط تكلفة الوحدة · USD", yaxis_title="الموديل", coloraxis_showscale=False)
        st.plotly_chart(chart_layout(fig, 370, dark), use_container_width=True, config=chart_cfg)
    with right:
        supplier = filtered.dropna(subset=["substrate_supplier", "qa_result"]).groupby("substrate_supplier").agg(passed=("is_pass", "sum"), tested=("qa_result", "size"), cost=("unit_cost_usd", "mean")).reset_index()
        supplier["اجتياز الجودة %"] = supplier.passed / supplier.tested * 100
        fig = px.scatter(supplier, x="cost", y="اجتياز الجودة %", size="tested", color="substrate_supplier", hover_data={"tested": True}, size_max=40)
        fig.update_layout(xaxis_title="متوسط التكلفة · USD", yaxis_title="اجتياز الجودة %", yaxis_ticksuffix="%")
        st.plotly_chart(chart_layout(fig, 370, dark), use_container_width=True, config=chart_cfg)
    fab = filtered.dropna(subset=["foundry_partner"]).groupby("foundry_partner", as_index=False).agg(units=("unit_serial", "size"), avg_process_node=("process_node_nm", "mean"), avg_cost=("unit_cost_usd", "mean"))
    st.markdown("#### توزيع الكميات حسب المسبك")
    fig = px.bar(fab, x="foundry_partner", y="units", color="avg_cost", color_continuous_scale="Tealgrn", hover_data={"avg_process_node": ":.1f", "avg_cost": ":.2f"})
    fig.update_layout(xaxis_title="المسبك", yaxis_title="عدد البطاقات", coloraxis_colorbar_title="متوسط التكلفة USD")
    st.plotly_chart(chart_layout(fig, 320, dark), use_container_width=True, config=chart_cfg)
else:
    st.markdown("### Action · توصيات قابلة للاختبار")
    kpi_row()
    st.caption(f"الملاحظات دي محسوبة من {len(filtered):,} سجل مطابق للفلاتر. اعتبرها فرصًا للتحقق بتجربة إنتاج محدودة.")
    for note in build_local_insights(filtered):
        st.markdown(f"<div class='insight'>💡 {note}</div>", unsafe_allow_html=True)
    st.info("اسأل سؤالًا من زر «اسأل AI» أعلى الصفحة، أو راجع التوصيات المحسوبة هنا.")

    st.markdown("#### صحة البيانات بعد التنظيف")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("السجلات الخام", f"{quality['raw_rows']:,}")
    q2.metric("التكرارات المحذوفة", f"{quality['duplicates_removed']:,}")
    q3.metric("القيم الرقمية غير الصالحة/الناقصة", f"{quality['missing_numeric']:,}")
    q4.metric("سجلات بعد التنظيف", f"{quality['clean_rows']:,}")
    st.markdown("#### خطوات التنظيف المنفذة")
    st.markdown("""
    - إزالة المسافات الزائدة وتوحيد اختلافات الكتابة والحروف في الفئات.
    - تحويل القيم الرقمية إلى أرقام، وتنظيف رمز العملة والفواصل.
    - اعتبار القيم غير المنطقية في الذاكرة والعقدة والساعة والطاقة والحرارة والتكلفة كقيم ناقصة.
    - إزالة السجلات المتطابقة مع توضيح عددها.
    - الإبقاء على نتائج الاختبار الفعلية Pass/Fail واستخدامها لحساب مؤشرات الجودة.
    """)

with st.expander("معاينة البيانات بعد التنظيف وتنزيلها"):
    st.caption("سجلات للتدريب فقط. الحقول الظاهرة لا تمثل معلومات داخلية فعلية من NVIDIA.")
    st.dataframe(filtered.drop(columns=["month", "is_pass", "is_fail", "has_defect"], errors="ignore"), use_container_width=True, hide_index=True)
    csv = filtered.drop(columns=["month", "is_pass", "is_fail", "has_defect"], errors="ignore").to_csv(index=False).encode("utf-8-sig")
    st.download_button("تنزيل البيانات المنظّفة للفلاتر الحالية", csv, "clean_nvidia_gpu_manufacturing.csv", "text/csv")
