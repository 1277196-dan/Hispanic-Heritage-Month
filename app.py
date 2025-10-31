"""
Streamlit app: Polynomial regression analysis (degree >= 3) using real historical data
from Our World in Data for up to three high-income Latin-American countries.

Fixes included:
- Use numpy.poly1d (not scipy.poly1d) to avoid ImportError reported in deployment logs.
- Uses live OWID grapher CSVs; no placeholder data.

Place this file at repository root alongside the 'requirements' file.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import matplotlib.pyplot as plt
import base64
import json
import datetime

st.set_page_config(layout="wide", page_title="Polynomial Regression — Latin countries (OWID data)")

# ---------- Constants & mapping ----------
OWID_BASE = "https://ourworldindata.org/grapher"

# Indicator mapping: OWID grapher slugs and units/notes.
INDICATOR_MAP = {
    "Population": {"slug": "population", "unit": "people", "note": "Total population (annual)"},
    "Life expectancy": {"slug": "life_expectancy", "unit": "years", "note": "Life expectancy at birth"},
    "Birth rate": {"slug": "crude-birth-rate", "unit": "births per 1,000 people", "note": "Crude birth rate"},
    "Education levels (0-25)": {"slug": "mean-years-of-schooling-long-run", "unit": "0-25 scaled", "note": "Mean years of schooling (rescaled to 0-25 for display)"},
    "Average wealth (GDP per capita, PPP)": {"slug": "gdp-per-capita-ppp", "unit": "intl $ (PPP)", "note": "GDP per capita (PPP) — proxy for average wealth/income"},
    "Average income (GDP per capita, current US$)": {"slug": "gdp-per-capita", "unit": "US$", "note": "GDP per capita (current US$)"},
    "Unemployment rate": {"slug": "unemployment-rate", "unit": "%", "note": "National unemployment rate (where available)"},
    "Immigration out of the country (net migration)": {"slug": "net-migration", "unit": "people", "note": "Net migration (positive = net immigration)"},
    "Murder Rate": {"slug": "intentional-homicides", "unit": "per 100,000 people", "note": "Intentional homicides per 100k"}
}

# Default countries chosen because they are among Latin America's higher-GDP-per-capita countries:
DEFAULT_COUNTRIES = ["Chile", "Panama", "Uruguay"]

# ---------- Helper functions ----------
def owid_fetch_csv(slug, countries=None):
    """Fetch OWID grapher CSV for slug; optional country filter (list)."""
    params = ""
    if countries:
        q = ",".join(countries)
        params = f"?country={requests.utils.requote_uri(q)}"
    url = f"{OWID_BASE}/{slug}.csv{params}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))

def pick_value_column(df):
    """Return the most-likely numeric value column from an OWID grapher CSV DataFrame."""
    # OWID often has columns: year, country, iso_code, and then 1 or more value columns.
    candidates = [c for c in df.columns if c not in ('year', 'country', 'iso_code')]
    if not candidates:
        raise ValueError("No candidate value column found in OWID CSV.")
    # Use the last non-meta column (most grapher CSVs place the value last).
    return candidates[-1]

def prepare_country_timeseries(df, country, value_col):
    sub = df[df['country'] == country][['year', value_col]].dropna()
    sub = sub.sort_values('year').reset_index(drop=True)
    sub = sub.rename(columns={value_col: country})
    return sub

def scale_education_to_25(series):
    """Rescale a series of mean years-of-schooling so max observed -> 25 (transparent mapping)."""
    max_val = np.nanmax(series)
    if not np.isfinite(max_val) or max_val <= 0:
        return series * 0.0, 0.0
    factor = 25.0 / max_val
    return series * factor, factor

def poly_fit_and_predict(x, y, degree, xs):
    """Fit polynomial of given degree using numpy.polyfit; return poly1d and predicted ys."""
    coefs = np.polyfit(x, y, deg=degree)
    p = np.poly1d(coefs)   # numpy.poly1d (correct)
    ys = p(xs)
    return p, coefs, ys

def polynomial_str(coefs):
    """Human-friendly polynomial string from coef array (highest -> constant)."""
    terms = []
    deg = len(coefs) - 1
    for i, c in enumerate(coefs):
        powr = deg - i
        if abs(c) < 1e-12:
            continue
        if powr == 0:
            terms.append(f"{c:.6g}")
        elif powr == 1:
            terms.append(f"{c:.6g}·x")
        else:
            terms.append(f"{c:.6g}·x^{powr}")
    return " + ".join(terms) if terms else "0"

def find_critical_points(p):
    """Return real roots of p'(x) and p'(x) poly1d object."""
    d1 = p.deriv()           # numpy.poly1d derivative
    roots = d1.roots        # complex roots possibly
    # keep only real within tolerance
    real_roots = [np.real(r) for r in roots if np.isreal(r)]
    return np.array(real_roots), d1

def classify_critical_points(p, roots):
    d2 = p.deriv(2)
    crit_info = []
    for r in roots:
        y = p(r)
        val2 = d2(r)
        if val2 > 0:
            typ = "local minimum"
        elif val2 < 0:
            typ = "local maximum"
        else:
            typ = "inflection / saddle (2nd deriv ≈ 0)"
        crit_info.append({"x": r, "y": y, "type": typ, "second_derivative": val2})
    return crit_info

def increasing_decreasing_intervals(p, xmin, xmax, num=800):
    xs = np.linspace(xmin, xmax, num)
    d1 = p.deriv()
    deriv_vals = d1(xs)
    intervals = []
    if len(xs) == 0:
        return intervals, d1, deriv_vals, xs
    start = xs[0]
    cur_sign = np.sign(deriv_vals[0])
    for i in range(1, len(xs)):
        sign = np.sign(deriv_vals[i])
        if sign != cur_sign:
            intervals.append({"type": "increasing" if cur_sign > 0 else ("decreasing" if cur_sign < 0 else "flat"),
                              "start": start, "end": xs[i-1]})
            start = xs[i]
            cur_sign = sign
    intervals.append({"type": "increasing" if cur_sign > 0 else ("decreasing" if cur_sign < 0 else "flat"),
                      "start": start, "end": xs[-1]})
    return intervals, d1, deriv_vals, xs

def find_fastest_change_points(p, xmin, xmax):
    # fastest change points where derivative reaches extremum -> solve derivative of derivative (2nd deriv) = 0
    d1 = p.deriv()
    d2 = p.deriv(2)
    roots = d2.roots
    real_roots = [np.real(r) for r in roots if np.isreal(r) and xmin - 1e-6 <= np.real(r) <= xmax + 1e-6]
    candidates = []
    for r in real_roots:
        candidates.append({"x": r, "derivative": d1(r)})
    # include endpoints
    candidates.append({"x": xmin, "derivative": d1(xmin)})
    candidates.append({"x": xmax, "derivative": d1(xmax)})
    if not candidates:
        return None, None
    fastest_inc = max(candidates, key=lambda a: a["derivative"])
    fastest_dec = min(candidates, key=lambda a: a["derivative"])
    return fastest_inc, fastest_dec

def make_printer_friendly_report(context):
    html = "<html><head><meta charset='utf-8'><title>Regression Report</title></head><body>"
    html += "<h1>Polynomial Regression Report</h1>"
    html += f"<p>Generated: {datetime.datetime.now().isoformat()}</p>"
    html += "<h2>Settings</h2>"
    html += f"<pre>{json.dumps(context['settings'], indent=2)}</pre>"
    html += "<h2>Raw data</h2>"
    html += context['table_html']
    html += "<h2>Model equation</h2>"
    html += f"<pre>{context['equation']}</pre>"
    html += "<h2>Function analysis</h2>"
    for line in context['analysis_lines']:
        html += f"<p>{line}</p>"
    html += "<h2>Plot</h2>"
    if 'plot_png' in context:
        img_b64 = base64.b64encode(context['plot_png']).decode('utf-8')
        html += f"<img src='data:image/png;base64,{img_b64}' style='max-width:100%;height:auto'/>"
    html += "</body></html>"
    return html

def fig_to_png_bytes(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf.read()

# ---------- UI ----------
st.title("Polynomial Regression — Real OWID data (Chile, Panama, Uruguay defaults)")

st.markdown("""
This app fetches **real historical data** from *Our World in Data* (OWID) and fits a polynomial regression
(degree 3–6) to the chosen series. No placeholder numbers — the app reads live OWID CSVs for each indicator.
If an indicator lacks long coverage for a country, the app will warn you.  
(If you have a CSV you prefer, upload it using the U.S. Latin-groups upload control.)
""")

left, right = st.columns([1, 2])
with left:
    st.subheader("Controls")
    category = st.selectbox("Select category", list(INDICATOR_MAP.keys()))
    countries = st.multiselect("Countries to compare (pick 1–3)", DEFAULT_COUNTRIES, default=DEFAULT_COUNTRIES)
    degree = st.slider("Polynomial degree (3–6)", 3, 6, 3)
    step = st.slider("Plot resolution (years per step)", 1, 10, 1)
    extrap_years = st.number_input("Extrapolate how many years beyond latest data?", min_value=0, max_value=100, value=10)
    st.markdown("Upload CSV (optional) for U.S. Latin-origin groups (columns: group,year,value)")
    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])
    if st.button("Download CSV template"):
        template = "group,year,value\nMexican,2000,100\nPuerto Rican,2000,50\nCuban,2000,20\n"
        st.download_button("Download template CSV", data=template, file_name="us_latin_groups_template.csv", mime="text/csv")

with right:
    st.subheader(f"Indicator: {category}")
    indicator = INDICATOR_MAP[category]
    slug = indicator['slug']
    st.caption(f"OWID slug: `{slug}` — unit: {indicator['unit']} — note: {indicator['note']}")
    # Fetch
    try:
        df_raw = owid_fetch_csv(slug, countries)
    except Exception as e:
        st.error(f"Failed to fetch OWID data for slug `{slug}`: {e}")
        st.stop()

    value_col = pick_value_column(df_raw)
    st.write(f"Using value column: `{value_col}` from OWID CSV.")
    # assemble table
    data_table = None
    for c in countries:
        ts = prepare_country_timeseries(df_raw, c, value_col)
        if ts.empty:
            st.warning(f"No data for {c} (indicator `{slug}`).")
            continue
        if data_table is None:
            data_table = ts
        else:
            data_table = pd.merge(data_table, ts, on='year', how='outer')
    if data_table is None or data_table.shape[0] == 0:
        st.error("No usable data for the selected countries/indicator.")
        st.stop()

    data_table = data_table.sort_values('year').reset_index(drop=True)
    min_year = int(data_table['year'].min())
    max_year = int(data_table['year'].max())
    st.write(f"Data coverage: {min_year} — {max_year} (rows: {data_table.shape[0]})")

    # Education scaling if requested
    if category.startswith("Education"):
        for c in countries:
            if c in data_table.columns:
                scaled, factor = scale_education_to_25(data_table[c].fillna(method='ffill').fillna(method='bfill'))
                data_table[c + " (orig_years)"] = data_table[c]
                data_table[c] = scaled
                st.write(f"{c}: education scaling factor applied so max observed maps to 25 (factor {factor:.4g}).")

    st.markdown("Editable raw data (edit then press Apply edits & fit)")
    edited = st.data_editor(data_table, num_rows="dynamic")
    if st.button("Apply edits & fit"):
        data_table = edited.copy()

    # Primary country for analysis
    primary_country = st.selectbox("Primary country for detailed analysis", countries, index=0)
    df_primary = data_table[['year', primary_country]].dropna()
    x = df_primary['year'].astype(float).values
    y = df_primary[primary_country].astype(float).values

    if len(x) < 4:
        st.warning("Very few data points — polynomial fits may be unstable.")

    xs_plot = np.arange(int(x.min()), int(max_year + extrap_years) + 1, step)

    # Fit
    p, coefs, ys = poly_fit_and_predict(x, y, degree, xs_plot)
    eq = polynomial_str(coefs)
    st.markdown("### Model equation (primary)")
    st.code(f"f(x) = {eq}")

    # Analysis
    crit_roots, d1 = find_critical_points(p)
    crit_info = classify_critical_points(p, crit_roots)
    intervals, d1obj, deriv_vals, deriv_xs = increasing_decreasing_intervals(p, xs_plot.min(), xs_plot.max())
    fastest_inc, fastest_dec = find_fastest_change_points(p, xs_plot.min(), xs_plot.max())
    fvals = p(xs_plot)
    domain_str = f"[{xs_plot.min()}, {xs_plot.max()}]"
    range_str = f"[{np.nanmin(fvals):.6g}, {np.nanmax(fvals):.6g}]"

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for c in countries:
        if c in data_table.columns:
            series = data_table[['year', c]].dropna()
            ax.scatter(series['year'], series[c], label=f"{c} (data)", s=20, alpha=0.6)
    # plot primary fit (full), and mark extrapolation part differently
    ax.plot(xs_plot, ys, label=f"{primary_country} (fit deg {degree})", linewidth=2)
    if extrap_years > 0:
        mask_ex = xs_plot > max_year
        if mask_ex.any():
            ax.plot(xs_plot[mask_ex], ys[mask_ex], linestyle='--', linewidth=2, label=f"{primary_country} (extrapolated)")
    ax.set_xlabel("Year")
    ax.set_ylabel(indicator['unit'])
    ax.set_title(f"{category} — polynomial fit (primary: {primary_country})")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Human-readable analysis lines
    st.subheader("Function analysis (primary)")
    analysis_lines = []
    if len(crit_info) == 0:
        analysis_lines.append(f"No real critical points found for {primary_country}'s modeled {category.lower()}.")
    else:
        for cc in crit_info:
            analysis_lines.append(f"The {primary_country} {category.lower()} reached a {cc['type']} at year {cc['x']:.3f}; modeled value ≈ {cc['y']:.6g} {indicator['unit']}.")

    for it in intervals:
        analysis_lines.append(f"Model is {it['type']} roughly from {it['start']:.1f} to {it['end']:.1f} (approx.).")

    if fastest_inc:
        analysis_lines.append(f"The modeled {category.lower()} was increasing fastest near year {fastest_inc['x']:.3f} with instantaneous rate ≈ {fastest_inc['derivative']:.6g} {indicator['unit']}/year.")
    if fastest_dec:
        analysis_lines.append(f"The modeled {category.lower()} was decreasing fastest near year {fastest_dec['x']:.3f} with instantaneous rate ≈ {fastest_dec['derivative']:.6g} {indicator['unit']}/year.")

    analysis_lines.append(f"Domain (plotted): {domain_str}. Range (plotted): {range_str}.")
    extrap_year = int(max_year + extrap_years)
    extrap_val = p(extrap_year)
    analysis_lines.append(f"Extrapolation example: model predicts {primary_country}'s {category.lower()} ≈ {extrap_val:.6g} {indicator['unit']} in {extrap_year} (extrapolation beyond observed {min_year}-{max_year}).")

    for ln in analysis_lines:
        st.write(ln)

    # Interpolate/extrapolate a specific year
    st.subheader("Interpolate/Extrapolate a year")
    query_year = st.number_input("Year to evaluate f(year)", value=int(max_year)+1, step=1)
    if st.button("Compute f(year)"):
        val = p(query_year)
        st.write(f"Model prediction for {query_year}: **{val:.6g} {indicator['unit']}**")
        st.write(f"Interpretation: According to the model, {primary_country}'s {category.lower()} will be {val:.6g} {indicator['unit']} in {int(query_year)}.")

    # Average rate of change between two years using model
    st.subheader("Average rate of change between two years (model)")
    col1, col2 = st.columns(2)
    with col1:
        year_a = st.number_input("From year (a)", value=int(min_year), step=1, key="A")
    with col2:
        year_b = st.number_input("To year (b)", value=int(max_year), step=1, key="B")
    if st.button("Compute average rate of change"):
        fa = p(year_a)
        fb = p(year_b)
        avg_rate = (fb - fa) / (year_b - year_a)
        st.write(f"Average rate from {year_a} to {year_b}: **{avg_rate:.6g} {indicator['unit']}/year**")
        st.write(f"Example sentence: Between {year_a} and {year_b}, the model predicts an average change of {avg_rate:.6g} {indicator['unit']} per year for {primary_country}.")

    # Printer-friendly report download
    st.subheader("Printer-friendly report")
    table_html = data_table.to_html(index=False)
    png = fig_to_png_bytes(fig)
    context = {
        "settings": {"category": category, "primary_country": primary_country, "countries": countries, "degree": degree, "extrap_years": extrap_years},
        "table_html": table_html, "equation": f"f(x) = {eq}", "analysis_lines": analysis_lines, "plot_png": png
    }
    html_report = make_printer_friendly_report(context)
    st.download_button("Download HTML report", data=html_report.encode('utf-8'), file_name="regression_report.html", mime="text/html")

    # Optional: handle uploaded CSV for US Latin-origin groups
    if uploaded_csv is not None:
        try:
            ug = pd.read_csv(uploaded_csv)
            st.write("Uploaded data preview:")
            st.dataframe(ug.head())
            if set(['group', 'year', 'value']).issubset(ug.columns):
                pivot = ug.pivot_table(index='year', columns='group', values='value', aggfunc='mean')
                fig2, ax2 = plt.subplots(figsize=(10,5))
                for grp in pivot.columns:
                    ax2.plot(pivot.index, pivot[grp], marker='o', label=str(grp))
                ax2.set_xlabel("Year")
                ax2.set_ylabel("Value")
                ax2.set_title("Uploaded U.S. Latin-origin group comparison")
                ax2.legend()
                ax2.grid(True)
                st.pyplot(fig2)
            else:
                st.error("Uploaded CSV must include columns: 'group', 'year', 'value'.")
        except Exception as e:
            st.error(f"Failed to parse uploaded CSV: {e}")

st.caption("Note: OWID series combine UN / World Bank / national data; coverage varies by country/indicator. Extrapolations are model-based projections, not causal forecasts.")
