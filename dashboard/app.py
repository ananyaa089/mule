"""MuleGuard dashboard.   Run:  streamlit run dashboard/app.py      (API must be running: uvicorn api.main:app --port 8000)"""
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API = os.environ.get("MULEGUARD_API", "http://localhost:8000")
COLORS = {"HIGH": "#c0392b", "MEDIUM": "#e08e0b", "LOW": "#2e8b57"}

st.set_page_config(page_title="MuleGuard", layout="wide")


def call(method, path, **kw):
    try:
        r = requests.request(method, API + path, timeout=kw.pop("timeout", 120), **kw)
    except requests.RequestException as e:
        st.error(f"Cannot reach the API at {API}. Start it with: uvicorn api.main:app --port 8000\n\n{e}")
        st.stop()
    if r.status_code >= 400:
        detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
        return None, detail
    return r.json(), None


def fmt_inr(x):
    return f"₹{x:,.0f}"


def flow_figure(tr):
    """Left = sources, centre = the account, right = recipients by hop depth."""
    acct = tr["account"]
    pos, edges = {acct: (0.0, 0.0)}, []
    cols = {}
    for lvl in tr["backward"]["hops"]:
        for h in lvl["shown"]:
            edges.append((h["from"], h["to"], h["amount"], -lvl["depth"]))
            cols.setdefault(-lvl["depth"], []).append(h["from"])
    for lvl in tr["forward"]["hops"]:
        for h in lvl["shown"]:
            edges.append((h["from"], h["to"], h["amount"], lvl["depth"]))
            cols.setdefault(lvl["depth"], []).append(h["to"])
    for x, names in cols.items():
        names = list(dict.fromkeys(n for n in names if n not in pos))
        for i, n in enumerate(names):
            pos[n] = (float(x), (i - (len(names) - 1) / 2) * 1.0)
    fig = go.Figure()
    for s, d, amt, _ in edges:
        if s in pos and d in pos:
            (x0, y0), (x1, y1) = pos[s], pos[d]
            fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                               arrowhead=2, arrowwidth=1.5, arrowcolor="#8a8f98")
            fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=fmt_inr(amt), showarrow=False, font=dict(size=10), bgcolor="rgba(255,255,255,0.7)")
    sus = {s["account"]: s for s in tr["suspects"]}
    fig.add_trace(go.Scatter(
        x=[p[0] for p in pos.values()], y=[p[1] for p in pos.values()], mode="markers+text", text=list(pos), textposition="top center",
        marker=dict(size=[26 if n == acct else 16 for n in pos],
                    color=["#1f3a93" if n == acct else ("#c0392b" if sus.get(n, {}).get("above_threshold") else "#9aa5b1") for n in pos]),
        hovertext=[f"{n}<br>ML score: {sus.get(n, {}).get('ml_score', tr['account_info'].get('ml_score'))}" for n in pos], hoverinfo="text"))
    fig.update_layout(height=380, showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def show_report(rep):
    n, lvl = rep["narrative"], rep["risk_level"]
    st.markdown(f"<span style='background:{COLORS[lvl]};color:white;padding:4px 14px;border-radius:6px;font-weight:700;font-size:20px'>{lvl} RISK</span>"
                f"&nbsp;&nbsp;account <code>{rep['account']}</code>", unsafe_allow_html=True)
    s = rep["scores"]
    c1, c2, c3 = st.columns(3)
    c1.metric("ML score", s["ml_score"], help=f"flag threshold {s['ml_threshold']}")
    c2.metric("Anomaly percentile", s["anomaly_pct"])
    c3.metric("Narrative source", n["source"])
    st.subheader("Summary")
    st.write(n["summary"])
    st.subheader("Key suspicious patterns")
    for k in n["key_patterns"]:
        st.markdown(f"- **{k['pattern']}**: {k['evidence']}")
    with st.expander("Rule findings (the numbers behind the risk level)", expanded=True):
        for x in rep["rule_findings"]:
            st.markdown(f"- {x}")
    st.subheader("Similar historical cases")
    if rep["similar_cases"]:
        st.dataframe(pd.DataFrame(rep["similar_cases"])[["case_id", "kind", "typology", "similarity", "summary"]], hide_index=True, use_container_width=True)
    st.caption("Similar cases come from the past (train) window. A high share of mules among neighbours partly reflects how many mules were indexed, so it is context, not a risk score.")
    for c in n["caveats"]:
        st.caption("⚠ " + c)
    if n.get("unverified_numbers"):
        st.warning(f"Numbers in the AI text that were not found in the evidence: {n['unverified_numbers']}")
    st.info("**Recommended action:** " + rep["recommended_action"])


st.title("🛡️ MuleGuard")
st.caption("Prototype on synthetic data. The AI recommends; a human decides.")
page = st.sidebar.radio("Page", ["Alerts", "Investigate", "Audit log"])
analyst = st.sidebar.text_input("Your name (recorded in the audit log)", value="analyst")
h, err = call("GET", "/health")
st.sidebar.write("API: " + ("🟢 up" if h else "🔴 " + str(err)))
if h:
    st.sidebar.write("LLM: " + ("🟢 key set" if h["llm_configured"] else "⚪ no key (template narrative)"))

if page == "Alerts":
    data, err = call("GET", "/alerts", params={"limit": 100})
    st.subheader(f"Accounts flagged by the model (score ≥ {data['threshold']:.3f}, test window)")
    df = pd.DataFrame(data["alerts"])
    st.dataframe(df, hide_index=True, use_container_width=True)
    if len(df):
        pick = st.selectbox("Open an account in Investigate", df["acct_id"])
        if st.button("Go to investigation"):
            st.session_state["acct"] = pick
            st.session_state["go"] = True
            st.info("Choose 'Investigate' in the sidebar; the account is pre-filled.")

elif page == "Investigate":
    acct = st.text_input("Account ID", value=st.session_state.get("acct", ""))
    use_llm = st.checkbox("Use AI narrative (needs GROQ_API_KEY on the API server)", value=False)
    if st.button("Investigate", type="primary") and acct:
        with st.spinner("Tracing funds and building the report..."):
            rep, err = call("POST", "/investigate", json={"account": acct.strip(), "use_llm": use_llm})
            tr, err2 = call("GET", f"/trace/{acct.strip()}")
        if err:
            st.error(err)
        else:
            st.session_state["rep"], st.session_state["trace"] = rep, tr
    rep, tr = st.session_state.get("rep"), st.session_state.get("trace")
    if rep:
        show_report(rep)
        if tr:
            st.subheader("Fund flow")
            f = tr["forward"]
            st.caption(f"Trigger {fmt_inr(tr['trigger']['amount'])} at {tr['trigger']['ts']}; {f['share_of_trigger']} of it forwarded, first outflow after {f['first_out_secs']} s. "
                       "Red = recipient the model also flags.")
            st.plotly_chart(flow_figure(tr), use_container_width=True)
        st.subheader("Decision")
        cands = rep["freeze_candidates"]
        if cands:
            chosen = st.multiselect("Accounts to freeze", cands + [a for a in rep["review_accounts"] if a not in cands], default=cands)
            reason = st.text_input("Reason", value=f"{rep['risk_level']} risk: rapid pass-through (investigation #{rep.get('investigation_id')})")
            ok = st.checkbox("I have reviewed the evidence and approve freezing these accounts")
            if st.button("Freeze selected accounts", disabled=not (ok and chosen)):
                out, err = call("POST", "/freeze", json={"accounts": chosen, "actor": analyst, "reason": reason, "confirm": True,
                                                         "investigation_id": rep.get("investigation_id")})
                st.error(err) if err else st.success(f"Frozen: {out['frozen']}  (already frozen: {out['skipped_already_frozen']})")
        else:
            st.write("No freeze recommended for this account.")
        reason2 = st.text_input("Dismiss reason", key="dr")
        if st.button("Dismiss as not a mule", disabled=len(reason2) < 3):
            out, err = call("POST", "/dismiss", json={"account": rep["account"], "actor": analyst, "reason": reason2,
                                                      "investigation_id": rep.get("investigation_id")})
            st.error(err) if err else st.success("Dismissal logged.")

else:
    data, err = call("GET", "/audit", params={"limit": 200})
    st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)
