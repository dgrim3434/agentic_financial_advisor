from __future__ import annotations

import json
from dataclasses import asdict
from typing import List, Dict, Any

import streamlit as st
import pandas as pd

from src.agents.models import UserProfile
from src.agents.crews.profiling_crew import profiling_step
from src.agents.crews.investment_strategy_crew import run_research_universe
from src.agents.crews.portfolio_decision_crew import run_portfolio_decision



# Streamlit page config & basic styling

st.set_page_config(
    page_title="Agentic AI Financial Advisor",
    page_icon="💹",
    layout="wide",
)

st.markdown(
    """
    <style>
    .small-text { font-size: 0.85rem; color: #666; }
    .tag {
        display:inline-block;
        padding:2px 8px;
        margin:2px 4px 2px 0;
        border-radius:999px;
        background-color:#f0f2f6;
        font-size:0.80rem;
    }
    .risk-badge-low { background-color:#e3f6e5; color:#1b7c2f; }
    .risk-badge-medium { background-color:#fff4d6; color:#b57700; }
    .risk-badge-high { background-color:#ffe4e4; color:#b3261e; }
    </style>
    """,
    unsafe_allow_html=True,
)



# Session-state setup

def init_session_state() -> None:
    """Initialize Streamlit session state on first load."""
    if "user_profile" not in st.session_state:
        st.session_state.user_profile: UserProfile = UserProfile()

    if "profile_history" not in st.session_state:
        st.session_state.profile_history: List[Dict[str, Any]] = []

    if "profile_done" not in st.session_state:
        st.session_state.profile_done: bool = False

    if "chat_messages" not in st.session_state:
        # Kick things off with a simple onboarding prompt
        st.session_state.chat_messages: List[Dict[str, str]] = [
            {
                "role": "assistant",
                "content": (
                    "Hi, I'm your AI financial advisor onboarding assistant.\n\n"
                    "Tell me about yourself as an investor: your goals, risk comfort, "
                    "experience, sectors you like, any tickers you're watching, and "
                    "roughly how much you'd like to invest."
                ),
            }
        ]

    # Strategy objects: research bundle + decision bundle
    if "strategy_research" not in st.session_state:
        st.session_state.strategy_research: Dict[str, Any] | None = None

    if "strategy_decision" not in st.session_state:
        st.session_state.strategy_decision: Dict[str, Any] | None = None


init_session_state()


# Helper: render profile snapshot in the sidebar

def _risk_badge(risk: str | None) -> str:
    """Return a small colored badge for the risk tolerance."""
    if not risk:
        return "<span class='tag'>—</span>"

    r = risk.lower()
    css_class = "risk-badge-medium"
    if r.startswith("low"):
        css_class = "risk-badge-low"
    elif r.startswith("high"):
        css_class = "risk-badge-high"

    return f"<span class='tag {css_class}'>{risk.title()}</span>"


def render_profile_snapshot(profile: UserProfile) -> None:
    """Compact view of the current investor profile."""
    st.markdown("### Investor Profile Snapshot")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Name**")
        st.write(profile.name or "—")

        st.markdown("**Age**")
        st.write(profile.age if profile.age is not None else "—")

        st.markdown("**Investment Horizon**")
        st.write(profile.investment_horizon or "—")

        st.markdown("**Experience Level**")
        st.write(profile.experience_level or "—")

    with col_b:
        st.markdown("**Risk Tolerance**")
        st.markdown(_risk_badge(profile.risk_tolerance), unsafe_allow_html=True)

        st.markdown("**Investment Amount**")
        if profile.investment_amount:
            st.write(str(profile.investment_amount))
        else:
            st.write("—")

        st.markdown("**Preferred Sectors**")
        if profile.preferred_sectors:
            tags = " ".join(
                f"<span class='tag'>{s}</span>" for s in profile.preferred_sectors
            )
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.write("—")

        st.markdown("**Watchlist Tickers**")
        if profile.ticker_watchlist:
            tags = " ".join(
                f"<span class='tag'>{t.upper()}</span>"
                for t in profile.ticker_watchlist
            )
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.write("—")

    if profile.constraints or profile.notes:
        st.markdown("**Notes & Constraints**")
        txt = ""
        if profile.constraints:
            txt += f"- Constraints: {profile.constraints}\n"
        if profile.notes:
            txt += f"- Notes: {profile.notes}\n"
        st.markdown(txt)

    # Developer view only – nice to keep around while iterating
    with st.expander("Developer view: raw profile JSON", expanded=False):
        st.json(asdict(profile))



# Layout section

st.title("💹 Agentic AI Financial Advisor")
st.caption(
    "A multi-agent research pipeline that profiles you as an investor, researches companies, "
    "and proposes a diversified, risk-aware strategy."
)

chat_col, side_col = st.columns([2.1, 1])


# Chat / Onboarding 

with chat_col:
    st.subheader("Onboarding Conversation")

    # Render existing messages
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Describe yourself as an investor...")
    if user_input:
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Run one profiling step
        with st.chat_message("assistant"):
            with st.spinner("Thinking about your profile..."):
                (
                    assistant_text,
                    updated_profile,
                    profile_done,
                    new_history,
                ) = profiling_step(
                    user_input,
                    st.session_state.user_profile,
                    st.session_state.chat_messages,
                )

            st.markdown(assistant_text)

        # Update session state
        st.session_state.user_profile = updated_profile
        st.session_state.profile_done = bool(profile_done)
        st.session_state.chat_messages = new_history


# Profile + Strategy Controls 

with side_col:
    render_profile_snapshot(st.session_state.user_profile)

    st.markdown("---")
    st.markdown("### Strategy Generation")

    if not st.session_state.profile_done:
        st.markdown(
            "<span class='small-text'>Keep chatting until the assistant says "
            "it has enough information to build a portfolio. "
            "You can still add details even after that.</span>",
            unsafe_allow_html=True,
        )

    ticker_list = st.session_state.user_profile.ticker_watchlist or []

    btn_disabled = len(ticker_list) == 0
    tip = (
        "Add at least one ticker to your watchlist in the conversation first."
        if btn_disabled
        else ""
    )

    if st.button(
        "🚀 Generate Investment Strategy",
        type="primary",
        disabled=btn_disabled,
        help=tip,
    ):
        with st.spinner("Running research & decision agents..."):
            # 1) Multi-agent research
            research = run_research_universe(st.session_state.user_profile)

            # 2) Portfolio decision agent on top of that research
            decision_bundle = run_portfolio_decision(
                st.session_state.user_profile, research
            )

        st.session_state.strategy_research = research
        st.session_state.strategy_decision = decision_bundle


# Strategy Output

st.divider()
st.header("📊 Investment Strategy & Research")

if not st.session_state.strategy_decision or not st.session_state.strategy_research:
    st.info(
        "Once you finish onboarding and click **Generate Investment Strategy**, "
        "your personalized allocation and research will appear here."
    )
else:
    research = st.session_state.strategy_research
    decision = st.session_state.strategy_decision

    allocation_mode = decision.get("allocation_mode", "percent")
    total_investment = decision.get("total_investment")
    positions = decision.get("positions", [])
    decision_notes = decision.get("notes", "")
    decision_narrative = decision.get("narrative", "")  # kept for future use

    tab_strategy, tab_research = st.tabs(
        ["Strategy & Allocation", "Per-Ticker Research"]
    )

    # Tab 1: Strategy & Allocation 
    with tab_strategy:
        st.markdown("## Portfolio Recommendation")

        if not positions:
            st.info(
                "The decision agent did not return any positions. "
                "Try adjusting your profile or watchlist and rerun."
            )
        else:
            rows = []
            for pos in positions:
                ticker = pos.get("ticker", "").upper()
                weight_pct = pos.get("weight_pct")
                rationale = pos.get("rationale", "")
                bucket = pos.get("bucket", "?")

                rows.append(
                    {
                        "Ticker": ticker,
                        "Bucket": bucket,
                        "Weight %": round(weight_pct, 2)
                        if isinstance(weight_pct, (int, float))
                        else None,
                        "Rationale": rationale,
                    }
                )

            df_alloc = (
                pd.DataFrame(rows)
                .sort_values("Weight %", ascending=False, na_position="last")
                .reset_index(drop=True)
            )

            col_table, col_chart = st.columns([2, 1])

            with col_table:
                st.markdown("### Allocation Table")

                # Let the rationale text wrap instead of getting visually cut off
                st.markdown(
                    """
                    <style>
                    div[data-testid="stDataFrame"] td {
                        white-space: normal !important;
                        overflow-wrap: anywhere !important;
                    }
                    div[data-testid="stDataFrame"] th {
                        white-space: normal !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                st.dataframe(
                    df_alloc,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Rationale": st.column_config.TextColumn(
                            "Rationale",
                            width="large",
                        )
                    },
                )

            with col_chart:
                st.markdown("### Allocation by Ticker")
                chart_df = df_alloc.dropna(subset=["Weight %"]).set_index("Ticker")
                if not chart_df.empty:
                    st.bar_chart(
                        chart_df["Weight %"],
                        use_container_width=True,
                    )
                else:
                    st.info("No weight percentages available to chart.")

            st.markdown("### Allocation Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Number of tickers", len(df_alloc))

            if total_investment is not None:
                c2.metric(
                    "Total capital allocated",
                    f"${float(total_investment):,.0f}",
                )
            else:
                c2.metric("Allocation mode", allocation_mode.title())

            if not df_alloc.empty and df_alloc["Weight %"].notna().any():
                top = df_alloc.sort_values("Weight %", ascending=False).iloc[0]
                c3.metric(
                    "Largest position",
                    f"{top['Ticker']} ({top['Weight %']}%)",
                )

            if decision_notes:
                st.markdown("### Notes from Decision Agent")
                st.markdown(decision_notes)

    # Tab 2: Per-Ticker Research
    with tab_research:
        st.markdown(
            "This tab shows the research produced by the agents for **every ticker** "
            "in the universe: market behavior, news & sentiment, and SEC fundamentals."
        )

        universe = research.get("universe", [])
        per_ticker = research.get("per_ticker", {})

        if not universe:
            st.info("No researched tickers found.")
        else:
            # One expander per ticker, so everything for that name lives in one place
            for ticker in universe:
                block = per_ticker.get(ticker, {}) or {}
                if not block:
                    continue

                market_md = (block.get("market") or "").strip()
                sentiment_md = (block.get("sentiment") or "").strip()
                sec_md = (block.get("sec") or "").strip()

                with st.expander(f"📌 {ticker}", expanded=False):
                    st.markdown(f"### {ticker}")

                    st.markdown("#### 📈 Market / Quant Characteristics")
                    if market_md:
                        st.markdown(market_md)
                    else:
                        st.info("No market/quant research available for this ticker.")

                    st.markdown("#### 📰 News & Sentiment")
                    if sentiment_md:
                        st.markdown(sentiment_md)
                    else:
                        st.info(
                            "No news & sentiment research available for this ticker."
                        )

                    st.markdown("#### 📜 SEC Filings & Fundamentals (RAG)")
                    if sec_md:
                        st.markdown(sec_md)
                    else:
                        st.info("No SEC RAG research available for this ticker.")

        # Optional: raw blobs for debugging
        with st.expander("Developer view: raw research blobs", expanded=False):
            st.markdown("**Market notes (all tickers):**")
            st.markdown(research.get("market_notes_raw", "") or "_none_")

            st.markdown("**Sentiment notes (all tickers):**")
            st.markdown(research.get("sentiment_notes_raw", "") or "_none_")

            st.markdown("**SEC summary notes (all tickers):**")
            st.markdown(research.get("sec_summary_raw", "") or "_none_")
