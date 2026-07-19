# ═══════════════════════════════════════════════════════
# app.py — EARNINGS STRADDLE ANALYSER
# Real data: Alpha Vantage + Yahoo Finance + CBOE
# Push to GitHub → auto-deploys on Streamlit Cloud
# ═══════════════════════════════════════════════════════

import streamlit as st
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import datetime
import os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title = "Earnings Straddle Analyser",
    page_icon  = "📈",
    layout     = "wide"
)

# ─────────────────────────────────────────────
# DATA SOURCE 1: ALPHA VANTAGE — EARNINGS DATES
# ─────────────────────────────────────────────
@st.cache_data(ttl=86400)  # cache 24 hours
def get_earnings_dates_av(ticker, api_key):
    url = (
        "https://www.alphavantage.co/query"
        f"?function=EARNINGS&symbol={ticker}&apikey={api_key}"
    )
    try:
        resp = requests.get(url, timeout=20)
        data = resp.json()
        if "quarterlyEarnings" not in data:
            return None, None
        today  = datetime.date.today()
        dates  = []
        for e in data["quarterlyEarnings"]:
            d = e.get("reportedDate")
            if d and d != "None":
                dates.append(datetime.date.fromisoformat(d))
        dates = sorted(dates)
        past  = [d for d in dates if d <= today]
        if past:
            last     = past[-1]
            est_next = last + datetime.timedelta(days=91)
            return str(last), str(est_next)
        return None, None
    except:
        return None, None


# ─────────────────────────────────────────────
# DATA SOURCE 2: YAHOO FINANCE — STOCK PRICES
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)  # cache 5 minutes
def get_stock_price_yf(ticker):
    try:
        hist  = yf.Ticker(ticker).history(period="1d")
        price = float(hist["Close"].iloc[-1])
        return price
    except:
        return None


@st.cache_data(ttl=3600)
def get_price_history_yf(ticker):
    try:
        df = yf.Ticker(ticker).history(period="5y")
        df.index = df.index.tz_convert(None)
        df.index = pd.Index([
            datetime.date(d.year, d.month, d.day)
            for d in df.index
        ])
        return df
    except:
        return None


# ─────────────────────────────────────────────
# DATA SOURCE 3: CBOE — REAL OPTIONS PRICING
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)  # cache 5 minutes
def get_cboe_term_structure(ticker):
    url = (
        f"https://cdn.cboe.com/api/global/delayed_quotes"
        f"/options/{ticker}.json"
    )
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer":    "https://www.cboe.com/"
    }
    try:
        resp  = requests.get(url, headers=headers, timeout=15)
        data  = resp.json()
        price = data["data"]["current_price"]
        opts  = pd.DataFrame(data["data"]["options"])
        today = datetime.date.today()

        def parse_opt(tkr):
            try:
                rest = tkr[len(ticker):]
                dp   = rest[:6]
                tp   = rest[6]
                sp   = rest[7:]
                exp  = datetime.date(
                    2000+int(dp[:2]),
                    int(dp[2:4]),
                    int(dp[4:6])
                )
                return exp, tp, int(sp)/1000
            except:
                return None, None, None

        parsed          = opts["option"].apply(parse_opt)
        opts["expiry"]  = parsed.apply(lambda x: x[0])
        opts["opttype"] = parsed.apply(lambda x: x[1])
        opts["strike"]  = parsed.apply(lambda x: x[2])
        opts["distance"]= (opts["strike"] - price).abs()

        opts = opts[
            opts["expiry"].apply(
                lambda x: 2 <= (x - today).days <= 90
                if x else False
            )
        ].copy()

        rows = []
        for expiry, grp in opts.groupby("expiry"):
            calls = grp[grp["opttype"] == "C"]
            puts  = grp[grp["opttype"] == "P"]
            if calls.empty or puts.empty:
                continue

            bc = calls.nsmallest(1, "distance")
            bp = puts.nsmallest(1, "distance")

            bc_mid = (float(bc["bid"].values[0]) +
                      float(bc["ask"].values[0])) / 2
            bp_mid = (float(bp["bid"].values[0]) +
                      float(bp["ask"].values[0])) / 2

            cp = bc_mid if bc_mid > 0 else float(
                bc["last_trade_price"].values[0]
            )
            pp = bp_mid if bp_mid > 0 else float(
                bp["last_trade_price"].values[0]
            )

            if cp <= 0 or pp <= 0:
                continue

            straddle = cp + pp
            cost_pct = (straddle / price) * 100
            days_out = (expiry - today).days

            rows.append({
                "expiry":    expiry,
                "days_out":  days_out,
                "strike":    float(bc["strike"].values[0]),
                "call_mid":  round(cp, 2),
                "put_mid":   round(pp, 2),
                "straddle":  round(straddle, 2),
                "cost_pct":  round(cost_pct, 2),
            })

        if not rows:
            return None, price

        sdf         = pd.DataFrame(rows).sort_values("days_out")
        sdf["jump"] = sdf["cost_pct"].diff().round(2)
        return sdf, price

    except Exception as e:
        return None, None


def identify_earnings_expiry(sdf, min_jump=1.5):
    sdf       = sdf.copy()
    sdf["jump"] = sdf["cost_pct"].diff()
    valid     = sdf[sdf["jump"] >= min_jump]
    if len(valid) == 0:
        return None
    row      = valid.nlargest(1, "jump").iloc[0]
    pre_idx  = sdf.index[sdf.index.get_loc(row.name) - 1]
    pre      = sdf.loc[pre_idx]
    isolated = row["cost_pct"] - pre["cost_pct"]
    return {
        "expiry":        str(row["expiry"]),
        "days_out":      int(row["days_out"]),
        "strike":        row["strike"],
        "call_price":    row["call_mid"],
        "put_price":     row["put_mid"],
        "straddle_cost": row["straddle"],
        "cost_pct":      row["cost_pct"],
        "isolated_move": round(isolated, 2),
        "pre_cost":      pre["cost_pct"],
    }


# ─────────────────────────────────────────────
# PAPER TRADE TRACKER — PERSISTENT STORAGE
# ─────────────────────────────────────────────
TRACKER_FILE = "paper_trades.csv"

def load_tracker():
    if os.path.exists(TRACKER_FILE):
        return pd.read_csv(TRACKER_FILE)
    cols = [
        "trade_id","ticker","earnings_date","expiry",
        "entry_date","stock_price_entry","strike",
        "call_price_entry","put_price_entry",
        "straddle_cost","cost_pct",
        "breakeven_up","breakeven_down","contracts",
        "earnings_source","price_source","options_source",
        "status","exit_date","stock_price_exit",
        "call_price_exit","put_price_exit",
        "straddle_exit_value","actual_move_pct",
        "pnl_pct","pnl_usd","outcome","notes"
    ]
    return pd.DataFrame(columns=cols)

def save_tracker(df):
    df.to_csv(TRACKER_FILE, index=False)


# ─────────────────────────────────────────────
# UI LAYOUT
# ─────────────────────────────────────────────
st.title("📈 Earnings Straddle Analyser")
st.caption(
    "Real data only: "
    "Earnings dates from Alpha Vantage · "
    "Stock prices from Yahoo Finance · "
    "Options pricing from CBOE"
)

tab1, tab2, tab3 = st.tabs([
    "🔍 Scanner",
    "📝 Paper Trade Log",
    "📊 Performance"
])


# ══════════════════════════════════════════════
# TAB 1: SCANNER
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Earnings Straddle Scanner")

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        ticker_input = st.text_input(
            "Stock Ticker",
            value="TSLA",
            placeholder="e.g. TSLA, META, AMZN"
        ).upper().strip()
    with c2:
        av_key = st.text_input(
            "Alpha Vantage API Key",
            placeholder="Your free key",
            type="password"
        )
        st.caption("Free at alphavantage.co/support/#api-key")
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        scan_btn = st.button(
            "🔍 Scan",
            use_container_width=True,
            type="primary"
        )

    if scan_btn and ticker_input:

        col_l, col_r = st.columns([1, 1])

        # Alpha Vantage: earnings dates
        with col_l:
            with st.spinner("Fetching earnings dates (Alpha Vantage)..."):
                last_earn, est_next = get_earnings_dates_av(
                    ticker_input, av_key
                ) if av_key else (None, None)

            st.markdown("**Earnings Dates — Alpha Vantage**")
            av_df = pd.DataFrame([{
                "Source":     "Alpha Vantage",
                "Last Earnings": last_earn or "N/A",
                "Est. Next":  est_next or "Check Yahoo Finance",
            }])
            st.dataframe(av_df, use_container_width=True,
                         hide_index=True)

        # Yahoo Finance: current price
        with col_r:
            with st.spinner("Fetching price (Yahoo Finance)..."):
                yf_price = get_stock_price_yf(ticker_input)

            st.markdown("**Current Price — Yahoo Finance**")
            if yf_price:
                st.metric(
                    f"{ticker_input} Current Price",
                    f"${yf_price:.2f}"
                )

        st.divider()

        # CBOE: term structure
        with st.spinner("Fetching options term structure (CBOE)..."):
            sdf, cboe_price = get_cboe_term_structure(ticker_input)

        if sdf is None:
            st.error("Could not fetch CBOE options data")
            st.stop()

        st.markdown("**Options Term Structure — CBOE**")
        st.caption(
            "The jump in cost% identifies where earnings "
            "risk is priced in"
        )

        display_sdf = sdf.copy()
        display_sdf["expiry"] = display_sdf["expiry"].astype(str)
        display_sdf["jump"]   = display_sdf["jump"].fillna("—")

        def highlight_earnings(row):
            try:
                jump = float(row["jump"])
                if jump >= 1.5:
                    return ["background-color: #1a3a1a"] * len(row)
            except:
                pass
            return [""] * len(row)

        st.dataframe(
            display_sdf[[
                "expiry","days_out","strike",
                "call_mid","put_mid","straddle",
                "cost_pct","jump"
            ]].style.apply(highlight_earnings, axis=1),
            use_container_width=True,
            hide_index=True
        )

        # Identify earnings expiry
        info = identify_earnings_expiry(sdf)

        if info:
            st.divider()
            st.markdown("**Signal**")

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Earnings Expiry",  info["expiry"])
            m2.metric("Isolated Move",    f"{info['isolated_move']:.2f}%",
                      help="What the market prices as the earnings move specifically")
            m3.metric("Total Straddle",   f"${info['straddle_cost']:.2f}")
            m4.metric("Cost %",           f"{info['cost_pct']:.2f}%")

            st.divider()

            # Term structure chart
            fig, ax = plt.subplots(figsize=(12, 4))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#1e2130")
            ax.tick_params(colors="white")
            ax.title.set_color("white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            for sp in ax.spines.values():
                sp.set_edgecolor("#444")

            colors = []
            for _, row in sdf.iterrows():
                try:
                    j = float(row["jump"]) if pd.notna(row["jump"]) else 0
                    colors.append("#ffd700" if j >= 1.5 else "#4c9be8")
                except:
                    colors.append("#4c9be8")

            ax.bar(
                range(len(sdf)),
                sdf["cost_pct"],
                color=colors, alpha=0.85
            )
            ax.set_xticks(range(len(sdf)))
            ax.set_xticklabels(
                [str(e) for e in sdf["expiry"]],
                rotation=45, ha="right", fontsize=8
            )
            ax.set_title(
                f"{ticker_input} — Options Term Structure "
                f"(gold = earnings expiry)",
                color="white"
            )
            ax.set_ylabel("Straddle Cost %", color="white")
            ax.grid(axis="y", color="#333", linewidth=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Entry form
            st.divider()
            st.markdown("**Log This Paper Trade**")
            st.caption(
                "Prices below are from CBOE. "
                "Verify earnings date on Yahoo Finance before entering."
            )

            ef1, ef2, ef3 = st.columns(3)
            with ef1:
                earn_date_input = st.text_input(
                    "Confirmed Earnings Date",
                    value=est_next or "",
                    placeholder="YYYY-MM-DD"
                )
            with ef2:
                contracts_input = st.number_input(
                    "Contracts (paper)",
                    min_value=1, max_value=10, value=1
                )
            with ef3:
                st.markdown("<br>", unsafe_allow_html=True)
                log_btn = st.button(
                    "📝 Log Paper Trade Entry",
                    use_container_width=True
                )

            if log_btn and earn_date_input:
                df_tracker = load_tracker()
                trade_id   = f"{ticker_input}_{earn_date_input}"

                if trade_id in df_tracker.get("trade_id", pd.Series()).values:
                    st.warning(f"Trade {trade_id} already logged")
                else:
                    straddle  = info["straddle_cost"]
                    price     = yf_price or cboe_price
                    cost_pct  = info["cost_pct"]
                    contracts = int(contracts_input)

                    new_trade = {
                        "trade_id":          trade_id,
                        "ticker":            ticker_input,
                        "earnings_date":     earn_date_input,
                        "expiry":            info["expiry"],
                        "entry_date":        str(datetime.date.today()),
                        "stock_price_entry": round(price, 2),
                        "strike":            info["strike"],
                        "call_price_entry":  info["call_price"],
                        "put_price_entry":   info["put_price"],
                        "straddle_cost":     straddle,
                        "cost_pct":          cost_pct,
                        "breakeven_up":   round(price + straddle, 2),
                        "breakeven_down": round(price - straddle, 2),
                        "contracts":         contracts,
                        "earnings_source":   "Alpha Vantage",
                        "price_source":      "Yahoo Finance",
                        "options_source":    "CBOE",
                        "status":            "OPEN",
                        "exit_date":         None,
                        "stock_price_exit":  None,
                        "call_price_exit":   None,
                        "put_price_exit":    None,
                        "straddle_exit_value": None,
                        "actual_move_pct":   None,
                        "pnl_pct":           None,
                        "pnl_usd":           None,
                        "outcome":           None,
                        "notes":             None,
                    }

                    df_tracker = pd.concat(
                        [df_tracker, pd.DataFrame([new_trade])],
                        ignore_index=True
                    )
                    save_tracker(df_tracker)

                    st.success(
                        f"✅ Paper trade logged: {trade_id}\n\n"
                        f"Entry: ${price:.2f} | "
                        f"Straddle: ${straddle:.2f} ({cost_pct:.2f}%) | "
                        f"Breakeven: >{cost_pct:.2f}% move"
                    )
        else:
            st.warning(
                "No clear earnings expiry identified in term structure. "
                "Earnings may have already passed or are far out."
            )


# ══════════════════════════════════════════════
# TAB 2: PAPER TRADE LOG
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Paper Trade Log")
    st.caption(
        "All prices from real sources: "
        "CBOE options pricing · Yahoo Finance stock prices"
    )

    df_tracker = load_tracker()

    # Open trades — log exit
    open_trades = df_tracker[df_tracker["status"] == "OPEN"]
    if len(open_trades) > 0:
        st.markdown(f"**Open Trades ({len(open_trades)})**")
        st.dataframe(
            open_trades[[
                "trade_id","entry_date","earnings_date",
                "expiry","stock_price_entry","straddle_cost",
                "cost_pct","breakeven_up","breakeven_down"
            ]],
            use_container_width=True,
            hide_index=True
        )

        st.markdown("**Log Exit for Open Trade**")
        st.caption(
            "Get exit prices from CBOE at market open "
            "the morning after earnings"
        )

        trade_ids = list(open_trades["trade_id"].values)
        sel_trade = st.selectbox("Select trade to close", trade_ids)

        if sel_trade:
            sel_row = open_trades[
                open_trades["trade_id"] == sel_trade
            ].iloc[0]

            st.info(
                f"**{sel_row['ticker']}** entered at "
                f"${sel_row['stock_price_entry']:.2f} | "
                f"Straddle cost: ${sel_row['straddle_cost']:.2f} "
                f"({sel_row['cost_pct']:.2f}%)"
            )

            x1, x2, x3 = st.columns(3)
            with x1:
                exit_stock = st.number_input(
                    "Stock price at exit [Yahoo Finance]",
                    min_value=0.01, value=float(
                        sel_row["stock_price_entry"]
                    ), step=0.01
                )
            with x2:
                exit_call = st.number_input(
                    "Call price at exit [CBOE]",
                    min_value=0.0,
                    value=float(sel_row["call_price_entry"]),
                    step=0.01
                )
            with x3:
                exit_put = st.number_input(
                    "Put price at exit [CBOE]",
                    min_value=0.0,
                    value=float(sel_row["put_price_entry"]),
                    step=0.01
                )

            exit_notes = st.text_input(
                "Notes (optional)",
                placeholder="e.g. Gapped up 8% on earnings beat"
            )

            # Live P&L preview
            entry_cost   = float(sel_row["straddle_cost"])
            entry_stock  = float(sel_row["stock_price_entry"])
            exit_val     = exit_call + exit_put
            pnl_pct_prev = ((exit_val - entry_cost) / entry_stock) * 100
            act_move_prev= ((exit_stock - entry_stock) / entry_stock) * 100

            p1, p2, p3 = st.columns(3)
            p1.metric("Actual Move",     f"{act_move_prev:+.2f}%")
            p2.metric("Exit Value",      f"${exit_val:.2f}")
            p3.metric(
                "P&L Preview",
                f"{pnl_pct_prev:+.2f}%",
                delta=f"{'WIN' if pnl_pct_prev > 0 else 'LOSS'}"
            )

            close_btn = st.button(
                "✅ Close This Trade",
                type="primary"
            )

            if close_btn:
                contracts  = int(sel_row["contracts"])
                pnl_usd    = (
                    (exit_val - entry_cost) * 100 * contracts
                )
                actual_move = act_move_prev
                outcome     = "WIN" if pnl_pct_prev > 0 else "LOSS"

                idx = df_tracker[
                    df_tracker["trade_id"] == sel_trade
                ].index[0]

                df_tracker.loc[idx, "exit_date"]           = str(datetime.date.today())
                df_tracker.loc[idx, "stock_price_exit"]    = round(exit_stock, 2)
                df_tracker.loc[idx, "call_price_exit"]     = round(exit_call, 2)
                df_tracker.loc[idx, "put_price_exit"]      = round(exit_put, 2)
                df_tracker.loc[idx, "straddle_exit_value"] = round(exit_val, 2)
                df_tracker.loc[idx, "actual_move_pct"]     = round(actual_move, 2)
                df_tracker.loc[idx, "pnl_pct"]             = round(pnl_pct_prev, 2)
                df_tracker.loc[idx, "pnl_usd"]             = round(pnl_usd, 2)
                df_tracker.loc[idx, "outcome"]             = outcome
                df_tracker.loc[idx, "status"]              = "CLOSED"
                df_tracker.loc[idx, "notes"]               = exit_notes
                save_tracker(df_tracker)

                if outcome == "WIN":
                    st.success(
                        f"✅ WIN — {sel_trade} closed | "
                        f"P&L: {pnl_pct_prev:+.2f}% | "
                        f"${pnl_usd:+.2f}"
                    )
                else:
                    st.error(
                        f"❌ LOSS — {sel_trade} closed | "
                        f"P&L: {pnl_pct_prev:+.2f}% | "
                        f"${pnl_usd:+.2f}"
                    )
    else:
        st.info(
            "No open trades. "
            "Use the Scanner tab to find and log a trade."
        )


# ══════════════════════════════════════════════
# TAB 3: PERFORMANCE
# ══════════════════════════════════════════════
with tab3:
    st.subheader("Performance Analytics")

    df_tracker = load_tracker()
    closed     = df_tracker[df_tracker["status"] == "CLOSED"]

    if len(closed) == 0:
        st.info(
            "No closed trades yet. "
            "Complete your first paper trade to see performance."
        )
    else:
        wins   = closed[closed["outcome"] == "WIN"]
        losses = closed[closed["outcome"] == "LOSS"]
        n      = len(closed)

        # Key metrics
        total_ret = closed["pnl_pct"].sum()
        win_rate  = len(wins) / n * 100
        avg_win   = wins["pnl_pct"].mean()   if len(wins)   > 0 else 0
        avg_loss  = losses["pnl_pct"].mean() if len(losses) > 0 else 0
        wl_ratio  = abs(avg_win/avg_loss)     if avg_loss   != 0 else 0

        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Total Return",  f"{total_ret:.1f}%")
        m2.metric("Win Rate",      f"{win_rate:.0f}%")
        m3.metric("Avg Win",       f"+{avg_win:.2f}%")
        m4.metric("Avg Loss",      f"{avg_loss:.2f}%")
        m5.metric("W/L Ratio",     f"{wl_ratio:.2f}x")

        if n >= 4:
            pnls   = closed["pnl_pct"].values.astype(float)
            sharpe = (np.mean(pnls)/np.std(pnls))*np.sqrt(4)
            max_dd = min(np.minimum.accumulate(np.cumsum(pnls)))
            s1, s2 = st.columns(2)
            s1.metric("Sharpe Ratio", f"{sharpe:.2f}",
                      delta="above benchmark" if sharpe > 1.0 else "below benchmark")
            s2.metric("Max Drawdown",  f"{max_dd:.1f}%")

        st.divider()

        # Equity curve
        if n >= 2:
            closed_sorted = closed.sort_values("exit_date")
            pnls_sorted   = closed_sorted["pnl_pct"].values.astype(float)
            cum_pnl       = np.cumsum(pnls_sorted)

            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            fig.patch.set_facecolor("#0e1117")
            for ax in axes:
                ax.set_facecolor("#1e2130")
                ax.tick_params(colors="white")
                ax.title.set_color("white")
                ax.yaxis.label.set_color("white")
                ax.xaxis.label.set_color("white")
                for sp in ax.spines.values():
                    sp.set_edgecolor("#444")

            # Equity curve
            axes[0].plot(
                range(len(cum_pnl)), cum_pnl,
                color="#4c9be8", linewidth=2.5
            )
            axes[0].axhline(0, color="white",
                            linestyle="--", linewidth=0.8)
            axes[0].fill_between(
                range(len(cum_pnl)), cum_pnl, 0,
                alpha=0.15, color="#4c9be8"
            )
            axes[0].set_title(
                "Cumulative P&L — Paper Trades",
                color="white"
            )
            axes[0].set_ylabel("Cumulative Return %", color="white")
            axes[0].set_xlabel("Trade Number", color="white")
            axes[0].grid(color="#333", linewidth=0.5)

            # P&L per trade
            colors_pnl = [
                "#2ecc71" if o == "WIN" else "#e74c3c"
                for o in closed_sorted["outcome"]
            ]
            axes[1].bar(
                range(len(pnls_sorted)), pnls_sorted,
                color=colors_pnl, alpha=0.85
            )
            axes[1].axhline(0, color="white", linewidth=0.8)
            axes[1].set_title(
                "P&L Per Trade",
                color="white"
            )
            axes[1].set_ylabel("Return %", color="white")
            axes[1].set_xlabel("Trade Number", color="white")
            axes[1].grid(color="#333", linewidth=0.5)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.divider()

        # Full trade log
        st.markdown("**Full Trade Log**")
        st.dataframe(
            closed[[
                "trade_id","entry_date","exit_date",
                "cost_pct","actual_move_pct",
                "pnl_pct","pnl_usd","outcome","notes"
            ]],
            use_container_width=True,
            hide_index=True
        )

        csv = closed.to_csv(index=False)
        st.download_button(
            "⬇ Download Trade Log CSV",
            csv,
            "paper_trades.csv",
            "text/csv"
        )
