
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
import requests
import datetime
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Earnings Straddle Analyser",
    page_icon="📈",
    layout="wide"
)

# ── Main Core functions usinh black scholes 
def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    try:
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    except:
        return 0.0

def get_next_trading_day(date, all_days):
    future = [d for d in all_days if d > date]
    return future[0] if future else None

def get_prev_trading_day(date, all_days):
    past = [d for d in all_days if d < date]
    return past[-1] if past else None

def calculate_hv(df, date, window=30):
    past = [d for d in df.index if d < date]
    if len(past) < window:
        return None
    prices  = df.loc[past[-window:], "Close"]
    returns = prices.pct_change().dropna()
    if len(returns) < 5:
        return None
    return returns.std() * np.sqrt(252)

def fetch_earnings_dates(ticker, api_key):
    url = (
        "https://www.alphavantage.co/query"
        f"?function=EARNINGS&symbol={ticker}&apikey={api_key}"
    )
    try:
        resp = requests.get(url, timeout=20)
        data = resp.json()
        if "quarterlyEarnings" not in data:
            return None, f"API error: {data}"
        earnings = []
        for item in data["quarterlyEarnings"][:16]:
            d = item.get("reportedDate")
            if d and d != "None":
                earnings.append({"report_date": d, "timing": "AMC"})
        earnings = sorted(earnings, key=lambda x: x["report_date"])
        return earnings, None
    except Exception as e:
        return None, str(e)

def load_price_data(ticker):
    try:
        df = yf.Ticker(ticker).history(period="5y")

        if df.empty or len(df) < 100:
            return None

        df.index = pd.to_datetime(df.index)

        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)

        df.index = pd.Index([
            datetime.date(d.year, d.month, d.day)
            for d in df.index
        ])

        return df

    except Exception as e:
        st.error(f"Price data error: {e}")
        return None
def run_backtest(earnings_list, df, iv_mult, iv_crush,
                 iv_max, contract_days, sl_pct):
    trading_days = list(df.index)
    trades = []
    T_entry = contract_days / 365
    T_exit  = (contract_days - 1) / 365
    r = 0.05

    for earning in earnings_list:
        report_date = datetime.date.fromisoformat(earning["report_date"])
        timing = earning["timing"]
        try:
            if timing == "BMO":
                impact_day   = report_date
                baseline_day = get_prev_trading_day(report_date, trading_days)
            else:
                impact_day   = get_next_trading_day(report_date, trading_days)
                baseline_day = report_date

            if (impact_day is None or baseline_day is None or
                impact_day not in df.index or
                baseline_day not in df.index):
                continue

            S       = float(df.loc[baseline_day, "Close"])
            S_close = float(df.loc[impact_day,   "Close"])
            if S <= 0:
                continue

            full_move = ((S_close - S) / S) * 100
            hv = calculate_hv(df, baseline_day)
            if hv is None or hv <= 0:
                continue

            iv_before = min(hv * iv_mult,  2.0)
            iv_after  = min(hv * iv_crush, 1.5)
            K = S

            call_e = black_scholes_price(S, K, T_entry, r, iv_before, "call")
            put_e  = black_scholes_price(S, K, T_entry, r, iv_before, "put")
            if call_e <= 0 or put_e <= 0:
                continue

            S_exit = S * (1 + full_move / 100)
            call_x = black_scholes_price(S_exit, K, T_exit, r, iv_after, "call")
            put_x  = black_scholes_price(S_exit, K, T_exit, r, iv_after, "put")

            SL_VAL = S * sl_pct
            if full_move > 0:
                win_pnl   = call_x - call_e
                lose_loss = min(put_e,  SL_VAL)
            else:
                win_pnl   = put_x  - put_e
                lose_loss = min(call_e, SL_VAL)

            net_pnl_pct  = ((win_pnl - lose_loss) / S) * 100
            implied_move = (iv_before * np.sqrt(T_entry)) * 100
            edge         = abs(full_move) - implied_move

            trades.append({
                "report_date":   report_date,
                "iv_before_pct": round(iv_before * 100, 2),
                "cost_pct":      round(((call_e + put_e) / S) * 100, 2),
                "implied_move":  round(implied_move,   2),
                "actual_move":   round(full_move,      2),
                "abs_move":      round(abs(full_move), 2),
                "edge":          round(edge,           2),
                "net_pnl_pct":   round(net_pnl_pct,   2),
                "outcome":       "WIN" if net_pnl_pct > 0 else "LOSS",
            })
        except Exception:
            continue

    if len(trades) < 4:
        return None

    trades_df = pd.DataFrame(trades)
    filtered  = trades_df[trades_df["iv_before_pct"] < iv_max].copy()
    if len(filtered) < 3:
        filtered = trades_df.copy()
    filtered["cumulative_pnl"] = filtered["net_pnl_pct"].cumsum()
    return filtered


# ── Page layout ────────────────────────────────────────────────
st.title("📈 Earnings Straddle Strategy Analyser")
st.caption(
    "Enter your stock ticker and Alpha Vantage API key "
    "to backtest the earnings straddle strategy with "
    "real verified earnings dates."
)

col_settings, col_results = st.columns([1, 3])
with col_settings:
    st.subheader("Start")

    api_key = st.text_input(
        "Alpha Vantage API Key",
        placeholder="Paste your free key here",
        type="password"
    )

    ticker_input = st.text_input(
        "Stock Ticker",
        value="META",
        placeholder="e.g. META, SNAP, TSLA"
    ).upper().strip()

    with st.expander("Advanced strategy settings"):
        iv_multiplier = st.number_input(
            "IV Premium Factor",
            min_value=1.0, max_value=2.5, value=1.5, step=0.1
        )

        iv_crush_factor = st.number_input(
            "IV Crush Factor",
            min_value=0.5, max_value=1.2, value=0.9, step=0.1
        )

        iv_filter = st.number_input(
            "Max IV Filter (%)",
            min_value=50, max_value=150, value=75, step=5
        )

        contract_days = st.number_input(
            "Contract Days to Expiry",
            min_value=7, max_value=30, value=14, step=1
        )

        sl_pct_input = st.number_input(
            "Stop Loss on Losing Leg (%)",
            min_value=1.0, max_value=5.0, value=2.0, step=0.5
        )

    sl_pct = sl_pct_input / 100

    run_button = st.button(
        "Run Analysis",
        use_container_width=True,
        type="primary"
    )

# ── Results column ─────────────────────────────────────────────
with col_results:

    if not run_button:
        st.info(
            "👈  Fill in your API key and ticker, "
            "then click **Run Analysis** to begin."
        )
        st.markdown("### How it works")
        h1, h2, h3, h4 = st.columns(4)
        h1.info("**1. Your API key**\nFree from Alpha Vantage — takes 60 seconds to get")
        h2.info("**2. Enter ticker**\nAny US stock e.g. META, SNAP, TSLA, NFLX")
        h3.info("**3. Backtest runs**\nBlack-Scholes pricing, IV filter, stop loss logic")
        h4.info("**4. Get signal**\nTrade or skip the next earnings event")

    if run_button:

        if not api_key:
            st.error("Please enter your Alpha Vantage API key in the settings panel.")
            st.stop()

        if not ticker_input:
            st.error("Please enter a stock ticker.")
            st.stop()

        # ── Fetch earnings dates ──────────────────────────────
        with st.spinner(
            f"Fetching verified earnings dates for "
            f"{ticker_input} from Alpha Vantage..."
        ):
            earnings, error = fetch_earnings_dates(ticker_input, api_key)

        if error or not earnings:
            st.error(f"Could not fetch earnings dates: {error}")
            st.info(
                "Check your API key is correct. "
                "Get a free key at alphavantage.co/support/#api-key"
            )
            st.stop()

        st.success(
            f"✅  {len(earnings)} verified earnings dates "
            f"loaded from Alpha Vantage"
        )

        with st.expander("View verified earnings dates"):
            st.dataframe(
                pd.DataFrame(earnings),
                use_container_width=True
            )

        # ── Load price data ───────────────────────────────────
        with st.spinner("Loading 5 years of price data from Yahoo Finance..."):
            df = load_price_data(ticker_input)

        if df is None:
            st.error(
                "Could not load price data. "
                "Check the ticker symbol is correct."
            )
            st.stop()

        # ── Run backtest ──────────────────────────────────────
        with st.spinner("Running backtest..."):
            trades_df = run_backtest(
                earnings_list = earnings,
                df            = df,
                iv_mult       = float(iv_multiplier),
                iv_crush      = float(iv_crush_factor),
                iv_max        = float(iv_filter),
                contract_days = int(contract_days),
                sl_pct        = sl_pct
            )

        if trades_df is None:
            st.error(
                "Not enough valid trades found. "
                "Try increasing the IV Filter or check the ticker."
            )
            st.stop()

        # ── Calculate stats ───────────────────────────────────
        wins      = trades_df[trades_df["outcome"] == "WIN"]
        losses    = trades_df[trades_df["outcome"] == "LOSS"]
        win_rate  = len(wins) / len(trades_df) * 100
        avg_win   = wins["net_pnl_pct"].mean()   if len(wins)   > 0 else 0
        avg_loss  = losses["net_pnl_pct"].mean() if len(losses) > 0 else 0
        wl_ratio  = abs(avg_win / avg_loss)       if avg_loss   != 0 else 9.99
        total_ret = trades_df["net_pnl_pct"].sum()
        beat_rate = (
            trades_df["abs_move"] > trades_df["implied_move"]
        ).mean() * 100
        best_removed = total_ret - trades_df["net_pnl_pct"].max()

        # ── Next earnings signal ──────────────────────────────
        next_date   = earnings[-1]["report_date"]
        last_hv     = calculate_hv(df, list(df.index)[-1])
        next_iv_est = (
            last_hv * float(iv_multiplier) * 100
        ) if last_hv else None

        st.divider()
        st.subheader(f"Results — {ticker_input}")
        st.markdown(f"### Next Earnings: `{next_date}`")

        if next_iv_est and next_iv_est < iv_filter:
            st.success(
                f"✅  TRADE — Estimated IV before earnings "
                f"({next_iv_est:.1f}%) is below your filter "
                f"({iv_filter}%). Strategy edge likely present."
            )
        elif next_iv_est:
            st.warning(
                f"⚠  SKIP — Estimated IV before earnings "
                f"({next_iv_est:.1f}%) exceeds your filter "
                f"({iv_filter}%). Options may be too expensive."
            )
        else:
            st.info("Could not estimate IV for next earnings.")

        st.divider()

        # ── Key metrics row ───────────────────────────────────
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("Total Return",    f"{total_ret:.1f}%")
        m2.metric("Win Rate",        f"{win_rate:.1f}%")
        m3.metric("Avg Win",         f"+{avg_win:.2f}%")
        m4.metric("Avg Loss",        f"{avg_loss:.2f}%")
        m5.metric("W/L Ratio",       f"{wl_ratio:.2f}x")
        m6.metric("Beat Rate",       f"{beat_rate:.1f}%")

        st.divider()

        # ── IV Regime table ───────────────────────────────────
        st.subheader("IV Regime Analysis")
        regime_rows = []
        for label, subset in [
            ("Low IV  < 50%",
             trades_df[trades_df["iv_before_pct"] < 50]),
            ("Mid IV 50–75%",
             trades_df[(trades_df["iv_before_pct"] >= 50) &
                       (trades_df["iv_before_pct"] <  75)]),
            ("High IV > 75%",
             trades_df[trades_df["iv_before_pct"] >= 75]),
        ]:
            if len(subset) > 0:
                w  = subset[subset["outcome"] == "WIN"]
                wr = len(w) / len(subset) * 100
                regime_rows.append({
                    "Regime":   label,
                    "Trades":   len(subset),
                    "Win Rate": f"{wr:.0f}%",
                    "Avg PnL":  f"{subset['net_pnl_pct'].mean():.2f}%",
                    "Signal":   "✅ Trade" if wr > 50 else "⚠  Caution"
                })
        st.dataframe(
            pd.DataFrame(regime_rows),
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # ── Charts ────────────────────────────────────────────
        st.subheader("Charts")

        fig, axes = plt.subplots(3, 1, figsize=(12, 14))
        fig.patch.set_facecolor("#0e1117")
        for ax in axes:
            ax.set_facecolor("#1e2130")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")

        fig.suptitle(
            f"{ticker_input} — Earnings Straddle Backtest",
            fontsize=14, fontweight="bold", color="white"
        )

        dates_str = [str(d) for d in trades_df["report_date"]]
        x  = list(range(len(trades_df)))
        bw = 0.35

        # Chart 1: Actual vs implied move
        axes[0].bar(
            [i - bw/2 for i in x], trades_df["abs_move"],
            width=bw, color="#4c9be8", alpha=0.85,
            label="Actual Move %"
        )
        axes[0].plot(
            x, trades_df["implied_move"],
            color="#ff6b6b", linewidth=2,
            marker="o", markersize=5,
            label="Implied Move % (your breakeven)"
        )
        axes[0].set_title("Actual Move vs Implied Move (Breakeven Line)")
        axes[0].set_ylabel("Move %", color="white")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(
            dates_str, rotation=45, ha="right", fontsize=7
        )
        axes[0].legend(facecolor="#1e2130", labelcolor="white")
        axes[0].grid(axis="y", color="#333", linewidth=0.5)

        # Chart 2: P&L per trade
        pnl_colors = [
            "#2ecc71" if o == "WIN" else "#e74c3c"
            for o in trades_df["outcome"]
        ]
        axes[1].bar(x, trades_df["net_pnl_pct"],
                    color=pnl_colors, alpha=0.85)
        axes[1].axhline(0, color="white", linewidth=0.8)
        axes[1].set_title("P&L Per Trade (green = win, red = loss)")
        axes[1].set_ylabel("Return %", color="white")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(
            dates_str, rotation=45, ha="right", fontsize=7
        )
        axes[1].grid(axis="y", color="#333", linewidth=0.5)

        # Chart 3: Equity curve
        axes[2].plot(
            x, trades_df["cumulative_pnl"],
            color="#4c9be8", linewidth=2.5
        )
        axes[2].axhline(0, color="white", linestyle="--", linewidth=0.8)
        axes[2].fill_between(
            x, trades_df["cumulative_pnl"],
            0, alpha=0.15, color="#4c9be8"
        )
        axes[2].set_title("Cumulative P&L — Equity Curve")
        axes[2].set_ylabel("Cumulative Return %", color="white")
        axes[2].set_xlabel("Trade Number", color="white")
        axes[2].grid(axis="y", color="#333", linewidth=0.5)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.divider()

        # ── Robustness check ──────────────────────────────────
        st.subheader("Robustness Check")
        r1, r2, r3 = st.columns(3)
        r1.metric(
            "Return after removing best trade",
            f"{best_removed:.1f}%",
            help="If still positive, edge is not just one lucky trade"
        )
        r2.metric(
            "Total trades analysed",
            f"{len(trades_df)}"
        )
        r3.metric(
            "Avg edge per trade",
            f"{trades_df['edge'].mean():.2f}%",
            help="Actual move minus implied move — your structural edge"
        )

        st.divider()

        # ── Full trade log ────────────────────────────────────
        st.subheader("Full Trade Log")
        display_df = trades_df[[
            "report_date", "iv_before_pct", "cost_pct",
            "implied_move", "actual_move", "edge",
            "net_pnl_pct", "outcome"
        ]].copy()
        display_df.columns = [
            "Date", "IV Before %", "Cost %",
            "Implied Move", "Actual Move", "Edge",
            "Net P&L %", "Outcome"
        ]
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # ── Download ──────────────────────────────────────────
        csv = display_df.to_csv(index=False)
        st.download_button(
            label     = "⬇  Download Trade Log as CSV",
            data      = csv,
            file_name = f"{ticker_input}_straddle_backtest.csv",
            mime      = "text/csv"
        )
