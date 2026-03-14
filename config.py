import os


class Config:
    # Exchange
    EXCHANGE        = os.getenv("EXCHANGE", "binance")
    SYMBOL          = os.getenv("SYMBOL", "BTC/USDT")
    API_KEY         = os.getenv("API_KEY", "")
    API_SECRET      = os.getenv("API_SECRET", "")

    # Timeframes
    TIMEFRAMES      = os.getenv("TIMEFRAMES", "15m,1h,4h,1d").split(",")
    CANDLE_LIMIT    = int(os.getenv("CANDLE_LIMIT", "300"))

    # Runtime
    RUN_ONCE                = os.getenv("RUN_ONCE", "false").lower() == "true"
    LOOP_INTERVAL_MINUTES   = int(os.getenv("LOOP_INTERVAL_MINUTES", "15"))

    # Risk Management
    RISK_PERCENT        = float(os.getenv("RISK_PERCENT", "1.0"))
    ACCOUNT_BALANCE     = float(os.getenv("ACCOUNT_BALANCE", "10000"))
    TP_LEVELS           = [1.0, 2.0, 3.0]

    # Indicators
    RSI_PERIOD      = int(os.getenv("RSI_PERIOD", "14"))
    RSI_OVERSOLD    = int(os.getenv("RSI_OVERSOLD", "30"))
    RSI_OVERBOUGHT  = int(os.getenv("RSI_OVERBOUGHT", "70"))

    MACD_FAST       = int(os.getenv("MACD_FAST", "12"))
    MACD_SLOW       = int(os.getenv("MACD_SLOW", "26"))
    MACD_SIGNAL     = int(os.getenv("MACD_SIGNAL", "9"))

    EMA_SHORT       = int(os.getenv("EMA_SHORT", "9"))
    EMA_LONG        = int(os.getenv("EMA_LONG", "21"))
    EMA_TREND       = int(os.getenv("EMA_TREND", "200"))

    ATR_PERIOD      = int(os.getenv("ATR_PERIOD", "14"))
    ATR_SL_MULT     = float(os.getenv("ATR_SL_MULT", "1.5"))

    # LSOB
    LSOB_LOOKBACK   = int(os.getenv("LSOB_LOOKBACK", "20"))
    LSOB_VOLUME_MULT = float(os.getenv("LSOB_VOLUME_MULT", "1.5"))

    # GUSS
    GUSS_MIN_GAP_PCT = float(os.getenv("GUSS_MIN_GAP_PCT", "0.3"))

    # Chart
    CHART_OUTPUT_DIR = os.getenv("CHART_OUTPUT_DIR", "/app/charts")
    CHART_CANDLES    = int(os.getenv("CHART_CANDLES", "80"))

    # Quality Filter
    QF_MIN_SCORE    = float(os.getenv("QF_MIN_SCORE",  "40"))   # 0–100
    QF_MIN_RR       = float(os.getenv("QF_MIN_RR",     "1.8"))  # hartes Minimum

    # GUSS Quality Parameter
    GUSS_MAX_CANDLES     = int(os.getenv("GUSS_MAX_CANDLES",   "15"))
    GUSS_EMA_TOUCH_PCT   = float(os.getenv("GUSS_EMA_TOUCH_PCT", "0.003"))
    GUSS_SWING_LOOKBACK  = int(os.getenv("GUSS_SWING_LOOKBACK", "50"))
    GUSS_SL_ATR_MULT     = float(os.getenv("GUSS_SL_ATR_MULT",  "1.8"))

    # Trendline Breakout Detector
    TL_PIVOT_WINDOW  = int(os.getenv("TL_PIVOT_WINDOW",  "5"))
    TL_MIN_PIVOTS    = int(os.getenv("TL_MIN_PIVOTS",    "2"))
    TL_LOOKBACK      = int(os.getenv("TL_LOOKBACK",      "80"))
    TL_VOL_MULT      = float(os.getenv("TL_VOL_MULT",    "1.2"))
    TL_SL_ATR_MULT   = float(os.getenv("TL_SL_ATR_MULT", "1.5"))
