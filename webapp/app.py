"""
Trading Bot Dashboard — Flask Backend
"""

import os
import re
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, jsonify, send_file, render_template, abort, request

# Bot-Code-Pfad für MacroAnalyzer-Import
BOT_DIR = os.getenv("BOT_DIR", "/app")
if BOT_DIR not in sys.path:
    sys.path.insert(0, BOT_DIR)

app = Flask(__name__)

CHARTS_DIR     = Path(os.getenv("CHART_OUTPUT_DIR", "/app/charts"))
RECENT_MINUTES = int(os.getenv("RECENT_MINUTES", "60"))

FILENAME_RE = re.compile(
    r"^(?P<date>\d{8})_(?P<time>\d{6})_(?P<tf>[\w]+)_(?P<dir>LONG|SHORT)_(?P<type>.+)\.png$"
)

SIGNAL_META = {
    "LSOB_LONG":           {"label": "Liquidity Sweep",  "category": "Smart Money", "color": "#f0c040"},
    "LSOB_SHORT":          {"label": "Liquidity Sweep",  "category": "Smart Money", "color": "#f0c040"},
    "DIV_BULL_RSI":        {"label": "RSI Divergenz",    "category": "Divergenz",   "color": "#4fc3f7"},
    "DIV_BEAR_RSI":        {"label": "RSI Divergenz",    "category": "Divergenz",   "color": "#4fc3f7"},
    "DIV_BULL_MACD_HIST":  {"label": "MACD Divergenz",   "category": "Divergenz",   "color": "#81d4fa"},
    "DIV_BEAR_MACD_HIST":  {"label": "MACD Divergenz",   "category": "Divergenz",   "color": "#81d4fa"},
    "GUSS_LONG":           {"label": "GUSS Long",        "category": "GUSS",        "color": "#a5d6a7"},
    "GUSS_SHORT":          {"label": "GUSS Short",       "category": "GUSS",        "color": "#ef9a9a"},
    "GUSS_LONG_GAP_UP":    {"label": "Gap Up Zone",      "category": "GUSS",        "color": "#a5d6a7"},
    "GUSS_SHORT_GAP_DOWN": {"label": "Gap Down Zone",    "category": "GUSS",        "color": "#ef9a9a"},
    "GUSS_LONG_FVG_BULL":  {"label": "Fair Value Gap",   "category": "GUSS",        "color": "#ce93d8"},
    "GUSS_SHORT_FVG_BEAR": {"label": "Fair Value Gap",   "category": "GUSS",        "color": "#ce93d8"},
    "EMA_GOLDEN_CROSS":    {"label": "Golden Cross",     "category": "EMA Cross",   "color": "#ffcc80"},
    "EMA_DEATH_CROSS":     {"label": "Death Cross",      "category": "EMA Cross",   "color": "#ff8a65"},
    "TL_BREAKOUT_LONG":   {"label": "TL Breakout",     "category": "Trendlinie",  "color": "#26a69a"},
    "TL_BREAKOUT_SHORT":  {"label": "TL Breakout",     "category": "Trendlinie",  "color": "#ef5350"},
}

GRADE_COLORS = {"S": "#f0c040", "A": "#26a69a", "B": "#4fc3f7", "C": "#ff9800", "F": "#ef5350"}


def load_score(png_path: Path):
    json_path = png_path.with_suffix(".json")
    if not json_path.exists():
        return None
    try:
        with open(json_path) as f:
            return json.load(f)
    except Exception:
        return None


def parse_chart(filename: str):
    m = FILENAME_RE.match(filename)
    if not m:
        return None

    d, t = m.group("date"), m.group("time")
    sig  = m.group("type")
    meta = SIGNAL_META.get(sig, {"label": sig, "category": "Other", "color": "#90a4ae"})

    try:
        dt = datetime.strptime(d + t, "%Y%m%d%H%M%S")
    except ValueError:
        return None

    file_path = CHARTS_DIR / filename
    size_kb   = round(file_path.stat().st_size / 1024, 1) if file_path.exists() else 0
    score_data = load_score(file_path)

    result = {
        "filename":  filename,
        "timestamp": dt.strftime("%d.%m.%Y %H:%M"),
        "ts_iso":    dt.isoformat(),
        "ts_date":   dt.strftime("%Y-%m-%d"),
        "ts_hour":   dt.strftime("%H"),
        "dt":        dt,
        "timeframe": m.group("tf"),
        "direction": m.group("dir"),
        "signal":    sig,
        "label":     meta["label"],
        "category":  meta["category"],
        "color":     meta["color"],
        "size_kb":   size_kb,
        "score":     None,
    }

    if score_data:
        result["score"] = {
            "grade":        score_data.get("grade", "?"),
            "total":        score_data.get("total_score", 0),
            "max":          score_data.get("max_score", 100),
            "pct":          score_data.get("pct", 0),
            "grade_color":  GRADE_COLORS.get(score_data.get("grade", ""), "#90a4ae"),
            "scores":       score_data.get("scores", {}),
            "reasons_pass": score_data.get("reasons_pass", []),
            "reasons_fail": score_data.get("reasons_fail", []),
            "trade":        score_data.get("trade", {}),
        }

    return result


def get_all_charts():
    if not CHARTS_DIR.exists():
        return []
    charts = []
    for f in sorted(CHARTS_DIR.glob("*.png"), reverse=True):
        parsed = parse_chart(f.name)
        if parsed:
            charts.append(parsed)
    return charts


def strip_dt(charts):
    out = []
    for c in charts:
        d = dict(c)
        d.pop("dt", None)
        out.append(d)
    return out


def build_stats(charts):
    if not charts:
        return {"total": 0, "long": 0, "short": 0, "by_tf": {}, "by_category": {},
                "latest": None, "avg_score": None, "grade_dist": {}}
    by_tf, by_cat, scores, grades = {}, {}, [], {}
    for c in charts:
        by_tf[c["timeframe"]]  = by_tf.get(c["timeframe"], 0) + 1
        by_cat[c["category"]]  = by_cat.get(c["category"], 0) + 1
        if c.get("score"):
            scores.append(c["score"]["pct"])
            g = c["score"]["grade"]
            grades[g] = grades.get(g, 0) + 1
    return {
        "total":       len(charts),
        "long":        sum(1 for c in charts if c["direction"] == "LONG"),
        "short":       sum(1 for c in charts if c["direction"] == "SHORT"),
        "by_tf":       dict(sorted(by_tf.items())),
        "by_category": dict(sorted(by_cat.items(), key=lambda x: -x[1])),
        "latest":      charts[0]["timestamp"] if charts else None,
        "avg_score":   round(sum(scores) / len(scores), 1) if scores else None,
        "grade_dist":  grades,
    }


# ── Chart API ──────────────────────────────────────────────────────────────

@app.route("/api/charts/recent")
def api_recent():
    cutoff = datetime.now() - timedelta(minutes=RECENT_MINUTES)
    all_c  = get_all_charts()
    recent = [c for c in all_c if c["dt"] >= cutoff]
    return jsonify({"charts": strip_dt(recent), "stats": build_stats(recent),
                    "cutoff_minutes": RECENT_MINUTES})


@app.route("/api/charts/archive")
def api_archive():
    cutoff  = datetime.now() - timedelta(minutes=RECENT_MINUTES)
    all_c   = get_all_charts()
    archive = [c for c in all_c if c["dt"] < cutoff]
    days    = sorted({c["ts_date"] for c in archive}, reverse=True)
    hours   = sorted({c["ts_hour"] for c in archive}, reverse=True)
    date_f  = request.args.get("date", "")
    hour_f  = request.args.get("hour", "")
    if date_f:
        archive = [c for c in archive if c["ts_date"] == date_f]
    if hour_f:
        archive = [c for c in archive if c["ts_hour"] == hour_f]
    return jsonify({
        "charts":          strip_dt(archive),
        "stats":           build_stats(archive),
        "available_days":  days,
        "available_hours": hours,
        "filter_date":     date_f,
        "filter_hour":     hour_f,
    })


@app.route("/api/charts/<filename>")
def api_chart_image(filename):
    path = CHARTS_DIR / filename
    if not path.exists() or not filename.endswith(".png"):
        abort(404)
    return send_file(str(path), mimetype="image/png")


# ── Makro-Analyse ──────────────────────────────────────────────────────────

_macro_analyzer = None

def _get_macro_analyzer():
    global _macro_analyzer
    if _macro_analyzer is None:
        try:
            from config import Config
            from analysis.macro_analyzer import MacroAnalyzer
            _macro_analyzer = MacroAnalyzer(Config())
        except Exception as e:
            return None, str(e)
    return _macro_analyzer, None


@app.route("/api/macro")
def api_macro():
    force    = request.args.get("force", "0") == "1"
    analyzer, err = _get_macro_analyzer()
    if analyzer is None:
        return jsonify({"error": f"Analyzer nicht verfügbar: {err}"}), 500
    try:
        data = analyzer.get_analysis(force=force)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/macro")
def macro_page():
    return render_template("macro.html")


# ── Pages ──────────────────────────────────────────────────────────────────


# ── Liquiditäts-Analyse ────────────────────────────────────────────────────

_liquidity_analyzer = None

def _get_liquidity_analyzer():
    global _liquidity_analyzer
    if _liquidity_analyzer is None:
        try:
            from config import Config
            from analysis.liquidity_analyzer import LiquidityAnalyzer
            _liquidity_analyzer = LiquidityAnalyzer(Config())
        except Exception as e:
            return None, str(e)
    return _liquidity_analyzer, None


@app.route("/api/liquidity")
def api_liquidity():
    force    = request.args.get("force", "0") == "1"
    analyzer, err = _get_liquidity_analyzer()
    if analyzer is None:
        return jsonify({"error": f"Analyzer nicht verfügbar: {err}"}), 500
    try:
        data = analyzer.get_analysis(force=force)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/liquidity")
def liquidity_page():
    return render_template("liquidity.html")

@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
