# -*- coding: utf-8 -*-
"""
KRİPTO 15dk Sinyal Tarayıcı — KIVY MOBİL SÜRÜM
════════════════════════════════════════════════════════════════════════
Orijinal masaüstü/terminal (Rich tabanlı) Python scriptinin Android
uyumlu Kivy uyarlamasıdır.

YAPILAN DEĞİŞİKLİKLER (orijinal koda göre):
  1) `pandas` TAMAMEN KALDIRILDI. Android'de pandas'ı buildozer/
     python-for-android ile derlemek son derece kırılgan ve çoğu zaman
     başarısız oluyor. Bütün rolling-mean / rolling-std / EWM / stoch
     hesapları saf NumPy ile yeniden yazıldı (bkz. "NUMPY GÖSTERGE
     YARDIMCILARI" bölümü). Sinyal ÜRETİM MANTIĞI (state machine'ler,
     eşikler, sayaçlar) BİREBİR KORUNDU.
  2) `rich` konsol arayüzü tamamen kaldırıldı; Kivy ekranlarıyla
     değiştirildi (Ayarlar ekranı + Canlı Panel ekranı).
  3) `input()` ile sorulan ayarlar artık dokunmatik form elemanlarıyla
     alınıyor (Spinner / TextInput / CheckBox).
  4) Sesli uyarı (winsound/bell) yerine Android'de basit bir "toast" +
     titreşim (varsa) kullanılıyor; masaüstünde sessizce loglanır.
  5) Kapsam dışı bırakılanlar: orijinal dosyadaki devasa "geçmiş
     backtest paneli" (run_full_backtest / _build_backtest_panel vb.)
     bu sürüme alınmadı — sinyal tarama + sanal (paper) pozisyon
     yönetimi tam olarak çalışıyor, ama geçmiş performans dağılım
     panosu yok. İstenirse ayrı bir ekran olarak eklenebilir.

ÖNEMLİ — BU BİR SİMÜLASYONDUR (PAPER TRADING).
HİÇBİR GERÇEK EMİR BİNANCE'E GÖNDERİLMEZ.
"""

import re
import time
import threading
import queue
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import requests

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.properties import StringProperty, BooleanProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label

try:
    from zoneinfo import ZoneInfo
    TR_TZ = ZoneInfo("Europe/Istanbul")
except Exception:
    TR_TZ = None


def now_tr():
    return datetime.now(TR_TZ) if TR_TZ else datetime.now()


# ═══════════════════════════════════════════════════════════════════════
# NUMPY GÖSTERGE YARDIMCILARI (pandas rolling/EWM karşılıkları)
# ═══════════════════════════════════════════════════════════════════════

def rolling_mean(arr, window):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    result = np.full(n, np.nan)
    if n < window or window <= 0:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(arr, window)
    result[window - 1:] = windows.mean(axis=1)
    return result


def rolling_std(arr, window, ddof=0):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    result = np.full(n, np.nan)
    if n < window or window <= 0:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(arr, window)
    result[window - 1:] = windows.std(axis=1, ddof=ddof)
    return result


def rolling_min(arr, window):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    result = np.full(n, np.nan)
    if n < window or window <= 0:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(arr, window)
    result[window - 1:] = windows.min(axis=1)
    return result


def rolling_max(arr, window):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    result = np.full(n, np.nan)
    if n < window or window <= 0:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(arr, window)
    result[window - 1:] = windows.max(axis=1)
    return result


def ewm_mean(arr, alpha, min_periods=1):
    """pandas .ewm(alpha=alpha, min_periods=min_periods, adjust=False).mean() karşılığı."""
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    result = np.full(n, np.nan)
    prev = None
    valid_count = 0
    for i in range(n):
        x = arr[i]
        if np.isnan(x):
            continue
        prev = x if prev is None else (alpha * x + (1 - alpha) * prev)
        valid_count += 1
        if valid_count >= min_periods:
            result[i] = prev
    return result


def calc_sma(values, length):
    return rolling_mean(values, length)


def calc_bollinger(values, length, mult):
    """(upper, lower) döndürür."""
    arr = np.asarray(values, dtype=float)
    basis = rolling_mean(arr, length)
    dev = mult * rolling_std(arr, length, ddof=0)
    return basis + dev, basis - dev


def calc_bollinger_full(values, length, mult):
    """(basis, upper, lower) döndürür — basis'e ihtiyaç duyan stratejiler için."""
    arr = np.asarray(values, dtype=float)
    basis = rolling_mean(arr, length)
    dev = mult * rolling_std(arr, length, ddof=0)
    return basis, basis + dev, basis - dev


def calculate_stochrsi_series(close_prices, rsi_len=14, stoch_len=14, smooth_k=3, smooth_d=3):
    close = np.asarray(close_prices, dtype=float)
    n = len(close)
    delta = np.full(n, np.nan)
    delta[1:] = close[1:] - close[:-1]

    gain = np.where(np.isnan(delta), np.nan, np.where(delta > 0, delta, 0.0))
    loss = np.where(np.isnan(delta), np.nan, np.where(delta < 0, -delta, 0.0))

    avg_gain = ewm_mean(gain, alpha=1 / rsi_len, min_periods=rsi_len)
    avg_loss = ewm_mean(loss, alpha=1 / rsi_len, min_periods=rsi_len)

    avg_loss_safe = np.where((avg_loss == 0) | np.isnan(avg_loss), 1e-10, avg_loss)
    rs = avg_gain / avg_loss_safe
    rsi = 100 - (100 / (1 + rs))

    lowest_rsi = rolling_min(rsi, stoch_len)
    highest_rsi = rolling_max(rsi, stoch_len)
    denom = highest_rsi - lowest_rsi
    denom_safe = np.where((denom == 0) | np.isnan(denom), 1e-10, denom)
    stoch = (rsi - lowest_rsi) / denom_safe * 100

    k_line = rolling_mean(stoch, smooth_k)
    d_line = rolling_mean(k_line, smooth_d)
    return rsi, k_line, d_line


# ═══════════════════════════════════════════════════════════════════════
# AYARLAR — BELLEK İÇİ (DOSYASIZ) — orijinal koddaki mantıkla aynı
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    "position_size_usd": 100.0,
    "leverage": 1,
    "max_open_positions": 3,
    "margin_mode": "ISOLATED",
    "tp_mode": "dynamic",
    "dynamic_tp_enabled": "open",
    "default_tp_pct": 3.0,
    "fixed_tp_pct": 3.0,
    "min_tp_pct": 0.5,
    "sl_pct": 3.0,
    "enable_orijinal": True,
    "enable_sniper": True,
    "enable_sniper1": True,
    "enable_sniper2": True,
    "market_modes": ["spot", "futures"],
}

CFG = dict(DEFAULT_CONFIG)


def _is_dynamic_tp_open():
    return str(CFG.get("dynamic_tp_enabled", "open")).strip().lower() != "close"


def _resolve_tp_pct(avg_target_pct):
    if not _is_dynamic_tp_open():
        tp = CFG["default_tp_pct"]
    elif CFG["tp_mode"] == "fixed":
        tp = CFG["fixed_tp_pct"]
    else:
        tp = avg_target_pct if (avg_target_pct is not None and avg_target_pct > 0) else CFG["default_tp_pct"]
    return max(CFG["min_tp_pct"], tp)


# ═══════════════════════════════════════════════════════════════════════
# BORSA UÇ NOKTALARI / VERİ ÇEKME
# ═══════════════════════════════════════════════════════════════════════
BINANCE_SPOT_BASE = "https://api.binance.com"
BINANCE_FUTURES_BASE = "https://fapi.binance.com"
MIN_QUOTE_VOLUME_24H = 30_000_000
MIN_SIGNAL_TARGET_PCT = 2.0
EXCLUDE_BASES = {
    "BTC", "ETH", "USDC", "FDUSD", "TUSD", "USDP", "DAI", "PAX", "BUSD", "EUR", "GBP", "TRY",
}
LEVERAGED_SUFFIXES = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Linux; Android) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36",
    "Accept": "application/json",
}


def _fetch_with_retry(url, params=None, retries=3, timeout=15):
    last_err = None
    for _ in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_err = e
            time.sleep(1.2)
    if last_err:
        log_msg(f"İstek başarısız ({url}): {str(last_err)[:80]}")
    return None


def is_valid_altcoin_symbol(sym: str) -> bool:
    if not sym or not re.fullmatch(r"[A-Z0-9]{2,20}USDT", sym):
        return False
    if sym.endswith(LEVERAGED_SUFFIXES):
        return False
    base = sym[:-4]
    if base in EXCLUDE_BASES:
        return False
    return True


def fetch_binance_24hr(market: str):
    url = f"{BINANCE_FUTURES_BASE}/fapi/v1/ticker/24hr" if market == "futures" else f"{BINANCE_SPOT_BASE}/api/v3/ticker/24hr"
    resp = _fetch_with_retry(url)
    if resp is None:
        return []
    try:
        return resp.json()
    except Exception:
        return []


def get_confirmed_movers(market_modes):
    confirmed = {}
    for market in market_modes:
        log_msg(f"↻ Binance {market.upper()} 24s verisi çekiliyor...")
        tickers = fetch_binance_24hr(market)
        if not tickers:
            log_msg(f"⚠ Binance {market.upper()} 24s verisi alınamadı")
            continue
        count = 0
        for t in tickers:
            sym = t.get("symbol", "")
            if not is_valid_altcoin_symbol(sym):
                continue
            try:
                pct = float(t.get("priceChangePercent", 0))
                qvol = float(t.get("quoteVolume", 0))
            except (TypeError, ValueError):
                continue
            if qvol < MIN_QUOTE_VOLUME_24H or pct == 0:
                continue
            key = f"{sym}_{market}"
            confirmed[key] = {
                "sym": sym, "market": market, "binance_pct": pct,
                "quote_volume": qvol, "direction": "UP" if pct > 0 else "DOWN",
            }
            count += 1
        log_msg(f"✓ {market.upper()}: {count} coin bulundu (≥30M hacim)")
    return confirmed


def fetch_klines(symbol: str, market: str, interval: str = "15m", limit: int = 1000):
    """Binance klines -> {'Open','High','Low','Close': np.ndarray} sözlüğü (pandas YOK)."""
    url = f"{BINANCE_FUTURES_BASE}/fapi/v1/klines" if market == "futures" else f"{BINANCE_SPOT_BASE}/api/v3/klines"
    resp = _fetch_with_retry(url, params={"symbol": symbol, "interval": interval, "limit": limit})
    if resp is None:
        return None
    try:
        raw = resp.json()
        if not isinstance(raw, list) or len(raw) < 260:
            return None
        opens = np.array([float(r[1]) for r in raw], dtype=float)
        highs = np.array([float(r[2]) for r in raw], dtype=float)
        lows = np.array([float(r[3]) for r in raw], dtype=float)
        closes = np.array([float(r[4]) for r in raw], dtype=float)
        return {"Open": opens, "High": highs, "Low": lows, "Close": closes}
    except Exception:
        return None


def fetch_15m_data(symbol, market="spot"):
    return fetch_klines(symbol, market, interval="15m", limit=1000)


def get_4h_sma200(symbol: str, market: str = "spot"):
    data = fetch_klines(symbol, market, interval="4h", limit=210)
    if data is None or len(data["Close"]) < 20:
        return None, None
    closes = data["Close"]
    sma200 = float(np.mean(closes[-200:])) if len(closes) >= 200 else float(np.mean(closes))
    return sma200, float(closes[-1])


def _fetch_last_price(symbol, market):
    url = (f"{BINANCE_FUTURES_BASE}/fapi/v1/ticker/price" if market == "futures"
           else f"{BINANCE_SPOT_BASE}/api/v3/ticker/price")
    resp = _fetch_with_retry(url, params={"symbol": symbol}, retries=1, timeout=8)
    if resp is None:
        return None
    try:
        return float(resp.json().get("price"))
    except Exception:
        return None


def _fmt_price(v: float) -> str:
    if v >= 100:
        return f"{v:,.2f}"
    if v >= 1:
        return f"{v:.4f}"
    if v >= 0.01:
        return f"{v:.6f}"
    return f"{v:.8f}"


# ═══════════════════════════════════════════════════════════════════════
# STRATEJİ PARAMETRELERİ — orijinal dosyadan birebir
# ═══════════════════════════════════════════════════════════════════════
DEFAULT_STRATEGY_PARAMS = {
    "bb_length": 20, "bb_mult": 2.0,
    "rsi_len": 14, "stoch_len": 14, "smooth_k": 3, "smooth_d": 3,
    "lookback": 20, "hundred_touch": 3.0, "zero_touch": 3.0,
    "cancel_level_l": 20, "cross_min_l": 50, "cross_cancel_l": 80,
    "cancel_level_s": 20, "cross_min_s": 50, "cross_cancel_s": 80,
    "signal_filter_l": True, "allow1_l": True, "allow2_l": True, "allow3_l": True, "allow4_l": False, "allow5_l": False,
    "signal_filter_s": True, "allow1_s": True, "allow2_s": True, "allow3_s": True, "allow4_s": False, "allow5_s": False,
}
SNIPER_LONG_PARAMS = {
    "sma_len": 200, "rsi_len": 14, "stoch_len": 14, "smooth_k": 3, "smooth_d": 3,
    "move_threshold_pct": 2.0, "look_forward": 20, "bb_length": 20, "bb_mult": 2.0,
    "bb_bars_required": 20, "min_bars_between_signals": 20,
    "long_sl_pct": 3.0, "long_tp_pct": 3.0, "short_sl_pct": 3.0, "short_tp_pct": 3.0, "signal1_only": True,
}
SNIPER1_PARAMS = {
    "sma_len": 200, "rsi_len": 14, "stoch_len": 14, "smooth_k": 3, "smooth_d": 3,
    "lookback_bars": 20, "sma_touch_lookback": 3, "short_block_bars": 200,
    "signal_filter_l": True, "allow1_l": True, "signal_filter_s": True, "allow1_s": True,
}
SNIPER2_PARAMS = {
    "sma_len": 200, "rsi_len": 14, "stoch_len": 14, "smooth_k": 3, "smooth_d": 3,
    "bb_length": 20, "bb_mult": 2.0, "upper_level": 100.0, "lower_level": 0.0,
    "min_dusus_mum": 4, "max_bar_window": 30, "confirm_lookback": 50,
    "max_bars_after_cross": 100, "allow_long": True, "allow_short": True,
}

MAX_LOOKBACK_BARS = 5000


def _build_avg_target(signal_history, sig_type, sig_no, close_last):
    """Verilen sinyal tipi/numarası için geçmiş ortalama hedef hareketi hesaplar
    (orijinal koddaki tekrar eden blok, tek fonksiyona indirgendi)."""
    n_hist = len(signal_history)
    targets = []
    for pos, (sig_i, s_type, entry_price, s_no) in enumerate(signal_history):
        if s_type != sig_type or s_no != sig_no:
            continue
        if entry_price <= 0:
            continue
        opposite_type = "SHORT" if s_type == "LONG" else "LONG"
        exit_price = close_last
        for future_i, future_type, future_price, future_no in signal_history[pos + 1:]:
            if future_type == opposite_type:
                exit_price = future_price
                break
        if s_type == "LONG":
            pct_move = (exit_price - entry_price) / entry_price * 100
        else:
            pct_move = (entry_price - exit_price) / entry_price * 100
        targets.append(pct_move)
    if targets:
        return float(np.mean(targets)), len(targets)
    return None, 0


# ── ORİJİNAL STRATEJİ — 200SMA + BB + StokRSI Long&Short (4H filtreli) ──
def analyze_signal(symbol, params=None, data=None, market="spot"):
    p = params or DEFAULT_STRATEGY_PARAMS
    _empty = (None, None, None)
    try:
        if data is None:
            data = fetch_15m_data(symbol, market)
        if data is None or len(data["Close"]) < 260:
            return _empty

        close = data["Close"]
        n = len(close)
        sma200 = calc_sma(close, 200)
        bb_upper, bb_lower = calc_bollinger(close, p["bb_length"], p["bb_mult"])
        rsi_series, k_series, d_series = calculate_stochrsi_series(
            close, p["rsi_len"], p["stoch_len"], p["smooth_k"], p["smooth_d"]
        )

        sma200_4h, price_4h = get_4h_sma200(symbol, market)
        if sma200_4h is None:
            return _empty
        above_4h = close[-1] > sma200_4h
        below_4h = close[-1] < sma200_4h

        warmup = 200
        start_i = warmup + max(p["stoch_len"], p["smooth_k"] + p["smooth_d"]) + 5
        if n - start_i < p["lookback"] + 5:
            return _empty

        waiting_long = False
        trigger_idx_long = None
        cond6_met_long = False
        above_sma_signal_count = 0
        prev_below_sma200 = None
        waiting_short = False
        trigger_idx_short = None
        cond6_met_short = False
        below_sma_signal_count = 0
        prev_above_sma200 = None
        long_signal_at_last = False
        short_signal_at_last = False
        long_signal_num = 0
        short_signal_num = 0
        signal_history = []

        for i in range(start_i, n):
            if np.isnan(sma200[i]) or np.isnan(bb_upper[i]) or np.isnan(k_series[i]) or np.isnan(d_series[i]):
                continue
            if np.isnan(k_series[i - 1]) or np.isnan(d_series[i - 1]):
                continue

            c = close[i]
            k = k_series[i]
            d = d_series[i]
            k_prev = k_series[i - 1]
            d_prev = d_series[i - 1]
            above_sma200 = c > sma200[i]
            below_sma200 = c < sma200[i]

            touched_100 = (k >= (100 - p["hundred_touch"])) or (d >= (100 - p["hundred_touch"]))
            touched_100_prev = (k_prev >= (100 - p["hundred_touch"])) or (d_prev >= (100 - p["hundred_touch"]))
            hundred_confirm = touched_100_prev and (k < k_prev) and (d < d_prev) and (not touched_100)
            inside_bb = (c <= bb_upper[i]) and (c >= bb_lower[i])
            trigger_bar_l = above_sma200 and inside_bb and hundred_confirm
            if trigger_bar_l:
                trigger_idx_long = i
                waiting_long = True
                cond6_met_long = False
            bullish_cross = (k_prev <= d_prev) and (k > d)
            cross_valid_l = bullish_cross and (k <= p["cross_min_l"]) and (k > (100 - p["cross_cancel_l"]))
            if waiting_long and trigger_idx_long is not None:
                bars_since = i - trigger_idx_long
                if 0 <= bars_since <= p["lookback"]:
                    if cross_valid_l and not cond6_met_long:
                        cond6_met_long = True
                elif bars_since > p["lookback"]:
                    waiting_long = False
                    trigger_idx_long = None
                    cond6_met_long = False
            close_above_upper = c > bb_upper[i]
            entry_cancel_high = (k >= (100 - p["cancel_level_l"])) or (d >= (100 - p["cancel_level_l"]))
            can_enter_long = (
                waiting_long and trigger_idx_long is not None
                and 0 <= (i - trigger_idx_long) <= p["lookback"]
                and close_above_upper and cond6_met_long
                and not entry_cancel_high and above_4h
            )
            crossed_above = above_sma200 and (prev_below_sma200 is True)
            if crossed_above:
                above_sma_signal_count = 0
            if below_sma200:
                above_sma_signal_count = 0
            if can_enter_long and above_sma200:
                above_sma_signal_count += 1
            long_num = above_sma_signal_count
            long_allowed = (not p["signal_filter_l"]) or (
                (long_num == 1 and p["allow1_l"]) or (long_num == 2 and p["allow2_l"]) or
                (long_num == 3 and p["allow3_l"]) or (long_num == 4 and p["allow4_l"]) or
                (long_num == 5 and p["allow5_l"]) or (long_num > 5)
            )
            long_sig = can_enter_long and long_allowed
            if long_sig:
                signal_history.append((i, "LONG", c, long_num))
                waiting_long = False
                trigger_idx_long = None
                cond6_met_long = False
            prev_below_sma200 = below_sma200

            touched_0 = (k <= p["zero_touch"]) or (d <= p["zero_touch"])
            touched_0_prev = (k_prev <= p["zero_touch"]) or (d_prev <= p["zero_touch"])
            zero_confirm = touched_0_prev and (k > k_prev) and (d > d_prev) and (not touched_0)
            trigger_bar_s = below_sma200 and inside_bb and zero_confirm
            if trigger_bar_s:
                trigger_idx_short = i
                waiting_short = True
                cond6_met_short = False
            bearish_cross = (k_prev >= d_prev) and (k < d)
            cross_valid_s = bearish_cross and (k >= p["cross_min_s"]) and (k < p["cross_cancel_s"])
            if waiting_short and trigger_idx_short is not None:
                bars_since = i - trigger_idx_short
                if 0 <= bars_since <= p["lookback"]:
                    if cross_valid_s and not cond6_met_short:
                        cond6_met_short = True
                elif bars_since > p["lookback"]:
                    waiting_short = False
                    trigger_idx_short = None
                    cond6_met_short = False
            close_below_lower = c < bb_lower[i]
            entry_cancel_low = (k <= p["cancel_level_s"]) or (d <= p["cancel_level_s"])
            can_enter_short = (
                waiting_short and trigger_idx_short is not None
                and 0 <= (i - trigger_idx_short) <= p["lookback"]
                and close_below_lower and cond6_met_short
                and not entry_cancel_low and below_4h
            )
            above_sma200_s = c >= sma200[i]
            crossed_below = below_sma200 and (prev_above_sma200 is True)
            if crossed_below:
                below_sma_signal_count = 0
            if above_sma200_s:
                below_sma_signal_count = 0
            if can_enter_short and below_sma200:
                below_sma_signal_count += 1
            short_num = below_sma_signal_count
            short_allowed = (not p["signal_filter_s"]) or (
                (short_num == 1 and p["allow1_s"]) or (short_num == 2 and p["allow2_s"]) or
                (short_num == 3 and p["allow3_s"]) or (short_num == 4 and p["allow4_s"]) or
                (short_num == 5 and p["allow5_s"]) or (short_num > 5)
            )
            short_sig = can_enter_short and short_allowed
            if short_sig:
                signal_history.append((i, "SHORT", c, short_num))
                waiting_short = False
                trigger_idx_short = None
                cond6_met_short = False
            prev_above_sma200 = above_sma200_s

            if i == n - 1:
                long_signal_at_last = long_sig
                short_signal_at_last = short_sig
                long_signal_num = long_num
                short_signal_num = short_num

        signal = None
        confidence = 0
        if long_signal_at_last:
            signal = "LONG"
            confidence = max(0.5, 1 - (k_series[-1] / 100))
        elif short_signal_at_last:
            signal = "SHORT"
            confidence = max(0.5, k_series[-1] / 100)

        current_signal_no = long_signal_num if long_signal_at_last else (short_signal_num if short_signal_at_last else 0)
        avg_long_target_pct, long_hist_count = (None, 0)
        avg_short_target_pct, short_hist_count = (None, 0)
        if long_signal_at_last:
            avg_long_target_pct, long_hist_count = _build_avg_target(signal_history, "LONG", current_signal_no, close[-1])
        if short_signal_at_last:
            avg_short_target_pct, short_hist_count = _build_avg_target(signal_history, "SHORT", current_signal_no, close[-1])

        return signal, confidence, {
            "price": float(close[-1]),
            "avg_long_target_pct": avg_long_target_pct,
            "avg_short_target_pct": avg_short_target_pct,
            "long_hist_count": long_hist_count,
            "short_hist_count": short_hist_count,
            "signal_no": current_signal_no,
        }
    except Exception:
        return _empty


# ── SNIPER LONG/SHORT ──
def analyze_sniper_long(symbol, params=None, data=None, market="spot"):
    p = params or SNIPER_LONG_PARAMS
    _empty = (None, None, None)
    try:
        if data is None:
            data = fetch_15m_data(symbol, market)
        if data is None or len(data["Close"]) < 260:
            return _empty

        close = data["Close"]
        n = len(close)
        look_forward = p["look_forward"]
        move_th = p["move_threshold_pct"]
        bb_bars_required = p["bb_bars_required"]
        min_gap = p["min_bars_between_signals"]

        sma = calc_sma(close, p["sma_len"])
        rsi_series, k_series, d_series = calculate_stochrsi_series(
            close, p["rsi_len"], p["stoch_len"], p["smooth_k"], p["smooth_d"]
        )
        bb_basis, bb_upper, bb_lower = calc_bollinger_full(close, p["bb_length"], p["bb_mult"])

        warmup = max(p["sma_len"], p["bb_length"])
        start_i = warmup + max(p["stoch_len"], p["smooth_k"] + p["smooth_d"]) + look_forward + 5
        if n - start_i < min_gap + 5:
            return _empty

        rise_k_levels, drop_k_levels = [], []
        bars_above_basis_below_upper = 0
        bars_below_basis_above_lower = 0
        bars_since_last_rise = 9999
        bars_since_last_drop = 9999
        long_signal_counter = 0
        short_signal_counter = 0
        current_long_signal_no = 0
        current_short_signal_no = 0
        long_signal_at_last = False
        short_signal_at_last = False
        long_signal_num = 0
        short_signal_num = 0
        signal_history = []

        for i in range(start_i, n):
            if np.isnan(sma[i]) or np.isnan(k_series[i]) or np.isnan(bb_basis[i]):
                continue
            past_i = i - look_forward
            if past_i < 0 or np.isnan(sma[past_i]) or np.isnan(k_series[past_i]):
                continue

            c = close[i]
            k = k_series[i]
            k_prev = k_series[i - 1] if i > 0 and not np.isnan(k_series[i - 1]) else None
            past_close = close[past_i]
            past_sma = sma[past_i]
            past_k = k_series[past_i]
            pct_change = (c - past_close) / past_close * 100 if past_close else 0.0
            is_uptrend = past_close > past_sma
            is_downtrend = past_close < past_sma
            is_big_rise = (pct_change >= move_th) and is_uptrend
            is_big_drop = (pct_change <= -move_th) and is_downtrend
            if is_big_rise and not np.isnan(past_k):
                rise_k_levels.append(past_k)
            if is_big_drop and not np.isnan(past_k):
                drop_k_levels.append(past_k)
            rise_avg = float(np.mean(rise_k_levels)) if rise_k_levels else None
            drop_avg = float(np.mean(drop_k_levels)) if drop_k_levels else None
            above_basis_below_upper = (c > bb_basis[i]) and (c < bb_upper[i])
            below_basis_above_lower = (c < bb_basis[i]) and (c > bb_lower[i])
            bars_above_basis_below_upper = (bars_above_basis_below_upper + 1) if above_basis_below_upper else 0
            bars_below_basis_above_lower = (bars_below_basis_above_lower + 1) if below_basis_above_lower else 0
            bb_rise_condition = bars_above_basis_below_upper >= bb_bars_required
            bb_drop_condition = bars_below_basis_above_lower >= bb_bars_required
            current_uptrend = c > sma[i]
            current_downtrend = c < sma[i]
            k_cross_up_rise = (k_prev is not None and rise_avg is not None and k_prev <= rise_avg and k > rise_avg)
            k_cross_down_drop = (k_prev is not None and drop_avg is not None and k_prev >= drop_avg and k < drop_avg)
            rise_signal_raw = rise_avg is not None and current_uptrend and k_cross_up_rise and bb_rise_condition
            drop_signal_raw = drop_avg is not None and current_downtrend and k_cross_down_drop and bb_drop_condition
            bars_since_last_rise += 1
            bars_since_last_drop += 1
            rise_signal = rise_signal_raw and bars_since_last_rise > min_gap
            drop_signal = drop_signal_raw and bars_since_last_drop > min_gap
            if rise_signal:
                bars_since_last_rise = 0
            if drop_signal:
                bars_since_last_drop = 0
            if i > 0 and not np.isnan(sma[i - 1]):
                sma_cross_up = (close[i - 1] <= sma[i - 1]) and (c > sma[i])
                sma_cross_down = (close[i - 1] >= sma[i - 1]) and (c < sma[i])
            else:
                sma_cross_up = sma_cross_down = False
            if sma_cross_up:
                long_signal_counter = 0
            if sma_cross_down:
                short_signal_counter = 0
            if rise_signal:
                long_signal_counter += 1
                current_long_signal_no = long_signal_counter
            if drop_signal:
                short_signal_counter += 1
                current_short_signal_no = short_signal_counter
            rise_signal_final = rise_signal and (current_long_signal_no == 1)
            drop_signal_final = drop_signal and (current_short_signal_no == 1)
            if rise_signal_final:
                signal_history.append((i, "LONG", c, current_long_signal_no))
            if drop_signal_final:
                signal_history.append((i, "SHORT", c, current_short_signal_no))
            if i == n - 1:
                long_signal_at_last = rise_signal_final
                short_signal_at_last = drop_signal_final
                long_signal_num = current_long_signal_no
                short_signal_num = current_short_signal_no

        signal = "LONG" if long_signal_at_last else ("SHORT" if short_signal_at_last else None)
        if signal == "LONG":
            confidence = max(0.5, 1 - (k_series[-1] / 100))
        elif signal == "SHORT":
            confidence = max(0.5, k_series[-1] / 100)
        else:
            confidence = 0

        avg_target_pct, hist_count, signal_no = None, 0, 0
        if signal == "LONG":
            signal_no = long_signal_num
            avg_target_pct, hist_count = _build_avg_target(signal_history, "LONG", signal_no, close[-1])
        elif signal == "SHORT":
            signal_no = short_signal_num
            avg_target_pct, hist_count = _build_avg_target(signal_history, "SHORT", signal_no, close[-1])

        return signal, confidence, {
            "price": float(close[-1]), "signal_no": signal_no,
            "avg_target_pct": avg_target_pct, "hist_count": hist_count,
        }
    except Exception:
        return _empty


# ── SNIPER 1 (State-Machine) ──
def analyze_sniper1(symbol, params=None, data=None, market="spot"):
    p = params or SNIPER1_PARAMS
    _empty = (None, None, None)
    try:
        if data is None:
            data = fetch_15m_data(symbol, market)
        if data is None or len(data["Close"]) < 260:
            return _empty

        open_ = data["Open"]
        high = data["High"]
        low = data["Low"]
        close = data["Close"]
        n = len(close)

        sma200 = calc_sma(close, p["sma_len"])
        rsi_series, k_series, d_series = calculate_stochrsi_series(
            close, p["rsi_len"], p["stoch_len"], p["smooth_k"], p["smooth_d"]
        )
        lookback_bars = p["lookback_bars"]
        sma_touch_lb = p["sma_touch_lookback"]
        short_block_bars = p["short_block_bars"]
        warmup = p["sma_len"]
        start_i = warmup + max(p["stoch_len"], p["smooth_k"] + p["smooth_d"]) + sma_touch_lb + 6
        if n - start_i < lookback_bars + 5:
            return _empty

        state_l = 0
        state_bar_l = 0
        sequence_ready_l = False
        state_s = 0
        state_bar_s = 0
        sequence_ready_s = False
        above_sma_signal_count = 0
        below_sma_signal_count = 0
        last_cross_below_sma_bar = None
        long_signal_at_last = False
        short_signal_at_last = False
        long_signal_num = 0
        short_signal_num = 0
        signal_history = []

        for i in range(start_i, n):
            if (np.isnan(sma200[i]) or np.isnan(k_series[i]) or np.isnan(d_series[i])
                    or np.isnan(k_series[i - 1]) or np.isnan(d_series[i - 1])):
                continue

            c = close[i]
            o = open_[i]
            k = k_series[i]
            d = d_series[i]
            price_above_sma = c > sma200[i]
            price_below_sma = c < sma200[i]

            touched_sma_recently = False
            for j in range(0, sma_touch_lb + 1):
                idx = i - j
                if idx < 0 or np.isnan(sma200[idx]):
                    continue
                if low[idx] <= sma200[idx] <= high[idx]:
                    touched_sma_recently = True
                    break

            below_count = 0
            for j in range(1, 6):
                idx = i - j
                if idx >= 0 and not np.isnan(sma200[idx]) and close[idx] < sma200[idx]:
                    below_count += 1
            max_two_below_sma = below_count < 2

            touched_100 = k >= 100
            touched_0 = k <= 0
            cross_over_any = (k_series[i - 1] <= d_series[i - 1]) and (k > d)
            cross_under_any = (k_series[i - 1] >= d_series[i - 1]) and (k < d)
            cross_over_20 = cross_over_any and k < 20 and d < 20
            cross_under_20 = cross_under_any and k < 20
            k_above_20 = k > 20
            cross_under_80 = cross_under_any and k > 80 and d > 80
            cross_over_80 = cross_over_any and k > 80
            k_below_80 = k < 80

            long_entry_raw = False
            short_entry_raw = False

            if state_l > 0 and (i - state_bar_l) > lookback_bars:
                state_l = 0
                sequence_ready_l = False
            if state_l == 0 and touched_100:
                state_l = 1
                state_bar_l = i
            if state_l == 1 and cross_over_20:
                state_l = 2
                state_bar_l = i
            if state_l == 2:
                if k_above_20:
                    state_l = 0
                elif cross_under_20:
                    state_l = 3
                    state_bar_l = i
            if state_l == 3:
                if k_above_20:
                    state_l = 0
                elif touched_0:
                    last4_bearish = all(close[i - m] < open_[i - m] for m in range(0, 4) if i - m >= 0) and (i - 3 >= 0)
                    state_l = 4 if last4_bearish else 0
                    if last4_bearish:
                        state_bar_l = i
            if state_l == 4 and state_bar_l == i:
                sequence_ready_l = True
            if sequence_ready_l:
                if (i - state_bar_l) > lookback_bars:
                    sequence_ready_l = False
                    state_l = 0
                elif c > o and cross_over_any:
                    long_entry_raw = price_above_sma and touched_sma_recently and max_two_below_sma
                    sequence_ready_l = False
                    state_l = 0

            if state_s > 0 and (i - state_bar_s) > lookback_bars:
                state_s = 0
                sequence_ready_s = False
            if state_s == 0 and touched_0:
                state_s = 1
                state_bar_s = i
            if state_s == 1 and cross_under_80:
                state_s = 2
                state_bar_s = i
            if state_s == 2:
                if k_below_80:
                    state_s = 0
                elif cross_over_80:
                    state_s = 3
                    state_bar_s = i
            if state_s == 3:
                if k_below_80:
                    state_s = 0
                elif touched_100:
                    last4_bullish = all(close[i - m] > open_[i - m] for m in range(0, 4) if i - m >= 0) and (i - 3 >= 0)
                    state_s = 4 if last4_bullish else 0
                    if last4_bullish:
                        state_bar_s = i
            if state_s == 4 and state_bar_s == i:
                sequence_ready_s = True
            if sequence_ready_s:
                if (i - state_bar_s) > lookback_bars:
                    sequence_ready_s = False
                    state_s = 0
                elif c < o and cross_under_any:
                    short_entry_raw = price_below_sma
                    sequence_ready_s = False
                    state_s = 0

            below_sma200_now = c <= sma200[i]
            prev_below_sma200 = close[i - 1] <= sma200[i - 1] if not np.isnan(sma200[i - 1]) else False
            crossed_above_sma = price_above_sma and prev_below_sma200
            if crossed_above_sma:
                above_sma_signal_count = 0
            if below_sma200_now:
                above_sma_signal_count = 0
            if long_entry_raw and price_above_sma:
                above_sma_signal_count += 1
            current_long_no = above_sma_signal_count
            long_allowed = (not p["signal_filter_l"]) or (current_long_no == 1 and p["allow1_l"])
            long_sig = long_entry_raw and long_allowed

            above_sma200_now = c >= sma200[i]
            prev_above_sma200 = close[i - 1] >= sma200[i - 1] if not np.isnan(sma200[i - 1]) else False
            crossed_below_sma = price_below_sma and prev_above_sma200
            if crossed_below_sma:
                below_sma_signal_count = 0
            if above_sma200_now:
                below_sma_signal_count = 0
            if short_entry_raw and price_below_sma:
                below_sma_signal_count += 1
            current_short_no = below_sma_signal_count
            short_allowed = (not p["signal_filter_s"]) or (current_short_no == 1 and p["allow1_s"])

            crossed_below_sma200 = (close[i - 1] >= sma200[i - 1]) and (c < sma200[i]) if not np.isnan(sma200[i - 1]) else False
            if crossed_below_sma200:
                last_cross_below_sma_bar = i
            short_blocked_by_cooldown = (
                last_cross_below_sma_bar is not None and (i - last_cross_below_sma_bar) > short_block_bars
            )
            short_sig = short_entry_raw and short_allowed and not short_blocked_by_cooldown

            if long_sig:
                signal_history.append((i, "LONG", c, current_long_no))
            if short_sig:
                signal_history.append((i, "SHORT", c, current_short_no))
            if i == n - 1:
                long_signal_at_last = long_sig
                short_signal_at_last = short_sig
                long_signal_num = current_long_no
                short_signal_num = current_short_no

        signal = "LONG" if long_signal_at_last else ("SHORT" if short_signal_at_last else None)
        if signal == "LONG":
            confidence = max(0.5, 1 - (k_series[-1] / 100))
        elif signal == "SHORT":
            confidence = max(0.5, k_series[-1] / 100)
        else:
            confidence = 0

        avg_target_pct, hist_count, signal_no = None, 0, 0
        if signal == "LONG":
            signal_no = long_signal_num
            avg_target_pct, hist_count = _build_avg_target(signal_history, "LONG", signal_no, close[-1])
        elif signal == "SHORT":
            signal_no = short_signal_num
            avg_target_pct, hist_count = _build_avg_target(signal_history, "SHORT", signal_no, close[-1])

        return signal, confidence, {
            "price": float(close[-1]), "signal_no": signal_no,
            "avg_target_pct": avg_target_pct, "hist_count": hist_count,
        }
    except Exception:
        return _empty


# ── SNIPER 2 (Streak / Trigger-Confirm) ──
def analyze_sniper2(symbol, params=None, data=None, market="spot"):
    p = params or SNIPER2_PARAMS
    _empty = (None, None, None)
    try:
        if data is None:
            data = fetch_15m_data(symbol, market)
        if data is None or len(data["Close"]) < 260:
            return _empty

        open_ = data["Open"]
        close = data["Close"]
        n = len(close)

        sma_line = calc_sma(close, p["sma_len"])
        rsi_series, k_series, d_series = calculate_stochrsi_series(
            close, p["rsi_len"], p["stoch_len"], p["smooth_k"], p["smooth_d"]
        )
        bb_basis, bb_upper, bb_lower = calc_bollinger_full(close, p["bb_length"], p["bb_mult"])

        upper_level = p["upper_level"]
        lower_level = p["lower_level"]
        min_streak = p["min_dusus_mum"]
        max_bar_window = p["max_bar_window"]
        confirm_lookback = p["confirm_lookback"]
        max_bars_after_cross = p["max_bars_after_cross"]
        allow_long = p["allow_long"]
        allow_short = p["allow_short"]

        warmup = max(p["sma_len"], p["bb_length"])
        start_i = warmup + max(p["stoch_len"], p["smooth_k"] + p["smooth_d"]) + 5
        if n - start_i < max_bar_window + 5:
            return _empty

        bear_streak = 0
        bull_streak = 0
        bars_since_upper_touch = 9999
        bars_since_lower_touch = 9999
        bars_since_cross_up = 9999
        bars_since_cross_down = 9999
        long_first_trigger_price = None
        long_first_trigger_bar = None
        short_first_trigger_price = None
        short_first_trigger_bar = None
        above_sma_signal_count = 0
        below_sma_signal_count = 0
        long_signal_at_last = False
        short_signal_at_last = False
        long_signal_num = 0
        short_signal_num = 0
        signal_history = []

        for i in range(start_i, n):
            if (np.isnan(sma_line[i]) or np.isnan(k_series[i]) or np.isnan(bb_basis[i])
                    or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i])):
                continue

            c = close[i]
            o = open_[i]
            k = k_series[i]
            is_bear = c < o
            is_bull = c > o
            bear_streak = bear_streak + 1 if is_bear else 0
            bull_streak = bull_streak + 1 if is_bull else 0
            above_sma = c > sma_line[i]
            below_sma = c < sma_line[i]
            inside_lower_band = (c <= bb_lower[i]) or (bb_lower[i] < c < bb_basis[i])
            inside_upper_band = (c >= bb_upper[i]) or (bb_basis[i] < c < bb_upper[i])
            stoch_upper_touch = k >= upper_level
            stoch_lower_touch = k <= lower_level
            bars_since_upper_touch = 0 if stoch_upper_touch else bars_since_upper_touch + 1
            bars_since_lower_touch = 0 if stoch_lower_touch else bars_since_lower_touch + 1
            long_upper_confirmed = bars_since_upper_touch <= confirm_lookback
            short_lower_confirmed = bars_since_lower_touch <= confirm_lookback

            if i > 0 and not np.isnan(sma_line[i - 1]):
                cross_up_sma = (close[i - 1] <= sma_line[i - 1]) and (c > sma_line[i])
                cross_down_sma = (close[i - 1] >= sma_line[i - 1]) and (c < sma_line[i])
            else:
                cross_up_sma = cross_down_sma = False
            bars_since_cross_up = 0 if cross_up_sma else bars_since_cross_up + 1
            bars_since_cross_down = 0 if cross_down_sma else bars_since_cross_down + 1
            long_cross_block = bars_since_cross_up > max_bars_after_cross
            short_cross_block = bars_since_cross_down > max_bars_after_cross

            long_first_trigger = above_sma and long_upper_confirmed and stoch_lower_touch and is_bear and not long_cross_block
            short_first_trigger = below_sma and short_lower_confirmed and stoch_upper_touch and is_bull and not short_cross_block
            if long_first_trigger:
                long_first_trigger_price = c
                long_first_trigger_bar = i
            if short_first_trigger:
                short_first_trigger_price = c
                short_first_trigger_bar = i

            long_in_window = long_first_trigger_bar is not None and 0 < (i - long_first_trigger_bar) <= max_bar_window
            short_in_window = short_first_trigger_bar is not None and 0 < (i - short_first_trigger_bar) <= max_bar_window
            long_second_confirm = (
                long_in_window and stoch_lower_touch and is_bear and (bear_streak >= min_streak)
                and long_first_trigger_price is not None and c > long_first_trigger_price
                and not long_cross_block and inside_lower_band
            )
            short_second_confirm = (
                short_in_window and stoch_upper_touch and is_bull and (bull_streak >= min_streak)
                and short_first_trigger_price is not None and c < short_first_trigger_price
                and not short_cross_block and inside_upper_band
            )

            if long_first_trigger_bar is not None and (i - long_first_trigger_bar) > max_bar_window:
                long_first_trigger_bar = None
                long_first_trigger_price = None
            if short_first_trigger_bar is not None and (i - short_first_trigger_bar) > max_bar_window:
                short_first_trigger_bar = None
                short_first_trigger_price = None

            final_long_signal = long_second_confirm and allow_long
            final_short_signal = short_second_confirm and allow_short
            if long_second_confirm:
                long_first_trigger_bar = None
                long_first_trigger_price = None
            if short_second_confirm:
                short_first_trigger_bar = None
                short_first_trigger_price = None

            below_sma_now = c <= sma_line[i]
            crossed_above_sma = above_sma and (i > 0 and not np.isnan(sma_line[i - 1]) and close[i - 1] <= sma_line[i - 1])
            if crossed_above_sma:
                above_sma_signal_count = 0
            if below_sma_now:
                above_sma_signal_count = 0
            if final_long_signal and above_sma:
                above_sma_signal_count += 1
            current_long_no = above_sma_signal_count

            above_sma_now = c >= sma_line[i]
            crossed_below_sma = below_sma and (i > 0 and not np.isnan(sma_line[i - 1]) and close[i - 1] >= sma_line[i - 1])
            if crossed_below_sma:
                below_sma_signal_count = 0
            if above_sma_now:
                below_sma_signal_count = 0
            if final_short_signal and below_sma:
                below_sma_signal_count += 1
            current_short_no = below_sma_signal_count

            if final_long_signal:
                signal_history.append((i, "LONG", c, current_long_no))
            if final_short_signal:
                signal_history.append((i, "SHORT", c, current_short_no))
            if i == n - 1:
                long_signal_at_last = final_long_signal
                short_signal_at_last = final_short_signal
                long_signal_num = current_long_no
                short_signal_num = current_short_no

        signal = "LONG" if long_signal_at_last else ("SHORT" if short_signal_at_last else None)
        if signal == "LONG":
            confidence = max(0.5, 1 - (k_series[-1] / 100))
        elif signal == "SHORT":
            confidence = max(0.5, k_series[-1] / 100)
        else:
            confidence = 0

        avg_target_pct, hist_count, signal_no = None, 0, 0
        if signal == "LONG":
            signal_no = long_signal_num
            avg_target_pct, hist_count = _build_avg_target(signal_history, "LONG", signal_no, close[-1])
        elif signal == "SHORT":
            signal_no = short_signal_num
            avg_target_pct, hist_count = _build_avg_target(signal_history, "SHORT", signal_no, close[-1])

        return signal, confidence, {
            "price": float(close[-1]), "signal_no": signal_no,
            "avg_target_pct": avg_target_pct, "hist_count": hist_count,
        }
    except Exception:
        return _empty


def analyze_symbol_all(symbol: str, market: str):
    results = []
    data = fetch_klines(symbol, market, interval="15m", limit=1000)
    if data is None:
        return results

    if CFG["enable_orijinal"]:
        try:
            signal, conf, details = analyze_signal(symbol, data=data, market=market)
            if signal:
                results.append({"sym": symbol, "strategy": "ORIJINAL", "signal": signal, "confidence": conf, "details": details or {}})
        except Exception:
            pass
    if CFG["enable_sniper"]:
        try:
            signal, conf, details = analyze_sniper_long(symbol, data=data, market=market)
            if signal:
                results.append({"sym": symbol, "strategy": "SNIPER_LONG", "signal": signal, "confidence": conf, "details": details or {}})
        except Exception:
            pass
    if CFG["enable_sniper1"]:
        try:
            signal, conf, details = analyze_sniper1(symbol, data=data, market=market)
            if signal:
                results.append({"sym": symbol, "strategy": "SNIPER_1", "signal": signal, "confidence": conf, "details": details or {}})
        except Exception:
            pass
    if CFG["enable_sniper2"]:
        try:
            signal, conf, details = analyze_sniper2(symbol, data=data, market=market)
            if signal:
                results.append({"sym": symbol, "strategy": "SNIPER_2", "signal": signal, "confidence": conf, "details": details or {}})
        except Exception:
            pass
    return results


# ═══════════════════════════════════════════════════════════════════════
# PAPER TRADING (SANAL POZİSYONLAR)
# ═══════════════════════════════════════════════════════════════════════
OPEN_POSITIONS = {}
CLOSED_POSITIONS = deque(maxlen=200)
POSITIONS_LOCK = threading.Lock()


def try_open_paper_position(sig_key, sig):
    with POSITIONS_LOCK:
        if sig_key in OPEN_POSITIONS:
            return
        sym = sig["sym"]
        market = sig.get("market", "spot")
        for pos in OPEN_POSITIONS.values():
            if pos["sym"] == sym and pos["market"] == market:
                return
        if len(OPEN_POSITIONS) >= CFG["max_open_positions"]:
            return
        entry_price = sig.get("price", 0)
        if not entry_price or entry_price <= 0:
            return
        direction = sig["signal"]
        tp_pct = _resolve_tp_pct(sig.get("avg_target_pct"))
        sl_pct = CFG["sl_pct"]
        if direction == "LONG":
            tp_price = entry_price * (1 + tp_pct / 100)
            sl_price = entry_price * (1 - sl_pct / 100)
        else:
            tp_price = entry_price * (1 - tp_pct / 100)
            sl_price = entry_price * (1 + sl_pct / 100)
        qty = (CFG["position_size_usd"] * CFG["leverage"]) / entry_price
        OPEN_POSITIONS[sig_key] = {
            "sym": sym, "market": market, "strategy": sig.get("strategy", "?"),
            "direction": direction, "entry_price": entry_price, "entry_time": now_tr(),
            "qty": qty, "position_usd": CFG["position_size_usd"], "leverage": CFG["leverage"],
            "tp_pct": tp_pct, "sl_pct": sl_pct, "tp_price": tp_price, "sl_price": sl_price,
            "last_price": entry_price, "pnl_pct": 0.0, "pnl_usd": 0.0,
        }
        log_msg(f"🟢 SANAL GİRİŞ: {sym} {direction} @ {_fmt_price(entry_price)} (TP %{tp_pct:.2f} / SL %{sl_pct:.2f})")


def update_paper_positions():
    with POSITIONS_LOCK:
        keys = list(OPEN_POSITIONS.keys())
    for key in keys:
        with POSITIONS_LOCK:
            pos = OPEN_POSITIONS.get(key)
        if pos is None:
            continue
        price = _fetch_last_price(pos["sym"], pos["market"])
        if price is None:
            continue
        entry = pos["entry_price"]
        if pos["direction"] == "LONG":
            pnl_pct = (price - entry) / entry * 100 * pos["leverage"]
        else:
            pnl_pct = (entry - price) / entry * 100 * pos["leverage"]
        pnl_usd = pos["position_usd"] * (pnl_pct / 100)
        hit_tp = (price >= pos["tp_price"]) if pos["direction"] == "LONG" else (price <= pos["tp_price"])
        hit_sl = (price <= pos["sl_price"]) if pos["direction"] == "LONG" else (price >= pos["sl_price"])
        with POSITIONS_LOCK:
            if key not in OPEN_POSITIONS:
                continue
            OPEN_POSITIONS[key]["last_price"] = price
            OPEN_POSITIONS[key]["pnl_pct"] = pnl_pct
            OPEN_POSITIONS[key]["pnl_usd"] = pnl_usd
            if hit_tp or hit_sl:
                closed = dict(OPEN_POSITIONS[key])
                closed["exit_price"] = price
                closed["exit_time"] = now_tr()
                closed["result"] = "TP" if hit_tp else "SL"
                CLOSED_POSITIONS.append(closed)
                del OPEN_POSITIONS[key]
                sonuc = "✅ TP" if hit_tp else "🛑 SL"
                log_msg(f"{sonuc}: {closed['sym']} {closed['direction']} kapandı @ {_fmt_price(price)} (P&L: {pnl_pct:+.2f}% / {pnl_usd:+.2f}$)")


# ═══════════════════════════════════════════════════════════════════════
# ORTAK DURUM / LOG KUYRUĞU (arka plan iş parçacığı -> Kivy ana iş parçacığı)
# ═══════════════════════════════════════════════════════════════════════
STOP_EVENT = threading.Event()
SIGNALS = {}
SIGNALS_LOCK = threading.Lock()
LOG_QUEUE = queue.Queue()
LAST_UPDATE = None
NEXT_SCAN_TIME = None
SCAN_IN_PROGRESS = False


def log_msg(message: str):
    ts = now_tr().strftime("%H:%M:%S")
    LOG_QUEUE.put(f"[{ts}] {message}")


def _seconds_until_next_candle_close(interval_minutes=15, buffer_seconds=20):
    now = now_tr()
    minute_block = (now.minute // interval_minutes + 1) * interval_minutes
    if minute_block >= 60:
        next_close = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1, minutes=minute_block - 60))
    else:
        next_close = now.replace(minute=minute_block, second=0, microsecond=0)
    wait = (next_close - now).total_seconds() + buffer_seconds
    return max(30, wait)


def background_scanner():
    global SIGNALS, LAST_UPDATE, NEXT_SCAN_TIME, SCAN_IN_PROGRESS
    first_run = True
    import concurrent.futures

    while not STOP_EVENT.is_set():
        try:
            SCAN_IN_PROGRESS = True
            log_msg("🔄 Binance yükselen/düşen listesi kontrol ediliyor..." if first_run else "↻ Liste yenileniyor...")
            confirmed_movers = get_confirmed_movers(CFG["market_modes"])

            if confirmed_movers:
                log_msg(f"📊 {len(confirmed_movers)} onaylı coin taranıyor (15dk)...")
                with SIGNALS_LOCK:
                    previous_keys = set(SIGNALS.keys())
                new_signals = {}
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    futures = {
                        executor.submit(analyze_symbol_all, info["sym"], info["market"]): (key, info)
                        for key, info in confirmed_movers.items()
                    }
                    for future in concurrent.futures.as_completed(futures):
                        key, info = futures[future]
                        if STOP_EVENT.is_set():
                            break
                        try:
                            sym_results = future.result()
                        except Exception:
                            continue
                        for r in sym_results:
                            strategy = r["strategy"]
                            signal = r["signal"]
                            details = r["details"] or {}
                            sig_key = f"{key}__{strategy}"
                            if strategy == "ORIJINAL" and signal == "LONG":
                                avg_target = details.get("avg_long_target_pct")
                            elif strategy == "ORIJINAL" and signal == "SHORT":
                                avg_target = details.get("avg_short_target_pct")
                            else:
                                avg_target = details.get("avg_target_pct")
                            if avg_target is None or avg_target < MIN_SIGNAL_TARGET_PCT:
                                continue
                            new_signals[sig_key] = {
                                "sym": info["sym"], "market": info["market"], "strategy": strategy,
                                "signal": signal, "time": now_tr().strftime("%H:%M:%S"),
                                "confidence": r["confidence"], "price": details.get("price", 0),
                                "avg_target_pct": avg_target,
                            }
                            try_open_paper_position(sig_key, new_signals[sig_key])

                with SIGNALS_LOCK:
                    SIGNALS = new_signals
                LAST_UPDATE = now_tr()
                up = sum(1 for s in new_signals.values() if s["signal"] == "LONG")
                down = sum(1 for s in new_signals.values() if s["signal"] == "SHORT")
                log_msg(f"✓ Tarama tamamlandı: {up} LONG, {down} SHORT")
            else:
                log_msg("✗ Binance listesinden coin bulunamadı!")
                with SIGNALS_LOCK:
                    SIGNALS = {}

            first_run = False
            SCAN_IN_PROGRESS = False
            wait_seconds = _seconds_until_next_candle_close()
            NEXT_SCAN_TIME = now_tr() + timedelta(seconds=wait_seconds)
            log_msg(f"⏳ Sıradaki tarama: {NEXT_SCAN_TIME.strftime('%H:%M:%S')}")
            for _ in range(int(wait_seconds)):
                if STOP_EVENT.is_set():
                    break
                time.sleep(1)
        except Exception as e:
            SCAN_IN_PROGRESS = False
            log_msg(f"✗ Tarama hatası: {str(e)[:100]}")
            time.sleep(30)


def position_watcher():
    while not STOP_EVENT.is_set():
        try:
            update_paper_positions()
        except Exception as e:
            log_msg(f"✗ Pozisyon güncelleme hatası: {str(e)[:80]}")
        time.sleep(5)


# ═══════════════════════════════════════════════════════════════════════
# KIVY ARAYÜZÜ
# ═══════════════════════════════════════════════════════════════════════
KV = """
ScreenManager:
    SettingsScreen:
    DashboardScreen:

<SettingsScreen>:
    name: "settings"
    BoxLayout:
        orientation: "vertical"
        padding: dp(16)
        spacing: dp(10)
        canvas.before:
            Color:
                rgba: 0.07, 0.08, 0.10, 1
            Rectangle:
                pos: self.pos
                size: self.size

        Label:
            text: "🪙 Kripto Sinyal Tarayıcı"
            font_size: "22sp"
            bold: True
            size_hint_y: None
            height: dp(40)

        Label:
            text: "⚠ SİMÜLASYON — gerçek emir gönderilmez"
            color: 1, 0.8, 0.2, 1
            size_hint_y: None
            height: dp(24)
            font_size: "13sp"

        ScrollView:
            BoxLayout:
                orientation: "vertical"
                size_hint_y: None
                height: self.minimum_height
                spacing: dp(12)
                padding: dp(4)

                Label:
                    text: "Piyasa"
                    size_hint_y: None
                    height: dp(22)
                    halign: "left"
                    text_size: self.size
                Spinner:
                    id: market_spinner
                    text: "İkisi de (Spot+Futures)"
                    values: ["Sadece Spot", "Sadece Futures", "İkisi de (Spot+Futures)"]
                    size_hint_y: None
                    height: dp(44)

                Label:
                    text: "Pozisyon Büyüklüğü (USD)"
                    size_hint_y: None
                    height: dp(22)
                    halign: "left"
                    text_size: self.size
                TextInput:
                    id: position_usd
                    text: "100"
                    input_filter: "float"
                    multiline: False
                    size_hint_y: None
                    height: dp(44)

                Label:
                    text: "Kaldıraç (x)"
                    size_hint_y: None
                    height: dp(22)
                    halign: "left"
                    text_size: self.size
                TextInput:
                    id: leverage
                    text: "1"
                    input_filter: "int"
                    multiline: False
                    size_hint_y: None
                    height: dp(44)

                Label:
                    text: "Maksimum Açık Pozisyon"
                    size_hint_y: None
                    height: dp(22)
                    halign: "left"
                    text_size: self.size
                TextInput:
                    id: max_positions
                    text: "3"
                    input_filter: "int"
                    multiline: False
                    size_hint_y: None
                    height: dp(44)

                Label:
                    text: "Aktif Stratejiler"
                    size_hint_y: None
                    height: dp(28)
                    bold: True
                    halign: "left"
                    text_size: self.size

                BoxLayout:
                    size_hint_y: None
                    height: dp(40)
                    CheckBox:
                        id: cb_orijinal
                        active: True
                        size_hint_x: None
                        width: dp(40)
                    Label:
                        text: "ORİJİNAL"
                        halign: "left"
                        text_size: self.size

                BoxLayout:
                    size_hint_y: None
                    height: dp(40)
                    CheckBox:
                        id: cb_sniper
                        active: True
                        size_hint_x: None
                        width: dp(40)
                    Label:
                        text: "SNIPER LONG-SHORT"
                        halign: "left"
                        text_size: self.size

                BoxLayout:
                    size_hint_y: None
                    height: dp(40)
                    CheckBox:
                        id: cb_sniper1
                        active: True
                        size_hint_x: None
                        width: dp(40)
                    Label:
                        text: "SNIPER 1 (State-Machine)"
                        halign: "left"
                        text_size: self.size

                BoxLayout:
                    size_hint_y: None
                    height: dp(40)
                    CheckBox:
                        id: cb_sniper2
                        active: True
                        size_hint_x: None
                        width: dp(40)
                    Label:
                        text: "SNIPER 2 (Streak)"
                        halign: "left"
                        text_size: self.size

                Label:
                    text: "Kâr Al / Zarar Durdur Modu"
                    size_hint_y: None
                    height: dp(28)
                    bold: True
                    halign: "left"
                    text_size: self.size
                Spinner:
                    id: tp_mode_spinner
                    text: "Dinamik TP"
                    values: ["Dinamik TP", "Sabit % TP/SL"]
                    size_hint_y: None
                    height: dp(44)

                BoxLayout:
                    size_hint_y: None
                    height: dp(44)
                    Label:
                        text: "Sabit TP %"
                        text_size: self.size
                        halign: "left"
                    TextInput:
                        id: fixed_tp
                        text: "3.0"
                        input_filter: "float"
                        multiline: False

                BoxLayout:
                    size_hint_y: None
                    height: dp(44)
                    Label:
                        text: "SL %"
                        text_size: self.size
                        halign: "left"
                    TextInput:
                        id: sl_pct
                        text: "3.0"
                        input_filter: "float"
                        multiline: False

        Button:
            text: "▶ Taramayı Başlat"
            size_hint_y: None
            height: dp(52)
            font_size: "18sp"
            background_color: 0.2, 0.6, 0.9, 1
            on_release: root.start_scanning()

<DashboardScreen>:
    name: "dashboard"
    BoxLayout:
        orientation: "vertical"
        padding: dp(10)
        spacing: dp(6)
        canvas.before:
            Color:
                rgba: 0.07, 0.08, 0.10, 1
            Rectangle:
                pos: self.pos
                size: self.size

        BoxLayout:
            size_hint_y: None
            height: dp(36)
            Label:
                id: status_label
                text: "Başlatılıyor..."
                font_size: "13sp"
                halign: "left"
                text_size: self.size
            Button:
                text: "Durdur"
                size_hint_x: None
                width: dp(90)
                background_color: 0.8, 0.3, 0.3, 1
                on_release: root.stop_scanning()

        Label:
            text: "📶 AKTİF SİNYALLER"
            bold: True
            size_hint_y: None
            height: dp(26)

        ScrollView:
            size_hint_y: 0.35
            BoxLayout:
                id: signals_box
                orientation: "vertical"
                size_hint_y: None
                height: self.minimum_height
                spacing: dp(4)

        Label:
            text: "💼 AÇIK SANAL POZİSYONLAR"
            bold: True
            size_hint_y: None
            height: dp(26)

        ScrollView:
            size_hint_y: 0.25
            BoxLayout:
                id: positions_box
                orientation: "vertical"
                size_hint_y: None
                height: self.minimum_height
                spacing: dp(4)

        Label:
            text: "📝 GÜNLÜK"
            bold: True
            size_hint_y: None
            height: dp(26)

        ScrollView:
            id: log_scroll
            size_hint_y: 0.3
            BoxLayout:
                id: log_box
                orientation: "vertical"
                size_hint_y: None
                height: self.minimum_height
                spacing: dp(2)
"""


class SignalRow(BoxLayout):
    def __init__(self, sig, **kwargs):
        super().__init__(orientation="horizontal", size_hint_y=None, height=34, **kwargs)
        color = (0.3, 0.85, 0.4, 1) if sig["signal"] == "LONG" else (0.9, 0.3, 0.3, 1)
        txt = f"{sig['sym']} · {sig['strategy']} · {sig['signal']} @ {_fmt_price(sig['price'])} (hedef ~%{sig['avg_target_pct']:.1f})"
        lbl = Label(text=txt, color=color, font_size="12sp", halign="left", valign="middle")
        lbl.bind(size=lambda inst, val: setattr(inst, "text_size", val))
        self.add_widget(lbl)


class PositionRow(BoxLayout):
    def __init__(self, pos, **kwargs):
        super().__init__(orientation="horizontal", size_hint_y=None, height=34, **kwargs)
        color = (0.3, 0.85, 0.4, 1) if pos["pnl_pct"] >= 0 else (0.9, 0.3, 0.3, 1)
        txt = (f"{pos['sym']} {pos['direction']} @ {_fmt_price(pos['entry_price'])} "
               f"→ {_fmt_price(pos['last_price'])} | {pos['pnl_pct']:+.2f}% ({pos['pnl_usd']:+.2f}$)")
        lbl = Label(text=txt, color=color, font_size="12sp", halign="left", valign="middle")
        lbl.bind(size=lambda inst, val: setattr(inst, "text_size", val))
        self.add_widget(lbl)


class LogRow(Label):
    def __init__(self, text, **kwargs):
        super().__init__(text=text, size_hint_y=None, height=20, font_size="11sp",
                          color=(0.75, 0.78, 0.8, 1), halign="left", valign="middle", **kwargs)
        self.bind(size=lambda inst, val: setattr(inst, "text_size", val))


class SettingsScreen(Screen):
    def start_scanning(self):
        ids = self.ids
        market_txt = ids.market_spinner.text
        if market_txt == "Sadece Spot":
            CFG["market_modes"] = ["spot"]
        elif market_txt == "Sadece Futures":
            CFG["market_modes"] = ["futures"]
        else:
            CFG["market_modes"] = ["spot", "futures"]

        try:
            CFG["position_size_usd"] = float(ids.position_usd.text.replace(",", ".") or 100)
        except ValueError:
            CFG["position_size_usd"] = 100.0
        try:
            CFG["leverage"] = int(ids.leverage.text or 1)
        except ValueError:
            CFG["leverage"] = 1
        try:
            CFG["max_open_positions"] = int(ids.max_positions.text or 3)
        except ValueError:
            CFG["max_open_positions"] = 3

        CFG["enable_orijinal"] = ids.cb_orijinal.active
        CFG["enable_sniper"] = ids.cb_sniper.active
        CFG["enable_sniper1"] = ids.cb_sniper1.active
        CFG["enable_sniper2"] = ids.cb_sniper2.active

        if ids.tp_mode_spinner.text == "Dinamik TP":
            CFG["dynamic_tp_enabled"] = "open"
            CFG["tp_mode"] = "dynamic"
        else:
            CFG["dynamic_tp_enabled"] = "open"
            CFG["tp_mode"] = "fixed"
            try:
                CFG["fixed_tp_pct"] = float(ids.fixed_tp.text.replace(",", ".") or 3.0)
                CFG["default_tp_pct"] = CFG["fixed_tp_pct"]
            except ValueError:
                pass
        try:
            CFG["sl_pct"] = float(ids.sl_pct.text.replace(",", ".") or 3.0)
        except ValueError:
            CFG["sl_pct"] = 3.0

        app = App.get_running_app()
        app.start_background_threads()
        self.manager.current = "dashboard"


class DashboardScreen(Screen):
    def on_enter(self):
        Clock.schedule_interval(self.refresh, 1.0)

    def on_leave(self):
        Clock.unschedule(self.refresh)

    def stop_scanning(self):
        STOP_EVENT.set()
        self.manager.current = "settings"

    def refresh(self, dt):
        # Log kuyruğunu boşalt
        log_box = self.ids.log_box
        while not LOG_QUEUE.empty():
            try:
                msg = LOG_QUEUE.get_nowait()
            except queue.Empty:
                break
            log_box.add_widget(LogRow(msg))
            if len(log_box.children) > 100:
                log_box.remove_widget(log_box.children[-1])
        self.ids.log_scroll.scroll_y = 0

        # Sinyaller
        signals_box = self.ids.signals_box
        signals_box.clear_widgets()
        with SIGNALS_LOCK:
            sigs = list(SIGNALS.values())
        if not sigs:
            signals_box.add_widget(LogRow("Şu an aktif sinyal yok."))
        else:
            for s in sigs[:40]:
                signals_box.add_widget(SignalRow(s))

        # Pozisyonlar
        positions_box = self.ids.positions_box
        positions_box.clear_widgets()
        with POSITIONS_LOCK:
            positions = list(OPEN_POSITIONS.values())
        if not positions:
            positions_box.add_widget(LogRow("Açık sanal pozisyon yok."))
        else:
            for p in positions:
                positions_box.add_widget(PositionRow(p))

        # Durum satırı
        status = "🟢 Taranıyor" if SCAN_IN_PROGRESS else "⏳ Bekliyor"
        if NEXT_SCAN_TIME:
            status += f" · Sıradaki: {NEXT_SCAN_TIME.strftime('%H:%M:%S')}"
        self.ids.status_label.text = status


class KriptoApp(App):
    def build(self):
        Window.clearcolor = (0.07, 0.08, 0.10, 1)
        return Builder.load_string(KV)

    def start_background_threads(self):
        STOP_EVENT.clear()
        threading.Thread(target=background_scanner, daemon=True).start()
        threading.Thread(target=position_watcher, daemon=True).start()

    def on_stop(self):
        STOP_EVENT.set()


if __name__ == "__main__":
    KriptoApp().run()
