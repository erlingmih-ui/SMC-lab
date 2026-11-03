from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class Zone:
    kind: str                      # "OB" ou "FVG"
    side: int                      # 1 haussier, -1 baissier
    bottom: float
    top: float
    start: int                     # bougie à partir de laquelle la zone est connue
    end: int | None = None         # bougie de mitigation, None si active
    event_index: int | None = None


@dataclass
class Event:
    index: int
    label: str                     # "BOS" ou "CHoCH"
    side: int
    level: float


@dataclass
class Setup:
    side: int
    entry: float
    stop: float
    target: float
    created: int
    expiry: int
    status: str = "en attente"     # en attente, en cours, stop, objectif, invalidé, expiré, remplacé
    fill_index: int | None = None
    fill_price: float | None = None
    exit_index: int | None = None
    exit_price: float | None = None


def atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=1).mean().to_numpy(float)


def _last_opposite(o, c, i, lookback, side):
    """Dernière bougie de sens opposé à l'impulsion, dans la fenêtre précédant i."""
    lo = max(0, i - lookback)
    for k in range(i - 1, lo - 1, -1):
        if side == 1 and c[k] < o[k]:
            return k
        if side == -1 and c[k] > o[k]:
            return k
    return None


def _has_fvg(h, l, k, i, side):
    """Vrai si un FVG du bon sens existe entre les bougies k et i."""
    for j in range(max(k + 1, 2), i + 1):
        if side == 1 and l[j] > h[j - 2]:
            return True
        if side == -1 and h[j] < l[j - 2]:
            return True
    return False


def analyse(df: pd.DataFrame, n: int = 3, ob_lookback: int = 5, require_fvg: bool = False):
    """
    Détecte pivots, BOS/CHoCH, FVG et Order Blocks, bougie par bougie.
    Aucune donnée future n'est utilisée : un pivot n'est confirmé
    qu'après n bougies, et une cassure n'est retenue qu'à la clôture.
    """
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)

    trend = 0
    sh = None
    sl = None
    events: list[Event] = []
    zones: list[Zone] = []

    for i in range(len(df)):
        # 1) Mitigation : une bougie postérieure touche la zone
        for z in zones:
            if z.end is None and i > z.start:
                if (z.side == 1 and l[i] <= z.top) or (z.side == -1 and h[i] >= z.bottom):
                    z.end = i

        # 2) Cassure haussière sur clôture
        if sh is not None and not sh["broken"] and c[i] > sh["price"]:
            sh["broken"] = True
            events.append(Event(i, "CHoCH" if trend == -1 else "BOS", 1, sh["price"]))
            trend = 1
            k = _last_opposite(o, c, i, ob_lookback, side=1)
            if k is not None and (not require_fvg or _has_fvg(h, l, k, i, 1)):
                zones.append(Zone("OB", 1, l[k], h[k], start=i, event_index=i))

        # 3) Cassure baissière sur clôture
        if sl is not None and not sl["broken"] and c[i] < sl["price"]:
            sl["broken"] = True
            events.append(Event(i, "CHoCH" if trend == 1 else "BOS", -1, sl["price"]))
            trend = -1
            k = _last_opposite(o, c, i, ob_lookback, side=-1)
            if k is not None and (not require_fvg or _has_fvg(h, l, k, i, -1)):
                zones.append(Zone("OB", -1, l[k], h[k], start=i, event_index=i))

        # 4) FVG sur trois bougies
        if i >= 2:
            if l[i] > h[i - 2]:
                zones.append(Zone("FVG", 1, h[i - 2], l[i], start=i))
            if h[i] < l[i - 2]:
                zones.append(Zone("FVG", -1, h[i], l[i - 2], start=i))

        # 5) Pivots : le candidat p = i - n n'est confirmé qu'à la bougie i
        p = i - n
        if p - n >= 0:
            hw = h[p - n: p + n + 1]
            if h[p] == hw.max() and np.count_nonzero(hw == h[p]) == 1:
                sh = {"price": float(h[p]), "index": p, "broken": False}
            lw = l[p - n: p + n + 1]
            if l[p] == lw.min() and np.count_nonzero(lw == l[p]) == 1:
                sl = {"price": float(l[p]), "index": p, "broken": False}

    return events, zones


def build_setups(df, events, zones, rr: float = 2.0, buffer_atr: float = 0.2, max_wait: int = 30):
    """Un setup par cassure : entrée au bord de l'OB, stop au-delà, objectif à ratio R."""
    a = atr(df)
    by_event = {z.event_index: z for z in zones if z.event_index is not None}
    setups: list[Setup] = []

    for ev in events:
        z = by_event.get(ev.index)
        if z is None:
            continue
        i = ev.index
        buf = buffer_atr * a[i]
        if z.side == 1:
            entry = z.top
            stop = z.bottom - buf
            risk = entry - stop
            target = entry + rr * risk
        else:
            entry = z.bottom
            stop = z.top + buf
            risk = stop - entry
            target = entry - rr * risk
        if risk <= 0:
            continue
        setups.append(Setup(side=z.side, entry=entry, stop=stop, target=target,
                            created=i, expiry=i + max_wait))
    return setups


def backtest(df: pd.DataFrame, setups: list[Setup], spread: float = 0.0):
    """
    Simulation bougie par bougie : un trade à la fois, ordre limite à l'entrée.
    Si stop et objectif sont touchés dans la même bougie, le stop est retenu (hypothèse prudente).
    Retourne (trades_df, setups) ; les setups sont mis à jour avec leur statut.
    """
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)

    by_bar: dict[int, list[Setup]] = {}
    for s in setups:
        by_bar.setdefault(s.created, []).append(s)

    pending: Setup | None = None
    open_t: Setup | None = None
    closed: list[Setup] = []

    for j in range(len(df)):
        # a) Gestion du trade ouvert (à partir de la bougie suivant l'entrée)
        if open_t is not None and j > open_t.fill_index:
            s = open_t
            if s.side == 1:
                hit_sl, hit_tp = l[j] <= s.stop, h[j] >= s.target
            else:
                hit_sl, hit_tp = h[j] >= s.stop, l[j] <= s.target
            if hit_sl or hit_tp:
                price = s.stop if hit_sl else s.target
                s.exit_index, s.exit_price = j, price
                s.status = "stop" if hit_sl else "objectif"
                closed.append(s)
                open_t = None

        # b) Setup en attente : expiration, remplissage ou invalidation
        if pending is not None and open_t is None and j > pending.created:
            s = pending
            if s.side == 1:
                filled = l[j] <= s.entry
            else:
                filled = h[j] >= s.entry

            if j > s.expiry:
                s.status = "expiré"
                pending = None
            elif filled:
                s.fill_price = min(o[j], s.entry) if s.side == 1 else max(o[j], s.entry)
                s.fill_index = j
                s.status = "en cours"
                open_t = s
                pending = None
            elif (s.side == 1 and c[j] < s.stop) or (s.side == -1 and c[j] > s.stop):
                s.status = "invalidé"
                pending = None

        # c) Nouveaux setups créés à la clôture de la bougie j
        for s in by_bar.get(j, []):
            if pending is not None:
                pending.status = "remplacé"
            pending = s

    trades = pd.DataFrame([asdict(s) for s in closed])
    if not trades.empty:
        trades["r"] = [
            ((s.exit_price - s.fill_price) * s.side) / abs(s.fill_price - s.stop)
            - spread / abs(s.fill_price - s.stop)
            for s in closed
        ]
    return trades, setups


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {}
    r = trades["r"].astype(float)
    gains = r[r > 0].sum()
    losses = -r[r <= 0].sum()
    equity = pd.concat([pd.Series([0.0]), r.cumsum()], ignore_index=True)
    return {
        "n": int(len(r)),
        "win_rate": float((r > 0).mean()),
        "expectancy": float(r.mean()),
        "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
        "total_r": float(r.sum()),
        "max_dd_r": float((equity.cummax() - equity).max()),
    }
