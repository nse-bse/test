from datetime import datetime, timedelta
from typing import Tuple, Optional
from config import IST, now_ist

def market_bounds(now: datetime) -> Tuple[datetime, datetime]:
    d = now.date()
    start = datetime(d.year, d.month, d.day, 9, 15, 0, tzinfo=IST if IST else None)
    end   = datetime(d.year, d.month, d.day, 15, 30, 0, tzinfo=IST if IST else None)
    return start, end

def next_refresh_in_seconds(now: datetime) -> Optional[int]:
    mstart, mend = market_bounds(now)
    if now < mstart:
        target = mstart
    else:
        first_gap = mstart + timedelta(minutes=2)
        if now < first_gap:
            target = first_gap
        else:
            base = first_gap
            elapsed = (now - base).total_seconds()
            next_mult = int(elapsed // 180) * 180 + 180
            target = base + timedelta(seconds=next_mult)
    if target > mend:
        return None
    remaining = int(max(0, (target - now).total_seconds()))
    return remaining

def on_market_tick(now: datetime, last_fetch_ts: float) -> bool:
    if last_fetch_ts == 0.0: return True
    remaining = next_refresh_in_seconds(now)
    if remaining is None: return False
    return remaining <= 4
