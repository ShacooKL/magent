"""Application-wide Beijing time helpers."""

from datetime import datetime
from zoneinfo import ZoneInfo


BEIJING_TZ = ZoneInfo("Asia/Shanghai")
TIMEZONE_NAME = "Asia/Shanghai"


def now_beijing() -> datetime:
    return datetime.now(BEIJING_TZ)
