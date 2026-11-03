"""Single-attempt HTTP policy and bounded retries owned by recording/metering callers.

Never put retries inside a raw model client: each send must be independently
recorded and, when budgeted, reserved by the caller before it reaches the wire.
"""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.client import IncompleteRead
import math
import random
import re
import socket
import ssl
import time
import traceback
from urllib.error import HTTPError, URLError
from urllib.parse import quote, quote_plus, urlsplit, urlunsplit
from urllib.request import getproxies, proxy_bypass


def transport_policy(settings=None, base_url=""):
    domestic = urlsplit(base_url).hostname == "dashscope.aliyuncs.com"
    result = {"proxy_mode": "direct" if domestic else "environment",
              "max_retries": 2 if domestic else 0,
              "retry_base_seconds": 1.0, "retry_max_seconds": 8.0}
    if settings is not None:
        if not isinstance(settings, dict) or set(settings) - set(result):
            raise ValueError("api_transport must contain only supported transport settings")
        result.update(settings)
    if not isinstance(result["proxy_mode"], str) or result["proxy_mode"] not in {"direct", "environment"}:
        raise ValueError("api_transport.proxy_mode must be direct or environment")
    if type(result["max_retries"]) is not int or not 0 <= result["max_retries"] <= 3:
        raise ValueError("api_transport.max_retries must be an integer between 0 and 3")
    for key in ("retry_base_seconds", "retry_max_seconds"):
        value = result[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 30:
            raise ValueError("api_transport retry delays must be finite numbers in (0, 30]")
    if result["retry_base_seconds"] > result["retry_max_seconds"]:
        raise ValueError("api_transport base delay cannot exceed maximum delay")
    return result


def _chain(error):
    seen = set()
    while isinstance(error, BaseException) and id(error) not in seen:
        seen.add(id(error))
        yield error
        error = (error.reason if isinstance(error, URLError) and isinstance(error.reason, BaseException)
                 else error.__cause__ or (None if error.__suppress_context__ else error.__context__))


def is_transient(error):
    chain = list(_chain(error))
    http = next((e for e in chain if isinstance(e, HTTPError)), None)
    if http is not None:
        return http.code in {408, 429, 500, 502, 503, 504}
    if any(isinstance(e, ssl.SSLError) and not isinstance(e, ssl.SSLEOFError) for e in chain):
        return False
    return any(isinstance(e, (ConnectionError, TimeoutError, IncompleteRead, ssl.SSLEOFError))
               or (isinstance(e, socket.gaierror) and e.errno == socket.EAI_AGAIN) for e in chain)


def retry_delay(error, attempt, policy):
    if attempt > policy["max_retries"] or not is_transient(error):
        return None
    delay = min(policy["retry_max_seconds"], policy["retry_base_seconds"] * 2 ** (attempt - 1))
    delay = min(policy["retry_max_seconds"], delay + random.uniform(0, delay * .25))
    http = next((e for e in _chain(error) if isinstance(e, HTTPError)), None)
    raw = http.headers.get("Retry-After") if http is not None and http.headers else None
    if raw:
        try:
            required = float(raw)
        except ValueError:
            try:
                required = (parsedate_to_datetime(raw) - datetime.now(timezone.utc)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                return None  # Do not guess an earlier retry for an invalid server instruction.
        if not math.isfinite(required) or required > policy["retry_max_seconds"]:
            return None
        delay = max(delay, required)
    return delay


def with_retries(operation, policy, on_retry):
    """operation(attempt) owns ALL logging/reservations and performs exactly one send."""
    for attempt in range(1, policy["max_retries"] + 2):
        try:
            return operation(attempt)
        except Exception as error:
            delay = retry_delay(error, attempt, policy)
            if delay is None:
                raise
            on_retry(attempt, delay)
            time.sleep(delay)


def redact(text, api_key=None):
    text = str(text)
    if api_key:
        for secret in (api_key, quote(api_key, safe=""), quote_plus(api_key)):
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[REDACTED]@", text)
    return text[:4000]


def error_details(error, api_key=None):
    chain = list(_chain(error))
    return {"error": redact(error, api_key), "error_type": type(error).__name__,
            "retryable": is_transient(error),
            "http_status": next((e.code for e in chain if isinstance(e, HTTPError)), None),
            "exception_chain": [{"type": type(e).__name__, "message": redact(e, api_key),
                "frames": [{"file": redact(f.filename, api_key), "line": f.lineno, "function": f.name}
                           for f in traceback.extract_tb(e.__traceback__)[-12:]]} for e in chain[:6]]}


def transport_details(base_url, policy):
    """No credentials, full proxy URL, query string, headers or stack locals."""
    url = urlsplit(base_url)
    host = url.hostname or ""
    if ":" in host:
        host = "[" + host + "]"
    if url.port is not None:
        host += ":" + str(url.port)
    result = {"proxy_mode": policy["proxy_mode"],
              "endpoint": urlunsplit((url.scheme, host, url.path, "", "")),
              "proxy": None}
    if policy["proxy_mode"] == "environment" and url.hostname and not proxy_bypass(url.hostname):
        proxy = getproxies().get(url.scheme)
        if proxy:
            parsed = urlsplit(proxy if "://" in proxy else "http://" + proxy)
            result["proxy"] = {"scheme": parsed.scheme, "host": parsed.hostname, "port": parsed.port}
    return result
