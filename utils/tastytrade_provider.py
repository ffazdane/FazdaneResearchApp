"""
Tastytrade market-data adapter.

This follows the same connection order used by the Portfolio Management app:
  1. OAuth2 refresh grant: TT_CLIENT_ID, TT_SECRET, TT_REFRESH
  2. Username/password fallback: TASTYTRADE_USERNAME, TASTYTRADE_PASSWORD

TASTYTRADE_ENVIRONMENT or TASTYTRADE_API_BASE_URL can be used to select
production versus sandbox endpoints.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, replace
from datetime import datetime
import json
import os
from typing import Any, Iterable

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional during partial installs
    load_dotenv = None

if load_dotenv:
    load_dotenv()


USER_AGENT = "FazDaneResearch/1.0"
PRODUCTION_BASE_URL = "https://api.tastyworks.com"
SANDBOX_BASE_URL = "https://api.cert.tastyworks.com"
_SESSION_CACHE: dict[str, "_DirectSession"] = {}


@dataclass(frozen=True)
class TastytradeConfig:
    base_url: str
    environment: str
    token: str | None
    client_id: str | None
    client_secret: str | None
    refresh_token: str | None
    username: str | None
    password: str | None

    @property
    def is_configured(self) -> bool:
        return bool(self.token or (self.client_secret and self.refresh_token) or (self.username and self.password))


class TastytradeProviderError(RuntimeError):
    """Raised when Tastytrade cannot provide requested data."""


class _DirectSession:
    """Lightweight Tastytrade REST session matching the Portfolio Management pattern."""

    def __init__(self, session_token: str, base_url: str, is_test: bool = False):
        self.session_token = session_token
        self.base_url = base_url
        self.is_test = is_test
        self.headers = {
            "Authorization": session_token,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}{endpoint}",
            headers=self.headers,
            params=params,
            timeout=20,
        )
        if response.status_code != 200:
            raise TastytradeProviderError(f"Tastytrade API error ({response.status_code}): {response.text[:200]}")
        return response.json()


def load_config() -> TastytradeConfig:
    environment = _secret("TASTYTRADE_ENVIRONMENT", "tastytrade_environment", default="production").strip()
    is_test = environment.lower() == "sandbox"
    default_base_url = SANDBOX_BASE_URL if is_test else PRODUCTION_BASE_URL
    base_url = _secret("TASTYTRADE_API_BASE_URL", default=default_base_url).rstrip("/")

    client_id = _secret("TT_CLIENT_ID").strip() or None
    client_secret = _secret("TT_SECRET").strip() or None
    refresh_token = _secret("TT_REFRESH").strip() or None

    if not client_id and refresh_token and "." in refresh_token:
        client_id = _client_id_from_refresh_token(refresh_token)
    if not client_id:
        client_id = client_secret

    return TastytradeConfig(
        base_url=base_url,
        environment=environment,
        token=_secret("TASTYTRADE_ACCESS_TOKEN").strip()
        or _secret("TASTYTRADE_SESSION_TOKEN").strip()
        or None,
        client_id=client_id,
        client_secret=client_secret,
        refresh_token=refresh_token,
        username=_secret("TASTYTRADE_USERNAME").strip() or None,
        password=_secret("TASTYTRADE_PASSWORD").strip() or None,
    )


def clear_session_cache() -> None:
    _SESSION_CACHE.clear()


def get_tastytrade_session(config: TastytradeConfig | None = None) -> tuple[_DirectSession | None, str | None]:
    """Create or return a cached Tastytrade session."""
    config = config or load_config()
    if not config.is_configured:
        return None, "No Tastytrade credentials configured."

    cache_key = f"{config.token or config.client_secret or config.username}_{config.environment}_{config.base_url}"
    if cache_key in _SESSION_CACHE:
        return _SESSION_CACHE[cache_key], None

    if config.token:
        session = _DirectSession(config.token, config.base_url, config.environment.lower() == "sandbox")
        _SESSION_CACHE[cache_key] = session
        return session, None

    if config.client_secret and config.refresh_token:
        session, error = _oauth_session(config)
        if session:
            _SESSION_CACHE[cache_key] = session
            return session, None
        if not (config.username and config.password):
            return None, error

    if config.username and config.password:
        session, error = _password_session(config)
        if session:
            _SESSION_CACHE[cache_key] = session
            return session, None
        return None, error

    return None, "Unable to create Tastytrade session."


def fetch_nested_option_chain(symbol: str, config: TastytradeConfig | None = None) -> pd.DataFrame:
    """Fetch equity option-chain metadata from Tastytrade."""
    config = config or load_config()
    payload = _session_get_with_retry(config, f"/option-chains/{symbol.upper()}/nested")
    return _nested_chain_payload_to_frame(payload, symbol)


def fetch_market_data_by_type(
    equities: Iterable[str] | None = None,
    options: Iterable[str] | None = None,
    config: TastytradeConfig | None = None,
) -> pd.DataFrame:
    """Fetch equity and equity-option market data from Tastytrade."""
    config = config or load_config()
    frames = []

    for equity_chunk, option_chunk in _market_data_chunks(equities, options, limit=100):
        params: dict[str, list[str]] = {}
        if equity_chunk:
            params["equity"] = equity_chunk
        if option_chunk:
            params["equity-option"] = option_chunk
        if not params:
            continue

        payload = _session_get_with_retry(config, "/market-data/by-type", params=params)
        data = payload.get("data", payload)
        items = data.get("items", [])
        if items:
            frames.append(pd.DataFrame([_normalize_market_data_item(item) for item in items]))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _session_get_with_retry(
    config: TastytradeConfig,
    endpoint: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    session, error = get_tastytrade_session(config)
    if error or session is None:
        raise TastytradeProviderError(error or "Unable to create Tastytrade session.")

    try:
        return session.get(endpoint, params=params)
    except TastytradeProviderError as exc:
        if _is_unauthorized(exc) and config.token and (config.client_secret or config.username):
            clear_session_cache()
            retry_config = replace(config, token=None)
            session, error = get_tastytrade_session(retry_config)
            if error or session is None:
                raise TastytradeProviderError(
                    f"Tastytrade token unauthorized; credential fallback failed: "
                    f"{error or 'Unable to create Tastytrade session.'}"
                ) from exc
            return session.get(endpoint, params=params)
        raise


def _market_data_chunks(
    equities: Iterable[str] | None,
    options: Iterable[str] | None,
    limit: int,
) -> Iterable[tuple[list[str], list[str]]]:
    equity_list = [str(symbol).upper() for symbol in equities or [] if str(symbol).strip()]
    option_list = [str(symbol) for symbol in options or [] if str(symbol).strip()]

    if equity_list:
        yield equity_list[:limit], []

    for start in range(0, len(option_list), limit):
        yield [], option_list[start:start + limit]


def _normalize_market_data_item(item: dict[str, Any]) -> dict[str, Any]:
    instrument = item.get("instrument") or {}
    return {
        "market_symbol": item.get("symbol"),
        "instrument_type": item.get("instrument-type") or item.get("instrument_type"),
        "underlying_symbol": item.get("underlying-instrument") or instrument.get("underlying-instrument"),
        "bid": _to_float(item.get("bid")),
        "ask": _to_float(item.get("ask")),
        "mark": _to_float(item.get("mark")),
        "last_price": _to_float(item.get("last")),
        "close": _to_float(item.get("close") or item.get("prev-close")),
        "volume": _to_float(item.get("volume")),
        "open_interest": _to_float(item.get("open-interest") or item.get("open_interest")),
        "implied_volatility": _to_float(
            item.get("implied-volatility")
            or item.get("implied_volatility")
            or item.get("iv")
        ),
        "updated_at": item.get("updated-at") or item.get("updated_at"),
    }


def _secret(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value

    try:
        import streamlit as st

        for name in names:
            value = st.secrets.get(name)
            if value:
                return str(value)
    except Exception:
        pass

    return default


def _is_unauthorized(exc: Exception) -> bool:
    text = str(exc).lower()
    return "api error (401)" in text or "unauthorized" in text


def _oauth_session(config: TastytradeConfig) -> tuple[_DirectSession | None, str | None]:
    attempts: list[dict[str, str]] = []
    if config.client_id and config.client_id != config.client_secret:
        attempts.append({
            "grant_type": "refresh_token",
            "refresh_token": config.refresh_token or "",
            "client_id": config.client_id,
            "client_secret": config.client_secret or "",
        })
        attempts.append({
            "grant_type": "refresh_token",
            "refresh_token": config.refresh_token or "",
            "client_id": config.client_id,
        })

    attempts.append({
        "grant_type": "refresh_token",
        "refresh_token": config.refresh_token or "",
        "client_id": config.client_secret or "",
        "client_secret": config.client_secret or "",
    })

    last_error = "OAuth2 failed."
    for payload in attempts:
        try:
            response = requests.post(
                f"{config.base_url}/oauth/token",
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": USER_AGENT},
                timeout=20,
            )
            body = _response_json(response)
            if response.status_code in (200, 201) and "error" not in body and "error_code" not in body:
                token = body.get("access_token") or body.get("session-token")
                if token:
                    if body.get("access_token"):
                        token = f"Bearer {token}"
                    return _DirectSession(token, config.base_url, config.environment.lower() == "sandbox"), None
            last_error = body.get("error_description") or body.get("error") or response.text[:300]
        except Exception as exc:
            last_error = str(exc)

    return None, f"OAuth2 login failed: {last_error}"


def _password_session(config: TastytradeConfig) -> tuple[_DirectSession | None, str | None]:
    try:
        response = requests.post(
            f"{config.base_url}/sessions",
            json={"login": config.username, "password": config.password, "remember-me": True},
            headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
            timeout=20,
        )
        body = _response_json(response)
        if response.status_code == 403:
            return None, "2FA is enabled; use TT_SECRET and TT_REFRESH OAuth credentials."
        if response.status_code not in (200, 201):
            message = body.get("error", {}).get("message") if isinstance(body.get("error"), dict) else body.get("error")
            return None, f"Login failed ({response.status_code}): {message or response.text[:200]}"

        token = body.get("data", {}).get("session-token")
        if not token:
            return None, "No session token received from Tastytrade."
        return _DirectSession(token, config.base_url, config.environment.lower() == "sandbox"), None
    except Exception as exc:
        return None, f"Login failed: {exc}"


def _nested_chain_payload_to_frame(payload: dict[str, Any], symbol: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    today = datetime.today().date()

    for item in payload.get("data", {}).get("items", []):
        underlying = item.get("underlying-symbol") or symbol.upper()
        for expiration in item.get("expirations", []):
            expiration_date = expiration.get("expiration-date")
            if not expiration_date:
                continue
            try:
                exp_date = datetime.strptime(expiration_date, "%Y-%m-%d").date()
                dte = int(expiration.get("days-to-expiration", (exp_date - today).days))
            except ValueError:
                continue

            for strike in expiration.get("strikes", []):
                strike_price = _to_float(strike.get("strike-price"))
                if strike_price is None:
                    continue

                for option_type, contract_key, streamer_key in (
                    ("Call", "call", "call-streamer-symbol"),
                    ("Put", "put", "put-streamer-symbol"),
                ):
                    contract = strike.get(contract_key)
                    if not contract:
                        continue
                    rows.append({
                        "symbol": underlying,
                        "option_type": option_type,
                        "expiration": expiration_date,
                        "dte": dte,
                        "strike": strike_price,
                        "contract": contract,
                        "streamer_symbol": strike.get(streamer_key),
                        "data_source": "Tastytrade",
                    })

    return pd.DataFrame(rows)


def _client_id_from_refresh_token(refresh_token: str) -> str | None:
    try:
        payload = refresh_token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        decoded = json.loads(base64.urlsafe_b64decode(payload))
        return decoded.get("aud")
    except Exception:
        return None


def _response_json(response: requests.Response) -> dict[str, Any]:
    try:
        return response.json()
    except Exception:
        return {}


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
