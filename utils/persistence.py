"""
Database Persistence Manager for FazDane Research Application
=============================================================
Handles backup and restore of SQLite databases (portfolio_performance,
earnings_calendar, options_liquidity) to/from cloud storage.

This resolves the ephemeral filesystem issue on containerized hosts (like Streamlit Cloud).
"""

import os
import shutil
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("FazDanePersistence")

REPO_ROOT = Path(__file__).resolve().parents[1]

# Tracked databases and their environment variables/local paths
DATABASES = {
    "portfolio_performance": {
        "env_var": "PORTFOLIO_PERFORMANCE_DB_PATH",
        "default_path": REPO_ROOT / "data" / "portfolio_performance" / "portfolio_performance.sqlite",
    },
    "earnings_calendar": {
        "env_var": "EARNINGS_CALENDAR_DB_PATH",
        "default_path": REPO_ROOT / "data" / "earnings_calendar" / "earnings_calendar.sqlite",
    },
    "options_liquidity": {
        "env_var": "OPTIONS_LIQUIDITY_DB_PATH",
        "default_path": REPO_ROOT / "data" / "options_liquidity" / "options_liquidity.sqlite",
    },
    "option_search": {
        "env_var": "OPTION_SEARCH_DB_PATH",
        "default_path": REPO_ROOT / "data" / "option_search.db",
    },
}

BACKUP_DIR = REPO_ROOT / "data" / "backups"


# ════════════════════════════════════════════════════════════
# Configuration Helpers
# ════════════════════════════════════════════════════════════

def get_db_path(db_name: str) -> Path:
    """Resolve the local file path for a database name."""
    if db_name not in DATABASES:
        raise ValueError(f"Unknown database: {db_name}")
    config = DATABASES[db_name]
    path_str = os.getenv(config["env_var"])
    if path_str:
        return Path(path_str).expanduser().resolve()
    return config["default_path"].resolve()


def _get_backend() -> str:
    """Read the configured backend: none | github | s3."""
    try:
        import streamlit as st
        return st.secrets.get("database", {}).get("backend", "none").strip().lower()
    except Exception:
        return os.getenv("DB_BACKEND", "none").strip().lower()


def _get_github_config() -> Dict[str, str]:
    """Load GitHub Release API configurations."""
    try:
        import streamlit as st
        gh = st.secrets.get("database", {}).get("github", {})
    except Exception:
        gh = {}
    return {
        "token": (gh.get("token") or os.getenv("GH_DB_TOKEN") or "").strip(),
        "owner": (gh.get("owner") or os.getenv("GH_DB_OWNER") or "").strip(),
        "repo":  (gh.get("repo")  or os.getenv("GH_DB_REPO")  or "").strip(),
        "tag":   (gh.get("tag")   or os.getenv("GH_DB_TAG")   or "db-backup").strip(),
    }


def _get_s3_config() -> Dict[str, str]:
    """Load AWS S3 API configurations."""
    try:
        import streamlit as st
        s3 = st.secrets.get("database", {}).get("s3", {})
    except Exception:
        s3 = {}
    return {
        "bucket":            (s3.get("bucket")            or os.getenv("DB_S3_BUCKET") or "").strip(),
        "key_prefix":        (s3.get("key_prefix")        or os.getenv("DB_S3_KEY_PREFIX") or "databases/").strip(),
        "region":            (s3.get("region")            or os.getenv("AWS_DEFAULT_REGION") or "us-east-1").strip(),
        "access_key_id":     (s3.get("access_key_id")     or os.getenv("AWS_ACCESS_KEY_ID") or "").strip(),
        "secret_access_key": (s3.get("secret_access_key") or os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip(),
    }


# ════════════════════════════════════════════════════════════
# Database State Checkers
# ════════════════════════════════════════════════════════════

def initialize_volatility_cache_tables():
    """Create the volatility caching tables if they don't already exist.

    Called at app startup to guarantee the schema is present before any
    module attempts to read or write cached volatility data.
    """
    try:
        db_path = get_db_path("options_liquidity")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, timeout=10)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS volatility_page_snapshots (
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                last_price REAL,
                atm_iv REAL,
                hv20 REAL,
                hv30 REAL,
                hvr REAL,
                expected_move REAL,
                regime_label TEXT,
                trend_label TEXT,
                vix_current REAL,
                vix_pct REAL,
                vvix_current REAL,
                term_shape TEXT,
                term_structure_json TEXT,
                otm_put_iv REAL,
                otm_call_iv REAL,
                skew_label TEXT,
                liq_label TEXT,
                liq_detail_json TEXT,
                days_to_earnings INTEGER,
                strategy_name TEXT,
                strategy_json TEXT,
                PRIMARY KEY (symbol, timestamp)
            );

            CREATE TABLE IF NOT EXISTS options_chains_cache (
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                expiry_date TEXT NOT NULL,
                strike REAL NOT NULL,
                option_type TEXT NOT NULL,
                bid REAL,
                ask REAL,
                implied_volatility REAL,
                volume INTEGER,
                open_interest INTEGER,
                PRIMARY KEY (symbol, timestamp, expiry_date, strike, option_type)
            );

            CREATE INDEX IF NOT EXISTS idx_vol_snapshots_symbol_ts
                ON volatility_page_snapshots(symbol, timestamp);
            CREATE INDEX IF NOT EXISTS idx_options_chains_cache_symbol_ts
                ON options_chains_cache(symbol, timestamp);
        """)
        conn.close()
        logger.info("Volatility cache tables verified/created in options_liquidity.sqlite")
    except Exception as e:
        logger.warning(f"Failed to initialize volatility cache tables: {e}")


def db_exists_and_has_data(db_name: str) -> bool:
    """Check if the SQLite database file exists and contains user tables."""
    try:
        db_path = get_db_path(db_name)
        if not db_path.exists():
            return False
        if db_path.stat().st_size < 4096:
            return False

        conn = sqlite3.connect(db_path, timeout=5)
        # Select count of non-system tables
        row = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchone()
        conn.close()
        return bool(row and row[0] > 0)
    except Exception as e:
        logger.warning(f"Failed checking database state for {db_name}: {e}")
        return False


# ════════════════════════════════════════════════════════════
# Core Public API (Restore / Backup)
# ════════════════════════════════════════════════════════════

def log_persistence_event(event_type: str, db_name: str, success: bool, message: str):
    """Log database persistence events to a local file for history tracking."""
    try:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        log_file = BACKUP_DIR / "persistence_history.log"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_str = "SUCCESS" if success else "FAILED"
        log_entry = f"[{timestamp}] {event_type.upper()} {db_name.upper()} - {status_str}: {message}\n"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        logger.error(f"Failed writing persistence history log: {e}")


def restore_database(db_name: str) -> Tuple[bool, str]:
    """Restore a database binary from the cloud backend."""
    backend = _get_backend()
    db_path = get_db_path(db_name)

    if backend == "none":
        msg = f"Backend is 'none'; starting with a new local database for {db_name}."
        log_persistence_event("restore", db_name, False, msg)
        return False, msg

    logger.info(f"Restoring database '{db_name}' using backend '{backend}'...")
    if backend == "github":
        ok, msg = _restore_from_github(db_name, db_path)
    elif backend == "s3":
        ok, msg = _restore_from_s3(db_name, db_path)
    else:
        ok, msg = False, f"Unknown database backend: '{backend}'"

    log_persistence_event("restore", db_name, ok, msg)
    return ok, msg


def restore_all_databases(force: bool = False) -> Tuple[List[str], List[str]]:
    """Scan and restore all configured databases that are empty, missing, or when force is True."""
    restored = []
    failed = []
    
    backend = _get_backend()
    if backend == "none":
        logger.info("No cloud database restore executed: backend is set to 'none'.")
        return restored, failed

    for db_name in DATABASES:
        if force or not db_exists_and_has_data(db_name):
            ok, msg = restore_database(db_name)
            if ok:
                restored.append(db_name)
                logger.info(f"Database '{db_name}' restored: {msg}")
            else:
                failed.append(f"{db_name}: {msg}")
                logger.warning(f"Database '{db_name}' restore failed: {msg}")
        else:
            logger.info(f"Database '{db_name}' is already initialized on disk. Skipping restore.")
            
    return restored, failed


def backup_database(db_name: str, reason: str = "auto") -> Tuple[bool, str]:
    """Upload a database binary to the configured cloud backend."""
    if db_name not in DATABASES:
        return False, f"Unknown database name: {db_name}"

    db_path = get_db_path(db_name)
    if not db_path.exists():
        msg = f"No database file found at {db_path} to backup."
        log_persistence_event("backup", db_name, False, msg)
        return False, msg

    # 1. Local copy (always created in data/backups)
    local_msg = ""
    try:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_backup = BACKUP_DIR / f"{db_name}_backup_{timestamp}.sqlite"
        shutil.copy2(db_path, local_backup)
        logger.info(f"Local backup created for {db_name}: {local_backup.name}")
        local_msg = f"Local backup: {local_backup.name}"
    except Exception as e:
        logger.error(f"Local backup failed for {db_name}: {e}")
        local_msg = f"Local backup failed: {e}"

    # 2. Cloud copy
    backend = _get_backend()
    if backend == "none":
        msg = f"Local backup only. {local_msg}"
        log_persistence_event("backup", db_name, True, msg)
        return True, msg
    elif backend == "github":
        ok, msg = _backup_to_github(db_name, db_path, reason)
    elif backend == "s3":
        ok, msg = _backup_to_s3(db_name, db_path, reason)
    else:
        ok, msg = False, f"Unknown backend: '{backend}'"

    log_persistence_event("backup", db_name, ok, f"{msg} | {local_msg}")
    return ok, msg


# ════════════════════════════════════════════════════════════
# GitHub Release Implementation
# ════════════════════════════════════════════════════════════

def _restore_from_github(db_name: str, db_path: Path) -> Tuple[bool, str]:
    try:
        import requests
    except ImportError:
        return False, "requests package is missing. Run: pip install requests"

    cfg = _get_github_config()
    if not all([cfg["token"], cfg["owner"], cfg["repo"]]):
        return False, "GitHub credentials missing or incomplete in secrets/env variables."

    headers = {
        "Authorization": f"token {cfg['token']}",
        "Accept": "application/vnd.github.v3+json",
    }
    base_url = f"https://api.github.com/repos/{cfg['owner']}/{cfg['repo']}"
    filename = db_path.name

    try:
        resp = requests.get(f"{base_url}/releases/tags/{cfg['tag']}", headers=headers, timeout=30)
        if resp.status_code == 404:
            return False, f"No release backup tag '{cfg['tag']}' exists yet. Starting fresh."
        resp.raise_for_status()

        assets = resp.json().get("assets", [])
        asset = next((a for a in assets if a["name"] == filename), None)
        if not asset:
            return False, f"Asset '{filename}' not found under release tag '{cfg['tag']}'. Starting fresh."

        dl = requests.get(asset["url"], headers={**headers, "Accept": "application/octet-stream"}, timeout=120)
        dl.raise_for_status()

        db_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = db_path.with_suffix(".tmp")
        with open(temp_path, "wb") as f:
            f.write(dl.content)
        
        if db_path.exists():
            try:
                db_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove old database file {db_path}: {e}")
        temp_path.rename(db_path)

        return True, f"Database restored from GitHub release tag '{cfg['tag']}' ({len(dl.content):,} bytes)."
    except Exception as e:
        return False, f"GitHub download failed: {e}"


def _backup_to_github(db_name: str, db_path: Path, reason: str) -> Tuple[bool, str]:
    try:
        import requests
    except ImportError:
        return False, "requests package is missing."

    cfg = _get_github_config()
    if not all([cfg["token"], cfg["owner"], cfg["repo"]]):
        return False, "GitHub credentials missing or incomplete in secrets/env variables."

    headers = {
        "Authorization": f"token {cfg['token']}",
        "Accept": "application/vnd.github.v3+json",
    }
    base_url = f"https://api.github.com/repos/{cfg['owner']}/{cfg['repo']}"
    filename = db_path.name

    try:
        release_id, upload_url = _gh_get_or_create_release(headers, base_url, cfg["tag"])
        _gh_delete_asset(headers, base_url, release_id, filename)

        with open(db_path, "rb") as f:
            data = f.read()

        clean_url = upload_url.replace("{?name,label}", "")
        resp = requests.post(
            clean_url,
            params={
                "name": filename,
                "label": f"Backup: {db_name} ({reason}) — {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
            },
            headers={**headers, "Content-Type": "application/octet-stream"},
            data=data,
            timeout=120,
        )
        resp.raise_for_status()
        return True, f"Successfully backed up {filename} to GitHub release tag '{cfg['tag']}'."
    except Exception as e:
        return False, f"GitHub upload failed: {e}"


def _gh_get_or_create_release(headers: dict, base: str, tag: str) -> Tuple[int, str]:
    import requests
    resp = requests.get(f"{base}/releases/tags/{tag}", headers=headers, timeout=30)
    if resp.status_code == 200:
        data = resp.json()
        return data["id"], data["upload_url"]

    create = requests.post(
        f"{base}/releases",
        headers=headers,
        json={
            "tag_name": tag,
            "name": "FazDane Research Database Backups",
            "body": (
                "⚠️ Automated database backups managed by the FazDane Research Application.\n"
                "Do NOT delete this release — it hosts the persistent SQLite files for production."
            ),
            "draft": False,
            "prerelease": True,
        },
        timeout=30,
    )
    create.raise_for_status()
    data = create.json()
    return data["id"], data["upload_url"]


def _gh_delete_asset(headers: dict, base: str, release_id: int, name: str) -> None:
    import requests
    resp = requests.get(f"{base}/releases/{release_id}/assets", headers=headers, timeout=30)
    if resp.status_code != 200:
        return
    for asset in resp.json():
        if asset["name"] == name:
            requests.delete(f"{base}/releases/assets/{asset['id']}", headers=headers, timeout=30)


# ════════════════════════════════════════════════════════════
# S3 Implementation
# ════════════════════════════════════════════════════════════

def _restore_from_s3(db_name: str, db_path: Path) -> Tuple[bool, str]:
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        return False, "boto3 is missing. Run: pip install boto3"

    cfg = _get_s3_config()
    if not cfg["bucket"]:
        return False, "S3 bucket name is missing in secrets/env variables."

    key = f"{cfg['key_prefix']}{db_path.name}"

    try:
        s3 = boto3.client(
            "s3",
            region_name=cfg["region"],
            aws_access_key_id=cfg["access_key_id"] or None,
            aws_secret_access_key=cfg["secret_access_key"] or None,
        )
        db_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = db_path.with_suffix(".tmp")
        s3.download_file(cfg["bucket"], key, str(temp_path))
        
        if db_path.exists():
            try:
                db_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove old database file {db_path}: {e}")
        temp_path.rename(db_path)
        
        size = db_path.stat().st_size
        return True, f"Database restored from s3://{cfg['bucket']}/{key} ({size:,} bytes)."
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("404", "NoSuchKey"):
            return False, f"No backup asset found in s3://{cfg['bucket']}/{key}. Starting fresh."
        return False, f"S3 download failed: {exc}"
    except Exception as e:
        return False, f"S3 restore failed: {e}"


def _backup_to_s3(db_name: str, db_path: Path, reason: str) -> Tuple[bool, str]:
    try:
        import boto3
    except ImportError:
        return False, "boto3 is missing."

    cfg = _get_s3_config()
    if not cfg["bucket"]:
        return False, "S3 bucket name is missing."

    key = f"{cfg['key_prefix']}{db_path.name}"

    try:
        s3 = boto3.client(
            "s3",
            region_name=cfg["region"],
            aws_access_key_id=cfg["access_key_id"] or None,
            aws_secret_access_key=cfg["secret_access_key"] or None,
        )
        s3.upload_file(
            str(db_path),
            cfg["bucket"],
            key,
            ExtraArgs={"Metadata": {
                "db_name": db_name,
                "backup_reason": reason,
                "timestamp": datetime.now().isoformat(),
            }},
        )
        return True, f"Successfully backed up database to s3://{cfg['bucket']}/{key}."
    except Exception as e:
        return False, f"S3 upload failed: {e}"


# ════════════════════════════════════════════════════════════
# UI Control Panel helper
# ════════════════════════════════════════════════════════════

def render_db_control_panel():
    """Renders a database management interface inside a Streamlit expander."""
    import streamlit as st
    
    with st.expander("🗄️ Database Management", expanded=False):
        backend = _get_backend()
        st.write(f"**Backend**: `{backend}`")
        
        # Show database paths and sizes
        for db_name in DATABASES:
            path = get_db_path(db_name)
            size_kb = (path.stat().st_size / 1024) if path.exists() else 0
            st.caption(f"**{db_name}**: {size_kb:.1f} KB")

        st.divider()
        
        # Manual actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Backup DBs", key="db_manual_backup", use_container_width=True):
                with st.spinner("Backing up..."):
                    success_dbs = []
                    failed_dbs = []
                    for db_name in DATABASES:
                        if get_db_path(db_name).exists():
                            ok, msg = backup_database(db_name, reason="manual UI push")
                            if ok:
                                success_dbs.append(db_name)
                            else:
                                failed_dbs.append(db_name)
                    if success_dbs:
                        st.session_state["db_action_status"] = f"Backup success: {', '.join(success_dbs)}"
                    if failed_dbs:
                        st.session_state["db_action_error"] = f"Backup failed: {', '.join(failed_dbs)}"
                st.rerun()
                
        with col2:
            if st.button("Restore DBs", key="db_manual_restore", use_container_width=True):
                with st.spinner("Restoring..."):
                    restored, failed = restore_all_databases(force=True)
                    if restored:
                        st.session_state["db_action_status"] = f"Restored: {', '.join(restored)}"
                    if failed:
                        st.session_state["db_action_error"] = f"Restore failed: {', '.join(failed)}"
                st.rerun()

        # Display results of manual actions
        status = st.session_state.pop("db_action_status", None)
        error = st.session_state.pop("db_action_error", None)
        if status:
            st.success(status)
        if error:
            st.error(error)

        st.divider()
        if st.button("Rebuild/Patch DB from Online", key="db_manual_ingest", use_container_width=True, help="Download historical price data from Yahoo Finance and calculate option expiries directly in production to rebuild the SQLite database from scratch."):
            with st.spinner("Downloading historical data and rebuilding database..."):
                try:
                    from scripts.ingest_and_patch import main as run_ingestion
                    run_ingestion()
                    st.session_state["db_action_status"] = "Database successfully rebuilt from online sources!"
                except Exception as e:
                    st.session_state["db_action_error"] = f"Ingestion failed: {e}"
            st.rerun()

        # Show recent logs
        st.divider()
        st.markdown("**Recent Actions Log**")
        log_file = BACKUP_DIR / "persistence_history.log"
        if log_file.exists():
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                recent_lines = lines[-8:]
                if recent_lines:
                    st.code("".join(recent_lines), language="text")
                else:
                    st.info("No activity logged yet.")
            except Exception as e:
                st.error(f"Failed to read logs: {e}")
        else:
            st.info("No activity logged yet.")

        # ── Volatility Cache Status ──
        st.divider()
        st.markdown("**📊 Volatility Engine Cache Status**")
        try:
            db_path = get_db_path("options_liquidity")
            if db_path.exists():
                conn = sqlite3.connect(db_path, timeout=5)
                # Check if tables exist
                tables = [r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('volatility_page_snapshots','options_chains_cache')"
                ).fetchall()]

                if "volatility_page_snapshots" in tables:
                    # Get total snapshot count
                    total_snapshots = conn.execute("SELECT COUNT(*) FROM volatility_page_snapshots").fetchone()[0]
                    # Get unique symbols cached
                    symbols = conn.execute("SELECT DISTINCT symbol FROM volatility_page_snapshots ORDER BY symbol").fetchall()
                    symbol_list = [r[0] for r in symbols]

                    if total_snapshots > 0:
                        st.caption(f"**Total Snapshots**: {total_snapshots}  |  **Symbols Cached**: {', '.join(symbol_list)}")

                        # Show last cached entry per symbol
                        recent = conn.execute("""
                            SELECT symbol, timestamp, last_price, atm_iv, regime_label, strategy_name
                            FROM volatility_page_snapshots
                            WHERE (symbol, timestamp) IN (
                                SELECT symbol, MAX(timestamp) FROM volatility_page_snapshots GROUP BY symbol
                            )
                            ORDER BY timestamp DESC
                        """).fetchall()

                        if recent:
                            import pandas as pd
                            cache_df = pd.DataFrame(recent, columns=[
                                "Symbol", "Last Cached", "Price", "ATM IV", "Regime", "Strategy"
                            ])
                            # Format columns
                            cache_df["Price"] = cache_df["Price"].apply(lambda x: f"${x:,.2f}" if x else "N/A")
                            cache_df["ATM IV"] = cache_df["ATM IV"].apply(lambda x: f"{x:.1f}%" if x else "N/A")
                            st.dataframe(cache_df, use_container_width=True, hide_index=True)

                        # Show chain cache counts
                        if "options_chains_cache" in tables:
                            chain_summary = conn.execute("""
                                SELECT symbol, timestamp, COUNT(*) as contracts
                                FROM options_chains_cache
                                WHERE (symbol, timestamp) IN (
                                    SELECT symbol, MAX(timestamp) FROM options_chains_cache GROUP BY symbol
                                )
                                GROUP BY symbol, timestamp
                                ORDER BY timestamp DESC
                            """).fetchall()
                            if chain_summary:
                                chain_info = ", ".join([f"{r[0]}: {r[2]} contracts" for r in chain_summary])
                                st.caption(f"**Cached Option Chains**: {chain_info}")
                    else:
                        st.info("No volatility data cached yet. Load the Volatility Engine page to populate the cache.")
                else:
                    st.info("Cache tables not initialized yet.")
                conn.close()
            else:
                st.info("options_liquidity database not found.")
        except Exception as e:
            st.warning(f"Could not read cache status: {e}")
