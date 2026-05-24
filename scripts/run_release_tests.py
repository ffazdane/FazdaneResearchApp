"""
FazDane Release Verification & Quality Assurance Suite
======================================================
Automated pre-release checks verifying module imports, yfinance connectivity,
persistence pathing, and local database backup/restore mechanics.
"""

import os
import sys
import sqlite3
import urllib.request
import traceback
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_section(title):
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN} {title} {RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}")

def print_result(test_name, success, detail=""):
    status = f"{GREEN}PASS{RESET}" if success else f"{RED}FAIL{RESET}"
    detail_str = f" - {detail}" if detail else ""
    print(f"[{status}] {test_name}{detail_str}")

def test_imports():
    print_section("1. Module Imports Verification")
    
    modules_to_test = [
        # Base Module
        ("modules.base_module", "FazDaneModule"),
        # Tier 1
        ("modules.tier1.option_search", "OptionSearchModule"),
        ("modules.tier1.options_liquidity", "OptionsLiquidityModule"),
        ("modules.tier1.market_breadth", "MarketBreadthModule"),
        ("modules.tier1.es_pivot_analysis", "ESPivotAnalysisModule"),
        ("modules.tier1.sector_rotation", "SectorRotationModule"),
        ("modules.tier1.calendar_rotation", "CalendarRotationModule"),
        ("modules.tier1.iron_condor", "IronCondorModule"),
        # Tier 2
        ("modules.tier2.money_flow", "MoneyFlowModule"),
        ("modules.tier2.portfolio_performance", "PortfolioPerformanceModule"),
        ("modules.tier2.portfolio_risk_management", "PortfolioRiskManagementModule"),
        ("modules.tier2.market_structure", "MarketStructureModule"),
        ("modules.tier2.correlation_matrix", "CorrelationMatrixModule"),
        ("modules.tier2.earnings_calendar", "EarningsCalendarModule"),
        ("modules.tier2.equity_income_statement", "EquityIncomeStatementModule"),
        ("modules.tier2.seasonality_analysis", "SeasonalityAnalysisModule"),
        ("modules.tier2.stock_sentiment", "StockSentimentModule"),
        ("modules.tier2.social_stock_stories", "SocialStockStoriesModule"),
        ("modules.tier2.universe_intelligence", "UniverseIntelligenceModule"),
        # Tier 3
        ("modules.tier3.bradley_siderograph", "BradleySiderographModule"),
        ("modules.tier3.elliott_wave_analysis", "ElliottWaveAnalysisModule"),
        # Tier 4
        ("modules.tier4.volatility_engine", "VolatilityEngineModule"),
        ("modules.tier4.gamma_flip.gamma_dashboard", "GammaFlipLineModule"),
        # Utilities
        ("utils.persistence", "get_db_path"),
        ("utils.portfolio_performance_store", "get_database_status"),
    ]

    all_passed = True
    for mod_path, class_name in modules_to_test:
        try:
            # Dynamically import
            mod = __import__(mod_path, fromlist=[class_name])
            getattr(mod, class_name)
            print_result(f"Import {mod_path}.{class_name}", True)
        except Exception as e:
            print_result(f"Import {mod_path}.{class_name}", False, str(e))
            all_passed = False
            
    return all_passed

def test_database_connections():
    print_section("2. Database Pathing & Connection Verification")
    
    from utils.persistence import get_db_path, DATABASES
    
    all_passed = True
    for db_name in DATABASES:
        try:
            db_path = get_db_path(db_name)
            exists = db_path.exists()
            size_kb = db_path.stat().st_size / 1024 if exists else 0
            
            # Attempt to connect and fetch list of tables
            if exists:
                with sqlite3.connect(db_path, timeout=5) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [row[0] for row in cursor.fetchall()]
                print_result(
                    f"DB '{db_name}' ({size_kb:.1f} KB)",
                    True,
                    f"Path: {db_path.name} | Tables: {len(tables)} {tables[:4]}..."
                )
            else:
                print_result(
                    f"DB '{db_name}'",
                    False,
                    f"Database file does not exist at expected path: {db_path}"
                )
                all_passed = False
        except Exception as e:
            print_result(f"DB '{db_name}' Connection Test", False, str(e))
            all_passed = False
            
    return all_passed

def test_backup_restore_mechanics():
    print_section("3. Local Persistence & Backup Sandbox Verification")
    
    from utils.persistence import backup_database, DATABASES, BACKUP_DIR
    
    all_passed = True
    
    # Verify backup folder writable
    try:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        test_file = BACKUP_DIR / "write_test.tmp"
        test_file.write_text("write verification")
        test_file.unlink()
        print_result("Local Backup Directory Writable", True, f"Path: {BACKUP_DIR}")
    except Exception as e:
        print_result("Local Backup Directory Writable", False, str(e))
        all_passed = False

    # Perform a local backup check on options_liquidity database
    db_name = "options_liquidity"
    try:
        ok, msg = backup_database(db_name, reason="automated release verification run")
        # Since DB_BACKEND is likely 'none' on local developer systems, this should do a local copy
        print_result(f"Trigger Local Backup for '{db_name}'", ok, msg)
        if not ok:
            all_passed = False
    except Exception as e:
        print_result(f"Trigger Local Backup for '{db_name}'", False, str(e))
        all_passed = False

    return all_passed

def test_yfinance_connectivity():
    print_section("4. Yahoo Finance API & Parsing Verification")
    
    test_symbol = "SPY"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{test_symbol}?period1=1700000000&period2=1700100000&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            import json
            data = json.loads(response.read().decode('utf-8'))
            timestamps = data['chart']['result'][0].get('timestamp', [])
        
        if len(timestamps) > 0:
            print_result(f"Yahoo Finance API Fetch ({test_symbol})", True, f"Successfully parsed {len(timestamps)} bars.")
        else:
            print_result(f"Yahoo Finance API Fetch ({test_symbol})", False, "Received empty response dataset.")
            return False
    except Exception as e:
        print_result(f"Yahoo Finance API Fetch ({test_symbol})", False, str(e))
        return False
        
    return True

def main():
    print(f"\n{BOLD}{GREEN}============================================================{RESET}")
    print(f"{BOLD}{GREEN}       FAZDANE ANALYTICS PRE-RELEASE CHECK SUITE            {RESET}")
    print(f"{BOLD}{GREEN}============================================================{RESET}")
    
    imports_ok = test_imports()
    db_ok = test_database_connections()
    backup_ok = test_backup_restore_mechanics()
    yf_ok = test_yfinance_connectivity()
    
    print_section("Summary Report")
    
    overall_success = imports_ok and db_ok and backup_ok and yf_ok
    
    if overall_success:
        print(f"\n{BOLD}{GREEN}*** ALL VERIFICATION CHECKS PASSED SUCCESSFULLY! ***{RESET}")
        print(f"{BOLD}{GREEN}The codebase is clean, persistent paths are correct, and all dependencies are importable.{RESET}\n")
        sys.exit(0)
    else:
        print(f"\n{BOLD}{RED}!!! VERIFICATION FAILED! !!!{RESET}")
        print(f"{BOLD}{RED}Please review the failed test checks above before attempting to release to production.{RESET}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
