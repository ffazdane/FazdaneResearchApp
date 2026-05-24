# FazDane Release Verification & Testing Process

This document defines the mandatory Quality Assurance (QA) testing protocol that must be completed and passed in full prior to deploying any code updates to the production environment. 

---

## Phase 1: Automated Verification
Run the automated pre-release test suite to verify that all modules are syntax-clean, importable, and local database connections/persistence paths are properly resolved.

### Option A: Running from the Streamlit UI (Recommended)
1. Open the local dashboard: [http://localhost:8501](http://localhost:8501).
2. Go to the sidebar and expand **Database Management**.
3. Click the red/primary **🚀 Run Pre-Release Checks** button.
4. Verify that the checks run, display `Pre-Release Checks: PASSED`, and show the detailed test logs below it.

### Option B: Running from the Terminal
Execute the test script directly from your project root:
```powershell
.\.venv\Scripts\python.exe scripts/run_release_tests.py
```

### Verification Criteria:
- [ ] **Module Imports**: All module classes and decorators (25+ modules/utilities) must import successfully with `[PASS]`.
- [ ] **Database Connection**: The local SQLite database connections for the 4 databases must return table lists successfully with `[PASS]`.
- [ ] **Persistence Sandbox**: Backup directory must be writable and a local backup of `options_liquidity` must generate cleanly in `data/backups/`.
- [ ] **Yahoo Finance Connectivity**: The script must successfully perform a mock fetch to query historical bars for validation.
- [ ] **Exit Code / UI Status**: The script must terminate with `*** ALL VERIFICATION CHECKS PASSED SUCCESSFULLY! ***` or display a green status check in Streamlit.

---

## Phase 2: Database Backup & Restore Verification
Perform manual confirmation of the database control panel inside the Streamlit local environment.

1. **Access the Database Management Panel**:
   - Open [http://localhost:8501](http://localhost:8501) in your browser.
   - Look at the sidebar and expand the **Database Management** expander.

2. **Verify Backup Operations**:
   - [ ] Verify that current database sizes (in KB) are displayed correctly for all active databases.
   - [ ] Click the **Backup DBs** button. Wait for completion.
   - [ ] Verify that a success message is displayed (e.g. `Backup success: ...`) and the log at the bottom lists a new `BACKUP` entry with a timestamp.

3. **Verify Restore Operations**:
   - [ ] Click the **Restore DBs** button. Wait for completion.
   - [ ] Verify that a success message is displayed (e.g. `Restored: ...`) and the log lists a new `RESTORE` entry.

4. **Verify Ingestion/Rebuilding**:
   - [ ] Click **Rebuild/Patch DB from Online**.
   - [ ] Verify that the database is fully populated from scratch (which queries Yahoo Finance historical price bars and parses monthly expiries) and returns a success notification.

---

## Phase 3: Module & UI Option Testing Checklist
Manually navigate through each tier and select the specified modules. Test the following features to ensure nothing has broken:

### 🟢 Tier 1: Live Trading Execution
- [ ] **Option Search Universe Engine**:
  - Select different pre-configured universes (e.g., *Mag 7*, *Premium Selling Favorites*).
  - Verify that the scoring table renders and displays the highest-scoring tickers.
- [ ] **Options Liquidity Discovery**:
  - Input a ticker (e.g., `SPY`).
  - Verify that contract lists are generated with bid-ask spreads, volume, and open interest metrics.
- [ ] **Market Breadth Dashboard**:
  - Verify that breadth indicators (Advancing/Declining ratio, McClellan Oscillator, etc.) render charts.
- [ ] **Calendar Strategy Matrix**:
  - Test switching between monthly and quarterly views.
- [ ] **Iron Condor Analyzer**:
  - Enter `QQQ` and check the calculated credit received, break-evens, and probability of profit bands.
- [ ] **ES Pivot Analysis**:
  - Verify that daily pivot lines (R3/S3) are generated and displayed in a table.
- [ ] **Sector Rotation Monitor**:
  - Verify that the relative strength comparison graph renders Sector ETF trends.

---

### 🔵 Tier 2: Analysis & Intelligence
- [ ] **Universe Intelligence System**:
  - Toggle between different sectors and verify that ticker classifications load correctly.
- [ ] **Portfolio Performance**:
  - Check the daily performance chart and verify that the P/L calculation aggregates properly.
- [ ] **Portfolio Performance & Risk Management**:
  - Verify that beta-weighted portfolio delta, total theta, and sector concentration heatmaps display correctly.
- [ ] **Multi-Timeframe Money Flow**:
  - Verify that volume-based indicators (OBV, Chaikin, etc.) generate charts on selected symbols.
- [ ] **Market Structure Heatmap**:
  - Verify that the S&P 500 sector tiles render and display correct coloring based on percentage changes.
- [ ] **Correlation Matrix**:
  - Verify that selecting multiple tickers (e.g. `SPY, AAPL, MSFT, TLT`) renders a clean, color-graded correlation grid.
- [ ] **Earnings Calendar**:
  - Verify that upcoming earnings announcements for the next 7 and 30 days are rendered in table format.
- [ ] **Equity Income Statement**:
  - Verify that fundamental details (Revenues, Gross Margins, Net Income) render for a selected ticker.
- [ ] **Equity / Index Seasonality**:
  - Verify that month-of-year and day-of-week seasonal averages display their bar charts.
- [ ] **Stock Sentiment Analysis**:
  - Verify that sentiment indicators show historical sentiment progression charts.
- [ ] **Social Stock Stories**:
  - Check the rendering of social stock activity logs.

---

### 🟡 Tier 3: Forecasting & Cycles
- [ ] **Bradley Siderograph**:
  - Verify that the Bradley turning points chart displays correct dates and curves.
- [ ] **Elliott Wave Analysis**:
  - Check that waves (1-5 and A-C correction waves) are identified and drawn onto the price charts.

---

### 🔴 Tier 4: Volatility & Dealer Positioning
- [ ] **Volatility Strategy Engine**:
  - Enter a ticker (e.g., `TSLA` or `SPY`).
  - Verify that **metric indicators** (Last Price, ATM IV, HV20, HVR, expected move bands) display numerical values.
  - Verify the **Candlestick and IV vs realized volatility** chart renders.
  - Check the **Volatility Structure** tab and verify that the **IV Term Structure** curve renders.
  - Check the **Strategy Engine** tab and verify that a strategy recommendation badge (e.g., *AVOID SELLING*, *SELL IRON CONDOR*) displays.
- [ ] **Gamma Flip Line Module**:
  - Input a ticker (e.g., `SPY`).
  - Verify that zero-gamma flip points are calculated and the net dealer GEX exposure curve renders correctly.

---

## Phase 4: Release Sign-Off
Before merging the release branch into `main` (or pushing to production):

1. **Verify logs contain no active errors**:
   - Check `logs/fazdane.log` and ensure there are no unhandled exception stack traces.
2. **Execute build validation**:
   - Confirm that the codebase builds without errors.
3. **Database Cloud Sync**:
   - Confirm that database cloud backups (via GitHub or S3, if configured) sync correctly with the production environment.
