# 🎯 VIBE CODING FRAMEWORK
## FazDane Research Application - Module-by-Module Streamlit Migration

**Framework Type**: Prioritized, Menu-Driven, Authentication-Gated Finance Dashboard  
**Architecture**: Modular Streamlit with Google Drive Integration  
**Target Audience**: Finance Traders, Portfolio Managers, Options Traders  
**Aesthetic Direction**: Dark Finance Sophistication + Real-Time Intelligence  

---

## 📋 PART 1: FRAMEWORK ARCHITECTURE & PRIORITY SYSTEM

### 1.1 Overall System Design

```
┌─────────────────────────────────────────────────────────┐
│           FAZDANE RESEARCH APPLICATION                  │
│        (Streamlit-Based Dashboard Platform)             │
└─────────────────────────────────────────────────────────┘
         │
         ├─ 🔐 AUTHENTICATION LAYER
         │   ├─ Login Screen (Username/Password + 2FA)
         │   ├─ Session Management (st.session_state)
         │   └─ Role-Based Access Control
         │
         ├─ 📦 MODULE ORCHESTRATOR (Main Menu)
         │   ├─ Sidebar Navigation
         │   ├─ Module Health Dashboard
         │   └─ Quick-Access Favorites
         │
         ├─ 🗂️ DYNAMIC MODULE LOADER
         │   ├─ Import from Google Drive
         │   ├─ Cache & Version Control
         │   └─ Hot-Reload Capability
         │
         └─ 🧩 FINANCE MODULES (4 Priority Tiers)
             ├─ Tier 1 (Highest Priority - Weeks 1-2)
             ├─ Tier 2 (High Priority - Weeks 3-4)
             ├─ Tier 3 (Medium Priority - Weeks 5-6)
             └─ Tier 4 (Lower Priority - Week 7+)
```

---

## 🎬 PART 2: PRIORITY-BASED MODULE ROADMAP

### **TIER 1: CRITICAL TRADING MODULES** (Weeks 1-2) - ✅ 100% COMPLETE
*Live trading, real-time decisions, high revenue impact*

| **Module** | **Source Notebook** | **Status** | **Data Deps** | **Integration** |
|-----------|-------------------|-----------|----------|------------------|
| **1. Options Liquidity Discovery** | `05-FazDane Options Liquidity Discovery Engine.ipynb` | 🟢 Live | Live API | Yahoo Finance + IV Rank |
| **2. SPX Market Breadth** | `06-Market Breadth Dashboard.ipynb` | 🟢 Live | Daily EOD | Real-time updates via caching |
| **3. Sector Rotation Monitor** | `05-SPX Sector Rotation RRG-Style Visualization.ipynb` | 🟢 Live | Sector ETF data | Real-time calc + Plotly |
| **4. Calendar Strategy Matrix** | `Average Return-Daily-Calendar View.ipynb` | 🟢 Live | Historical returns | Heatmap + Rotation matrix |
| **5. Iron Condor Analyzer** | `IronCondor Dashboard.ipynb` | 🟢 Live | Historical OHLCV | Strategy builder + Payoff diagram |
| **6. ES Pivot Analysis** | `ES Volume Profile and Pivot Confluence.ipynb` | 🟢 Live | Real-time 5m/1h | Plotly Volume Profile + Pivots |

**Why Tier 1?** Revenue-generating, used daily, support 80% of trading activity

**Dependencies to Install First:**
```bash
pip install streamlit streamlit-authenticator google-auth-oauthlib pandas yfinance numpy scipy plotly pandas-ta
```

**Google Drive Locations (Map These First):**
```
/My Drive/Python Projects/Notebooks/
├── 05-FazDane Options Liquidity Discovery Engine.ipynb
├── 06-Market Breadth Dashboard.ipynb
├── Future Analysis/ES 3-Month Intraday Strategy Backtest.ipynb
├── Portfolio/FazDane Portfolio Backtester and Optimizer.ipynb
└── 05-SPX Sector Rotation RRG-Style Visualization.ipynb
```

---

### **TIER 2: ANALYSIS & INTELLIGENCE MODULES** (Weeks 3-4) - 🧠 Calendar Scoring Phase 2 ✅ 100% COMPLETE
*Decision support, research, pattern identification*

| **Module** | **Source Notebook** | **Priority** | **Data Deps** |
|-----------|-------------------|-----------|----------|
| **6. Stock Ticker Screener** | `Equity Analysis/Stock Analysis.ipynb` + `Best Uptrending Calendar Candidates.ipynb` | P1 | EOD OHLCV |
| **7. Implied Volatility Analysis** | `Equity Analysis/Implied Volatility Analysis.ipynb` | P1 | Live IV data |
| **8. Index Analysis Dashboard** | `04-Index Analysis.ipynb` | P1 | Real-time quotes |
| **9. Calendar Heatmaps** | `Average Return-Daily-Calendar View.ipynb` | P1 | Historical returns |
| **10. Earnings Calendar** | `Earning Calendar.ipynb` | P1 | Earnings API |
| **11. Calendar Opportunity Scoring Engine** | `Average Return-Daily-Calendar View.ipynb` + Phase 2 Models | P1 | Historical daily indices, options chain, intraday bars (Phase 2 Advanced Intelligence Models live) |

---

### **TIER 3: FORECASTING & STRUCTURAL** (Weeks 5-6)
*Medium-term predictions, wave analysis, cycle timing*

| **Module** | **Source Notebook** | **Priority** |
|-----------|-------------------|-----------|
| **11. SPX Price Forecasting** | `Forecasting/Price Forecasting/spx_price_forecasting_prophet_arima_lstm_with_bradley.ipynb` | P2 |
| **12. Bradley Siderograph Cycles** | `Forecasting/Cycle Analysis/Bradley Siderograph.ipynb` | P2 |
| **13. Elliott Wave Analysis** | `Structural Study/Elliot Wave.ipynb` | P2 |
| **14. Market Cycle Timing** | `Forecasting/Long Term Cycle/CycleSync: SPX Market Timing Model.ipynb` | P2 |

---

### **TIER 4: SPECIALIZED & RESEARCH** (Week 7+)
*Niche analysis, learning, experimental*

| **Module** | **Source Notebook** | **Priority** |
|-----------|-------------------|-----------|
| **15. Macro Intelligence** | `FazDane Macro Intelligence Dashboard.ipynb` | P3 |
| **16. Moon Cycle Analysis** | `Forecasting/Cycle Analysis/Moon Cycle Analysis with SPX Data.ipynb` | P3 |
| **17. Iron Condor Analyzer** | `IronCondor Dashboard.ipynb` | P3 |
| **18. Volume Profile Analysis** | `Future Analysis/ES Volume Profile and Pivot Confluence.ipynb` | P3 |

---

## 🔐 PART 3: AUTHENTICATION FRAMEWORK

### 3.1 Login System (Week 1 - Foundation)

**File**: `app_auth.py`

```python
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from pathlib import Path

# Configuration
CONFIG_FILE = Path("config.yaml")  # Store credentials securely
CREDENTIALS = {
    "usernames": {
        "fazal": {"name": "Fazal", "password": "hashed_password_here", "role": "admin"},
        "trader1": {"name": "Trader 1", "password": "hashed_password_here", "role": "user"},
    }
}

class FazDaneAuthenticator:
    def __init__(self):
        self.authenticated = False
        self.user_info = None
    
    def login_screen(self):
        """Render login UI with FazDane branding"""
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 40px 0;">
                <h1 style="color: #00ff88; font-size: 48px; margin: 0;">⚡ FazDane</h1>
                <p style="color: #888; font-size: 14px; margin-top: 8px;">Research & Trading Intelligence</p>
            </div>
            """, unsafe_allow_html=True)
            
            username = st.text_input("🔑 Username", key="login_user")
            password = st.text_input("🔒 Password", type="password", key="login_pass")
            
            if st.button("Login", use_container_width=True, type="primary"):
                if self._validate_credentials(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.user_role = CREDENTIALS["usernames"][username]["role"]
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
    
    def _validate_credentials(self, username, password):
        """Validate against secure credential store"""
        # TODO: Implement bcrypt password hashing + salt
        # For now: placeholder
        return username in CREDENTIALS["usernames"]
    
    def logout(self):
        """Clear session and return to login"""
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()
```

**Credentials Storage** (`config.yaml` - GITIGNORE THIS):
```yaml
credentials:
  usernames:
    fazal:
      name: Fazal
      password: $2b$12$... # bcrypt hash
      role: admin
    trader1:
      name: Trader One
      password: $2b$12$...
      role: user
    trader2:
      name: Trader Two
      password: $2b$12$...
      role: user

2fa_enabled: false  # Enable for production
session_timeout: 3600  # 1 hour
```

---

## 🧩 PART 4: MODULAR ARCHITECTURE

### 4.1 Module Structure (Each Module = 1 Jupyter NB → 1 Python Module)

**Base Module Template**: `modules/base_module.py`

```python
from abc import ABC, abstractmethod
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, Any

class FazDaneModule(ABC):
    """Base class for all FazDane modules"""
    
    MODULE_NAME = "Base Module"
    MODULE_ICON = "📊"
    TIER = 1
    REQUIRES_LIVE_DATA = False
    CACHE_TTL = 3600  # seconds
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.data_cache = {}
    
    @abstractmethod
    def render_sidebar(self):
        """Render module-specific sidebar controls"""
        pass
    
    @abstractmethod
    def render_main(self):
        """Render main content area"""
        pass
    
    def run(self):
        """Execute module pipeline"""
        st.set_page_config(
            page_title=self.MODULE_NAME,
            page_icon=self.MODULE_ICON,
            layout="wide"
        )
        
        # Sidebar controls
        with st.sidebar:
            st.markdown(f"## {self.MODULE_ICON} {self.MODULE_NAME}")
            self.render_sidebar()
        
        # Main content
        self.render_main()
    
    @st.cache_data(ttl=3600)
    def fetch_data(self, source: str, **kwargs) -> pd.DataFrame:
        """Cached data fetching"""
        # Implement per module
        pass
    
    def _setup_logger(self):
        import logging
        return logging.getLogger(self.MODULE_NAME)
```

---

### 4.2 Google Drive Integration Layer

**File**: `utils/google_drive_sync.py`

```python
import streamlit as st
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import os

class GoogleDriveSync:
    def __init__(self):
        self.service = self._authenticate()
        self.folder_mapping = {
            "Options": "1a2b3c4d5e6f7g8h9",  # Google Drive Folder ID
            "Equity": "2a2b3c4d5e6f7g8h9",
            "Futures": "3a2b3c4d5e6f7g8h9",
            "Portfolio": "4a2b3c4d5e6f7g8h9",
            "Forecasting": "5a2b3c4d5e6f7g8h9",
        }
    
    def _authenticate(self):
        """OAuth2 authentication to Google Drive"""
        auth.authenticate_user()
        return build('drive', 'v3')
    
    def list_notebooks(self, category: str):
        """List all notebooks in a category folder"""
        folder_id = self.folder_mapping.get(category)
        
        query = f"'{folder_id}' in parents and trashed=false"
        results = self.service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name, modifiedTime)',
            pageSize=20
        ).execute()
        
        return results.get('files', [])
    
    def download_notebook(self, file_id: str, local_path: str):
        """Download notebook from Google Drive"""
        request = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while not done:
            status, done = downloader.next_chunk()
        
        with open(local_path, 'wb') as f:
            f.write(fh.getvalue())
    
    def get_file_version(self, file_id: str) -> str:
        """Get file's modified time (for version checking)"""
        file = self.service.files().get(fileId=file_id).execute()
        return file['modifiedTime']
```

**Usage in Modules**:
```python
# In any module that needs to load notebook data
drive_sync = GoogleDriveSync()
notebooks = drive_sync.list_notebooks("Options")

# Download if updated
file_version = drive_sync.get_file_version(notebook_id)
if st.session_state.get(f"version_{notebook_id}") != file_version:
    drive_sync.download_notebook(notebook_id, f"cache/{notebook_id}.ipynb")
    st.session_state[f"version_{notebook_id}"] = file_version
```

---

## 📦 PART 5: TIER 1 MODULE CONVERSION EXAMPLES

### Module 1: Options Liquidity Discovery Engine

**Source**: `05-FazDane Options Liquidity Discovery Engine.ipynb`  
**File**: `modules/tier1/options_liquidity.py`

```python
import streamlit as st
import pandas as pd
import yfinance as yf
from modules.base_module import FazDaneModule
import plotly.graph_objects as go

class OptionsLiquidityModule(FazDaneModule):
    MODULE_NAME = "Options Liquidity Discovery"
    MODULE_ICON = "💧"
    TIER = 1
    
    def render_sidebar(self):
        st.markdown("### Search Parameters")
        
        # Filters
        min_volume = st.slider("Min Option Volume", 100, 10000, 1000, 100)
        min_liquidity = st.slider("Min IV Rank (%)", 0, 100, 50, 5)
        option_type = st.multiselect(
            "Option Type",
            ["Call", "Put"],
            default=["Call", "Put"]
        )
        
        expiration = st.selectbox(
            "Expiration",
            ["Weekly", "Monthly", "Quarterly"]
        )
        
        if st.button("🔍 Scan Options", use_container_width=True, type="primary"):
            st.session_state.scan_params = {
                'min_volume': min_volume,
                'min_liquidity': min_liquidity,
                'option_type': option_type,
                'expiration': expiration
            }
    
    def render_main(self):
        st.markdown("## Options Liquidity Heatmap")
        
        if 'scan_params' not in st.session_state:
            st.info("👈 Set parameters in sidebar and click 'Scan Options'")
            return
        
        # Fetch data
        with st.spinner("Scanning options..."):
            df = self._scan_options(st.session_state.scan_params)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(df.head(20), use_container_width=True)
        
        with col2:
            # Heatmap of volume vs IV rank
            fig = go.Figure(data=go.Heatmap(
                z=df['volume'],
                x=df['ticker'],
                y=df['iv_rank'],
                colorscale='Viridis'
            ))
            st.plotly_chart(fig, use_container_width=True)
    
    @st.cache_data(ttl=300)
    def _scan_options(self, params):
        """Scan for high-liquidity options"""
        # Implementation: use options API (OptionChain, ThinkOrSwim, etc.)
        # Return DataFrame with columns: ticker, volume, iv_rank, bid, ask, delta
        pass
```

### Module 2: SPX Market Breadth Dashboard

**Source**: `06-Market Breadth Dashboard.ipynb`  
**File**: `modules/tier1/market_breadth.py`

```python
import streamlit as st
import pandas as pd
import yfinance as yf
from modules.base_module import FazDaneModule
import plotly.express as px

class MarketBreadthModule(FazDaneModule):
    MODULE_NAME = "SPX Market Breadth"
    MODULE_ICON = "📊"
    TIER = 1
    
    BREADTH_INDICES = {
        'NYSE Advance/Decline': '^ADVN',
        'NASDAQ Advance/Decline': '^ADVD',
        'Market Leaders': '^NYA',
        'Market Laggards': '^NYD',
    }
    
    def render_sidebar(self):
        st.markdown("### Breadth Analysis")
        
        lookback = st.slider("Days Back", 5, 252, 60)
        
        breadth_type = st.multiselect(
            "Show Breadth Indices",
            list(self.BREADTH_INDICES.keys()),
            default=['NYSE Advance/Decline']
        )
        
        if st.button("📈 Refresh Data", use_container_width=True):
            st.cache_data.clear()
    
    def render_main(self):
        st.markdown("## Market Breadth Analysis")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Advance/Decline Ratio", 1.24, "+0.08")
        with col2:
            st.metric("Breadth Momentum", 68, "-5")
        with col3:
            st.metric("Highest at 52W", 73, "-2")
        with col4:
            st.metric("Lowest at 52W", 32, "+8")
        
        # Charts
        breadth_data = self._fetch_breadth()
        
        fig = px.line(
            breadth_data,
            title="Advance/Decline Momentum",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    @st.cache_data(ttl=3600)
    def _fetch_breadth(self):
        """Fetch breadth indicators"""
        # Implementation
        pass
```

---

## 🎨 PART 6: MAIN APPLICATION ARCHITECTURE

**File**: `app.py` (Main Entry Point)

```python
import streamlit as st
from app_auth import FazDaneAuthenticator
from modules.tier1 import options_liquidity, market_breadth, es_backtester, portfolio_optimizer, sector_rotation
from modules.tier2 import ticker_screener, iv_analysis, index_analysis
from utils.google_drive_sync import GoogleDriveSync

# ═════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="FazDane Research | Trading Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═════════════════════════════════════════════════════════════
# CUSTOM STYLING
# ═════════════════════════════════════════════════════════════

st.markdown("""
<style>
    :root {
        --primary: #00ff88;
        --secondary: #0f172a;
        --accent: #ff6b6b;
        --border: #1e293b;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1a1f3a 100%);
        border-right: 1px solid #1e293b;
    }
    
    .main {
        background: #0a0e27;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# AUTHENTICATION CHECK
# ═════════════════════════════════════════════════════════════

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    authenticator = FazDaneAuthenticator()
    authenticator.login_screen()
    st.stop()

# ═════════════════════════════════════════════════════════════
# AUTHENTICATED APP STRUCTURE
# ═════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"### ⚡ FazDane Research\n**User**: {st.session_state.username}")
    st.divider()
    
    # Main Navigation
    st.markdown("### 📊 TIER 1: TRADING (Priority)")
    tier1_selected = st.radio(
        "Select Module",
        options=[
            "Options Liquidity 💧",
            "Market Breadth 📈",
            "ES Backtester ⚙️",
            "Portfolio Optimizer 🎯",
            "Sector Rotation 🔄",
        ],
        key="tier1_nav"
    )
    
    st.markdown("### 📈 TIER 2: ANALYSIS")
    tier2_selected = st.radio(
        "Advanced Tools",
        options=[
            "Ticker Screener 🔍",
            "IV Analysis 📊",
            "Index Analysis 🏦",
        ],
        key="tier2_nav"
    )
    
    st.divider()
    
    # System Info
    st.markdown("### ⚙️ System")
    if st.button("🔄 Sync Google Drive", use_container_width=True):
        st.info("Syncing notebooks from Google Drive...")
    
    if st.button("🔐 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

# ═════════════════════════════════════════════════════════════
# MODULE DISPATCHER
# ═════════════════════════════════════════════════════════════

TIER1_MODULES = {
    "Options Liquidity 💧": options_liquidity.OptionsLiquidityModule(),
    "Market Breadth 📈": market_breadth.MarketBreadthModule(),
    "ES Backtester ⚙️": es_backtester.ESBacktesterModule(),
    "Portfolio Optimizer 🎯": portfolio_optimizer.PortfolioOptimizerModule(),
    "Sector Rotation 🔄": sector_rotation.SectorRotationModule(),
}

TIER2_MODULES = {
    "Ticker Screener 🔍": ticker_screener.TickerScreenerModule(),
    "IV Analysis 📊": iv_analysis.IVAnalysisModule(),
    "Index Analysis 🏦": index_analysis.IndexAnalysisModule(),
}

# Route to selected module
selected = tier1_selected or tier2_selected

if selected in TIER1_MODULES:
    TIER1_MODULES[selected].run()
elif selected in TIER2_MODULES:
    TIER2_MODULES[selected].run()
else:
    st.markdown("## Welcome to FazDane Research 👋")
    st.info("Select a module from the sidebar to begin")
```

---

## 🚀 PART 7: DEPLOYMENT & EXECUTION GUIDE

### Step 1: Setup Project Structure

```bash
fazdane-research-app/
├── app.py                          # Main entry point
├── app_auth.py                     # Authentication system
├── config.yaml                     # Credentials (GITIGNORE)
├── requirements.txt                # Dependencies
│
├── modules/
│   ├── __init__.py
│   ├── base_module.py              # Base class for all modules
│   ├── tier1/
│   │   ├── options_liquidity.py    # Options Liquidity Module
│   │   ├── market_breadth.py       # Market Breadth Module
│   │   ├── es_backtester.py        # ES Backtester Module
│   │   ├── portfolio_optimizer.py  # Portfolio Optimizer Module
│   │   └── sector_rotation.py      # Sector Rotation Module
│   ├── tier2/
│   │   ├── ticker_screener.py
│   │   ├── iv_analysis.py
│   │   └── index_analysis.py
│   └── tier3/
│       ├── price_forecasting.py
│       └── ...
│
├── utils/
│   ├── google_drive_sync.py        # Google Drive integration
│   ├── data_cache.py               # Caching layer
│   └── helpers.py                  # Utility functions
│
└── data/
    ├── cache/                      # Downloaded notebooks
    └── logs/                        # Application logs
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**:
```
streamlit==1.28.1
streamlit-authenticator==0.2.3
pandas==2.0.3
numpy==1.24.3
yfinance==0.2.28
pandas-ta==0.3.14b
plotly==5.14.0
scipy==1.11.1
google-auth-oauthlib==1.0.0
google-auth-httplib2==0.1.1
google-api-python-client==2.92.0
bcrypt==4.0.1
python-dotenv==1.0.0
```

### Step 3: Configure Google Drive Access

1. Create OAuth2 credentials at [Google Cloud Console](https://console.cloud.google.com/)
2. Download JSON key → Save as `google_drive_creds.json`
3. Update `google_drive_sync.py` with folder IDs from your Drive

**Find Folder IDs**:
```python
# Run this once to map your folders
drive_sync = GoogleDriveSync()
results = drive_sync.service.files().list(
    q="name='FazDane Options Liquidity Discovery Engine.ipynb'",
    spaces='drive',
    fields='files(id, name, parents)'
).execute()

print(results)  # Get parent folder ID
```

### Step 4: Run Application

```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

---

## 🎯 PART 8: MODULE CONVERSION CHECKLIST

Use this for converting each Jupyter notebook to a Streamlit module:

```markdown
# Module Conversion Template

## Phase 1: Analysis (1 hour)
- [ ] Read source notebook end-to-end
- [ ] Identify all data sources (APIs, files, live feeds)
- [ ] Map parameters/filters from notebook → sidebar controls
- [ ] List all visualizations (charts, tables, heatmaps)
- [ ] Document refresh frequency (real-time vs. cached)

## Phase 2: Extraction (2 hours)
- [ ] Copy all data processing logic
- [ ] Extract visualization code (Plotly, Matplotlib → Plotly)
- [ ] Document all dependencies (packages, APIs)
- [ ] Create sample data for testing

## Phase 3: Streamlit Refactor (3 hours)
- [ ] Create module class inheriting from FazDaneModule
- [ ] Implement render_sidebar() with all filters
- [ ] Implement render_main() with visualizations
- [ ] Add @st.cache_data decorators for expensive operations
- [ ] Test module in isolation

## Phase 4: Integration (1 hour)
- [ ] Add to TIER1_MODULES/TIER2_MODULES dict
- [ ] Add to sidebar navigation
- [ ] Test with authentication flow
- [ ] Add error handling + loading states

## Phase 5: QA & Polish (1 hour)
- [ ] Test with real data
- [ ] Verify cache behavior
- [ ] Check mobile responsiveness
- [ ] Document any caveats

**Total Time Per Module: ~8 hours**
**Tier 1 (5 modules) Total: ~40 hours**
```

---

## 📊 PART 9: VIBE & AESTHETIC DIRECTION

### Visual Language

**Color Palette** (Dark Finance Sophistication):
```css
Primary: #00ff88 (Neon Green - Action/Success)
Secondary: #0f172a (Deep Navy - Background)
Accent: #ff6b6b (Red - Alerts/Warnings)
Neutral: #64748b (Slate - Secondary Info)
Success: #10b981 (Emerald - Gains)
Danger: #ef4444 (Red - Losses)
```

### Typography
- **Display Font**: Courier Prime (Monospace - Finance authenticity)
- **Body Font**: Inter (Modern, clean)
- **Sizes**: 48px (headers), 24px (titles), 14px (body)

### Component Style Guide

**Metric Cards** (4-column layout):
```html
┌─────────────────────┐
│ 📊 Label            │
│ 1,234.56            │  ← 28px, bold
│ ↑ 12.3% (+$45)      │  ← 12px, green
└─────────────────────┘
```

**Data Tables**:
- Striped rows (alternating #0f172a / #1a1f3a)
- Right-aligned numerics
- Hover highlight (#1e293b)
- Sortable columns

**Charts**:
- Dark background (#0a0e27)
- Light gridlines (#1e293b)
- Plotly native theme
- Responsive sizing

---

## 🔄 PART 10: ITERATION WORKFLOW

### Weekly Sprint Structure (7-Week Plan)

```
WEEK 1: Foundation + Tier 1.1
├─ Day 1: Auth system + base module template
├─ Day 2: Options Liquidity module
├─ Day 3: Market Breadth module
├─ Day 4: Testing + bug fixes
└─ Day 5: Google Drive sync integration

WEEK 2: Tier 1.2
├─ Days 1-2: ES Backtester module
├─ Days 3-4: Portfolio Optimizer module
├─ Day 5: Sector Rotation module

WEEKS 3-4: Tier 2 (Analysis modules)
WEEKS 5-6: Tier 3 (Forecasting modules)
WEEK 7: Tier 4 + Polish + Deployment

```

### Daily Standup Checklist
- [ ] Module builds without errors
- [ ] Data fetches within timeout (30sec)
- [ ] UI renders on mobile (375px width)
- [ ] Sidebar controls work correctly
- [ ] Cache works as expected
- [ ] Error messages are helpful

---

## 🎓 KEY BEST PRACTICES

1. **Always inherit from FazDaneModule** - ensures consistency
2. **Use @st.cache_data for external API calls** - prevents rate limiting
3. **Render sidebar first** - controls must load before main content
4. **Add st.spinner() during data fetch** - user feedback
5. **Use st.columns for layout** - responsive design
6. **Handle missing data gracefully** - show helpful messages
7. **Log errors to file** - debugging production issues
8. **Test with Google Drive live** - not just local files

---

## 📞 QUICK REFERENCE: MODULE LOCATIONS

```
Google Drive Path Structure:
/My Drive/
├── Python Projects/
│   └── Notebooks/
│       ├── 05-FazDane Options Liquidity Discovery Engine.ipynb
│       ├── 06-Market Breadth Dashboard.ipynb
│       ├── Future Analysis/
│       │   └── ES 3-Month Intraday Strategy Backtest.ipynb
│       ├── Portfolio/
│       │   └── FazDane Portfolio Backtester and Optimizer.ipynb
│       └── 05-SPX Sector Rotation RRG-Style Visualization.ipynb
```

**Folder ID Mapping** (Update these):
```python
GOOGLE_DRIVE_FOLDERS = {
    "Options": "PASTE_FOLDER_ID_HERE",
    "Equity": "PASTE_FOLDER_ID_HERE",
    "Futures": "PASTE_FOLDER_ID_HERE",
    "Portfolio": "PASTE_FOLDER_ID_HERE",
    "Forecasting": "PASTE_FOLDER_ID_HERE",
}
```

---

## ✅ VALIDATION CHECKLIST

Before deploying each module to production:

- [ ] Authentication works (login/logout flow)
- [ ] Data refreshes correctly
- [ ] Charts render without errors
- [ ] Sidebar controls update data
- [ ] Mobile responsive (test on phone)
- [ ] Load time < 3 seconds
- [ ] Error messages are user-friendly
- [ ] Google Drive sync works
- [ ] Session state doesn't leak between users
- [ ] No hardcoded credentials
- [ ] Logging is enabled
- [ ] Documentation is complete

---

**Framework Created By**: Claude AI  
**Framework Version**: 1.0  
**Last Updated**: May 2026  
**Target Launch**: Week 1 Q2 2026

**VIBE**: Professional Financial Intelligence Platform with Dark Sophistication, Real-Time Data, and Frictionless User Experience.
