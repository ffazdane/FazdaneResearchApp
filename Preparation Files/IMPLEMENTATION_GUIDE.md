# 🚀 FAZDANE STREAMLIT MIGRATION - IMPLEMENTATION GUIDE
## Module-by-Module Conversion Playbook

**Version**: 1.0  
**Last Updated**: May 2026  
**Estimated Timeline**: 7 weeks  

---

## 📋 TABLE OF CONTENTS

1. [Quick Start (Hour 1)](#quick-start)
2. [Environment Setup](#environment-setup)
3. [Tier 1 Module Conversion](#tier-1-conversion)
4. [Testing & Validation](#testing-validation)
5. [Deployment](#deployment)

---

## ⚡ QUICK START (Hour 1)

### Step 1: Clone/Create Project Structure

```bash
# Create project directory
mkdir fazdane-research-app
cd fazdane-research-app

# Create folder structure
mkdir -p {modules/tier1,modules/tier2,modules/tier3,utils,pages,data,logs}
touch {app.py,requirements.txt,.env,.gitignore}
```

### Step 2: Create Virtual Environment

```bash
# Create Python 3.10+ virtual environment
python3.10 -m venv venv

# Activate
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Create .env File

```bash
# .env (Keep this secret! Add to .gitignore)
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
GOOGLE_DRIVE_CREDENTIALS_PATH=./credentials/google_drive_creds.json
LOG_LEVEL=INFO
APP_ENV=development
```

### Step 5: Quick Test

```bash
# Copy app.py, pages_auth.py, modules_base_module.py from outputs
# Then run:
streamlit run app.py
```

---

## 🔧 ENVIRONMENT SETUP

### 1. Google Drive OAuth2 Setup

**In Google Cloud Console** (https://console.cloud.google.com/):

1. Create new project: "FazDane Research"
2. Enable APIs:
   - Google Drive API
   - Google Sheets API
   - Google Docs API
3. Create OAuth 2.0 Desktop credentials
4. Download JSON → Save as `credentials/google_drive_creds.json`
5. Add scopes:
   ```
   https://www.googleapis.com/auth/drive
   https://www.googleapis.com/auth/drive.file
   ```

**Test connection**:
```python
# test_drive_connection.py
from utils.google_drive_sync import GoogleDriveSync

drive = GoogleDriveSync()
notebooks = drive.list_notebooks("Options")
print(f"Found {len(notebooks)} notebooks")
```

### 2. Configuration Files

**config.yaml** (Secrets - Add to .gitignore):
```yaml
credentials:
  users:
    fazal:
      password_hash: $2b$12$...  # Use generate_password_hash()
      role: admin
      active: true
    trader1:
      password_hash: $2b$12$...
      role: user
      active: true

settings:
  session_timeout: 3600
  max_users: 10
  cache_enabled: true
  cache_ttl: 3600
  
google_drive:
  folder_ids:
    options: "1a2b3c4d5e..."
    equity: "2a2b3c4d5e..."
    futures: "3a2b3c4d5e..."
    portfolio: "4a2b3c4d5e..."
    forecasting: "5a2b3c4d5e..."
```

---

## 🧩 TIER 1 MODULE CONVERSION

### Conversion Workflow (Per Module: ~8 Hours)

#### PHASE 1: ANALYSIS (1 Hour)

**Example: Options Liquidity Discovery**

```python
# Step 1: Read source notebook end-to-end
# File: /My Drive/Python Projects/Notebooks/
#       05-FazDane Options Liquidity Discovery Engine.ipynb

# Step 2: Document all cells
cells = [
    "Cell 1: Imports (yfinance, numpy, pandas)",
    "Cell 2: Data fetch function (download options chain)",
    "Cell 3: Calculate IV Rank",
    "Cell 4: Filter by volume > 1000",
    "Cell 5: Create heatmap visualization",
]

# Step 3: Identify parameters
parameters = {
    'min_volume': 1000,           # From slider
    'min_iv_rank': 50,            # From slider
    'option_type': ['Call', 'Put'],  # From multiselect
    'expiration': 'Weekly',       # From selectbox
    'symbols': 'SPY, QQQ, IWM',   # From text input
}

# Step 4: List all visualizations
visualizations = [
    "Heatmap(x=symbol, y=iv_rank, z=volume)",
    "DataFrame(ticker, volume, iv_rank, bid, ask)",
    "Metrics(count, avg_volume, max_iv_rank)",
]

# Step 5: Check data dependencies
data_sources = [
    "yfinance - OHLCV data",
    "yfinance - Options chain",
    "Custom API - IV Rank calculation",
]
```

#### PHASE 2: EXTRACTION (2 Hours)

Create a working Python script with the extracted code:

```python
# modules/tier1/options_liquidity.py - EXTRACTION PHASE

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

class OptionsLiquidityExtractor:
    """Extract logic from Jupyter notebook"""
    
    @staticmethod
    def scan_options(symbols: list, min_volume: int = 1000, 
                     min_iv_rank: int = 50, option_type: str = 'Call'):
        """
        Extracted from notebook cells 1-4
        """
        results = []
        
        for symbol in symbols:
            try:
                # Cell 2: Get options chain
                ticker = yf.Ticker(symbol)
                expirations = ticker.options
                
                for exp in expirations[:1]:  # First expiration only
                    opts = ticker.option_chain(exp)
                    
                    if option_type.lower() == 'call':
                        df = opts.calls
                    else:
                        df = opts.puts
                    
                    # Cell 3: Calculate IV Rank (example)
                    df['iv_rank'] = (df['impliedVolatility'] * 100)
                    
                    # Cell 4: Filter
                    df_filtered = df[
                        (df['volume'] >= min_volume) &
                        (df['iv_rank'] >= min_iv_rank)
                    ]
                    
                    # Add symbol
                    df_filtered['symbol'] = symbol
                    results.append(df_filtered)
            
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue
        
        # Combine results
        if results:
            combined = pd.concat(results, ignore_index=True)
            return combined[['symbol', 'strike', 'volume', 'iv_rank', 
                           'bid', 'ask', 'impliedVolatility']]
        
        return pd.DataFrame()
    
    @staticmethod
    def create_heatmap(df):
        """
        Extracted from notebook cell 5
        """
        import plotly.graph_objects as go
        
        pivot = df.pivot_table(
            values='volume',
            index='iv_rank',
            columns='symbol',
            aggfunc='sum',
            fill_value=0
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='Viridis'
        ))
        
        return fig
```

#### PHASE 3: STREAMLIT REFACTOR (3 Hours)

Convert extracted code to Streamlit module:

```python
# modules/tier1/options_liquidity.py - STREAMLIT REFACTOR

import streamlit as st
import pandas as pd
import logging
from modules.base_module import FazDaneModule
from modules.tier1.options_liquidity_extractor import OptionsLiquidityExtractor

logger = logging.getLogger("OptionsLiquidity")

class OptionsLiquidityModule(FazDaneModule):
    """
    Options Liquidity Discovery Module
    Scans for high-liquidity options with elevated IV Rank
    """
    
    # Module metadata
    MODULE_NAME = "Options Liquidity Discovery"
    MODULE_ICON = "💧"
    MODULE_DESCRIPTION = "Real-time options scanning & liquidity analysis"
    TIER = 1
    SOURCE_NOTEBOOK = "05-FazDane Options Liquidity Discovery Engine.ipynb"
    
    DATA_SOURCES = ["yfinance", "options-chain-api"]
    CACHE_TTL = 300  # 5 minutes (options update frequently)
    REQUIRES_LIVE_DATA = True
    
    def __init__(self):
        super().__init__()
        self.extractor = OptionsLiquidityExtractor()
        
        # Initialize session state
        if 'scan_results' not in st.session_state:
            st.session_state.scan_results = None
    
    # ─────────────────────────────────────────────────────────────────────
    # SIDEBAR - Module Controls
    # ─────────────────────────────────────────────────────────────────────
    
    def render_sidebar(self):
        """Render sidebar with scanning parameters"""
        
        st.markdown("### Search Parameters")
        
        # Symbols input
        symbols_input = st.text_input(
            "Symbols (comma-separated)",
            value="SPY, QQQ, IWM, XLK, XLV",
            help="Symbols to scan for options"
        )
        self.symbols = [s.strip().upper() for s in symbols_input.split(',')]
        
        # Filters
        st.markdown("#### Filters")
        
        self.min_volume = st.slider(
            "Minimum Volume",
            min_value=0,
            max_value=10000,
            value=1000,
            step=100,
            help="Filter options by minimum volume"
        )
        
        self.min_iv_rank = st.slider(
            "Minimum IV Rank (%)",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            help="Filter by IV percentile rank"
        )
        
        # Option type
        self.option_type = st.multiselect(
            "Option Type",
            options=["Call", "Put"],
            default=["Call", "Put"],
            help="Which option types to include"
        )
        
        # Expiration
        self.expiration = st.selectbox(
            "Expiration",
            options=["Weekly", "Monthly", "Quarterly"],
            help="Option expiration preference"
        )
        
        # Action button
        st.markdown("#### Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Scan Options", use_container_width=True, 
                        type="primary", key="scan_btn"):
                self._perform_scan()
        
        with col2:
            if st.button("💾 Export", use_container_width=True, key="export_btn"):
                self._export_results()
        
        st.divider()
        
        # Statistics
        if st.session_state.scan_results is not None:
            st.markdown("#### Scan Statistics")
            
            df = st.session_state.scan_results
            
            st.metric("Total Opportunities", len(df))
            st.metric("Avg Volume", int(df['volume'].mean()))
            st.metric("Avg IV Rank", f"{df['iv_rank'].mean():.1f}%")
    
    # ─────────────────────────────────────────────────────────────────────
    # MAIN CONTENT
    # ─────────────────────────────────────────────────────────────────────
    
    def render_main(self):
        """Render main content area"""
        
        st.markdown("## Options Liquidity Heatmap")
        
        if st.session_state.scan_results is None:
            st.info("👈 Set parameters in sidebar and click 'Scan Options' to begin")
            return
        
        df = st.session_state.scan_results
        
        if df.empty:
            st.warning("⚠️ No options found matching your criteria")
            return
        
        # Display tabs
        tab1, tab2, tab3 = st.tabs(["📊 Heatmap", "📋 Data", "📈 Analytics"])
        
        # ─────────────────────────────────────────────────────────────────
        # TAB 1: HEATMAP
        # ─────────────────────────────────────────────────────────────────
        
        with tab1:
            st.markdown("### Volume × IV Rank Heatmap")
            
            fig = self.extractor.create_heatmap(df)
            st.plotly_chart(fig, use_container_width=True)
        
        # ─────────────────────────────────────────────────────────────────
        # TAB 2: DATA TABLE
        # ─────────────────────────────────────────────────────────────────
        
        with tab2:
            st.markdown("### Detailed Options Data")
            
            # Sortable columns
            sort_by = st.selectbox(
                "Sort By",
                options=["volume", "iv_rank", "bid", "ask"],
                help="Column to sort by"
            )
            
            df_sorted = df.sort_values(by=sort_by, ascending=False)
            
            st.dataframe(
                df_sorted,
                use_container_width=True,
                height=500,
                column_config={
                    "volume": st.column_config.NumberColumn(format="%d"),
                    "iv_rank": st.column_config.NumberColumn(format="%.1f%%"),
                    "bid": st.column_config.NumberColumn(format="$%.2f"),
                    "ask": st.column_config.NumberColumn(format="$%.2f"),
                }
            )
        
        # ─────────────────────────────────────────────────────────────────
        # TAB 3: ANALYTICS
        # ─────────────────────────────────────────────────────────────────
        
        with tab3:
            st.markdown("### Analytics & Insights")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Opportunities",
                    len(df),
                    help="Number of options matching criteria"
                )
            
            with col2:
                st.metric(
                    "Avg Volume",
                    int(df['volume'].mean()),
                    help="Average option volume"
                )
            
            with col3:
                st.metric(
                    "Max IV Rank",
                    f"{df['iv_rank'].max():.1f}%",
                    help="Highest IV rank in results"
                )
            
            with col4:
                st.metric(
                    "Avg Bid-Ask",
                    f"${(df['ask'] - df['bid']).mean():.2f}",
                    help="Average spread"
                )
            
            # Distribution charts
            st.markdown("#### Distribution Analysis")
            
            col_dist1, col_dist2 = st.columns(2)
            
            with col_dist1:
                st.markdown("**Volume Distribution**")
                fig_vol = self._create_histogram(df, 'volume', 'Volume')
                st.plotly_chart(fig_vol, use_container_width=True)
            
            with col_dist2:
                st.markdown("**IV Rank Distribution**")
                fig_iv = self._create_histogram(df, 'iv_rank', 'IV Rank (%)')
                st.plotly_chart(fig_iv, use_container_width=True)
    
    # ─────────────────────────────────────────────────────────────────────
    # PRIVATE METHODS
    # ─────────────────────────────────────────────────────────────────────
    
    def _perform_scan(self):
        """Execute options scan"""
        with st.spinner("🔍 Scanning options..."):
            try:
                df = self.extractor.scan_options(
                    symbols=self.symbols,
                    min_volume=self.min_volume,
                    min_iv_rank=self.min_iv_rank,
                    option_type=self.option_type[0] if self.option_type else 'Call'
                )
                
                st.session_state.scan_results = df
                logger.info(f"Scan completed: {len(df)} options found")
                st.success(f"✅ Found {len(df)} matching options")
            
            except Exception as e:
                logger.error(f"Scan failed: {str(e)}")
                st.error(f"❌ Scan failed: {str(e)}")
    
    def _export_results(self):
        """Export results to CSV"""
        if st.session_state.scan_results is None:
            st.warning("No results to export")
            return
        
        df = st.session_state.scan_results
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"options_scan_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    def _create_histogram(self, df, column, title):
        """Create histogram chart"""
        import plotly.express as px
        
        fig = px.histogram(
            df,
            x=column,
            nbins=30,
            title=f"{title} Distribution",
            labels={column: title}
        )
        
        return fig
```

#### PHASE 4: INTEGRATION (1 Hour)

Add to main app.py:

```python
# In app.py - Add to Tier 1 modules dispatcher

elif selected_module == "💧 Options Liquidity Discovery":
    from modules.tier1.options_liquidity import OptionsLiquidityModule
    module = OptionsLiquidityModule()
    module.run()
    logger.info(f"User {st.session_state.username} opened Options Liquidity module")
```

#### PHASE 5: TESTING & QA (1 Hour)

```python
# tests/test_options_liquidity.py

import pytest
from modules.tier1.options_liquidity import OptionsLiquidityModule
import pandas as pd

class TestOptionsLiquidityModule:
    
    @pytest.fixture
    def module(self):
        return OptionsLiquidityModule()
    
    def test_module_initialization(self, module):
        assert module.MODULE_NAME == "Options Liquidity Discovery"
        assert module.TIER == 1
    
    def test_scan_options(self, module):
        df = module.extractor.scan_options(
            symbols=['SPY'],
            min_volume=100,
            min_iv_rank=30
        )
        
        assert isinstance(df, pd.DataFrame)
        assert 'volume' in df.columns
        assert 'iv_rank' in df.columns
    
    def test_heatmap_creation(self, module):
        # Create sample data
        df = pd.DataFrame({
            'symbol': ['SPY', 'SPY', 'QQQ'],
            'volume': [1000, 1500, 800],
            'iv_rank': [50, 65, 45]
        })
        
        fig = module.extractor.create_heatmap(df)
        assert fig is not None

# Run tests
# pytest tests/test_options_liquidity.py -v
```

---

## ✅ TESTING & VALIDATION

### Pre-Deployment Checklist

For each module, verify:

- [ ] **Functionality**
  - [ ] All sidebar controls work
  - [ ] Data fetches without timeout
  - [ ] Visualizations render correctly
  - [ ] Export/download works

- [ ] **Performance**
  - [ ] Initial load < 3 seconds
  - [ ] Data refresh < 30 seconds
  - [ ] Cache works (2nd load < 1 second)

- [ ] **UI/UX**
  - [ ] Mobile responsive (375px)
  - [ ] Touch-friendly buttons
  - [ ] Helpful error messages

- [ ] **Error Handling**
  - [ ] Invalid symbols handled
  - [ ] Network timeouts caught
  - [ ] Missing data handled gracefully

- [ ] **Security**
  - [ ] No hardcoded credentials
  - [ ] User can only see own data
  - [ ] Proper logging (no passwords)

### Manual Testing

```bash
# Start app
streamlit run app.py

# Test with demo credentials
# Username: fazal
# Password: FazDane2026!

# Test each module:
# 1. Click through all sidebar controls
# 2. Verify data displays
# 3. Check charts render
# 4. Test export
# 5. Clear cache & refresh
# 6. Check mobile (F12 Dev Tools)
```

---

## 🚀 DEPLOYMENT

### Option 1: Local Development

```bash
streamlit run app.py
# Visit http://localhost:8501
```

### Option 2: Streamlit Cloud

```bash
# 1. Push to GitHub
git init
git add .
git commit -m "FazDane Streamlit App v1.0"
git push origin main

# 2. Deploy at https://streamlit.io/cloud
# Connect GitHub repo → Automatically deploys

# 3. Configure secrets at https://app.streamlit.io/[yourapp]/settings
# Add to Secrets:
google_drive_folder_id = "..."
api_key = "..."
```

### Option 3: Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

```bash
# Build & run
docker build -t fazdane-research .
docker run -p 8501:8501 fazdane-research
```

---

## 📊 IMPLEMENTATION TIMELINE

```
WEEK 1: Foundation & Initial Modules
├─ Day 1: Environment setup + auth system
├─ Day 2: Options Liquidity module
├─ Day 3: Market Breadth module
└─ Day 4: Sector Rotation Monitor

WEEK 2: Complete Tier 1 (100% DONE)
├─ Day 1: Calendar Strategy Matrix
├─ Day 2: Iron Condor Analyzer
├─ Day 3: ES Pivot Analysis
└─ Day 4: QA & Polish

WEEKS 3-4: Tier 2 modules (5 modules)
WEEKS 5-6: Tier 3 modules (4 modules)
WEEK 7: Polish + Production deployment

Total: Tier 1 is 100% COMPLETE! (6 Modules Live)
```

---

## 🎓 KEY CONVERSION PRINCIPLES

1. **Keep extraction separate** - Logic independent of Streamlit
2. **Use session_state** - For multi-page state management
3. **Cache aggressively** - @st.cache_data for all data fetches
4. **Handle errors** - Try/except with user-friendly messages
5. **Test early** - Write tests during PHASE 2, not after
6. **Document thoroughly** - Docstrings + README
7. **Log everything** - Debug production issues easily

---

**Ready to start? Begin with "Quick Start" section above!**
