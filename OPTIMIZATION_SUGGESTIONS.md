# FazDane Research Application — Analysis & Optimization Suggestions

Date: 2026-06-12 · App version: 2.05 · Streamlit 1.55.0 (bundled Python 3.12)

---

## 1. Sidebar bug — root cause and fix (FIXED in app.py)

**Symptom:** On opening the app, the side menu does not appear; reloading the frame manually brings it back.

**Root cause (confirmed in the Streamlit 1.55 frontend source shipped with the app):**

Streamlit persists the sidebar collapsed/expanded state in the browser's
`localStorage` under the key `stSidebarCollapsed-<appId>`. On every page load it does:

```js
const saved = getSavedSidebarState(appId);
return saved !== null ? saved : shouldCollapse(initialSidebarState, mdBreakpoint, innerWidth);
```

The saved value **takes precedence over `initial_sidebar_state="expanded"`**.
Once the sidebar gets collapsed even once — which happens automatically when the
app first renders inside a narrow preview iframe (the devcontainer config uses
`onAutoForward: "openPreview"`, i.e. the VS Code/Codespaces Simple Browser pane),
or via an accidental click on the collapse chevron — `true` is written to
localStorage and every subsequent open starts with the sidebar hidden.

Additionally, the existing CSS "force open" hack in `app.py` targeted
`[data-testid="stAppViewContainer"][data-sidebar-collapsed="true"]` — an attribute
that **does not exist in Streamlit 1.55** (verified by grepping the bundled JS).
It was dead CSS doing nothing.

**Fix applied to `app.py`:**

1. Removed the dead CSS block.
2. Injected a small JS watchdog (via `st.components.v1.html`) that runs on every load:
   - Resets any `stSidebarCollapsed-*` localStorage flag from `"true"` to `"false"`.
   - For ~10 s after load, if `section[data-testid="stSidebar"]` is collapsed or
     missing, it clicks Streamlit's own expand control
     (`[data-testid="stExpandSidebarButton"]`). Clicking the real button keeps
     Streamlit's internal state consistent — no layout hacks required.
3. Added CSS to hide `[data-testid="stSidebarNav"]` (see §2).

No manual reload should be needed anymore, in a normal browser tab or in the preview frame.

---

## 2. `pages/` directory name collides with Streamlit's multipage convention

`pages/auth.py` lives in a folder Streamlit treats as the magic multipage
directory, so Streamlit auto-generates a page nav ("app" / "auth") at the top of
the sidebar — and a user clicking "auth" lands on the raw auth module outside
your dispatcher.

- Short-term: the nav is now hidden via CSS (applied with the fix above).
- Proper fix: rename `pages/` to `views/` or `auth_ui/` and update the two imports
  (`from pages.auth import ...` in app.py and any module that imports it). This
  also removes the page from Streamlit's URL routing entirely.

## 3. Triple rerun on every navigation click

A module click currently triggers up to three full script executions:

1. `launch_module()` sets `pending_nav` + `st.rerun()`
2. The next run applies `pending_nav`, then the "module transition" detector
   (`last_active_module`) renders a "Loading module..." placeholder and calls
   `st.rerun()` again
3. The third run finally renders the module.

Each run re-executes the ~400-line CSS f-string, the auth check, sidebar build,
etc. Recommendation: drop the intermediate "Loading module..." rerun (steps in
lines around `last_active_module`). Streamlit already clears the previous frame
on rerun; if you want a visual cue, use `st.spinner` inside the module instead.
This alone cuts navigation latency by roughly a third to a half.

## 4. Blocking cloud DB restore on first load

`restore_all_databases(force=True)` runs synchronously inside the first
post-login render and blocks the UI. Suggestions:

- Run it lazily per database when a module actually needs it, or
- Move it to a background thread and surface status in the sidebar caption
  (you already have `db_restore_msg`/`db_restore_err` plumbing), or
- Keep `force=True` only behind the manual "Restore" control and use
  `force=False` (skip if local file is fresh) at startup.

## 5. Plotly monkey-patching is fragile

Patching `go.Figure.__init__` / `update_layout` globally works but:

- It silently swallows all exceptions (`except Exception: pass`) — theming bugs
  become invisible.
- It compares against the hard-coded `#0d1b2e` sentinel, so modules using any
  other explicit color bypass theming.

Cleaner approach: register one custom Plotly template per theme at startup
(`plotly.io.templates["fazdane_navy"] = ...`) and set
`plotly.io.templates.default = ...` when the theme changes. Modules then need no
patching at all.

## 6. `use_container_width` deprecation (logs are full of warnings)

Streamlit announced removal after 2025-12-31; 1.55 still warns on every render
of every button. Bulk replace:

- `use_container_width=True` → `width="stretch"`
- `use_container_width=False` → `width="content"`

This silences hundreds of log lines per session (`logs/streamlit_local.err.log`)
and protects against the next Streamlit upgrade breaking every button/image call.

## 7. Pin dependencies

`requirements.txt` pins almost nothing (`streamlit`, `pandas`, `plotly`, … all
unpinned). The sidebar bug itself is version-dependent behavior — an unpinned
Streamlit means any redeploy can change UI behavior. Recommend pinning at least:

```
streamlit==1.55.0
```

and ideally generating a full lock (`pip freeze > requirements.lock.txt`).

## 8. Module dispatcher: replace the if/elif chain with a registry

The ~270-line `elif` chain in app.py can become a dict:

```python
MODULE_REGISTRY = {
    "Search Module": ("modules.tier1.search_module", "SearchModule"),
    "Market Breadth Dashboard": ("modules.tier1.market_breadth", "MarketBreadthModule"),
    # ...
}

mod_path, cls_name = MODULE_REGISTRY[active_module]
cls = getattr(importlib.import_module(mod_path), cls_name)
cls().run()
```

Benefits: one place to add modules, uniform error handling (currently only some
modules are wrapped in try/except), and the tier menus / home-dashboard tab
lists can be generated from the same registry instead of being duplicated three
times (sidebar options, `menu_config`, `macro_module_tabs`).

## 9. Cache the theme CSS

The large CSS f-string is rebuilt and re-sent on every rerun. Wrap it:

```python
@st.cache_data
def build_css(theme_name: str) -> str: ...
st.markdown(build_css(st.session_state["app_theme_selection"]), unsafe_allow_html=True)
```

Minor CPU win, but it also makes the CSS testable and keeps app.py readable.

## 10. Housekeeping / security notes

- `.env` and `.streamlit/secrets.toml` exist in the project folder — verify both
  are in `.gitignore` and have never been committed (rotate keys if they have).
- `.python/` (a full Python 3.12 install) and `.venv-1/` inside the project make
  the folder huge and slow down OneDrive sync; consider moving the interpreter
  outside the synced folder.
- `logs/` is also OneDrive-synced — rotating log files cause constant sync churn;
  consider relocating logs or excluding the folder from sync.
- Duplicate sources in `Preparation Files/` (old app.py, auth.py copies) invite
  confusion — archive them outside the live project.

---

## Priority order

1. ~~Sidebar bug~~ (fixed)
2. §3 remove the extra navigation rerun — biggest perceived speed win
3. §6 `use_container_width` bulk replace — prevents future breakage
4. §7 pin Streamlit
5. §4 non-blocking DB restore
6. §2 rename `pages/` → `views/`
7. §8/§9 dispatcher registry + cached CSS (refactor, do together)
8. §5 Plotly templates
9. §10 housekeeping
