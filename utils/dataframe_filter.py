import streamlit as st
import pandas as pd

def render_dataframe_filter(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """
    Renders a dynamic filter UI for a dataframe.
    Features:
    1. Include/Exclude options.
    2. Uses all column names.
    3. Filter by unique column content.
    4. Filter multiple columns.
    5. Outputs a comma-delimited list of tickers.
    """
    if df.empty:
        return df

    st.markdown("### 🎛️ Dynamic Data Filter")
    
    rules_key = f"{key_prefix}_filter_rules"
    if rules_key not in st.session_state:
        st.session_state[rules_key] = []
        
    rules = st.session_state[rules_key]
    
    columns = list(df.columns)
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("➕ Add Filter", key=f"{key_prefix}_add_btn"):
            rules.append({"id": len(rules), "action": "Include", "column": columns[0], "values": []})
            st.rerun()
            
    with col2:
        if st.button("🗑️ Clear All", key=f"{key_prefix}_clear_btn"):
            st.session_state[rules_key] = []
            st.rerun()

    if rules:
        st.markdown("<div style='font-size:12px; color:#94a3b8; margin-bottom: 8px;'>Filter Rules (Applied in Order):</div>", unsafe_allow_html=True)

    rules_to_keep = []
    for i, rule in enumerate(rules):
        c1, c2, c3, c4 = st.columns([2, 3, 5, 1])
        
        col_name = rule.get("column", columns[0])
        
        with c1:
            default_idx = columns.index(col_name) if col_name in columns else 0
            rule["column"] = st.selectbox(
                "Column", 
                columns, 
                key=f"{key_prefix}_col_{i}", 
                index=default_idx,
                label_visibility="collapsed"
            )
            
        # Re-check numeric after user might have changed the column
        col_name = rule["column"]
        is_numeric = pd.api.types.is_numeric_dtype(df[col_name]) if col_name in df.columns else False
            
        with c2:
            if is_numeric:
                action_options = ["Greater Than", "Less Than", "Equal To", "Include", "Exclude"]
            else:
                action_options = ["Include", "Exclude"]
                
            current_action = rule.get("action", "Include")
            if current_action not in action_options:
                current_action = action_options[0]
                
            rule["action"] = st.selectbox(
                "Action", 
                action_options, 
                key=f"{key_prefix}_action_{i}", 
                index=action_options.index(current_action),
                label_visibility="collapsed"
            )
            
        with c3:
            if col_name in df.columns:
                action = rule["action"]
                if is_numeric and action in ["Greater Than", "Less Than", "Equal To"]:
                    default_val = 0.0
                    if isinstance(rule.get("values"), (int, float)):
                        default_val = float(rule["values"])
                    elif isinstance(rule.get("values"), list) and len(rule["values"]) > 0:
                        try:
                            default_val = float(rule["values"][0])
                        except:
                            pass
                    
                    rule["values"] = st.number_input(
                        "Value",
                        value=default_val,
                        key=f"{key_prefix}_num_val_{i}",
                        label_visibility="collapsed"
                    )
                else:
                    unique_vals = df[col_name].dropna().unique().tolist()
                    try:
                        unique_vals = sorted(unique_vals)
                    except TypeError:
                        pass
                    
                    current_values = rule.get("values", [])
                    if not isinstance(current_values, list):
                        current_values = [current_values]
                    current_values = [v for v in current_values if v in unique_vals]
                    
                    rule["values"] = st.multiselect(
                        "Values", 
                        options=unique_vals, 
                        default=current_values, 
                        key=f"{key_prefix}_vals_{i}",
                        label_visibility="collapsed",
                        placeholder=f"Select {col_name} values..."
                    )
            else:
                rule["values"] = []
            
        with c4:
            remove = st.button("❌", key=f"{key_prefix}_remove_{i}")
            if not remove:
                rules_to_keep.append(rule)
                
    if len(rules) != len(rules_to_keep):
        st.session_state[rules_key] = rules_to_keep
        st.rerun()
        
    # Apply Filters
    filtered_df = df.copy()
    
    for rule in rules_to_keep:
        col = rule.get("column")
        vals = rule.get("values")
        action = rule.get("action")
        
        if col and col in filtered_df.columns:
            if action == "Greater Than" and vals is not None:
                filtered_df = filtered_df[filtered_df[col] > vals]
            elif action == "Less Than" and vals is not None:
                filtered_df = filtered_df[filtered_df[col] < vals]
            elif action == "Equal To" and vals is not None:
                filtered_df = filtered_df[filtered_df[col] == vals]
            elif action == "Include" and vals:
                filtered_df = filtered_df[filtered_df[col].isin(vals)]
            elif action == "Exclude" and vals:
                filtered_df = filtered_df[~filtered_df[col].isin(vals)]

    # Ticker copy box
    if "Ticker" in filtered_df.columns:
        tickers = filtered_df["Ticker"].unique().tolist()
        ticker_str = ", ".join(tickers)
        st.markdown(f"**Filtered Tickers ({len(tickers)})**")
        st.code(ticker_str, language="text")
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    return filtered_df
