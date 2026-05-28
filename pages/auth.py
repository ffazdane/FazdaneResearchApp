"""
Research & Trading Intelligence Platform — Authentication System
Secure Login with bcrypt + Session Management
"""

import streamlit as st
import bcrypt
import time
import logging
from datetime import datetime
from utils.version import VERSION

logger = logging.getLogger("FazDaneAuth")

# ══════════════════════════════════════════════════════════════════════
# CREDENTIALS MANAGER
# In production: load from config.yaml or environment secrets
# ══════════════════════════════════════════════════════════════════════

class CredentialsManager:
    """Manage user credentials securely with bcrypt"""

    # Passwords are bcrypt-hashed — never store plain text
    # To generate a new hash: bcrypt.hashpw(b"YourPassword", bcrypt.gensalt()).decode()
    DEFAULT_CREDENTIALS = {
        "users": {
            "fazal": {
                # Password: FazDane2026!
                "password_hash": "$2b$12$vne2LyvN9nx9/R7cyO2wxeeZhiR7ryD2kyyV1MVFXZKt0TtuHurDS",
                "display_name": "Fazal",
                "email": "fazal@fazdane.com",
                "role": "admin",
                "active": True,
            },
            "trader1": {
                # Password: Trader123!
                "password_hash": "$2b$12$5jVa2qOG5yFxKqL2mJXU..vH7dRCxLpCPxrZVG8S9jKHY4VqxJQ3m",
                "display_name": "Trader One",
                "email": "trader1@fazdane.com",
                "role": "user",
                "active": True,
            },
        }
    }

    @staticmethod
    def hash_password(password: str) -> str:
        """Generate bcrypt hash for a new password"""
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")

    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """Verify plain password against stored bcrypt hash"""
        try:
            return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False

    @classmethod
    def validate_credentials(cls, username: str, password: str) -> tuple[bool, dict]:
        """Validate username + password, return (success, user_info)"""
        users = cls.DEFAULT_CREDENTIALS["users"]

        if username not in users:
            logger.warning(f"Login attempt — unknown user: {username}")
            return False, {}

        user = users[username]

        if not user.get("active", False):
            logger.warning(f"Login attempt — inactive user: {username}")
            return False, {}

        if cls.verify_password(password, user["password_hash"]):
            logger.info(f"Successful login: {username} ({user['role']})")
            return True, {
                "username": username,
                "display_name": user["display_name"],
                "email": user["email"],
                "role": user["role"],
                "login_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }

        logger.warning(f"Failed login attempt: {username}")
        return False, {}


# ══════════════════════════════════════════════════════════════════════
# FAZDANE AUTHENTICATOR — Login Screen
# ══════════════════════════════════════════════════════════════════════

class FazDaneAuthenticator:
    """Renders the branded login screen and manages authentication"""

    def __init__(self):
        self.credentials = CredentialsManager()

    def render_login_screen(self):
        """Full-page branded login with logo"""

        # Center column layout
        _, col, _ = st.columns([1, 1.6, 1])

        with col:
            # ── Logo ──────────────────────────────────────────────
            try:
                st.image("assets/logo.png", use_container_width=True)
            except Exception:
                st.markdown(
                    "<h1 style='text-align:center;color:#3ab54a;font-size:42px;'>Research & Trading Intelligence Platform</h1>",
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            # ── Tagline ───────────────────────────────────────────
            st.markdown(
                """
                <p style="
                    text-align:center;
                    color:#64748b;
                    font-size:13px;
                    letter-spacing:3px;
                    text-transform:uppercase;
                    margin-bottom:28px;
                ">Research & Trading Intelligence</p>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div style="
                    background: rgba(13,27,46,0.55);
                    border: 1px solid #1e3a5f;
                    border-radius: 10px;
                    padding: 12px 14px;
                    margin-bottom: 16px;
                    color: #e2e8f0;
                    font-size: 18px;
                    font-weight: 700;
                ">🔐 Sign In</div>
                """,
                unsafe_allow_html=True,
            )

            username = st.text_input(
                "User Name",
                placeholder="Enter your username",
                key="login_username",
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                key="login_password",
            )

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            if st.button("Sign In →", use_container_width=True, type="primary", key="login_btn"):
                if not username.strip() or not password.strip():
                    st.error("Please enter both username and password.")
                    return

                with st.spinner("Verifying credentials…"):
                    is_valid, user_info = self.credentials.validate_credentials(
                        username.strip(), password
                    )

                if is_valid:
                    st.session_state.authenticated = True
                    st.session_state.username = user_info["username"]
                    st.session_state.display_name = user_info["display_name"]
                    st.session_state.user_email = user_info["email"]
                    st.session_state.user_role = user_info["role"]
                    st.session_state.login_time = user_info["login_time"]
                    st.success(f"Welcome, {user_info['display_name']}! Loading dashboard…")
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password.")

            # ── Footer ────────────────────────────────────────────
            st.markdown(
                f"""
                <p style="text-align:center;color:#334155;font-size:11px;margin-top:20px;">
                    Research & Trading Intelligence Platform {VERSION} · © 2026 All Rights Reserved
                </p>
                """,
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════
# SESSION UTILITIES
# ══════════════════════════════════════════════════════════════════════

def is_authenticated() -> bool:
    return st.session_state.get("authenticated", False)

def is_admin() -> bool:
    return st.session_state.get("user_role") == "admin"

def get_current_user() -> dict:
    return {
        "username": st.session_state.get("username"),
        "display_name": st.session_state.get("display_name"),
        "email": st.session_state.get("user_email"),
        "role": st.session_state.get("user_role"),
        "login_time": st.session_state.get("login_time"),
    }

def logout():
    """Clear all session state and trigger login screen"""
    username = st.session_state.get("username", "unknown")
    for key in ["authenticated", "username", "display_name", "user_email", "user_role", "login_time"]:
        st.session_state.pop(key, None)
    logger.info(f"User {username} logged out")
    st.rerun()


def generate_password_hash(password: str) -> str:
    """Utility: generate bcrypt hash for a new password"""
    return CredentialsManager.hash_password(password)
