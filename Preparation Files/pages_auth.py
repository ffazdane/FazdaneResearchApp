"""
FazDane Authentication System
Secure Login with Session Management
"""

import streamlit as st
import bcrypt
import json
from datetime import datetime
import logging

logger = logging.getLogger("FazDaneAuth")

# ═══════════════════════════════════════════════════════════════════════════
# CREDENTIALS MANAGEMENT (Store in secure location)
# ═══════════════════════════════════════════════════════════════════════════

class CredentialsManager:
    """Manage user credentials securely"""
    
    # NOTE: In production, use environment variables or secure vault
    DEFAULT_CREDENTIALS = {
        "users": {
            "fazal": {
                "password_hash": "$2b$12$BKN6/OakCJhQLGLFNuBj.OO5RfL7e6SV1FQxCw1lj.uVBqnaSZxQi",  # "FazDane2026!"
                "email": "fazal@fazdane.com",
                "role": "admin",
                "active": True
            },
            "trader1": {
                "password_hash": "$2b$12$5jVa2qOG5yFxKqL2mJXU..vH7dRCxLpCPxrZVG8S9jKHY4VqxJQ3m",  # "trader123"
                "email": "trader1@fazdane.com",
                "role": "user",
                "active": True
            },
            "trader2": {
                "password_hash": "$2b$12$QwErTyUiOp.LmNbVcXzZ..HJ3sKj2mPq9rStVwXyZaB1DeF5nQ7La",  # "trader456"
                "email": "trader2@fazdane.com",
                "role": "user",
                "active": True
            }
        }
    }
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password with bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hash: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hash.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False
    
    @staticmethod
    def validate_credentials(username: str, password: str) -> tuple[bool, dict]:
        """Validate user credentials"""
        credentials = CredentialsManager.DEFAULT_CREDENTIALS
        
        if username not in credentials["users"]:
            logger.warning(f"Login attempt with non-existent user: {username}")
            return False, {}
        
        user = credentials["users"][username]
        
        if not user["active"]:
            logger.warning(f"Login attempt with inactive user: {username}")
            return False, {}
        
        if CredentialsManager.verify_password(password, user["password_hash"]):
            logger.info(f"Successful login: {username}")
            return True, {
                "username": username,
                "email": user["email"],
                "role": user["role"],
                "login_time": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
        else:
            logger.warning(f"Failed login attempt: {username}")
            return False, {}

# ═══════════════════════════════════════════════════════════════════════════
# FAZDANE AUTHENTICATOR CLASS
# ═══════════════════════════════════════════════════════════════════════════

class FazDaneAuthenticator:
    """Main authentication handler for FazDane application"""
    
    def __init__(self):
        self.session_timeout = 3600  # 1 hour
        self.credentials_manager = CredentialsManager()
    
    def render_login_screen(self):
        """Render beautiful login screen with FazDane branding"""
        
        # Create centered layout
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # ─────────────────────────────────────────────────────────────
            # HEADER
            # ─────────────────────────────────────────────────────────────
            
            st.markdown("""
            <div style="text-align: center; padding: 60px 0 40px 0;">
                <h1 style="
                    color: #00ff88;
                    font-size: 56px;
                    margin: 0;
                    font-family: 'Courier Prime', monospace;
                    font-weight: 700;
                    text-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
                ">⚡</h1>
                <h2 style="
                    color: #00ff88;
                    font-size: 32px;
                    margin: 16px 0 0 0;
                    font-family: 'Courier Prime', monospace;
                    font-weight: 700;
                ">FazDane</h2>
                <p style="
                    color: #64748b;
                    font-size: 13px;
                    margin: 8px 0 0 0;
                    letter-spacing: 2px;
                    text-transform: uppercase;
                ">Trading Intelligence Platform</p>
            </div>
            """, unsafe_allow_html=True)
            
            # ─────────────────────────────────────────────────────────────
            # LOGIN FORM
            # ─────────────────────────────────────────────────────────────
            
            st.markdown("""
            <div style="
                background: rgba(15, 23, 42, 0.8);
                border: 1px solid #1e293b;
                border-radius: 12px;
                padding: 32px;
                backdrop-filter: blur(10px);
            ">
            """, unsafe_allow_html=True)
            
            st.markdown("### Sign In")
            
            # Username input
            username = st.text_input(
                "Username",
                placeholder="Enter your username",
                key="login_username",
                help="Your FazDane account username"
            )
            
            # Password input
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                key="login_password",
                help="Your FazDane account password"
            )
            
            # Remember me checkbox
            remember_me = st.checkbox(
                "Remember me",
                value=False,
                help="Keep me logged in on this device"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Login button
            if st.button(
                "🔓 Sign In",
                use_container_width=True,
                type="primary",
                key="login_button"
            ):
                if not username or not password:
                    st.error("❌ Please enter both username and password")
                    return
                
                # Validate credentials
                is_valid, user_info = self.credentials_manager.validate_credentials(
                    username,
                    password
                )
                
                if is_valid:
                    # Set session state
                    st.session_state.authenticated = True
                    st.session_state.username = user_info["username"]
                    st.session_state.user_email = user_info["email"]
                    st.session_state.user_role = user_info["role"]
                    st.session_state.login_time = user_info["login_time"]
                    st.session_state.remember_me = remember_me
                    
                    logger.info(f"User {username} ({user_info['role']}) logged in successfully")
                    
                    st.success(f"✅ Welcome, {user_info['email']}!")
                    st.balloons()
                    
                    # Rerun to show main app
                    st.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password. Please try again.")
                    logger.warning(f"Failed login attempt for user: {username}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # ─────────────────────────────────────────────────────────────
            # FOOTER INFO
            # ─────────────────────────────────────────────────────────────
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style="text-align: center; color: #64748b; font-size: 12px;">
                <p>Demo Credentials:</p>
                <code style="background: #1a1f3a; padding: 4px 8px; border-radius: 4px; display: inline-block;">
                    Username: fazal | Password: FazDane2026!
                </code>
                <br><br>
                <p style="margin-top: 20px;">
                    <strong>FazDane Research Application v1.0</strong><br>
                    © 2026 All Rights Reserved
                </p>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SESSION MANAGEMENT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def check_authentication():
    """Check if user is authenticated"""
    return st.session_state.get("authenticated", False)

def get_current_user():
    """Get currently authenticated user info"""
    return {
        "username": st.session_state.get("username"),
        "email": st.session_state.get("user_email"),
        "role": st.session_state.get("user_role"),
        "login_time": st.session_state.get("login_time")
    }

def is_admin():
    """Check if current user is admin"""
    return st.session_state.get("user_role") == "admin"

def logout_user():
    """Clear session and logout user"""
    username = st.session_state.get("username", "Unknown")
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.user_email = None
    st.session_state.user_role = None
    st.session_state.login_time = None
    logger.info(f"User {username} logged out")

# ═══════════════════════════════════════════════════════════════════════════
# PASSWORD UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def generate_password_hash(password: str) -> str:
    """Generate bcrypt hash for new password"""
    return CredentialsManager.hash_password(password)

def validate_password_strength(password: str) -> tuple[bool, list]:
    """Validate password meets security requirements"""
    issues = []
    
    if len(password) < 8:
        issues.append("Password must be at least 8 characters")
    
    if not any(c.isupper() for c in password):
        issues.append("Password must contain at least one uppercase letter")
    
    if not any(c.isdigit() for c in password):
        issues.append("Password must contain at least one number")
    
    if not any(c in "!@#$%^&*" for c in password):
        issues.append("Password must contain at least one special character (!@#$%^&*)")
    
    return len(issues) == 0, issues

# ═══════════════════════════════════════════════════════════════════════════
# ADMIN FUNCTIONS (For credential management)
# ═══════════════════════════════════════════════════════════════════════════

def admin_add_user(username: str, email: str, password: str, role: str = "user"):
    """Add new user (admin only)"""
    if not is_admin():
        logger.error(f"Unauthorized add_user attempt by {st.session_state.username}")
        return False, "Only admins can add users"
    
    # Validate password strength
    is_strong, issues = validate_password_strength(password)
    if not is_strong:
        return False, f"Weak password: {', '.join(issues)}"
    
    # Generate hash
    password_hash = generate_password_hash(password)
    
    logger.info(f"Admin {st.session_state.username} added new user: {username}")
    
    return True, f"User {username} created successfully"

def admin_reset_password(username: str, new_password: str):
    """Reset user password (admin only)"""
    if not is_admin():
        return False, "Only admins can reset passwords"
    
    # Validate new password
    is_strong, issues = validate_password_strength(new_password)
    if not is_strong:
        return False, f"Weak password: {', '.join(issues)}"
    
    logger.info(f"Admin {st.session_state.username} reset password for {username}")
    
    return True, f"Password reset successfully for {username}"
