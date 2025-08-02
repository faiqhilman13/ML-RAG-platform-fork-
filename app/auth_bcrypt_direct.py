"""
Future-proof authentication module using bcrypt directly.
This module eliminates the passlib dependency and bcrypt compatibility issues.
Use this as a drop-in replacement for the current auth.py when ready.
"""

from fastapi import HTTPException, Request
import bcrypt
from typing import Optional
import logging
from app.config import ADMIN_PASSWORD

logger = logging.getLogger(__name__)

# Simple user storage (in production, use a database)
USERS = {
    "admin": {
        "username": "admin",
        "hashed_password": None,  # Will be set below
        "is_active": True
    }
}

# Direct bcrypt functions (no passlib dependency)
def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its hash using bcrypt directly."""
    try:
        return bcrypt.checkpw(
            plain_password.encode('utf-8'),
            hashed_password.encode('utf-8')
        )
    except Exception as e:
        logger.error("Password verification error occurred")
        return False

# Initialize admin password hash
def init_admin_password():
    """Initialize the admin password hash."""
    hashed = hash_password(ADMIN_PASSWORD)
    USERS["admin"]["hashed_password"] = hashed
    logger.info("Admin password hash initialized with direct bcrypt")
    return hashed

# Initialize on module load
admin_hash = init_admin_password()

def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Authenticate a user with username and password."""
    logger.info(f"Attempting to authenticate user: {username}")
    
    if username not in USERS:
        logger.warning(f"User not found: {username}")
        return None
    
    user = USERS[username]
    logger.info(f"User found: {user['username']}, active: {user['is_active']}")
    
    if not user["is_active"]:
        logger.warning(f"User inactive: {username}")
        return None
    
    if not verify_password(password, user["hashed_password"]):
        logger.warning(f"Password verification failed for user: {username}")
        return None
    
    logger.info(f"Authentication successful for user: {username}")
    return user

# Session management (unchanged from original)
def create_session(request: Request, username: str):
    """Create a session for the authenticated user."""
    request.session["user"] = username
    request.session["authenticated"] = True
    logger.info(f"Session created for user: {username}")

def destroy_session(request: Request):
    """Destroy the user session."""
    if "user" in request.session:
        user = request.session["user"]
        logger.info(f"Destroying session for user: {user}")
    request.session.clear()

def get_current_user(request: Request) -> Optional[str]:
    """Get the current authenticated user from session."""
    if request.session.get("authenticated") and request.session.get("user"):
        return request.session["user"]
    return None

def is_authenticated(request: Request) -> bool:
    """Check if the current request is authenticated."""
    return request.session.get("authenticated", False) and request.session.get("user") is not None

# Authentication dependency
def require_auth(request: Request) -> str:
    """FastAPI dependency that requires authentication."""
    if not is_authenticated(request):
        logger.warning("Authentication required but user not authenticated")
        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )
    
    user = get_current_user(request)
    logger.info(f"Authentication check passed for user: {user}")
    return user

# Optional authentication dependency
def optional_auth(request: Request) -> Optional[str]:
    """FastAPI dependency for optional authentication."""
    return get_current_user(request)