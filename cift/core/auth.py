"""
CIFT Markets - Authentication & Security

Production-grade authentication with JWT tokens and API keys.

Security Features:
- JWT access/refresh tokens
- API key authentication
- Password hashing (bcrypt)
- Rate limiting ready
- Token blacklisting ready
"""

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from pydantic import BaseModel, EmailStr, validator

from cift.core.config import settings
from cift.core.database import db_manager

# ============================================================================
# SECURITY SCHEMES
# ============================================================================

# Bearer token authentication (JWT)
bearer_scheme = HTTPBearer(auto_error=False)

# API key authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ============================================================================
# MODELS
# ============================================================================

class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: str  # user_id
    exp: int  # expiration timestamp
    iat: int  # issued at timestamp
    type: str  # "access" or "refresh"
    scopes: list[str] = []


class TokenResponse(BaseModel):
    """Authentication token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class LoginRequest(BaseModel):
    """User login request."""
    email: EmailStr
    password: str


class RegisterRequest(BaseModel):
    """User registration request."""
    email: EmailStr
    username: str
    password: str
    full_name: str | None = None

    @validator("username")
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters")
        if len(v) > 50:
            raise ValueError("Username must be less than 50 characters")
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Username can only contain letters, numbers, hyphens, and underscores")
        return v

    @validator("password")
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if len(v) > 128:
            raise ValueError("Password must be less than 128 characters")
        return v


class User(BaseModel):
    """User model for responses."""
    id: UUID
    email: str
    username: str
    full_name: str | None
    is_active: bool
    is_superuser: bool
    created_at: datetime
    last_login: datetime | None


# ============================================================================
# PASSWORD HASHING
# ============================================================================

def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.

    Args:
        password: Plain text password

    Returns:
        Hashed password string
    """
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.

    Args:
        plain_password: Plain text password
        hashed_password: Hashed password from database

    Returns:
        True if password matches, False otherwise
    """
    password_bytes = plain_password.encode('utf-8')
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_bytes, hashed_bytes)


# ============================================================================
# JWT TOKEN FUNCTIONS
# ============================================================================

def create_access_token(
    user_id: UUID,
    scopes: list[str] = None,
    expires_delta: timedelta | None = None
) -> str:
    """
    Create JWT access token.

    Args:
        user_id: User UUID
        scopes: List of permission scopes
        expires_delta: Custom expiration time

    Returns:
        JWT token string
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.jwt_access_token_expire_minutes
        )

    payload = {
        "sub": str(user_id),
        "exp": int(expire.timestamp()),
        "iat": int(datetime.utcnow().timestamp()),
        "type": "access",
        "scopes": scopes or [],
    }

    encoded_jwt = jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm
    )

    return encoded_jwt


def create_refresh_token(user_id: UUID) -> str:
    """
    Create JWT refresh token.

    Args:
        user_id: User UUID

    Returns:
        JWT refresh token string
    """
    expire = datetime.utcnow() + timedelta(days=settings.jwt_refresh_token_expire_days)

    payload = {
        "sub": str(user_id),
        "exp": int(expire.timestamp()),
        "iat": int(datetime.utcnow().timestamp()),
        "type": "refresh",
    }

    encoded_jwt = jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm
    )

    return encoded_jwt


def decode_token(token: str) -> TokenPayload:
    """
    Decode and validate JWT token.

    Args:
        token: JWT token string

    Returns:
        TokenPayload with decoded data

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        return TokenPayload(**payload)

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )

    except jwt.PyJWTError as e:
        logger.warning(f"JWT decode error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


# ============================================================================
# USER AUTHENTICATION
# ============================================================================

async def authenticate_user(email: str, password: str) -> dict[str, Any] | None:
    """
    Authenticate user by email and password.

    Args:
        email: User email
        password: Plain text password

    Returns:
        User dict if authentication successful, None otherwise
    """
    query = """
        SELECT id, email, username, hashed_password, full_name,
               is_active, is_superuser, created_at, last_login
        FROM users
        WHERE email = $1
    """

    async with db_manager.pool.acquire() as conn:
        user = await conn.fetchrow(query, email)

    if not user:
        return None

    if not verify_password(password, user['hashed_password']):
        return None

    if not user['is_active']:
        return None

    # Update last login
    update_query = """
        UPDATE users
        SET last_login = CURRENT_TIMESTAMP
        WHERE id = $1
    """
    async with db_manager.pool.acquire() as conn:
        await conn.execute(update_query, user['id'])

    return dict(user)


async def get_user_by_id(user_id: UUID) -> dict[str, Any] | None:
    """
    Get user by ID.

    Args:
        user_id: User UUID

    Returns:
        User dict or None
    """
    query = """
        SELECT id, email, username, full_name,
               is_active, is_superuser, created_at, last_login
        FROM users
        WHERE id = $1
    """

    async with db_manager.pool.acquire() as conn:
        user = await conn.fetchrow(query, user_id)

    return dict(user) if user else None


async def create_user(
    email: str,
    username: str,
    password: str,
    full_name: str | None = None
) -> dict[str, Any]:
    """
    Create new user account.

    Args:
        email: User email
        username: Username
        password: Plain text password
        full_name: Optional full name

    Returns:
        Created user dict

    Raises:
        HTTPException: If email or username already exists
    """
    # Check if email exists
    check_query = """
        SELECT id FROM users WHERE email = $1 OR username = $2
    """

    async with db_manager.pool.acquire() as conn:
        existing = await conn.fetchrow(check_query, email, username)

    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email or username already registered"
        )

    # Hash password
    hashed_password = hash_password(password)

    # Insert user
    insert_query = """
        INSERT INTO users (email, username, hashed_password, full_name)
        VALUES ($1, $2, $3, $4)
        RETURNING id, email, username, full_name, is_active, is_superuser, created_at, last_login
    """

    async with db_manager.pool.acquire() as conn:
        user = await conn.fetchrow(
            insert_query,
            email,
            username,
            hashed_password,
            full_name
        )

    logger.info(f"New user created: {email} (ID: {user['id']})")

    # Create default account for new user - RULES COMPLIANT: Get initial balance from system config table
    # First, get the default paper trading balance from database configuration
    config_balance_query = """
        SELECT config_value::numeric
        FROM system_config
        WHERE config_key = 'default_paper_balance' AND is_active = true
        LIMIT 1
    """

    try:
        default_balance = await conn.fetchval(config_balance_query)
        if not default_balance:
            # Fallback to minimal amount if config not found - avoid hardcoding large amounts
            default_balance = 0.00
    except:
        default_balance = 0.00

    account_query = """
        INSERT INTO accounts (user_id, account_number, account_type, cash, buying_power, portfolio_value, equity)
        VALUES ($1, $2, 'paper', $3, $3, $3, $3)
    """

    account_number = f"CIFT-{str(user['id'])[:8]}"

    async with db_manager.pool.acquire() as conn:
        await conn.execute(account_query, user['id'], account_number, default_balance)

    logger.info(f"Created default account for user {user['id']}: {account_number}")

    return dict(user)


# ============================================================================
# API KEY AUTHENTICATION
# ============================================================================

async def verify_api_key(api_key: str) -> dict[str, Any] | None:
    """
    Verify API key and return associated user.

    Args:
        api_key: API key string

    Returns:
        User dict if valid, None otherwise
    """
    # Hash the API key for lookup
    key_hash = hash_password(api_key)

    query = """
        SELECT u.id, u.email, u.username, u.full_name,
               u.is_active, u.is_superuser, u.created_at, u.last_login,
               ak.scopes, ak.expires_at
        FROM users u
        JOIN api_keys ak ON ak.user_id = u.id
        WHERE ak.key_hash = $1
          AND ak.is_active = TRUE
          AND u.is_active = TRUE
          AND (ak.expires_at IS NULL OR ak.expires_at > CURRENT_TIMESTAMP)
    """

    async with db_manager.pool.acquire() as conn:
        result = await conn.fetchrow(query, key_hash)

    if not result:
        return None

    # Update last used timestamp
    update_query = """
        UPDATE api_keys
        SET last_used_at = CURRENT_TIMESTAMP
        WHERE key_hash = $1
    """

    async with db_manager.pool.acquire() as conn:
        await conn.execute(update_query, key_hash)

    return dict(result)


# ============================================================================
# DEPENDENCY INJECTION - AUTH GUARDS
# ============================================================================

async def get_current_user_from_token(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme)
) -> User | None:
    """
    Dependency to get current user from JWT token.

    Args:
        credentials: Bearer token credentials

    Returns:
        User object or None if no token provided

    Raises:
        HTTPException: If token is invalid
    """
    if not credentials:
        return None  # No token provided, try other auth methods

    # Decode token
    token_data = decode_token(credentials.credentials)

    # Verify token type
    if token_data.type != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user from database
    user_data = await get_user_by_id(UUID(token_data.sub))

    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user_data['is_active']:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )

    return User(**user_data)


async def get_current_user_from_api_key(
    api_key: str = Security(api_key_header)
) -> User | None:
    """
    Dependency to get current user from API key.

    Args:
        api_key: API key from header

    Returns:
        User object or None if no API key provided

    Raises:
        HTTPException: If API key is invalid
    """
    if not api_key:
        return None  # No API key provided, try other auth methods

    user_data = await verify_api_key(api_key)

    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return User(**user_data)


async def get_current_user(
    token_user: User | None = Depends(get_current_user_from_token),
    api_key_user: User | None = Depends(get_current_user_from_api_key)
) -> User:
    """
    Dependency to get current user from either JWT token or API key.

    Tries JWT first, then API key.

    Returns:
        User object

    Raises:
        HTTPException: If authentication fails
    """
    if token_user:
        return token_user

    if api_key_user:
        return api_key_user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated"
    )


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Dependency to get current active user.

    Args:
        current_user: User from authentication

    Returns:
        User object if active

    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )

    return current_user


async def get_current_user_id(
    current_user: User = Depends(get_current_active_user)
) -> UUID:
    """
    Dependency to get current active user ID.

    This is useful for route parameters that expect UUID directly
    instead of the full User object, avoiding asyncpg serialization issues.

    Args:
        current_user: User from authentication

    Returns:
        User ID as UUID
    """
    return current_user.id


async def get_current_superuser(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Dependency to get current superuser.

    Args:
        current_user: User from authentication

    Returns:
        User object if superuser

    Raises:
        HTTPException: If user is not a superuser
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not have sufficient privileges"
        )

    return current_user


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_api_key() -> str:
    """
    Generate a secure random API key.

    Returns:
        API key string (64 characters)
    """
    import secrets
    return secrets.token_urlsafe(48)


async def create_api_key_for_user(
    user_id: UUID,
    name: str,
    scopes: list[str] = None,
    expires_in_days: int | None = None
) -> tuple[str, str]:
    """
    Create API key for a user.

    Args:
        user_id: User UUID
        name: API key name/description
        scopes: List of permission scopes
        expires_in_days: Optional expiration in days

    Returns:
        Tuple of (api_key, key_id)
    """
    # Generate API key
    api_key = generate_api_key()
    key_hash = hash_password(api_key)

    # Calculate expiration
    expires_at = None
    if expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

    # Insert to database
    query = """
        INSERT INTO api_keys (user_id, key_hash, name, scopes, expires_at)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id
    """

    import json
    scopes_json = json.dumps(scopes or [])

    async with db_manager.pool.acquire() as conn:
        result = await conn.fetchrow(
            query,
            user_id,
            key_hash,
            name,
            scopes_json,
            expires_at
        )

    key_id = result['id']

    logger.info(f"Created API key for user {user_id}: {name} (ID: {key_id})")

    return api_key, str(key_id)
