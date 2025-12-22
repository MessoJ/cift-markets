"""
CIFT Markets - Authentication API Routes

Endpoints for user registration, login, token refresh, and API key management.
"""

import os
import secrets
import string
from datetime import datetime
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from loguru import logger
from pydantic import BaseModel

from cift.core.auth import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    User,
    authenticate_user,
    create_access_token,
    create_api_key_for_user,
    create_refresh_token,
    create_user,
    decode_token,
    get_current_active_user,
)
from cift.core.config import settings
from cift.core.database import db_manager
from cift.core.limiter import limiter

# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ============================================================================
# MODELS
# ============================================================================


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""

    refresh_token: str


class APIKeyResponse(BaseModel):
    """API key creation response."""

    api_key: str
    key_id: str
    name: str
    expires_at: datetime | None
    message: str = "IMPORTANT: Save this API key securely. It will not be shown again."


class APIKeyCreateRequest(BaseModel):
    """API key creation request."""

    name: str
    scopes: list[str] = []
    expires_in_days: int | None = None


# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================


@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest):
    """
    Register a new user account.

    Creates user with default paper trading account ($100,000).

    **Requirements**:
    - Valid email address
    - Username (3-50 chars, alphanumeric + _-)
    - Password (8+ chars)
    """
    try:
        user_data = await create_user(
            email=request.email,
            username=request.username,
            password=request.password,
            full_name=request.full_name,
        )

        logger.info(f"User registered: {user_data['email']}")

        return User(**user_data)

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Registration error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user account",
        ) from e


# ============================================================================
# OAUTH HELPERS
# ============================================================================


def generate_random_password(length=32):
    alphabet = string.ascii_letters + string.digits + string.punctuation
    return "".join(secrets.choice(alphabet) for i in range(length))


async def get_or_create_oauth_user(email: str, username: str, full_name: str = None):
    """Find existing user by email or create a new one."""
    # Check if user exists by email
    query = "SELECT * FROM users WHERE email = $1"
    async with db_manager.pool.acquire() as conn:
        user = await conn.fetchrow(query, email)

    if user:
        return dict(user)

    # Create new user
    # Handle username collisions
    base_username = username
    while True:
        check_query = "SELECT id FROM users WHERE username = $1"
        async with db_manager.pool.acquire() as conn:
            existing = await conn.fetchrow(check_query, username)
        if not existing:
            break
        username = f"{base_username}_{secrets.token_hex(2)}"

    password = generate_random_password()

    # Create user using core logic
    try:
        return await create_user(email, username, password, full_name)
    except HTTPException:
        # Fallback if race condition occurred
        async with db_manager.pool.acquire() as conn:
            user = await conn.fetchrow(query, email)
        return dict(user)


async def complete_oauth_login(user_data: dict):
    """Generate tokens and redirect to frontend."""
    user_id = user_data["id"]
    access_token = create_access_token(user_id=user_id, scopes=["user"])
    refresh_token = create_refresh_token(user_id=user_id)

    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
    redirect_url = (
        f"{frontend_url}/auth/callback?access_token={access_token}&refresh_token={refresh_token}"
    )

    return RedirectResponse(url=redirect_url)


# ============================================================================
# OAUTH ENDPOINTS
# ============================================================================


@router.get("/github/login")
@limiter.limit("10/minute")
async def github_login(request: Request):
    """Initiate GitHub OAuth login."""
    client_id = getattr(settings, "github_client_id", os.getenv("GITHUB_CLIENT_ID"))
    if not client_id:
        raise HTTPException(status_code=500, detail="GitHub Client ID not configured")

    redirect_uri = (
        f"{getattr(settings, 'api_base_url', 'http://localhost:8000')}/api/v1/auth/github/callback"
    )
    return {
        "url": f"https://github.com/login/oauth/authorize?client_id={client_id}&redirect_uri={redirect_uri}&scope=user:email"
    }


@router.get("/github/callback")
@limiter.limit("10/minute")
async def github_callback(request: Request, code: str):
    """Handle GitHub OAuth callback."""
    client_id = getattr(settings, "github_client_id", os.getenv("GITHUB_CLIENT_ID"))
    client_secret = getattr(settings, "github_client_secret", os.getenv("GITHUB_CLIENT_SECRET"))

    if not client_id or not client_secret:
        raise HTTPException(status_code=500, detail="GitHub credentials not configured")

    async with httpx.AsyncClient() as client:
        # Exchange code for token
        token_resp = await client.post(
            "https://github.com/login/oauth/access_token",
            headers={"Accept": "application/json"},
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
            },
        )
        token_data = token_resp.json()
        if "error" in token_data:
            raise HTTPException(status_code=400, detail=token_data["error_description"])

        access_token = token_data["access_token"]

        # Get user info
        user_resp = await client.get(
            "https://api.github.com/user", headers={"Authorization": f"Bearer {access_token}"}
        )
        user_data = user_resp.json()

        # Get email (if private)
        email_resp = await client.get(
            "https://api.github.com/user/emails",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        emails = email_resp.json()
        primary_email = next((e["email"] for e in emails if e["primary"]), None)

        if not primary_email:
            raise HTTPException(status_code=400, detail="No primary email found on GitHub account")

        # Create or login user
        user = await get_or_create_oauth_user(
            email=primary_email, username=user_data["login"], full_name=user_data.get("name")
        )

        return await complete_oauth_login(user)


@router.get("/microsoft/login")
@limiter.limit("10/minute")
async def microsoft_login(request: Request):
    """Initiate Microsoft OAuth login."""
    client_id = getattr(settings, "microsoft_client_id", os.getenv("MICROSOFT_CLIENT_ID"))
    if not client_id:
        raise HTTPException(status_code=500, detail="Microsoft Client ID not configured")

    redirect_uri = f"{getattr(settings, 'api_base_url', 'http://localhost:8000')}/api/v1/auth/microsoft/callback"
    tenant = "common"
    return {
        "url": f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize?client_id={client_id}&response_type=code&redirect_uri={redirect_uri}&response_mode=query&scope=User.Read"
    }


@router.get("/microsoft/callback")
@limiter.limit("10/minute")
async def microsoft_callback(request: Request, code: str):
    """Handle Microsoft OAuth callback."""
    client_id = getattr(settings, "microsoft_client_id", os.getenv("MICROSOFT_CLIENT_ID"))
    client_secret = getattr(
        settings, "microsoft_client_secret", os.getenv("MICROSOFT_CLIENT_SECRET")
    )

    if not client_id or not client_secret:
        raise HTTPException(status_code=500, detail="Microsoft credentials not configured")

    redirect_uri = f"{getattr(settings, 'api_base_url', 'http://localhost:8000')}/api/v1/auth/microsoft/callback"

    async with httpx.AsyncClient() as client:
        # Exchange code for token
        token_resp = await client.post(
            "https://login.microsoftonline.com/common/oauth2/v2.0/token",
            data={
                "client_id": client_id,
                "scope": "User.Read",
                "code": code,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
                "client_secret": client_secret,
            },
        )
        token_data = token_resp.json()
        if "error" in token_data:
            raise HTTPException(
                status_code=400, detail=token_data.get("error_description", "Login failed")
            )

        access_token = token_data["access_token"]

        # Get user info
        user_resp = await client.get(
            "https://graph.microsoft.com/v1.0/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        user_data = user_resp.json()

        email = user_data.get("mail") or user_data.get("userPrincipalName")
        if not email:
            raise HTTPException(status_code=400, detail="No email found on Microsoft account")

        # Create or login user
        user = await get_or_create_oauth_user(
            email=email,
            username=user_data.get("displayName", email.split("@")[0]),
            full_name=user_data.get("displayName"),
        )

        return await complete_oauth_login(user)


@router.get("/google/login")
@limiter.limit("10/minute")
async def google_login(request: Request):
    """Initiate Google OAuth login."""
    client_id = getattr(settings, "google_client_id", os.getenv("GOOGLE_CLIENT_ID"))
    if not client_id:
        raise HTTPException(status_code=500, detail="Google Client ID not configured")

    redirect_uri = (
        f"{getattr(settings, 'api_base_url', 'http://localhost:8000')}/api/v1/auth/google/callback"
    )
    return {
        "url": f"https://accounts.google.com/o/oauth2/v2/auth?client_id={client_id}&redirect_uri={redirect_uri}&response_type=code&scope=email%20profile"
    }


@router.get("/google/callback")
@limiter.limit("10/minute")
async def google_callback(request: Request, code: str):
    """Handle Google OAuth callback."""
    client_id = getattr(settings, "google_client_id", os.getenv("GOOGLE_CLIENT_ID"))
    client_secret = getattr(settings, "google_client_secret", os.getenv("GOOGLE_CLIENT_SECRET"))

    if not client_id or not client_secret:
        raise HTTPException(status_code=500, detail="Google credentials not configured")

    redirect_uri = (
        f"{getattr(settings, 'api_base_url', 'http://localhost:8000')}/api/v1/auth/google/callback"
    )

    async with httpx.AsyncClient() as client:
        # Exchange code for token
        token_resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        token_data = token_resp.json()
        if "error" in token_data:
            raise HTTPException(
                status_code=400, detail=token_data.get("error_description", "Login failed")
            )

        access_token = token_data["access_token"]

        # Get user info
        user_resp = await client.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        user_data = user_resp.json()

        email = user_data.get("email")
        if not email:
            raise HTTPException(status_code=400, detail="No email found on Google account")

        # Create or login user
        user = await get_or_create_oauth_user(
            email=email, username=email.split("@")[0], full_name=user_data.get("name")
        )

        return await complete_oauth_login(user)


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """
    Authenticate and get access tokens.

    Returns JWT access token (30 min) and refresh token (7 days).

    **Usage**:
    ```bash
    curl -X POST /api/v1/auth/login \\
      -H "Content-Type: application/json" \\
      -d '{"email":"user@example.com","password":"password"}'
    ```
    """
    # Authenticate user
    user = await authenticate_user(request.email, request.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create tokens
    access_token = create_access_token(user_id=user["id"])
    refresh_token = create_refresh_token(user_id=user["id"])

    logger.info(f"User logged in: {user['email']}")

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=1800,  # 30 minutes
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest):
    """
    Refresh access token using refresh token.

    **Usage**:
    ```bash
    curl -X POST /api/v1/auth/refresh \\
      -H "Content-Type: application/json" \\
      -d '{"refresh_token":"your_refresh_token"}'
    ```
    """
    try:
        # Decode refresh token
        token_data = decode_token(request.refresh_token)

        # Verify it's a refresh token
        if token_data.type != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type"
            )

        # Create new tokens
        user_id = UUID(token_data.sub)
        access_token = create_access_token(user_id=user_id)
        new_refresh_token = create_refresh_token(user_id=user_id)

        logger.info(f"Token refreshed for user: {user_id}")

        return TokenResponse(
            access_token=access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=1800,
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        ) from e


@router.get("/me", response_model=User)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """
    Get current authenticated user information.

    Requires valid JWT token or API key.

    **Usage**:
    ```bash
    curl -X GET /api/v1/auth/me \\
      -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
    ```
    """
    return current_user


@router.post("/logout")
async def logout(current_user: User = Depends(get_current_active_user)):
    """
    Logout current user.

    Note: This is a placeholder. In production, you should:
    1. Add token to blacklist/revocation list
    2. Invalidate refresh tokens
    3. Clear client-side tokens
    """
    logger.info(f"User logged out: {current_user.email}")

    return {"message": "Successfully logged out", "user_id": str(current_user.id)}


# ============================================================================
# API KEY MANAGEMENT
# ============================================================================


@router.post("/api-keys", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    request: APIKeyCreateRequest, current_user: User = Depends(get_current_active_user)
):
    """
    Create new API key for authenticated user.

    **Important**: The API key is shown only once. Store it securely!

    **Usage**:
    ```bash
    curl -X POST /api/v1/auth/api-keys \\
      -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \\
      -H "Content-Type: application/json" \\
      -d '{"name":"Trading Bot Key","expires_in_days":90}'
    ```

    **Scopes** (optional):
    - `trading:read` - Read trading data
    - `trading:write` - Execute trades
    - `market_data:read` - Access market data
    - `account:read` - Read account info
    - `account:write` - Modify account
    """
    try:
        from datetime import timedelta

        api_key, key_id = await create_api_key_for_user(
            user_id=current_user.id,
            name=request.name,
            scopes=request.scopes,
            expires_in_days=request.expires_in_days,
        )

        expires_at = None
        if request.expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=request.expires_in_days)

        logger.info(f"API key created for user {current_user.email}: {request.name}")

        return APIKeyResponse(
            api_key=api_key, key_id=key_id, name=request.name, expires_at=expires_at
        )

    except Exception as e:
        logger.error(f"API key creation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create API key"
        ) from e


@router.get("/api-keys")
async def list_api_keys(current_user: User = Depends(get_current_active_user)):
    """
    List all API keys for authenticated user.

    Returns key metadata only (not the actual keys).
    """
    from cift.core.database import db_manager

    query = """
        SELECT id, name, scopes, is_active, expires_at, created_at, last_used_at
        FROM api_keys
        WHERE user_id = $1
        ORDER BY created_at DESC
    """

    async with db_manager.pool.acquire() as conn:
        rows = await conn.fetch(query, current_user.id)

    return [dict(row) for row in rows]


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(key_id: UUID, current_user: User = Depends(get_current_active_user)):
    """
    Revoke (deactivate) an API key.

    The key will no longer be valid for authentication.
    """
    from cift.core.database import db_manager

    query = """
        UPDATE api_keys
        SET is_active = FALSE
        WHERE id = $1 AND user_id = $2
        RETURNING id
    """

    async with db_manager.pool.acquire() as conn:
        result = await conn.fetchrow(query, key_id, current_user.id)

    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")

    logger.info(f"API key revoked: {key_id} by user {current_user.email}")

    return {"message": "API key revoked successfully", "key_id": str(key_id)}


# ============================================================================
# PASSWORD MANAGEMENT
# ============================================================================


class ChangePasswordRequest(BaseModel):
    """Change password request."""

    current_password: str
    new_password: str


@router.post("/change-password")
async def change_password(
    request: ChangePasswordRequest, current_user: User = Depends(get_current_active_user)
):
    """
    Change user password.

    Requires current password for verification.
    """
    from cift.core.auth import hash_password, verify_password
    from cift.core.database import db_manager

    # Get current hashed password
    query = "SELECT hashed_password FROM users WHERE id = $1"

    async with db_manager.pool.acquire() as conn:
        result = await conn.fetchrow(query, current_user.id)

    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Verify current password
    if not verify_password(request.current_password, result["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect current password"
        )

    # Hash new password
    new_hashed = hash_password(request.new_password)

    # Update password
    update_query = """
        UPDATE users
        SET hashed_password = $1, updated_at = CURRENT_TIMESTAMP
        WHERE id = $2
    """

    async with db_manager.pool.acquire() as conn:
        await conn.execute(update_query, new_hashed, current_user.id)

    logger.info(f"Password changed for user: {current_user.email}")

    return {"message": "Password changed successfully"}


# ============================================================================
# ADMIN ENDPOINTS
# ============================================================================


@router.get("/users", dependencies=[Depends(get_current_active_user)])
async def list_users(
    skip: int = 0,
    limit: int = 100,
):
    """
    List all users (admin only).

    TODO: Add superuser check
    """
    from cift.core.database import db_manager

    query = """
        SELECT id, email, username, full_name, is_active, is_superuser, created_at, last_login
        FROM users
        ORDER BY created_at DESC
        LIMIT $1 OFFSET $2
    """

    async with db_manager.pool.acquire() as conn:
        rows = await conn.fetch(query, limit, skip)

    return [User(**dict(row)) for row in rows]
