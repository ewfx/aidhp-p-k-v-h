from datetime import datetime, timedelta
from typing import Dict, Optional
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
import logging

from src.config.config import SECURITY_CONFIG

# Setup logging
logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)

def create_access_token(
    data: Dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=SECURITY_CONFIG["ACCESS_TOKEN_EXPIRE_MINUTES"]
        )
        
    to_encode.update({"exp": expire})
    
    try:
        encoded_jwt = jwt.encode(
            to_encode,
            SECURITY_CONFIG["JWT_SECRET"],
            algorithm=SECURITY_CONFIG["JWT_ALGORITHM"]
        )
        return encoded_jwt
    except Exception as e:
        logger.error(f"Error creating access token: {str(e)}")
        raise

async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict:
    """Get current user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            token,
            SECURITY_CONFIG["JWT_SECRET"],
            algorithms=[SECURITY_CONFIG["JWT_ALGORITHM"]]
        )
        
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
            
        # You might want to validate against your user database here
        return {"user_id": user_id}
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.JWTError:
        raise credentials_exception
    except Exception as e:
        logger.error(f"Error validating token: {str(e)}")
        raise credentials_exception

def validate_api_key(api_key: str) -> bool:
    """Validate API key"""
    # Implementation depends on your API key storage and validation logic
    pass

def check_permissions(user: Dict, required_permissions: list) -> bool:
    """Check if user has required permissions"""
    # Implementation depends on your permission system
    pass

def generate_api_key() -> str:
    """Generate new API key"""
    # Implementation depends on your API key generation logic
    pass

def revoke_token(token: str) -> bool:
    """Revoke a JWT token"""
    # Implementation depends on your token revocation strategy
    pass

def refresh_token(refresh_token: str) -> Dict[str, str]:
    """Refresh an access token using a refresh token"""
    # Implementation depends on your token refresh strategy
    pass

class RateLimiter:
    """Rate limiting implementation"""
    
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {}
        
    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed under rate limit"""
        now = datetime.utcnow()
        
        # Clean old requests
        self._clean_old_requests(now)
        
        # Get request history for key
        requests = self.requests.get(key, [])
        
        # Check rate limit
        if len(requests) >= self.max_requests:
            return False
            
        # Record request
        requests.append(now)
        self.requests[key] = requests
        
        return True
        
    def _clean_old_requests(self, now: datetime):
        """Clean requests older than time window"""
        cutoff = now - timedelta(seconds=self.time_window)
        
        for key in list(self.requests.keys()):
            self.requests[key] = [
                req for req in self.requests[key]
                if req > cutoff
            ]
            
            if not self.requests[key]:
                del self.requests[key]

class APIKeyAuth:
    """API Key authentication handler"""
    
    def __init__(self):
        self.api_keys = {}
        
    def validate_key(self, api_key: str) -> bool:
        """Validate API key"""
        return api_key in self.api_keys
        
    def create_key(self, user_id: str) -> str:
        """Create new API key"""
        # Implementation depends on your API key generation logic
        pass
        
    def revoke_key(self, api_key: str) -> bool:
        """Revoke API key"""
        if api_key in self.api_keys:
            del self.api_keys[api_key]
            return True
        return False 