"""
User data models.

This module defines Pydantic models for user entities and profiles.
"""

from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel, EmailStr, Field


class UserProfile(BaseModel):
    """User profile model."""

    first_name: Optional[str] = Field(None, description="User's first name")
    last_name: Optional[str] = Field(None, description="User's last name")
    company: Optional[str] = Field(None, description="User's company")
    phone: Optional[str] = Field(None, description="User's phone number")
    preferences: Dict[str, str] = Field(
        default_factory=dict, description="User preferences"
    )
    metadata: Dict[str, str] = Field(
        default_factory=dict, description="Additional user metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "first_name": "John",
                "last_name": "Doe",
                "company": "Real Estate Corp",
                "phone": "+1234567890",
                "preferences": {
                    "theme": "dark",
                    "notifications": "enabled",
                },
                "metadata": {},
            }
        }


class User(BaseModel):
    """User model representing a platform user."""

    id: str = Field(..., description="Unique user identifier")
    email: EmailStr = Field(..., description="User email address")
    profile: Optional[UserProfile] = Field(
        None, description="User profile information"
    )
    is_active: bool = Field(default=True, description="Whether user account is active")
    created_at: datetime = Field(
        default_factory=lambda: datetime.utcnow(), description="Account creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.utcnow(), description="Last update timestamp"
    )
    last_login_at: Optional[datetime] = Field(
        None, description="Last login timestamp"
    )

    def model_dump_for_firestore(self) -> dict:
        """Serialize model for Firestore (convert datetime to ISO strings)."""
        data = self.model_dump()
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        if self.last_login_at:
            data["last_login_at"] = self.last_login_at.isoformat()
        return data

    @classmethod
    def from_firestore(cls, user_id: str, data: dict) -> "User":
        """Create User instance from Firestore data."""
        # Parse datetime strings back to datetime objects
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data and isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        if "last_login_at" in data and data["last_login_at"]:
            if isinstance(data["last_login_at"], str):
                data["last_login_at"] = datetime.fromisoformat(data["last_login_at"])

        data["id"] = user_id
        return cls(**data)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "user_123456",
                "email": "user@example.com",
                "profile": {
                    "first_name": "John",
                    "last_name": "Doe",
                },
                "is_active": True,
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            }
        }
