from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    model_config = ConfigDict(extra="forbid")


class UserRead(BaseModel):
    id: int
    email: EmailStr
    is_active: bool

    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"  # noqa: S105


class PasswordUpdate(BaseModel):
    current_password: str = Field(min_length=8)
    new_password: str = Field(min_length=8)

    model_config = ConfigDict(extra="forbid")


class AccountDelete(BaseModel):
    password: str

    model_config = ConfigDict(extra="forbid")
