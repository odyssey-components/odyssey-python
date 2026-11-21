"""Common behavior for typed configuration models."""

from typing import Self

from pydantic import BaseModel, ConfigDict


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    def validate_settings(self) -> Self:
        """Revalidate a copy, including nested values changed since loading."""
        return type(self).model_validate(self.model_dump(by_alias=True, round_trip=True))
