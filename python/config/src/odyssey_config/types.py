"""Common configuration data types; applications construct their own clients."""

import re
from ipaddress import ip_address
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import ConfigDict, Field, SecretStr, field_validator, model_validator

from .model import Settings

_DURATION = re.compile(r"\+?(?:(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:ns|us|µs|μs|ms|s|m|h))+")


class _CommonConfig(Settings):
    model_config = ConfigDict(validate_default=True)


class Database(_CommonConfig):
    host: str
    port: int = Field(default=0, ge=0, le=65535)
    username: SecretStr | None = None
    password: SecretStr | None = None
    dbname: str = ""
    database_schema: str = Field(default="", alias="schema")
    sslmode: Literal["", "disable", "allow", "prefer", "require", "verify-ca", "verify-full"] = ""
    max_open_conns: int = Field(default=0, ge=0)
    max_idle_conns: int = Field(default=0, ge=0)

    @field_validator("host")
    @classmethod
    def validate_host(cls, value: str) -> str:
        try:
            ip_address(value)
        except ValueError:
            if not value.strip() or any(c in value for c in ":/\\?#@[] \t\r\n"):
                raise ValueError("expected a hostname or IP address") from None
        return value

    @model_validator(mode="after")
    def validate_pool(self) -> Self:
        if 0 < self.max_open_conns < self.max_idle_conns:
            raise ValueError("invalid connection pool limits")
        return self


class Web(_CommonConfig):
    listen_addr: str
    read_header_timeout: str = "0"
    read_timeout: str = "0"
    write_timeout: str = "0"
    idle_timeout: str = "0"

    @field_validator("listen_addr")
    @classmethod
    def validate_address(cls, value: str) -> str:
        host, separator, port = value.rpartition(":")
        valid_host = bool(re.fullmatch(r"[^:\[\]]*|\[[^\[\]]*\]", host))
        if not separator or not valid_host or not re.fullmatch(r"\+?[0-9]+", port) or not 1 <= int(port) <= 65535:
            raise ValueError("expected a listener address with a port between 1 and 65535")
        return value

    @field_validator("read_header_timeout", "read_timeout", "write_timeout", "idle_timeout")
    @classmethod
    def validate_timeout(cls, value: str) -> str:
        if value not in {"", "0", "+0", "-0"} and not _DURATION.fullmatch(value):
            raise ValueError("expected a nonnegative duration")
        return value


class ObjectStorage(_CommonConfig):
    endpoint: str = ""
    bucket: str = ""
    access_key: SecretStr | None = None
    secret_key: SecretStr | None = None

    @model_validator(mode="after")
    def validate_storage(self) -> Self:
        access_key = self.access_key.get_secret_value() if self.access_key is not None else ""
        secret_key = self.secret_key.get_secret_value() if self.secret_key is not None else ""
        if not any((self.endpoint, self.bucket, access_key, secret_key)):
            return self
        if not all(value.strip() for value in (self.bucket, access_key, secret_key)):
            raise ValueError("endpoint, bucket, access_key and secret_key must be configured together")
        try:
            url = urlsplit(self.endpoint)
            valid = (
                url.scheme in {"http", "https"} and bool(url.hostname)
                and url.username is None and url.path in {"", "/"}
                and not url.query and not url.fragment
            )
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("endpoint must be an HTTP or HTTPS origin")
        return self


class Logging(_CommonConfig):
    level: str = ""
    format: Literal["", "console", "json"] = ""
    output: str = ""

    @field_validator("level")
    @classmethod
    def validate_level(cls, value: str) -> str:
        if value.lower() not in {"", "debug", "info", "warn", "error", "dpanic", "panic", "fatal"}:
            raise ValueError("invalid log level")
        return value

    @field_validator("output")
    @classmethod
    def validate_output(cls, value: str) -> str:
        if value and not value.strip():
            raise ValueError("expected stdout, stderr, or a file path")
        return value
