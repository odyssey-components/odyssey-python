"""Shared startup for applications with an argparse entry point."""

import argparse
from collections.abc import Callable, Sequence
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import cast, get_args, get_type_hints

from pydantic import BaseModel

from ._loader import Loader
from .errors import ConfigurationError


def _bind_runtime_arguments(parser: argparse.ArgumentParser, model: type[object]) -> None:
    if not is_dataclass(model):
        raise TypeError("must be called with a dataclass type or instance")
    annotations = get_type_hints(model)
    for field in fields(model):
        flag = field.metadata.get("cli")
        if not field.init or flag is None:
            continue
        annotation = annotations[field.name]
        alternatives = get_args(annotation)
        if len(alternatives) == 2 and type(None) in alternatives:
            annotation = next(item for item in alternatives if item is not type(None))
        parser.add_argument(
            flag,
            dest=field.name,
            type=annotation,
            required=field.default is MISSING and field.default_factory is MISSING,
            default=argparse.SUPPRESS,
            help=field.metadata.get("help"),
        )


def run_app[ConfigT: BaseModel, RuntimeT](
    *,
    argv: Sequence[str] | None = None,
    name: str,
    loader: Loader[ConfigT],
    description: str | None = None,
    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None,
    runtime_model: type[RuntimeT] | None = None,
    start: Callable[[ConfigT, RuntimeT], int | None],
    version: Callable[[], str],
) -> int:
    """Handle CLI metadata requests, then load configuration before starting."""
    parser = argparse.ArgumentParser(prog=name, description=description)
    parser.add_argument("--version", action=_VersionAction, const=version)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--profile")
    loader.bind_arguments(parser)
    # argparse exposes option-name collisions, but allows duplicate destinations.
    config_destinations = {action.dest for action in parser._actions}
    if add_arguments is not None or runtime_model is not None:
        before = len(parser._actions)
        if runtime_model is not None:
            if add_arguments is not None:
                raise ValueError("choose either runtime_model or add_arguments")
            _bind_runtime_arguments(parser, runtime_model)
        elif add_arguments is not None:
            add_arguments(parser)
        app_destinations = {action.dest for action in parser._actions[before:]}
        if config_destinations & app_destinations:
            raise ValueError("application argument conflicts with a configuration destination")
    values = vars(parser.parse_args(argv))
    path = values.pop("config")
    profile = values.pop("profile")
    config_dir = values.pop("config_dir") if loader.directory_name is not None else None
    overrides = {key: values.pop(key) for key in config_destinations & values.keys()}
    try:
        config = loader.load(path, profile=profile, config_dir=config_dir, overrides=overrides)
    except ConfigurationError as error:
        parser.exit(2, f"{parser.prog}: {error}\n")
    except (ValueError, OSError):
        parser.exit(2, f"{parser.prog}: invalid configuration\n")
    runtime = runtime_model(**values) if runtime_model is not None else argparse.Namespace(**values)
    result = start(config, cast("RuntimeT", runtime))
    return 0 if result is None else result


class _VersionAction(argparse.Action):
    def __init__(self, option_strings: Sequence[str], dest: str, *, const: Callable[[], str]) -> None:
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS, help="show the program version and exit")
        self.version = const

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        print(self.version())
        parser.exit(0)
