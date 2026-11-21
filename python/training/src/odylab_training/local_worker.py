"""Watch the owning local web process and stop a fixed training module on exit."""

import argparse
import os
import runpy
import signal
import sys
import threading

def main(modules: tuple[str, ...] = ()) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.args[1:] if args.args and args.args[0] == "--" else args.args
    if not command or command[0] not in modules:
        parser.error("unsupported worker module")
    parent = args.parent_pid
    if parent <= 1 or os.getppid() != parent:
        parser.error("training parent unavailable")
    stopped = threading.Event()

    def watch_parent() -> None:
        while not stopped.wait(0.5):
            if os.getppid() != parent:
                os.kill(os.getpid(), signal.SIGTERM)
                return

    threading.Thread(target=watch_parent, daemon=True).start()
    try:
        sys.argv = command
        runpy.run_module(command[0], run_name="__main__")
    finally:
        stopped.set()


if __name__ == "__main__":
    main()
