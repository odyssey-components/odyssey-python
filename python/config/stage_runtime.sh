#!/bin/sh
# Stage the root workspace's pinned dependencies for a Linux x86_64 Worker.
set -eu
if [ "$#" -ne 2 ]; then
    echo "usage: stage_runtime.sh <worker-python-version> <context>/config-runtime" >&2
    exit 2
fi
python_version=$1
destination=$2
if [ -e "$destination" ]; then
    echo "configuration runtime target already exists: $destination" >&2
    exit 2
fi
root=$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd)
mkdir -p "$(dirname -- "$destination")"
destination=$(CDPATH='' cd -- "$(dirname -- "$destination")" && pwd)/$(basename -- "$destination")
cd "$root"
requirements=$(uv export --locked --package odyssey-config --no-emit-workspace --no-dev --no-hashes)
printf "%s\n" "$requirements" |
    uv pip install --no-deps --only-binary :all: --python-platform x86_64-manylinux_2_17 \
        --python-version "$python_version" --target "$destination" -r -
cp -R python/config/src/odyssey_config "$destination/"
find "$destination/odyssey_config" -name __pycache__ -type d -prune -exec rm -rf {} +
