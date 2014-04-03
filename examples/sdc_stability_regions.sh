#!/bin/sh

python3 "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/sdc_stability_regions.py "$@" | grep -e '^\[' --color=never

