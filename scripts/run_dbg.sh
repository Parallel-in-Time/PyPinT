#!/bin/bash

SCRIPT=$1
shift

PYTHONPATH=$PYPINT python3 $SCRIPT "$@" | sed -n 's/^.*: [!]*> //p'

