#!/bin/bash

VERBOSITY=$1
shift
SCRIPT=$1
shift

if [[ $VERBOSITY -eq 1 ]]; then
  PYTHONPATH=$PYPINT python3 $SCRIPT "$@" | sed -n 's/^.*: !> //p'
elif [[ $VERBOSITY -eq 2 ]]; then
  PYTHONPATH=$PYPINT python3 $SCRIPT "$@" | sed -n 's/^.*: [!>]> //p'
else
  PYTHONPATH=$PYPINT python3 $SCRIPT "$@" | sed -n 's/^.*: [!> ]> //p'
fi

