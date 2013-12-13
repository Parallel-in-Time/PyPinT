#!/bin/bash

SCRIPT=$1
PYPINT=${2:-`pwd`}

PYTHONPATH=$PYPINT python3 $SCRIPT | sed -n 's/^.*: > //p'

