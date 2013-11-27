#!/bin/bash

SCRIPT=$1
PYPINT=${2:-`pwd`}

PYTHONPATH=$PYPINT python $SCRIPT | sed -n 's/^.*: [!]*> //p'

