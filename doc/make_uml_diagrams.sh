#!/bin/bash
CWD="`pwd`"
PUML_OPTS="-tsvg -r -failfast2 -duration -nbthread auto -overwrite"

SRCS="${CWD}/source"
DEST="${CWD}/source/_static/imgs"

plantuml ${PUML_OPTS} -o "${DEST}" "${SRCS}/**/*.puml"
