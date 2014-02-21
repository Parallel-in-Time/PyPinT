#!/bin/bash

new_version=$1
if [[ "$new_version" == "" ]]; then
  new_version=$(git describe)
fi
new_version=${new_version//./\\.}

file="pypint/__init__.py"

old_version=$(grep '__version__' "${file}" | grep -o "'.*'")

SED_VERSION="s/${old_version}/'${new_version}'/"

old="${file}"
new="${file}.version"
sed "${SED_VERSION}" <"$old" >"$new"
mv $new $file
