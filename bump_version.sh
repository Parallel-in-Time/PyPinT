#!/bin/bash

old_version=$1
old_version=${old_version//./\\.}
old_release=$3
new_version=$2
new_version=${new_version//./\\.}
new_release=$4

files="setup.py doc/source/conf.py"

SED_VERSION="s/${old_version}/${new_version}/"
SED_RELEASE="s/${old_release}/${new_release}/"

for file in $files
do
  old="${file}"
  new="${file}.version"
  sed "${SED_VERSION}" <"$old" >"$new"
  if [[ "${new_release}" != "" ]]; then
    old="${new}"
    new="${file}.release"
    sed "${SED_RELEASE}" <"$old" >"$new"
  fi
  mv $new $file
done

