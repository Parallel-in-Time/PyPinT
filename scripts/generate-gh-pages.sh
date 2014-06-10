#!/bin/sh

git update-index --refresh --unmerged -q >/dev/null || :
if git diff-index --quiet HEAD; then
  echo "Generating gh-pages"
else
  echo "Repository is dirty. Not doing anything to avoid data loss."
fi

GH_PAGES_SOURCES="doc/source doc/Makefile pypint examples README.md CHANGELOG.rst MANIFEST.in setup.cfg setup.py"

git checkout gh-pages
rm -rf ./*
git checkout master $GH_PAGES_SOURCES
git reset HEAD

python setup.py build

cd doc
make html
cd ..

mv -fv doc/build/html/* ./
rm -rf $GH_PAGES_SOURCES doc PyPinT.egg-info build
touch .nojekyll

git add -A
git commit -m "Generated gh-pages for `git log master -1 --pretty=short --abbrev-commit`" \
  && git push origin gh-pages
git checkout master

