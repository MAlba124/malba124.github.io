#!/usr/bin/env sh

set -x

pandoc --toc beating_the_auto_vectorizer.md --standalone --highlight-style tango -o beating_the_auto_vectorizer.html
