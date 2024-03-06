
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
# Source: https://github.com/pytorch/torchx/blob/main/scripts/spellcheck.sh
set -ex
sudo apt-get install aspell

if [[ -z "$@" ]]; then
    sources=$(find -name '*.md')
else
    sources=$@
fi

sources_arg=""
for src in $sources; do
        sources_arg="${sources_arg} -S $src"
done

if [ ! "$sources_arg" ]; then
	echo "No files to spellcheck"
else
	pyspelling -c scripts/spellcheck_conf/spellcheck.yaml --name Markdown $sources_arg
fi
