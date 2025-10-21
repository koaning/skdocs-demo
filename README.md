This repository was created with a `justfile` in [this pr](https://github.com/scikit-learn/scikit-learn/pull/32427). 


```
[working-directory: 'doc']
build-docs:
    make html

[working-directory: '../skdocs-demo']
cleanup: 
    find . -mindepth 1 -not -name 'README.md' -not -path './.git*' -delete


docs: cleanup build-docs
    cp -r doc/_build/html/stable/**/* ../skdocs-demo

serve:
    uv run python -m http.server --directory ../skdocs-demo 12345```
