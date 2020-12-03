#!/bin/bash

. /Users/javin/.local/share/virtualenvs/singletrunner-JCI44ZZV/bin/activate
PIPENV_VERBOSITY=-1 pipenv run python /Users/javin/work/steven/graph/main.py
deactivate
