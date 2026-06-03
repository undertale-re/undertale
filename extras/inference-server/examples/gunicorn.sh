#!/bin/sh

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
gunicorn --workers=4 --bind 127.0.0.1:5000 inference.api:app
