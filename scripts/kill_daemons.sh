#!/bin/bash

PID=$(pidof daemon_d)

if [ -n "$PID" ]; then
    echo "Killing all daemons."
    kill -15 "$PID"
else
    echo "daemon_d is not running."
fi
