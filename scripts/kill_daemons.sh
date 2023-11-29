#!/bin/bash

# Find PIDs of all daemons running
pids=$(pgrep -f qresourcemanager_d)

# Check if any matching processes were found
if [ -n "$pids" ]; then
    # Loop through each PID and send a SIGTERM signal (-15) to kill the daemon
    for pid in $pids; do
        echo "Killing QRM daemon with PID: $pid"
        kill -15 "$pid"
    done
else
    echo "The QRM daemon is not running."
fi
