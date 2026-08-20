#!/bin/bash
# Double-click this file to launch the FPL Squad Advisor locally and
# open it in your browser. Leave this window open while using the
# site; close it (or press Ctrl+C) to stop the site.

cd "$(dirname "$0")"

echo "=== FPL Squad Advisor ==="
echo ""
echo "Setting up (only takes a while the first time)..."

pip3 install -r requirements.txt --quiet 2>/tmp/fpl_pip_err.log
if [ $? -ne 0 ]; then
  echo "Retrying install (common on newer Macs)..."
  pip3 install -r requirements.txt --quiet --break-system-packages 2>>/tmp/fpl_pip_err.log
fi

if [ $? -ne 0 ]; then
  echo ""
  echo "Could not install the required packages. Details:"
  echo "----------------------------------------------------"
  cat /tmp/fpl_pip_err.log
  echo "----------------------------------------------------"
  echo ""
  read -p "Press Enter to close this window..."
  exit 1
fi

echo "Starting the site (this can take a few seconds while it loads data)..."
python3 app.py &
SERVER_PID=$!

# poll instead of guessing a fixed wait -- the site loads player data
# on startup, which can take longer than a couple seconds
READY=0
for i in $(seq 1 30); do
  if curl -s -o /dev/null http://localhost:8888; then
    READY=1
    break
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    break
  fi
  sleep 1
done

if [ $READY -ne 1 ]; then
  echo ""
  echo "The site didn't start in time. Scroll up to see if there's an error above."
  echo ""
  read -p "Press Enter to close this window..."
  exit 1
fi

open "http://localhost:8888"

echo ""
echo "The site is running at http://localhost:8888"
echo "Leave this window open while you're using it."
echo "Close this window (or press Ctrl+C) to stop the site."
echo ""

wait $SERVER_PID
echo ""
read -p "Site stopped. Press Enter to close this window..."
