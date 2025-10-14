#!/usr/bin/bash
# Monitor memory usage of FL processes

echo "Federated Learning Memory Monitor"
echo "=================================="
echo

# Find server process
SERVER_PID=$(ps aux | grep "python.*server/main.py" | grep -v grep | awk '{print $2}' | head -1)

if [ -z "$SERVER_PID" ]; then
    echo "Server not running"
    exit 1
fi

echo "Monitoring Server PID: $SERVER_PID"
echo
echo "Time     | RSS (MB) | VMS (MB) | %MEM | Threads | Delta"
echo "---------|----------|----------|------|---------|-------"

PREV_RSS=0

while true; do
    # Get memory info
    MEM_INFO=$(ps -p $SERVER_PID -o rss=,vsz=,%mem=,nlwp= 2>/dev/null)

    if [ -z "$MEM_INFO" ]; then
        echo "Process $SERVER_PID no longer exists"
        exit 0
    fi

    RSS_KB=$(echo $MEM_INFO | awk '{print $1}')
    VSZ_KB=$(echo $MEM_INFO | awk '{print $2}')
    MEM_PCT=$(echo $MEM_INFO | awk '{print $3}')
    THREADS=$(echo $MEM_INFO | awk '{print $4}')

    RSS_MB=$((RSS_KB / 1024))
    VSZ_MB=$((VSZ_KB / 1024))

    # Calculate delta
    if [ $PREV_RSS -eq 0 ]; then
        DELTA="-"
    else
        DELTA_MB=$((RSS_MB - PREV_RSS))
        if [ $DELTA_MB -gt 0 ]; then
            DELTA="+${DELTA_MB}"
        else
            DELTA="$DELTA_MB"
        fi
    fi

    PREV_RSS=$RSS_MB

    # Current time
    TIME=$(date +%H:%M:%S)

    printf "%s | %8d | %8d | %4.1f | %7d | %s\n" \
        "$TIME" "$RSS_MB" "$VSZ_MB" "$MEM_PCT" "$THREADS" "$DELTA"

    sleep 5
done
