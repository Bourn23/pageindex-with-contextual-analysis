#!/bin/bash
# Monitor the test_end_to_end.py progress

echo "Monitoring test_end_to_end.py progress..."
echo "=========================================="
echo ""

# Check if test is running
if pgrep -f "python test_end_to_end.py" > /dev/null; then
    echo "✓ Test is running (PID: $(pgrep -f 'python test_end_to_end.py'))"
    echo ""
    
    # Show most recent log files
    echo "Recent log files:"
    ls -lt logs/*.json 2>/dev/null | head -3 | awk '{print "  " $9 " - " $6 " " $7 " " $8}'
    echo ""
    
    # Show last few log entries
    echo "Latest log entries:"
    latest_log=$(ls -t logs/*.json 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        echo "  From: $latest_log"
        tail -5 "$latest_log" | jq -r '.message // .mode // empty' 2>/dev/null | sed 's/^/    /'
    fi
    echo ""
    
    # Show process info
    echo "Process info:"
    ps -p $(pgrep -f 'python test_end_to_end.py') -o pid,time,rss,command | tail -1 | awk '{print "  PID: " $1 ", Time: " $2 ", Memory: " $3/1024 " MB"}'
    
else
    echo "✗ Test is not running"
    echo ""
    echo "Check for output or errors in the terminal where you ran the test"
fi

echo ""
echo "To see live updates, run:"
echo "  watch -n 2 ./monitor_test.sh"
