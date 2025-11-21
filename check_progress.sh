#!/bin/bash

# Quick script to check if PageIndex is making progress

echo "Checking PageIndex progress..."
echo ""

# Check if process is running
if pgrep -f "run_pageindex.py" > /dev/null; then
    echo "✓ PageIndex is running"
    
    # Show recent log activity
    if [ -d "logs" ]; then
        echo ""
        echo "Recent log activity (last 10 lines):"
        echo "======================================"
        tail -10 logs/*.json 2>/dev/null || echo "No logs found"
    fi
    
    # Check CPU usage
    echo ""
    echo "CPU usage:"
    echo "======================================"
    ps aux | grep "run_pageindex.py" | grep -v grep
    
else
    echo "✗ PageIndex is not running"
fi

echo ""
echo "Tip: If it's been running for >2 minutes on a 4-page PDF, something might be wrong"
echo "     Normal time: ~30-60 seconds for medium, ~60-120 seconds for fine"
