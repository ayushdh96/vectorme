#!/bin/bash
# Test script for vectorme

set -e

echo "=== vectorme test suite ==="
echo

# Check if test files exist
if [ ! -f "conversation.m4a" ]; then
    echo "Warning: conversation.m4a not found, some tests will be skipped"
fi

echo "1. List speakers in database"
vectorme --list
echo

echo "2. Test single file embedding (doug.m4a)"
if [ -f "doug.m4a" ]; then
    vectorme --file doug.m4a 2>/dev/null | head -c 200
    echo "..."
else
    echo "Skipped: doug.m4a not found"
fi
echo

echo "3. Test diarization (CLI mode)"
if [ -f "conversation.m4a" ]; then
    echo "First 10 segments:"
    vectorme --file conversation.m4a --diarize 2>/dev/null | grep segment | head -10
else
    echo "Skipped: conversation.m4a not found"
fi
echo

echo "4. Test HTTP server (if running)"
if curl -s http://127.0.0.1:3120/health > /dev/null 2>&1; then
    echo "Server health check:"
    curl -s http://127.0.0.1:3120/health | jq
    echo
    
    echo "List speakers via API:"
    curl -s http://127.0.0.1:3120/v1/speakers | jq
    echo
    
    if [ -f "conversation.m4a" ]; then
        echo "Diarization via API (first 5 lines):"
        curl -N -s -X POST http://127.0.0.1:3120/v1/audio/transcriptions \
            -F "file=@conversation.m4a" \
            -F "response_format=diarized_json" \
            -F "stream=true" | head -5
    fi
else
    echo "Server not running. Start with: vectorme --serve"
fi
echo

echo "=== Tests complete ==="
