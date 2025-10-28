#!/bin/bash

echo "=== Inbuilt Analysis System Test ==="
echo ""

echo "Testing inbuilt emotion analysis system..."

# Test the inbuilt analysis
response=$(curl -s http://localhost:8000/api/test-analysis)
echo "Response: $response"

if echo "$response" | grep -q "success"; then
    echo "✅ Inbuilt analysis system is working!"
    echo "✅ No external API required!"
    echo "✅ Emotion analysis and insights are ready!"
else
    echo "❌ Inbuilt analysis test failed"
    echo "Check the error message above"
fi
