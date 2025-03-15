#!/bin/bash

echo "Feature Engineering App Stopper"
echo "=============================="

# Function to stop a process by PID file
stop_process() {
    local pid_file=$1
    local name=$2
    
    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        
        echo "Stopping $name with PID: $PID"
        
        # Check if the process is still running
        if ps -p $PID > /dev/null; then
            # Kill the process
            kill $PID
            sleep 1
            
            # Check if process is still alive and force kill if necessary
            if ps -p $PID > /dev/null; then
                echo "$name is still running, force killing..."
                kill -9 $PID
            fi
            
            echo "$name stopped successfully."
        else
            echo "$name is not running."
        fi
        
        # Remove the PID file
        rm "$pid_file"
    else
        echo "$name is not running or PID file not found."
    fi
}

# Stop the backend
stop_backend() {
    echo "Stopping backend server..."
    stop_process "$(dirname "$0")/backend/backend.pid" "Backend server"
}

# Stop the frontend
stop_frontend() {
    echo "Stopping frontend server..."
    stop_process "$(dirname "$0")/frontend.pid" "Frontend server"
}

# Stop both servers
stop_backend
#stop_frontend

echo ""
echo "Feature Engineering App has been stopped."
