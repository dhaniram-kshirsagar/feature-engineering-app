#!/bin/bash

echo "Feature Engineering App Starter"
echo "=============================="

# Function to handle errors
error_exit() {
    echo "ERROR: $1"
    exit 1
}

# Start the backend
start_backend() {
    echo "Starting backend server..."
    
    # Navigate to the backend directory
    cd "$(dirname "$0")/backend" || error_exit "Could not navigate to backend directory"
    
    # Check for virtual environment
    # if [ ! -d "venv" ]; then
    #     echo "Creating virtual environment..."
    #     python -m venv venv || error_exit "Failed to create virtual environment"
    # fi
    
    # # Activate virtual environment
    # source venv/bin/activate || error_exit "Failed to activate virtual environment"
    
    # # Install requirements if needed
    # if [ -f "requirements.txt" ]; then
    #     echo "Installing backend dependencies..."
    #     pip install -r requirements.txt || error_exit "Failed to install backend dependencies"
    # fi
    
    # # Make sure ai-data-science-team is installed
    # echo "Installing ai-data-science-team package..."
    # pip install -e /Users/dhani/GitHub/ai-data-science-team || error_exit "Failed to install ai-data-science-team"
    
    # Start the backend server in the background
    uvicorn main:app --host 0.0.0.0 --port 8001 &
    BACKEND_PID=$!
    
    # Save the PID to a file
    echo $BACKEND_PID > backend.pid
    echo "Backend server started with PID: $BACKEND_PID"
    echo "Backend logs being saved to backend.log"
    
    # Return to the original directory
    cd - > /dev/null
}

# Start the frontend
start_frontend() {
    echo "Starting frontend server..."
    
    # Navigate to the frontend directory
    cd "$(dirname "$0")/frontend" || error_exit "Could not navigate to frontend directory"
    
    Install dependencies if needed
    if [ ! -d "node_modules" ] || [ ! -f "node_modules/.bin/react-scripts" ]; then
        echo "Installing frontend dependencies..."
        npm install || error_exit "Failed to install frontend dependencies"
    fi
    
    Start the React development server in the background
    npm start > ../frontend.log 2>&1 &
    FRONTEND_PID=$!
    
    # Save the PID to a file
    echo $FRONTEND_PID > ../frontend.pid
    echo "Frontend server started with PID: $FRONTEND_PID"
    echo "Frontend logs being saved to frontend.log"
    npm start
    # Return to the original directory
    cd - > /dev/null
}

# Start both servers
#start_backend
start_frontend

echo ""
echo "Feature Engineering App is now running!"
#echo "- Backend: http://localhost:8000"
echo "- Frontend: http://localhost:3000"
echo ""
echo "To stop the application, run ./stop_app.sh"
