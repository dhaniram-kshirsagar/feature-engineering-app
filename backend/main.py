import os
import sys
import uuid
import json
import logging
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to sys.path to import the ai_data_science_team package
sys.path.append('/Users/dhani/GitHub')

# Import the feature engineering agent
try:
    from ai_data_science_team.agents.feature_engineering_agent import FeatureEngineeringAgent
    from langchain_openai import ChatOpenAI
except ImportError:
    print("Error importing required modules. Make sure ai_data_science_team is installed.")
    FeatureEngineeringAgent = None
    ChatOpenAI = None

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/feature_engineering_api.log", delay=True),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("main")

# Create the FastAPI app
app = FastAPI(title="Feature Engineering API", 
              description="API for Human-in-the-Loop Feature Engineering with GenAI")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create data directory if it doesn't exist
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
tasks_file = data_dir / "tasks.pickle"

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Dictionary to store task data
tasks: Dict[str, Dict[str, Any]] = {}

# Load tasks from disk if file exists
def load_tasks_from_disk():
    global tasks
    if tasks_file.exists():
        try:
            with open(tasks_file, 'rb') as f:
                loaded_tasks = pickle.load(f)
                # Filter out any task data that can't be serialized
                for task_id, task_data in loaded_tasks.items():
                    # Remove agent from saved data
                    if 'agent' in task_data:
                        del task_data['agent']
                tasks = loaded_tasks
                logger.info(f"Loaded {len(tasks)} tasks from disk")
        except Exception as e:
            logger.error(f"Error loading tasks from disk: {str(e)}")
            tasks = {}
    else:
        tasks = {}

# Save tasks to disk
def save_tasks_to_disk():
    try:
        # Create a copy of tasks without agents (not serializable)
        serializable_tasks = {}
        for task_id, task_data in tasks.items():
            task_copy = task_data.copy()
            # Remove non-serializable data
            if 'agent' in task_copy:
                del task_copy['agent']
            if 'raw_data' in task_copy and isinstance(task_copy['raw_data'], pd.DataFrame):
                # Convert DataFrame to serializable format
                task_copy['raw_data'] = task_copy['raw_data'].to_dict()
            if 'data_engineered' in task_copy and isinstance(task_copy['data_engineered'], pd.DataFrame):
                # Convert DataFrame to serializable format
                task_copy['data_engineered'] = task_copy['data_engineered'].to_dict()
            serializable_tasks[task_id] = task_copy
        
        with open(tasks_file, 'wb') as f:
            pickle.dump(serializable_tasks, f)
            logger.info(f"Saved {len(serializable_tasks)} tasks to disk")
    except Exception as e:
        logger.error(f"Error saving tasks to disk: {str(e)}", exc_info=True)

# Load tasks at startup
load_tasks_from_disk()

class FeatureEngineeringRequest(BaseModel):
    instructions: str
    target_variable: Optional[str] = None
    model_name: str = "gpt-4o-mini"  # Default model

class FeatureEngineeringResponse(BaseModel):
    task_id: str
    status: str

class HumanFeedbackRequest(BaseModel):
    task_id: str
    feedback: str
    accept_recommendations: bool

class FeatureEngineeringFeedbackRequest(BaseModel):
    feedback: str
    accept: bool

@app.get("/")
def read_root():
    return {"message": "Feature Engineering API is running"}

@app.post("/api/feature-engineering/upload", response_model=FeatureEngineeringResponse)
async def upload_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    instructions: str = Form(...),
    target_variable: Optional[str] = Form(None),
    model_name: str = Form("gpt-4o-mini")
):
    """
    Upload a CSV file and start the feature engineering process.
    Returns a task ID that can be used to check the status.
    """
    try:
        # Generate a unique task ID
        task_id = str(uuid.uuid4())
        
        # Create data directory if it doesn't exist
        if not os.path.exists("data"):
            os.makedirs("data")
        
        # Save file to disk
        file_path = f"data/{file.filename}"
        contents = await file.read()
        
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Read the file into a pandas DataFrame
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read CSV file: {str(e)}")
        
        # Initialize task data
        tasks[task_id] = {
            "status": "processing",
            "file_path": file_path,  # Store the file path for potential reloading
            "instructions": instructions,
            "target_variable": target_variable,
            "model_name": model_name,
            "uploaded_at": datetime.now().isoformat(),
            "data_shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "results": None,
            "recommended_steps": None,
            "error": None,
            "human_feedback_required": False,
            "human_feedback": None
        }
        
        # Store a preview of the loaded data in the format expected by the frontend
        sample_records = df.head(10).to_dict(orient="records")
        tasks[task_id]["data_sample"] = {str(i): row for i, row in enumerate(sample_records)}
        tasks[task_id]["data_shape"] = [df.shape[0], df.shape[1]]
        tasks[task_id]["columns"] = df.columns.tolist()
        
        # Initialize empty results structure - will be populated when processing completes
        tasks[task_id]["results"] = {
            "data_sample": {str(i): row for i, row in enumerate(sample_records)},
            "data_shape": [df.shape[0], df.shape[1]],
            "columns": df.columns.tolist(),
            "function_code": "# Feature engineering not yet complete"
        }
        
        logger.info(f"Created initial data preview with {len(sample_records)} rows")
        
        # Save a pickled version of the data for reliable persistence
        if not os.path.exists("data"):
            os.makedirs("data")
        
        pickle_path = f"data/pickled_data_{task_id}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(df, f)
        
        tasks[task_id]["pickled_data_path"] = pickle_path
        logger.info(f"Saved pickled data for task {task_id} to {pickle_path}")
        
        # Start feature engineering in the background
        background_tasks.add_task(
            process_feature_engineering,
            task_id,
            df,
            instructions,
            target_variable,
            model_name
        )
        
        save_tasks_to_disk()
        
        return {"task_id": task_id, "status": "processing"}
    
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Class to handle DataFrame serialization for the checkpointer
class DataFrameSerializer:
    @staticmethod
    def to_serializable(df: pd.DataFrame) -> Dict[str, Any]:
        """Convert a pandas DataFrame to a serializable dictionary format.
        Returns a column-oriented dictionary that can be directly used with pd.DataFrame.from_dict()
        """
        if df is None:
            return None
        
        # Convert to a column-oriented dictionary (what pd.DataFrame.from_dict expects)
        # Each key is a column name, and each value is a list of values for that column
        return df.to_dict(orient="dict")
    
    @staticmethod
    def from_serializable(data: Dict[str, Any]) -> pd.DataFrame:
        """Convert a serializable dictionary back to a pandas DataFrame."""
        if data is None:
            return None
        
        # Create DataFrame from column-oriented dict
        return pd.DataFrame.from_dict(data)

@app.post("/api/feature-engineering/feedback", response_model=FeatureEngineeringResponse)
def provide_feedback(
    background_tasks: BackgroundTasks,
    feedback_request: HumanFeedbackRequest
):
    """
    Provide human feedback to the feature engineering process.
    """
    task_id = feedback_request.task_id
    
    if task_id not in tasks:
        logger.error(f"Feedback requested for non-existent task: {task_id}")
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    if task["status"] not in ["awaiting_feedback", "processing"]:
        # If task is already completed, just return its status
        if task["status"] == "completed":
            logger.info(f"Feedback provided for already completed task: {task_id}")
            return {"task_id": task_id, "status": "completed"}
        
        logger.error(f"Task {task_id} not awaiting feedback. Current status: {task['status']}")
        raise HTTPException(status_code=400, detail=f"Task not awaiting feedback. Current status: {task['status']}")
    
    # Update the task status to processing
    tasks[task_id]["status"] = "processing"
    logger.info(f"Received feedback for task {task_id}, accept={feedback_request.accept_recommendations}")
    
    # Store the feedback
    tasks[task_id]["human_feedback"] = {
        "accept": feedback_request.accept_recommendations,
        "feedback": feedback_request.feedback,
        "timestamp": datetime.now().isoformat()
    }
    
    # Process feedback in the background
    background_tasks.add_task(
        continue_feature_engineering_with_feedback,
        task_id, 
        feedback_request.feedback,
        feedback_request.accept_recommendations
    )
    
    save_tasks_to_disk()
    
    return {"task_id": task_id, "status": "processing"}

def continue_feature_engineering_with_feedback(task_id: str, feedback: str, accept_recommendations: bool):
    """
    Continue the feature engineering process with human feedback.
    """
    try:
        logger.info(f"Processing feedback for task {task_id}, accept={accept_recommendations}")
        
        # Check if task exists
        if task_id not in tasks:
            load_tasks_from_disk()
            
            if task_id not in tasks:
                raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        # Get task data
        task = tasks[task_id]
        
        # Check if agent is available
        if "agent" not in task:
            raise HTTPException(status_code=400, detail="Agent not available for this task")
        
        # Get data and instructions from the task
        data = task.get("data")
        instructions = task.get("instructions", "")
        target_variable = task.get("target_variable")
        
        # Handle case when data is None
        if data is None:
            logger.warning(f"Data is None for task {task_id}, attempting to retrieve from storage")
            # First check if we have pickled data
            pickled_data_path = task.get("pickled_data_path")
            
            if pickled_data_path and os.path.exists(pickled_data_path):
                try:
                    logger.info(f"Loading pickled data from {pickled_data_path}")
                    with open(pickled_data_path, 'rb') as f:
                        data = pickle.load(f)
                    logger.info(f"Successfully loaded pickled data for task {task_id}")
                except Exception as e:
                    logger.error(f"Error loading pickled data: {e}")
                    # Continue to next fallback option
            
            # Check if there's a file path stored in the task as fallback
            if data is None:
                original_df_path = task.get("file_path")
                
                if original_df_path and os.path.exists(original_df_path):
                    try:
                        # Load the original data from file
                        logger.info(f"Loading data from file: {original_df_path}")
                        if original_df_path.endswith('.csv'):
                            data = pd.read_csv(original_df_path)
                        elif original_df_path.endswith('.xlsx') or original_df_path.endswith('.xls'):
                            data = pd.read_excel(original_df_path)
                        else:
                            # Try pickle as fallback
                            logger.info(f"Trying to load pickled data for task {task_id}")
                            pickle_path = f"data/pickled_data_{task_id}.pkl"
                            if os.path.exists(pickle_path):
                                with open(pickle_path, 'rb') as f:
                                    data = pickle.load(f)
                            else:
                                raise ValueError(f"Unsupported file format: {original_df_path}")
                    except Exception as e:
                        logger.error(f"Error loading data from file: {e}")
                        raise HTTPException(status_code=500, detail=f"Failed to load data: {e}")
                else:
                    # As a last resort, check if there's a raw_df field
                    raw_df = task.get("raw_df")
                    if raw_df is not None:
                        data = raw_df
                        logger.info(f"Using raw_df from task data for task {task_id}")
                    else:
                        # If we still don't have data, raise an error
                        logger.error(f"No data available for task {task_id}")
                        raise HTTPException(status_code=500, detail="No data available to continue feature engineering")
        
        # Update task status
        task["status"] = "processing"
        save_tasks_to_disk()
        
        # Get agent
        agent = task["agent"]
        
        # Use feedback to continue feature engineering
        human_response = "yes" if accept_recommendations else "no"
        
        # Record human feedback in the task data
        task["feedback"] = feedback
        task["accept_recommendations"] = accept_recommendations
        
        # Create config with required keys for the checkpointer
        config = {
            "configurable": {
                "thread_id": task_id,
                "checkpoint_ns": "feature_engineering",
                "checkpoint_id": f"feature_engineering_{task_id}",
                "execution_globals": {
                    "pd": pd,
                    "np": __import__('numpy'),
                }
            }
        }
        
        # Convert original data to the proper format if needed
        # We need to pass this data again when continuing with feedback
        if isinstance(data, pd.DataFrame):
            # Use the DataFrame directly
            data_dict = data.to_dict('list')  # Use 'list' orientation for better compatibility
        elif isinstance(data, dict):
            # If it's already a dict, check if it has a proper DataFrame structure
            if all(isinstance(v, list) for v in data.values()) and len(set(len(v) for v in data.values())) == 1:
                # Looks like a column-oriented dictionary, good to use
                data_dict = data
            else:
                # Try to convert to DataFrame and back to ensure proper format
                try:
                    temp_df = pd.DataFrame(data)
                    data_dict = temp_df.to_dict('list')
                except Exception as e:
                    logger.error(f"Error converting data to DataFrame: {e}")
                    raise HTTPException(status_code=500, detail=f"Data format error: {e}")
        else:
            logger.error(f"Unsupported data type: {type(data)}")
            raise HTTPException(status_code=500, detail=f"Unsupported data type: {type(data)}")
        
        # Log data structure for debugging
        logger.info(f"Data dict keys: {list(data_dict.keys())}")
        logger.info(f"First few values of first column: {list(data_dict.values())[0][:5] if data_dict and list(data_dict.values()) else []}")
        logger.info(f"Data dictionary shape: {len(data_dict.keys())} columns, {len(next(iter(data_dict.values()))) if data_dict and next(iter(data_dict.values()), None) else 0} rows")
        
        # Print sample of the data for debugging
        sample_data = {k: v[:3] for k, v in data_dict.items() if v}
        logger.info(f"Sample data: {sample_data}")
        
        # Continue feature engineering with human feedback
        # Pass all required state variables along with the human response
        response = agent.invoke({
            "human_response": human_response,
            "user_instructions": instructions,
            "target_variable": target_variable,
            "data_raw": data_dict,  # Original data needs to be passed again
            "feedback": feedback    # Include the detailed feedback text
        }, config=config)
        
        # Store the response
        task["response"] = response
        
        # Store the updated recommendations if they exist
        if "recommended_steps" in response:
            task["recommended_steps"] = response["recommended_steps"]
            logger.info(f"Updated recommendations based on feedback for task {task_id}")
            
            # If the user didn't accept the recommendations, set the status back to awaiting_feedback
            if not accept_recommendations:
                task["status"] = "awaiting_feedback"
                logger.info(f"Task {task_id} set back to awaiting_feedback with updated recommendations")
                save_tasks_to_disk()
                return {"status": task["status"]}
        
        # Check if the feature engineering is complete
        if "data_engineered" in response and response["data_engineered"] is not None:
            # Successful feature engineering with engineered data available
            task["status"] = "completed"
            task["data_engineered"] = response["data_engineered"]
            task["feature_engineering_function"] = response.get("feature_engineer_function", "")
            
            # Store the results in a format compatible with the download endpoint
            task["results"] = {
                "function_code": response.get("feature_engineer_function", "") or task.get("feature_engineering_function", "# No function code available"),
                "data": response["data_engineered"]
            }
            
            logger.info(f"Feature engineering completed for task {task_id}")
        else:
            # Feature engineering is not complete yet
            task["status"] = "failed"
            error_message = response.get("feature_engineer_error", "Unknown error during feature engineering")
            task["error"] = error_message
            logger.error(f"Error during feature engineering for task {task_id}: {error_message}")
        
        # Save task status
        save_tasks_to_disk()
        
        return {"status": task["status"]}
        
    except Exception as e:
        logger.error(f"Error during agent operation: {str(e)}")
        if task_id in tasks:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = str(e)
            save_tasks_to_disk()
        raise HTTPException(status_code=500, detail=str(e))

def process_feature_engineering(
    task_id: str,
    df: pd.DataFrame,
    instructions: str,
    target_variable: Optional[str],
    model_name: str
):
    """
    Process the feature engineering task.
    This runs in the background and updates the task status.
    """
    try:
        # Initialize the language model
        try:
            llm = ChatOpenAI(model=model_name)
        except Exception as e:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = f"Failed to initialize language model: {str(e)}"
            save_tasks_to_disk()
            return
        
        # Store original DataFrame for later use
        tasks[task_id]["raw_data"] = df
        
        # Initialize the feature engineering agent with human-in-the-loop enabled
        agent = FeatureEngineeringAgent(
            model=llm,
            log=True,
            log_path="logs/",
            human_in_the_loop=True,  # Enable human-in-the-loop
            n_samples=min(30, len(df)),  # Limit samples for larger datasets
            bypass_explain_code=True
        )
        
        # Store the agent in the task data for future reference
        tasks[task_id]["agent"] = agent
        
        # Convert DataFrame to the expected dict format
        # The agent expects a dictionary where each key is a column name and values are lists
        # Using 'list' orientation for better compatibility with pandas
        serializable_df = df.to_dict('list')
        
        # Log data structure for debugging
        logger.info(f"Data dict keys: {list(serializable_df.keys())}")
        logger.info(f"First few values of first column: {list(serializable_df.values())[0][:5] if serializable_df and list(serializable_df.values()) else []}")
        
        logger.info(f"Starting feature engineering for task {task_id}")
        
        # Add additional configuration to ensure pandas is available in the execution environment
        # This ensures that the generated code can properly import pandas
        config = {
            "configurable": {
                "thread_id": task_id,
                "checkpoint_ns": "feature_engineering",
                "checkpoint_id": f"feature_engineering_{task_id}",
                "execution_globals": {
                    "pd": pd,
                    "np": __import__('numpy'),
                }
            }
        }
        
        # Start the feature engineering process
        response = agent.invoke({
            "user_instructions": instructions,
            "target_variable": target_variable,
            "data_raw": serializable_df,
            "max_retries": 3,
            "retry_count": 0
        }, config=config)
        
        # After receiving the feature_engineer_function, ensure it has pandas imported
        if "feature_engineer_function" in response and response["feature_engineer_function"]:
            # Extract the function code
            function_code = response["feature_engineer_function"]
            
            # Check if pandas is imported in the function
            if "import pandas" not in function_code and "import pd" not in function_code:
                # Add pandas import to the top of the function
                function_code = "import pandas as pd\n" + function_code
                response["feature_engineer_function"] = function_code
                
                # Save the updated function code back to file if one exists
                if "feature_engineer_function_path" in response and response["feature_engineer_function_path"]:
                    with open(response["feature_engineer_function_path"], 'w') as f:
                        f.write(function_code)
                        logger.info(f"Updated feature engineering function with pandas import")
        
        # Check if we need human feedback based on agent's state
        try:
            # Get recommended steps directly from the agent
            recommended_steps = agent.get_recommended_feature_engineering_steps()
            
            if recommended_steps:
                logger.info(f"Agent generated recommended steps for task {task_id}, waiting for human feedback")
                
                # Make sure recommendations are stored in the expected format for the frontend
                # The frontend expects a markdown-formatted string
                if isinstance(recommended_steps, dict):
                    # Convert dict to markdown string
                    markdown_recommendations = "## Feature Engineering Recommendations\n\n"
                    for key, value in recommended_steps.items():
                        markdown_recommendations += f"### {key}\n{value}\n\n"
                    tasks[task_id]["recommended_steps"] = markdown_recommendations
                elif isinstance(recommended_steps, list):
                    # Convert list to markdown string
                    markdown_recommendations = "## Feature Engineering Recommendations\n\n"
                    for item in recommended_steps:
                        if isinstance(item, dict):
                            for key, value in item.items():
                                markdown_recommendations += f"### {key}\n{value}\n\n"
                        else:
                            markdown_recommendations += f"- {item}\n"
                    tasks[task_id]["recommended_steps"] = markdown_recommendations
                else:
                    # Assume it's already a string
                    tasks[task_id]["recommended_steps"] = recommended_steps
                
                # Explicitly set the status to awaiting_feedback
                tasks[task_id]["status"] = "awaiting_feedback"
                logger.info(f"Set task {task_id} status to awaiting_feedback with recommendations")
            else:
                # No recommended steps, but maybe we have results already
                data_engineered = agent.get_data_engineered()
                if data_engineered is not None:
                    logger.info(f"Agent completed feature engineering for task {task_id} without human feedback")
                    tasks[task_id]["status"] = "completed"
                    
                    # Format the data for preview and storage
                    try:
                        # Store the full data for download
                        tasks[task_id]["data_engineered"] = data_engineered
                        
                        # Create sample data for preview
                        if hasattr(data_engineered, 'head') and hasattr(data_engineered, 'to_dict'):
                            # For DataFrame objects
                            logger.info(f"Creating preview from DataFrame with shape {data_engineered.shape}")
                            
                            # Extract the first 10 rows for the preview
                            sample_records = data_engineered.head(10).to_dict(orient="records")
                            data_sample = {str(i): row for i, row in enumerate(sample_records)}
                            
                            tasks[task_id]["results"] = {
                                "data_sample": data_sample,
                                "data_shape": data_engineered.shape,
                                "columns": data_engineered.columns.tolist(),
                                "function_code": agent.get_feature_engineer_function()
                            }
                            
                            logger.info(f"Created preview with {len(sample_records)} rows from DataFrame")
                        elif isinstance(data_engineered, dict):
                            # For dictionary format (column-oriented)
                            logger.info("Creating preview from dictionary data")
                            
                            # Convert to DataFrame temporarily for consistent handling
                            temp_df = pd.DataFrame(data_engineered)
                            
                            # Extract records for preview
                            sample_records = temp_df.head(10).to_dict(orient="records")
                            data_sample = {str(i): row for i, row in enumerate(sample_records)}
                            
                            tasks[task_id]["results"] = {
                                "data_sample": data_sample,
                                "data_shape": [len(temp_df), len(temp_df.columns)],
                                "columns": temp_df.columns.tolist(),
                                "function_code": agent.get_feature_engineer_function()
                            }
                            
                            logger.info(f"Created preview with {len(sample_records)} rows from dictionary")
                        else:
                            # For other data types, fallback to simpler format
                            logger.warning(f"Unknown data type for preview: {type(data_engineered)}")
                            tasks[task_id]["results"] = {
                                "data": data_engineered,
                                "function_code": agent.get_feature_engineer_function(),
                                "message": "Unable to create standard preview format due to unexpected data type"
                            }
                    except Exception as e:
                        logger.error(f"Error creating data preview: {str(e)}")
                        # Create a minimal result set if we encounter an error
                        tasks[task_id]["results"] = {
                            "function_code": agent.get_feature_engineer_function(),
                            "error_message": f"Error creating preview: {str(e)}"
                        }
                        
                        # If we have access to the raw data, use that for the preview
                        if "raw_data" in tasks[task_id] and isinstance(tasks[task_id]["raw_data"], pd.DataFrame):
                            try:
                                raw_df = tasks[task_id]["raw_data"]
                                sample_records = raw_df.head(10).to_dict(orient="records")
                                
                                tasks[task_id]["results"]["data_sample"] = {str(i): row for i, row in enumerate(sample_records)}
                                tasks[task_id]["results"]["data_shape"] = raw_df.shape
                                tasks[task_id]["results"]["columns"] = raw_df.columns.tolist()
                                tasks[task_id]["results"]["note"] = "Showing original data due to error in feature engineering preview"
                                
                                logger.info("Successfully created fallback preview from raw data")
                            except Exception as fallback_error:
                                logger.error(f"Error creating fallback preview: {str(fallback_error)}")
                else:
                    # No results yet, still in progress
                    logger.info(f"Feature engineering for task {task_id} in progress")
        
        except Exception as e:
            logger.error(f"Error getting recommended steps from agent: {str(e)}", exc_info=True)
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = f"Error getting recommended steps: {str(e)}"
        except Exception as e:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = f"Error during feature engineering: {str(e)}"
            logger.error(f"Error during feature engineering for task {task_id}: {str(e)}", exc_info=True)
    
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = f"Error during feature engineering: {str(e)}"
        logger.error(f"Error during feature engineering for task {task_id}: {str(e)}", exc_info=True)
        
    finally:
        # Clean up temporary file
        if os.path.exists(tasks[task_id]["file_path"]):
            try:
                os.remove(tasks[task_id]["file_path"])
            except:
                pass
        
        save_tasks_to_disk()

@app.get("/api/feature-engineering/status/{task_id}")
def get_task_status(task_id: str):
    """
    Get the current status of a feature engineering task.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    task = tasks[task_id]
    logger.info(f"Getting status for task {task_id} - Status: {task['status']}")
    
    response = {
        "task_id": task_id,
        "status": task["status"],
    }
    
    # Add data preview for the initial file upload
    if "data_sample" in task and "columns" in task and "data_shape" in task:
        response["data_sample"] = task["data_sample"]
        response["columns"] = task["columns"]
        response["data_shape"] = task["data_shape"]
        logger.info(f"Added initial data preview to response for task {task_id}")
    
    if task["status"] == "awaiting_feedback" and "recommended_steps" in task:
        response["recommended_steps"] = task["recommended_steps"]
    
    if "human_feedback_required" in task:
        response["human_feedback_required"] = task["human_feedback_required"]
    
    if "uploaded_at" in task:
        response["uploaded_at"] = task["uploaded_at"]
    
    if task["status"] == "completed" and "results" in task:
        response["results"] = task["results"]
        
        # Make sure we have the necessary fields for the data preview
        if "data_sample" not in response["results"]:
            # Try to generate the preview data if it's not in the results
            try:
                if "data_engineered" in task and task["data_engineered"] is not None:
                    logger.info(f"Generating data preview for task {task_id}")
                    
                    # If data_engineered is a dict (list-oriented)
                    if isinstance(task["data_engineered"], dict):
                        import pandas as pd
                        try:
                            # Create a temporary DataFrame from the dictionary
                            temp_df = pd.DataFrame(task["data_engineered"])
                            
                            # Get sample data (first 10 rows as records)
                            sample_records = temp_df.head(10).to_dict(orient="records")
                            
                            # Add the required fields for the frontend preview
                            response["results"]["data_sample"] = {str(i): row for i, row in enumerate(sample_records)}
                            response["results"]["data_shape"] = [len(temp_df), len(temp_df.columns)]
                            response["results"]["columns"] = temp_df.columns.tolist()
                            
                            logger.info(f"Created data preview with shape {response['results']['data_shape']} and {len(sample_records)} rows")
                        except Exception as e:
                            logger.error(f"Error creating DataFrame from dictionary: {str(e)}")
                            # Fall back to raw data in case of error
                            if "data_raw" in task and task["data_raw"] is not None:
                                logger.info("Falling back to raw data for preview")
                                raw_df = pd.DataFrame(task["data_raw"])
                                sample_records = raw_df.head(10).to_dict(orient="records")
                                response["results"]["data_sample"] = {str(i): row for i, row in enumerate(sample_records)}
                                response["results"]["data_shape"] = [len(raw_df), len(raw_df.columns)]
                                response["results"]["columns"] = raw_df.columns.tolist()
                    
                    # If data_engineered is a DataFrame
                    elif hasattr(task["data_engineered"], "to_dict"):
                        try:
                            # Get sample data (first 10 rows as records)
                            sample_records = task["data_engineered"].head(10).to_dict(orient="records")
                            
                            # Add the required fields for the frontend preview
                            response["results"]["data_sample"] = {str(i): row for i, row in enumerate(sample_records)}
                            response["results"]["data_shape"] = [len(task["data_engineered"]), len(task["data_engineered"].columns)]
                            response["results"]["columns"] = task["data_engineered"].columns.tolist()
                            
                            logger.info(f"Created data preview with shape {response['results']['data_shape']} and {len(sample_records)} rows")
                        except Exception as e:
                            logger.error(f"Error getting preview from DataFrame: {str(e)}")
                            # Fall back to raw data in case of error
                            if "data_raw" in task and task["data_raw"] is not None:
                                logger.info("Falling back to raw data for preview")
                                raw_df = pd.DataFrame(task["data_raw"])
                                sample_records = raw_df.head(10).to_dict(orient="records")
                                response["results"]["data_sample"] = {str(i): row for i, row in enumerate(sample_records)}
                                response["results"]["data_shape"] = [len(raw_df), len(raw_df.columns)]
                                response["results"]["columns"] = raw_df.columns.tolist()
                            
                # Also check if the data might be in the results["data"] field
                elif "data" in response["results"]:
                    logger.info(f"Using data from results for preview (task {task_id})")
                    try:
                        # Data might be in records format already
                        if isinstance(response["results"]["data"], list):
                            sample_records = response["results"]["data"][:10]
                            # Convert to the format expected by the frontend
                            response["results"]["data_sample"] = {str(i): row for i, row in enumerate(sample_records)}
                            
                            # Get columns from the first record if available
                            if sample_records and len(sample_records) > 0:
                                response["results"]["columns"] = list(sample_records[0].keys())
                            else:
                                response["results"]["columns"] = []
                                
                            response["results"]["data_shape"] = [len(response["results"]["data"]), len(response["results"]["columns"])]
                            logger.info(f"Created data preview from results data with shape {response['results']['data_shape']}")
                    except Exception as e:
                        logger.error(f"Error creating preview from results data: {str(e)}")
            except Exception as e:
                logger.error(f"Error generating data preview: {str(e)}")
                # If we can't generate the preview, set some placeholder values
                response["results"]["data_sample"] = {}
                response["results"]["data_shape"] = [0, 0]
                response["results"]["columns"] = []
    
    if task["status"] == "failed" and "error" in task:
        response["error"] = task["error"]
    
    save_tasks_to_disk()
    
    return response

@app.get("/api/feature-engineering/download/{task_id}")
def download_processed_data(task_id: str):
    """
    Download the feature engineered data as JSON.
    """
    if task_id not in tasks:
        logger.error(f"Task {task_id} not found")
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    logger.info(f"Preparing download for task: {task_id}, status: {task['status']}")
    
    if task["status"] != "completed":
        logger.error(f"Task {task_id} not completed, status: {task['status']}")
        raise HTTPException(status_code=400, detail="Feature engineering not completed")
    
    if not task.get("results"):
        logger.error(f"Task {task_id} has no results")
        raise HTTPException(status_code=400, detail="No results available")
    
    try:
        # First try to use cached engineered data (preferred approach)
        if "data_engineered" in task and task["data_engineered"] is not None:
            logger.info(f"Using cached engineered data for download (task {task_id})")
            data_engineered = task["data_engineered"]
            
            # Debug information about the data_engineered
            logger.info(f"data_engineered type: {type(data_engineered)}")
            if isinstance(data_engineered, dict):
                logger.info(f"data_engineered keys: {list(data_engineered.keys())}")
                # If data_engineered is already a dict, convert it to the records format expected
                # Convert from list-oriented dict to records-oriented format
                try:
                    # Create a temporary DataFrame from the list-oriented dictionary
                    import pandas as pd
                    temp_df = pd.DataFrame(data_engineered)
                    
                    # Ensure we include ALL rows in the download, not just the preview
                    data_dict = temp_df.to_dict(orient="records")
                    logger.info(f"Converted dict to DataFrame and then to records, records count: {len(data_dict)}")
                except Exception as e:
                    logger.error(f"Error converting dictionary to DataFrame: {str(e)}")
                    # Fallback: try to use the dictionary directly
                    data_dict = data_engineered
            else:
                # Original behavior for DataFrame
                data_dict = data_engineered.to_dict(orient="records")
                logger.info(f"Converted DataFrame to records, records count: {len(data_dict)}")
            
            # Get the function code from multiple possible sources
            function_code = (
                # First check results
                task["results"].get("function_code") or 
                # Then check if it's directly in the task
                task.get("feature_engineering_function") or 
                # Finally check for response
                (task.get("response", {}) or {}).get("feature_engineer_function", "# No function code available")
            )
            
            # Prepare the response with both the data and the function code
            response_data = {
                "data": data_dict,
                "function_code": function_code
            }
            
            logger.info(f"Successfully prepared download data for task {task_id}")
            return JSONResponse(content=response_data)
        else:
            # Fallback to getting data from agent
            logger.info(f"Getting engineered data from agent for download (task {task_id})")
            agent = task.get("agent")
            if not agent:
                logger.error(f"Agent not available for task {task_id}")
                raise HTTPException(status_code=400, detail="Agent not available")
            
            data_engineered = agent.get_data_engineered()
            if data_engineered is None:
                logger.error(f"No engineered data available from agent for task {task_id}")
                raise HTTPException(status_code=400, detail="No engineered data available")
            
            logger.info(f"Agent data_engineered type: {type(data_engineered)}")
            data_dict = data_engineered.to_dict(orient="records")
            function_code = agent.get_feature_engineer_function()
        
        # Prepare the response with both the data and the function code
        response_data = {
            "data": data_dict,
            "function_code": function_code
        }
        
        logger.info(f"Successfully prepared download data for task {task_id}")
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"Error preparing download data for task {task_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error preparing download data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configuration to exclude logs directory from reload
    reload_dirs = [str(Path(__file__).parent.resolve())]
    reload_excludes = [str(logs_dir.resolve())]
    
    logger.info(f"Starting server with reload_dirs={reload_dirs} and reload_excludes={reload_excludes}")
    
    # Run the server with specific reload settings
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8001, 
        reload=True,
        reload_dirs=reload_dirs,
        reload_excludes=reload_excludes
    )
