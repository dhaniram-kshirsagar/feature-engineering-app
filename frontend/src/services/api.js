import axios from 'axios';

const API_URL = 'http://localhost:8001/api';

// Create axios instance
const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Upload a dataset for feature engineering
 * @param {File} file The CSV file to upload
 * @param {string} instructions Feature engineering instructions
 * @param {string} targetVariable Optional target variable
 * @returns {Promise} Response with task ID
 */
export const uploadDataset = async (file, instructions, targetVariable = '') => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('instructions', instructions);
  
  if (targetVariable) {
    formData.append('target_variable', targetVariable);
  }
  
  try {
    const response = await axios.post(
      `${API_URL}/feature-engineering/upload`, 
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error uploading dataset:', error);
    throw new Error(error.response?.data?.detail || 'Failed to upload dataset');
  }
};

/**
 * Fetch the status of a feature engineering task
 * @param {string} taskId The ID of the task
 * @returns {Promise} Task status
 */
export const fetchTaskStatus = async (taskId) => {
  try {
    const response = await apiClient.get(`/feature-engineering/status/${taskId}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching task status:', error);
    const errorMessage = error.response?.data?.detail || 'Failed to fetch task status';
    
    // Add more detailed logging for debugging
    if (error.response?.status === 404) {
      console.error(`Task ${taskId} not found. This might be due to a server restart.`);
    }
    
    throw new Error(errorMessage);
  }
};

/**
 * Provide human feedback to the feature engineering process
 * @param {string} taskId The ID of the task
 * @param {string} feedback Human feedback text
 * @param {boolean} accept Whether to accept recommendations
 * @returns {Promise} Response with updated task status
 */
export const provideFeedback = async (taskId, feedback, accept) => {
  try {
    const response = await apiClient.post('/feature-engineering/feedback', {
      task_id: taskId,
      feedback,
      accept_recommendations: accept, // Use new parameter name for better compatibility
    });
    return response.data;
  } catch (error) {
    console.error('Error providing feedback:', error);
    
    // Add more detailed error information
    let errorMessage = 'Failed to provide feedback';
    if (error.response?.data?.detail) {
      errorMessage = error.response.data.detail;
    } else if (error.response?.status === 404) {
      errorMessage = 'Task not found. The server may have restarted and lost task information.';
    }
    
    throw new Error(errorMessage);
  }
};

/**
 * Download the feature engineered results
 * @param {string} taskId The ID of the task
 * @returns {Promise} Response with data and function code
 */
export const downloadResults = async (taskId) => {
  try {
    const response = await apiClient.get(`/feature-engineering/download/${taskId}`);
    
    // Convert to JSON and trigger download
    const blob = new Blob([JSON.stringify(response.data, null, 2)], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `feature_engineered_data_${taskId}.json`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
    
    return response.data;
  } catch (error) {
    console.error('Error downloading results:', error);
    
    let errorMessage = 'Failed to download results';
    if (error.response?.data?.detail) {
      errorMessage = error.response.data.detail;
    } else if (error.response?.status === 404) {
      errorMessage = 'Task not found. The server may have restarted and lost task information.';
    }
    
    throw new Error(errorMessage);
  }
};
