import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  TextField,
  Grid,
  Paper,
  Stepper,
  Step,
  StepLabel,
  CircularProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Card,
  CardContent,
  Divider
} from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import ReactMarkdown from 'react-markdown';
import { fetchTaskStatus, uploadDataset, provideFeedback, downloadResults } from '../services/api';

const steps = ['Upload Dataset', 'Review Recommendations', 'View Results'];

const Home = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [file, setFile] = useState(null);
  const [instructions, setInstructions] = useState('');
  const [targetVariable, setTargetVariable] = useState('');
  const [taskId, setTaskId] = useState(null);
  const [taskStatus, setTaskStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dataPreview, setDataPreview] = useState(null);
  const [feedback, setFeedback] = useState('');
  const [pollingInterval, setPollingInterval] = useState(null);
  const [recommendations, setRecommendations] = useState('');
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [recommendationsUpdated, setRecommendationsUpdated] = useState(false);

  // Function to handle file selection
  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);
    setError(null);
  };

  // Function to handle dataset upload
  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file to upload');
      return;
    }

    if (!instructions.trim()) {
      setError('Please provide feature engineering instructions');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await uploadDataset(file, instructions, targetVariable);
      setTaskId(response.task_id);
      
      // Start polling for task status
      const interval = setInterval(() => {
        pollTaskStatus(response.task_id);
      }, 5000);
      
      setPollingInterval(interval);
      setActiveStep(1);
    } catch (err) {
      setError(`Error uploading dataset: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Function to poll for task status
  const pollTaskStatus = async (id) => {
    try {
      const status = await fetchTaskStatus(id);
      console.log('Received task status:', status); // Add debugging
      setTaskStatus(status);

      if (status.status === 'awaiting_feedback') {
        console.log('Recommendations received:', status.recommended_steps);
        
        // Check if recommendations have changed after feedback
        if (feedbackSubmitted && recommendations !== status.recommended_steps) {
          setRecommendationsUpdated(true);
        }
        
        // Always reset these states when we're in awaiting_feedback state
        setFeedbackSubmitted(false);
        setLoading(false);
        // Don't clear feedback text field automatically
        
        setRecommendations(status.recommended_steps || '');
        clearInterval(pollingInterval);
      } else if (status.status === 'completed') {
        setActiveStep(2);
        setLoading(false);
        clearInterval(pollingInterval);
      } else if (status.status === 'failed') {
        setError(`Feature engineering failed: ${status.error}`);
        setLoading(false);
        clearInterval(pollingInterval);
      }
    } catch (err) {
      console.error('Error in pollTaskStatus:', err);
      setLoading(false);
      
      // Check if we got a 404 Not Found error, which might happen after server restart
      if (err.message.includes('404') || err.message.includes('not found')) {
        console.log('Task not found, possibly due to server restart');
        setError(
          'Task information was lost, possibly due to a server restart. ' +
          'Please try again or check server logs for more information.'
        );
        
        // Stop polling if we can't find the task
        clearInterval(pollingInterval);
      } else {
        setError(`Error fetching task status: ${err.message}`);
      }
    }
  };

  // Function to handle feedback submission
  const handleSubmitFeedback = async (accept) => {
    if (!accept && !feedback.trim()) {
      setError('Please provide feedback or click Accept if you approve the recommendations');
      return;
    }

    setLoading(true);
    setError(null);
    setFeedbackSubmitted(true);
    setRecommendationsUpdated(false);

    try {
      await provideFeedback(taskId, feedback, accept);
      
      // Resume polling
      const interval = setInterval(() => {
        pollTaskStatus(taskId);
      }, 5000);
      
      setPollingInterval(interval);
    } catch (err) {
      console.error('Error in handleSubmitFeedback:', err);
      setLoading(false);
      setFeedbackSubmitted(false); // Reset this state on error
      
      // Special handling for 404 errors which might happen after server restart
      if (err.message.includes('404') || err.message.includes('not found')) {
        setError(
          'Task information was lost, possibly due to a server restart. ' +
          'Please try again or check server logs for more information.'
        );
      } else {
        setError(`Error submitting feedback: ${err.message}`);
      }
    }
  };

  // Function to handle results download
  const handleDownload = async () => {
    try {
      await downloadResults(taskId);
    } catch (err) {
      setError(`Error downloading results: ${err.message}`);
    }
  };

  // Clean up interval on component unmount
  useEffect(() => {
    return () => {
      if (pollingInterval) {
        clearInterval(pollingInterval);
      }
    };
  }, [pollingInterval]);

  return (
    <Box className="centered-container">
      <Box className="hero-section">
        <Typography variant="h4" gutterBottom>
          Human-in-the-Loop Feature Engineering with GenAI
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Upload your dataset, provide instructions, and let AI help you with feature engineering recommendations that you can review and modify.
        </Typography>
      </Box>

      <Stepper activeStep={activeStep} sx={{ mb: 4, width: '100%', maxWidth: 800 }}>
        {steps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      {error && (
        <Alert severity="error" sx={{ mb: 3, width: '100%', maxWidth: 800 }}>
          {error}
        </Alert>
      )}

      {activeStep === 0 && (
        <Box className="upload-section">
          <Typography variant="h6" gutterBottom>
            Upload Your Dataset
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Button
                variant="outlined"
                component="label"
                startIcon={<UploadFileIcon />}
                fullWidth
              >
                Select CSV File
                <input
                  type="file"
                  accept=".csv"
                  hidden
                  onChange={handleFileChange}
                />
              </Button>
              {file && (
                <Typography variant="body2" sx={{ mt: 1 }}>
                  Selected file: {file.name}
                </Typography>
              )}
            </Grid>
            
            <Grid item xs={12}>
              <TextField
                label="Feature Engineering Instructions"
                multiline
                rows={4}
                value={instructions}
                onChange={(e) => setInstructions(e.target.value)}
                placeholder="Enter instructions for feature engineering (e.g., 'Encode categorical variables and normalize numeric features')"
                fullWidth
                required
              />
            </Grid>
            
            <Grid item xs={12}>
              <TextField
                label="Target Variable (optional)"
                value={targetVariable}
                onChange={(e) => setTargetVariable(e.target.value)}
                placeholder="Enter the name of the target variable if applicable"
                fullWidth
              />
            </Grid>
            
            <Grid item xs={12}>
              <Button
                variant="contained"
                onClick={handleUpload}
                disabled={loading || !file}
                fullWidth
              >
                {loading ? <CircularProgress size={24} /> : 'Upload & Process'}
              </Button>
            </Grid>
          </Grid>
        </Box>
      )}

      {activeStep === 1 && (
        <Box className="feedback-section">
          <Typography variant="h6" gutterBottom>
            Review AI Recommendations
          </Typography>
          
          {taskStatus?.status === 'awaiting_feedback' ? (
            <>
              <Paper elevation={2} sx={{ p: 3, mb: 3, maxHeight: '400px', overflow: 'auto' }}>
                {recommendationsUpdated && (
                  <Alert severity="success" sx={{ mb: 2 }}>
                    Recommendations have been updated based on your feedback!
                  </Alert>
                )}
                
                {recommendations ? (
                  <ReactMarkdown>
                    {recommendations}
                  </ReactMarkdown>
                ) : (
                  <Typography variant="body1">
                    No recommendations available yet. Please wait...
                  </Typography>
                )}
              </Paper>
              
              <Alert severity="info" sx={{ mb: 3 }}>
                {feedbackSubmitted ? 
                  'Processing your feedback. Please wait...' : 
                  'Review the recommendations and provide feedback if needed, or accept them to continue.'}
              </Alert>
              
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <TextField
                    label="Your Feedback"
                    multiline
                    rows={4}
                    value={feedback}
                    onChange={(e) => setFeedback(e.target.value)}
                    placeholder="If you want to modify the recommendations, describe your changes here. Otherwise, click 'Accept Recommendations'."
                    fullWidth
                    disabled={loading || feedbackSubmitted}
                  />
                </Grid>
                
                <Grid item xs={12} sm={6}>
                  <Button
                    variant="outlined"
                    onClick={() => handleSubmitFeedback(false)}
                    disabled={loading || !feedback.trim() || feedbackSubmitted}
                    fullWidth
                  >
                    {loading ? <CircularProgress size={24} /> : 'Send Feedback'}
                  </Button>
                </Grid>
                
                <Grid item xs={12} sm={6}>
                  <Button
                    variant="contained"
                    color="primary"
                    onClick={() => handleSubmitFeedback(true)}
                    disabled={loading || feedbackSubmitted}
                    fullWidth
                  >
                    {loading ? <CircularProgress size={24} /> : 'Accept Recommendations'}
                  </Button>
                </Grid>
              </Grid>
            </>
          ) : (
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', p: 3 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <CircularProgress />
                <Typography variant="body1" sx={{ ml: 2 }}>
                  {feedbackSubmitted ? 
                    'Processing your feedback...' : 
                    'Processing your dataset and generating recommendations...'}
                </Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                This may take a moment. The page will automatically update when recommendations are ready.
              </Typography>
            </Box>
          )}
        </Box>
      )}

      {activeStep === 2 && taskStatus?.status === 'completed' && (
        <Box className="results-section">
          <Typography variant="h6" gutterBottom>
            Feature Engineering Results
          </Typography>
          
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                Processed Data Preview
              </Typography>
              
              {taskStatus.results?.data_sample && (
                <>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Shape: {taskStatus.results.data_shape[0]} rows × {taskStatus.results.data_shape[1]} columns
                  </Typography>
                  
                  <TableContainer component={Paper} sx={{ maxHeight: 300, mb: 2 }}>
                    <Table size="small" stickyHeader>
                      <TableHead>
                        <TableRow>
                          {taskStatus.results.columns.map((column, index) => (
                            <TableCell key={index}>{column}</TableCell>
                          ))}
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {Object.keys(taskStatus.results.data_sample).slice(0, 10).map((rowKey, rowIndex) => (
                          <TableRow key={rowIndex}>
                            {taskStatus.results.columns.map((column, colIndex) => (
                              <TableCell key={colIndex}>
                                {String(taskStatus.results.data_sample[rowKey][column] !== undefined ? 
                                  taskStatus.results.data_sample[rowKey][column] : 'null')}
                              </TableCell>
                            ))}
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </>
              )}
            </CardContent>
          </Card>
          
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                Feature Engineering Function
              </Typography>
              <Divider sx={{ mb: 2 }} />
              <Box className="code-block">
                <pre>{taskStatus.results?.function_code}</pre>
              </Box>
            </CardContent>
          </Card>
          
          <Button
            variant="contained"
            color="primary"
            onClick={handleDownload}
            startIcon={<CheckCircleIcon />}
          >
            Download Results
          </Button>
        </Box>
      )}
    </Box>
  );
};

export default Home;
