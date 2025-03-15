# Human-in-the-Loop GenAI Feature Engineering Application

A modern web application that provides an intuitive user interface for interactive feature engineering using GenAI, powered by the `ai_data_science_team` project's FeatureEngineeringAgent.

![Feature Engineering Process](https://raw.githubusercontent.com/business-science/ai-data-science-team/master/img/feature-engineer-demo.png)

## 🌟 Features

- **Interactive Feature Engineering**: Upload your datasets and leverage AI to perform sophisticated feature engineering with human oversight
- **Human-in-the-Loop**: Review, modify, and improve AI-generated feature engineering recommendations
- **Modern UI**: Clean, responsive interface built with React and Material UI
- **Python Backend**: Powerful FastAPI backend that integrates with the AI Data Science Team library
- **No-Code Solution**: Perform complex feature engineering without writing code
- **Reproducible Workflows**: Download the generated feature engineering functions for use in your own projects
- **Enhanced Data Previews**: View comprehensive previews of both your original and transformed data
- **Robust Error Handling**: Gracefully handles processing errors with fallback mechanisms

## 🔧 Technologies

- **Frontend**: React.js, Material UI, Axios
- **Backend**: FastAPI, Pandas, LangChain, OpenAI
- **AI Components**: FeatureEngineeringAgent from ai_data_science_team

## 📋 Prerequisites

- Python 3.10+
- Node.js 16+
- Access to the `ai_data_science_team` package
- OpenAI API key (set as OPENAI_API_KEY environment variable)

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/feature-engineering-app.git
cd feature-engineering-app
```

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install /path/to/ai-data-science-team  # Install the ai-data-science-team package
```

4. Set your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY="your-openai-api-key"  # On Windows: set OPENAI_API_KEY=your-openai-api-key
```

5. Make the start script executable and run the FastAPI server:
```bash
chmod +x start.sh
./start.sh
```

The backend server will be available at http://localhost:8000.

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd ../frontend
```

2. Install Node.js dependencies:
```bash
npm install
```

3. Start the React development server:
```bash
npm start
```

The frontend application will be available at http://localhost:3000.

## 📊 Usage Flow

1. **Upload Dataset**: Select a CSV file to upload
2. **Provide Instructions**: Enter specific feature engineering instructions or leave it to the AI
3. **Review Recommendations**: Examine the AI-generated feature engineering plan
4. **Provide Feedback**: Accept the recommendations or suggest modifications
5. **View Results**: Explore the transformed dataset and the generated feature engineering code
6. **Download Output**: Save the results for use in your own projects

## 🔄 Application Workflow

1. **Data Upload Phase**: Users upload CSV data and specify feature engineering instructions
2. **AI Processing Phase**: The backend leverages the FeatureEngineeringAgent to analyze the data and generate recommendations
3. **Human Feedback Phase**: Users review and provide feedback on the recommendations
4. **Execution Phase**: The backend processes the data according to the approved or modified plan
5. **Results Phase**: The transformed data and feature engineering code are presented to the user

## 🛠️ Development

### Backend API Endpoints

- `GET /`: Root endpoint to check if the API is running
- `POST /api/feature-engineering/upload`: Upload a dataset and start feature engineering
- `GET /api/feature-engineering/status/{task_id}`: Check the status of a feature engineering task
- `POST /api/feature-engineering/feedback`: Provide feedback on feature engineering recommendations
- `GET /api/feature-engineering/download/{task_id}`: Download the feature engineered results

### Frontend Components

- **Home**: Main application interface
- **UploadSection**: Handles file uploads and instructions
- **FeedbackSection**: Displays recommendations and collects user feedback
- **ResultsSection**: Shows the transformed data and generated code
- **DataPreview**: Displays comprehensive previews of both original and processed data

## 🆕 Recent Improvements

- **Enhanced Data Previews**: Added robust data preview capabilities showing at least 10 rows for both original and processed data
- **Improved Error Handling**: Added fallback mechanisms that ensure data is always displayed even when errors occur
- **Data Processing Resilience**: Enhanced the feature engineering pipeline to handle edge cases and unexpected data formats
- **Better User Feedback**: Improved logging and error messages for more transparent operation
- **Serialization Improvements**: Fixed data serialization to properly support human-in-the-loop functionality

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Business Science](https://www.business-science.io/) for the ai-data-science-team library
- [Matt Dancho](https://github.com/mdancho84) for the GenAI Data Science Workshop

## 📧 Contact

For questions or support, please open an issue on this repository or contact the maintainers directly.

---

*Built with ❤️ using AI Data Science Team*
