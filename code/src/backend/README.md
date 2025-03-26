# Customer Analytics Chatbot

A FastAPI-based chatbot application that provides customer analytics, transaction insights, and personalized recommendations.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Setup Instructions

1. Clone or download this repository to your local machine.

2. Create a virtual environment (recommended):
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the `backend` directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
   Note: If you don't have an OpenAI API key, the application will still work but with limited functionality.

5. (Optional) Place your customer data Excel file in the `backend/data/datasets` directory with the name `sample_customer_data.xlsx`. The file should contain three sheets:
   - Sheet1: Customer profiles
   - Sheet2: Social media sentiment
   - Sheet3: Transaction history
   
   If no Excel file is provided, the application will use mock data.

## Running the Application

1. Make sure you're in the `backend` directory:
   ```bash
   cd backend
   ```

2. Start the application:
   ```bash
   python main.py
   ```

3. Open your web browser and navigate to:
   ```
   http://localhost:8001
   ```

## Features

- Customer profile analysis
- Transaction history and patterns
- Social media sentiment analysis
- Personalized recommendations
- Interactive chat interface

## Troubleshooting

1. If you get a "port already in use" error:
   - The application uses port 8001 by default
   - You can modify the port in `main.py` if needed
   - Or close any other applications using the same port

2. If you get dependency errors:
   - Make sure you've activated your virtual environment
   - Try reinstalling the requirements: `pip install -r requirements.txt`

3. If the chatbot isn't responding:
   - Check your internet connection
   - Verify your OpenAI API key is correct
   - Check the console for error messages

## Directory Structure

```
backend/
├── data/
│   └── datasets/
│       └── sample_customer_data.xlsx (optional)
├── services/
│   ├── chatbot_service.py
│   ├── data_service.py
│   └── genai_service.py
├── templates/
│   └── chat.html
├── main.py
├── requirements.txt
└── README.md
```

## Support

If you encounter any issues or have questions, please check the troubleshooting section or create an issue in the repository. 