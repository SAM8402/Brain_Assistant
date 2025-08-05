# Brain Region Assistant Flask App

This Flask application provides an interactive interface for exploring brain regions using the Brain Region Assistant with LangChain.

## Features

- Interactive SVG brain diagram with clickable regions
- Custom brain region search
- AI-powered chatbot for brain region information
- Three modes: Fast, Detailed (Web Search), and Ultra-fast
- Follow-up question capability
- Real-time chat interface

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flask app:
```bash
python app.py
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

1. **Click on Brain Regions**: Click on any highlighted region in the brain diagram to get information about it.

2. **Custom Search**: Enter any brain region name in the search box (e.g., "thalamus", "hypothalamus", "corpus callosum").

3. **Select Mode**:
   - **Fast**: Quick AI response with concise information
   - **Detailed (Web)**: Enhanced search with web results
   - **Ultra-fast**: Minimal, key facts only

4. **Ask Questions**: After selecting a region, ask follow-up questions in the chat interface.

## Brain Regions Available

- Frontal Lobe
- Parietal Lobe
- Temporal Lobe
- Occipital Lobe
- Cerebellum
- Hippocampus
- Amygdala
- Brainstem
- And any other brain region via custom search

## API Endpoints

- `GET /` - Main interface
- `POST /api/brain-region` - Get brain region information
- `POST /api/ask-question` - Ask follow-up questions
- `POST /api/validate-region` - Validate if input is a brain region