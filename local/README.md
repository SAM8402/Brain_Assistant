# Brain Viewer Application

A comprehensive brain atlas viewer and AI assistant application that provides interactive brain exploration with intelligent chatbot functionality.

## Overview

This application combines a sophisticated brain atlas viewer with an AI-powered chatbot assistant to provide an immersive brain exploration experience. Users can view brain sections, interact with anatomical regions, and get detailed information through natural language conversations.

## Current Architecture

The application currently uses a monolithic architecture with all functionality contained in two main files:

```
local/
├── combined_viewer.py      # Main Flask backend with all server logic
├── combined_viewer.html    # Complete frontend with embedded CSS/JS
├── requirements.txt        # Python dependencies
└── README.md              # Documentation
```

## Key Features

### 🧠 Interactive Brain Atlas Viewer
- **Zoomable Brain Images**: High-resolution brain section viewing with smooth zoom/pan
- **Interactive Annotations**: Click on brain regions for detailed information
- **Adjustable Overlays**: Opacity control for annotation layers
- **Region Highlighting**: Visual feedback for selected anatomical structures
- **Toggle Visibility**: Show/hide annotations with eye icon control

### 🖼️ Thumbnail Navigation System  
- **Grid Layout**: Responsive thumbnail grid for section browsing
- **Pagination Controls**: Navigate through large datasets efficiently
- **Visual Selection**: Clear indication of currently selected section
- **Quick Navigation**: One-click access to any brain section
- **Series Support**: Multiple staining types (NISL, Luxol, H&E)

### 🤖 AI-Powered Brain Assistant
- **Streaming Responses**: Real-time text generation for immediate feedback
- **Contextual Understanding**: Maintains conversation context about selected regions
- **Web Search Integration**: Enhanced answers with up-to-date research
- **Markdown Formatting**: Rich text responses with proper formatting
- **Session Management**: Persistent conversation history during session
- **Auto-clear on Refresh**: Fresh start with each page reload

### 🎨 Modern User Interface
- **Ultra-Modern Design**: Glassmorphism effects and smooth animations
- **Dark/Light Theme**: Toggle between themes with persistence
- **Responsive Layout**: Adapts to different screen sizes
- **Smooth Transitions**: Polished animations throughout the interface
- **Accessibility**: Keyboard navigation and proper contrast ratios

## Technical Stack

### Backend Technologies
- **Flask**: Web framework for Python
- **SQLAlchemy**: Database ORM for metadata queries
- **PyMySQL**: MySQL database connector
- **OpenAI API**: Integration for AI assistant functionality
- **LangChain**: AI agent framework with tool integration
- **Server-Sent Events**: Real-time streaming responses

### Frontend Technologies  
- **OpenLayers**: Interactive map rendering for brain atlas
- **Vanilla JavaScript**: Custom UI interactions and API communication
- **CSS3**: Modern styling with custom properties and animations
- **HTML5**: Semantic markup and responsive design
- **WebSockets/SSE**: Real-time communication with backend

### External Services
- **MySQL Database**: Brain metadata and annotation storage
- **IIP Image Server**: High-performance image serving
- **DuckDuckGo Search**: Web search capabilities for enhanced responses
- **Local LLM**: Llama model for AI responses

## Installation & Setup

### Prerequisites
- Python 3.8+
- MySQL database access
- Network access to external APIs

### Installation Steps

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure Database**:
   - Update database credentials in `combined_viewer.py`
   - Ensure MySQL server is accessible

3. **Run Application**:
```bash
python3 combined_viewer.py
```

4. **Access Interface**:
   - Open browser to `http://localhost:5001`
   - Application will be ready for use

## Usage Guide

### Basic Navigation

#### Atlas Viewer Controls
- **Zoom**: Mouse wheel or pinch gestures
- **Pan**: Click and drag to move around
- **Region Selection**: Click on anatomical regions
- **Opacity**: Use left-side slider to adjust overlay transparency
- **Visibility**: Eye icon toggles annotation visibility

#### Thumbnail Navigation
- **Section Selection**: Click any thumbnail to load that section
- **Pagination**: Use arrow buttons to navigate pages
- **Series Selection**: Dropdown to choose staining type
- **Visual Feedback**: Selected thumbnail highlighted in blue

#### Chatbot Interface
- **Open Assistant**: Click brain icon (bottom-right) or select a region
- **Ask Questions**: Type in input field and press Enter
- **Web Search**: Toggle checkbox for enhanced research capabilities  
- **Theme Toggle**: Button in top-right (hidden when chatbot open)
- **Close Assistant**: X button in chatbot header

### Advanced Features

#### URL Parameters
Customize the initial view with URL parameters:
```
http://localhost:5001/?biosample=244&section=916&series=1
```
- `biosample`: Brain sample ID
- `section`: Section number
- `series`: Staining series type

#### Keyboard Shortcuts
- `Escape`: Close chatbot if open
- `Enter`: Send message in chat (without Shift)
- `Shift+Enter`: New line in chat input

## API Documentation

### Core Endpoints

#### Viewer APIs
- `GET /` - Main application interface
- `POST /api/geojson` - Get GeoJSON annotation data
- `POST /api/thumbnails` - Get thumbnail grid data
- `POST /api/light-weight-viewer` - Get viewer configuration

#### Chatbot APIs
- `POST /api/brain-region-stream` - Stream brain region information
- `POST /api/ask-question-stream` - Stream follow-up responses
- `GET /api/chat-history` - Retrieve conversation history
- `DELETE /api/chat-history` - Clear conversation history

### Request/Response Format

#### Brain Region Request
```json
{
  "region": "Hippocampus",
  "mode": "detailed"
}
```

#### Streaming Response
```
data: {"chunk": "The hippocampus is...", "success": true}
data: {"complete": true, "region": "Hippocampus"}
```

## Configuration

### Database Settings
```python
MySQL_db_user = "root"
MySQL_db_password = "Health#123"
MySQL_db_host = "apollo2.humanbrain.in"
MySQL_db_port = "3306" 
MySQL_db_name = "HBA_V2"
```

### External Service URLs
```python
base_url = "http://dgx3.humanbrain.in:10603"
atlas_url = "http://gliacoder.humanbrain.in:8000/atlas/getAtlasgeoJson"
image_server = "https://apollo2.humanbrain.in/iipsrv"
```

### AI Assistant Settings
```python
model_name = "Llama-3.3-70B-Instruct"
base_url = "http://dgx5.humanbrain.in:8999/v1"
max_tokens = 512
temperature = 0
```

## Development

### Code Organization
The monolithic structure keeps all functionality in two main files:

- **`combined_viewer.py`**: Contains all backend logic including:
  - Flask application setup
  - Database operations  
  - API endpoint handlers
  - Chatbot integration
  - Session management

- **`combined_viewer.html`**: Contains complete frontend including:
  - HTML structure
  - CSS styling and themes
  - JavaScript functionality
  - Event handlers and API calls

### Adding Features

#### Backend Changes
1. Add new functions to `combined_viewer.py`
2. Create new API endpoints as needed
3. Update database queries if required
4. Test with existing endpoints

#### Frontend Changes  
1. Modify HTML structure in `combined_viewer.html`
2. Add CSS styles in the `<style>` section
3. Implement JavaScript in the `<script>` section
4. Ensure responsive design compatibility

### Testing

#### Manual Testing
- Test all UI interactions
- Verify API responses
- Check responsive behavior
- Validate theme switching
- Test streaming functionality

#### Browser Testing
- Chrome/Chromium
- Firefox
- Safari  
- Edge
- Mobile browsers

## Performance Optimization

### Caching Strategy
- **In-Memory Caches**: GeoJSON, thumbnails, and viewer data
- **Session Storage**: Theme preferences and temporary data
- **Database Connection Pooling**: Efficient database access

### Loading Optimization
- **Lazy Loading**: Thumbnails load on demand
- **Streaming Responses**: Progressive content delivery
- **Image Optimization**: IIIF server provides optimized images
- **CDN Resources**: External libraries from CDN

## Security Considerations

### Input Validation
- All user inputs sanitized
- SQL injection protection via parameterized queries
- XSS prevention through proper escaping

### Session Security  
- Secure session management
- CORS properly configured
- No sensitive data in client storage

### API Security
- Rate limiting considerations
- Authentication for sensitive endpoints
- Proper error handling without information disclosure

## Troubleshooting

### Common Issues

#### Database Connection
- Verify MySQL server accessibility
- Check credentials and network connectivity
- Ensure database exists and has proper permissions

#### Image Loading Problems
- Confirm image server is running
- Check network access to image URLs  
- Verify CORS settings for cross-origin requests

#### Chatbot Issues
- Ensure AI model endpoint is accessible
- Check API keys and authentication
- Verify streaming endpoint functionality

#### Performance Issues
- Monitor database query performance
- Check image loading optimization
- Verify caching is working properly

### Debug Mode
Run with debug enabled for detailed error information:
```bash
export FLASK_DEBUG=true
python3 combined_viewer.py
```

## Deployment

### Production Considerations
- Use production WSGI server (Gunicorn, uWSGI)
- Configure proper logging
- Set up monitoring and health checks
- Implement proper backup strategies
- Configure SSL/HTTPS

### Docker Deployment
```dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
CMD ["python3", "combined_viewer.py"]
```

## Future Enhancements

### Planned Features
1. **Enhanced AI Capabilities**: More sophisticated brain analysis
2. **3D Visualization**: Three-dimensional brain rendering
3. **Collaborative Features**: Multi-user annotation and sharing
4. **Mobile App**: Native mobile application
5. **Offline Support**: Progressive Web App capabilities
6. **Advanced Analytics**: Usage tracking and insights

### Architecture Evolution
- **Microservices**: Split into focused services
- **WebSocket**: Full duplex communication
- **GraphQL**: More efficient API queries
- **Machine Learning**: Enhanced region detection
- **Cloud Integration**: Scalable cloud deployment

## Contributing

### Development Workflow
1. Fork the repository
2. Create feature branch
3. Make changes and test thoroughly
4. Submit pull request with detailed description

### Code Standards
- Follow PEP 8 for Python code
- Use consistent naming conventions
- Add comments for complex logic
- Test all new functionality

## Support

For issues and questions:
- Check troubleshooting section
- Review error logs
- Test with minimal configuration
- Document steps to reproduce issues

## License

This project is developed for research and educational purposes. Please respect all applicable licenses and terms of use for external services and data sources.