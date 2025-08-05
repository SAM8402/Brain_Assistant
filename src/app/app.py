from flask import Flask, render_template, jsonify, request, session
import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brain_region_assistant_langchain import UltraFastBrainAssistant

app = Flask(__name__)
app.secret_key = 'brain_assistant_secret_key_change_in_production'  # Change this in production

# Initialize the brain assistant
assistant = UltraFastBrainAssistant(use_web_search=True)

# Helper function to sync Flask session with assistant's conversation history
def sync_assistant_history():
    """Sync Flask session history with the assistant's internal history"""
    chat_history = get_chat_history()
    
    # Clear and rebuild assistant's conversation history from session
    assistant.clear_conversation_history()
    
    # Add conversations to assistant's memory
    for msg in chat_history:
        if msg['type'] == 'user_query':
            # Find the corresponding assistant response
            next_msg = None
            idx = chat_history.index(msg)
            if idx + 1 < len(chat_history):
                next_msg = chat_history[idx + 1]
            
            if next_msg and next_msg['type'] in ['assistant_response', 'region_info']:
                assistant.add_to_conversation(
                    msg['content'],
                    next_msg['content'],
                    msg['type']
                )

def get_chat_history():
    """Get chat history from session"""
    if 'chat_history' not in session:
        session['chat_history'] = []
    return session['chat_history']

def add_to_chat_history(message_type, content, region=None):
    """Add a message to chat history"""
    chat_history = get_chat_history()
    message = {
        'type': message_type,  # 'user_query', 'assistant_response', 'region_info'
        'content': content,
        'timestamp': datetime.now().isoformat(),
        'region': region
    }
    chat_history.append(message)
    session['chat_history'] = chat_history
    session.modified = True

def clear_chat_history():
    """Clear chat history from session"""
    session['chat_history'] = []
    session.pop('current_mode', None)  # Also clear stored mode
    session.modified = True
    # Also clear assistant's conversation history
    assistant.clear_conversation_history()

@app.route('/')
def index():
    # Clear chat history on page load/refresh
    clear_chat_history()
    assistant.current_region = None
    return render_template('index.html')

@app.route('/api/brain-region', methods=['POST'])
def process_brain_region():
    """Process brain region queries"""
    try:
        data = request.json
        region_name = data.get('region')
        mode = data.get('mode', 'fast')  # fast, web, or ultra
        
        if not region_name:
            return jsonify({
                'success': False,
                'message': 'Please provide a brain region name'
            }), 400
        
        # Sync conversation history before processing
        sync_assistant_history()
        
        # Add user query to chat history
        add_to_chat_history('user_query', f"Tell me about {region_name} (mode: {mode})", region_name)
        
        # Get brain region info
        is_valid, info = assistant.get_brain_region_info(region_name, mode)
        
        if is_valid:
            # Store current region and mode for follow-up questions
            assistant.current_region = region_name
            session['current_mode'] = mode
            session.modified = True
            
            # Add assistant response to chat history
            add_to_chat_history('region_info', info, region_name)
            
            return jsonify({
                'success': True,
                'message': info,
                'region': region_name,
                'mode': mode,
                'chat_history': get_chat_history()
            })
        else:
            # Add error response to chat history
            add_to_chat_history('assistant_response', info, region_name)
            
            return jsonify({
                'success': False,
                'message': info,
                'chat_history': get_chat_history()
            })
            
    except Exception as e:
        error_msg = f'Error: {str(e)}'
        add_to_chat_history('assistant_response', error_msg)
        return jsonify({
            'success': False,
            'message': error_msg,
            'chat_history': get_chat_history()
        }), 500

@app.route('/api/ask-question', methods=['POST'])
def ask_question():
    """Handle follow-up questions about current brain region"""
    try:
        data = request.json
        question = data.get('question')
        use_web = data.get('use_web', False)
        # Use stored mode from session, fallback to provided mode or 'fast'
        mode = session.get('current_mode', data.get('mode', 'fast'))
        
        if not question:
            return jsonify({
                'success': False,
                'message': 'Please provide a question'
            }), 400
        
        if not assistant.current_region:
            return jsonify({
                'success': False,
                'message': 'Please select a brain region first'
            }), 400
        
        # Sync conversation history before processing
        sync_assistant_history()
        
        # Automatically enable web search if mode is 'web' (Detailed mode)
        if mode == 'web':
            use_web = True
        
        # Add user question to chat history
        search_indicator = " (web search)" if use_web else ""
        add_to_chat_history('user_query', f"{question}{search_indicator}", assistant.current_region)
        
        # Get answer
        answer = assistant.ask_question(question, use_web)
        
        # Add assistant answer to chat history
        add_to_chat_history('assistant_response', answer, assistant.current_region)
        
        return jsonify({
            'success': True,
            'message': answer,
            'region': assistant.current_region,
            'chat_history': get_chat_history(),
            'web_search_used': use_web
        })
        
    except Exception as e:
        error_msg = f'Error: {str(e)}'
        add_to_chat_history('assistant_response', error_msg, assistant.current_region)
        return jsonify({
            'success': False,
            'message': error_msg,
            'chat_history': get_chat_history()
        }), 500

@app.route('/api/validate-region', methods=['POST'])
def validate_region():
    """Validate if input is a brain region"""
    try:
        data = request.json
        region_name = data.get('region')
        
        if not region_name:
            return jsonify({
                'success': False,
                'is_valid': False
            }), 400
        
        is_valid = assistant.validate_brain_region(region_name)
        
        return jsonify({
            'success': True,
            'is_valid': is_valid
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/chat-history', methods=['GET'])
def get_chat_history_endpoint():
    """Get current chat history"""
    try:
        chat_history = get_chat_history()
        return jsonify({
            'success': True,
            'chat_history': chat_history,
            'current_region': assistant.current_region,
            'current_mode': session.get('current_mode', 'fast')
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/chat-history', methods=['DELETE'])
def clear_chat_history_endpoint():
    """Clear chat history"""
    try:
        clear_chat_history()
        # Also reset current region
        assistant.current_region = None
        return jsonify({
            'success': True,
            'message': 'Chat history cleared successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/my-questions', methods=['GET'])
def get_user_questions():
    """Get only the questions asked by the user"""
    try:
        chat_history = get_chat_history()
        # Filter only user queries
        user_questions = [
            {
                'question': msg['content'],
                'timestamp': msg['timestamp'],
                'region': msg.get('region'),
                'index': idx
            }
            for idx, msg in enumerate(chat_history) 
            if msg['type'] == 'user_query'
        ]
        
        return jsonify({
            'success': True,
            'questions': user_questions,
            'total_questions': len(user_questions)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/search-config', methods=['POST'])
def configure_search():
    """Configure search parameters"""
    try:
        data = request.json
        max_sources = data.get('max_sources', 3)
        
        if max_sources < 1 or max_sources > 15:
            return jsonify({
                'success': False,
                'message': 'max_sources must be between 1 and 15 (default: 12)'
            }), 400
        
        # Update search configuration
        assistant.set_search_sources(max_sources)
        
        return jsonify({
            'success': True,
            'message': f'Search configured to use maximum {max_sources} sources',
            'max_sources': max_sources
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/search-config', methods=['GET'])
def get_search_config():
    """Get current search configuration"""
    try:
        max_sources = getattr(assistant.web_search, 'max_sources', 12) if hasattr(assistant, 'web_search') else 12
        
        return jsonify({
            'success': True,
            'max_sources': max_sources,
            'default_sources': 12,
            'maximum_sources': 15,
            'available_sources': [
                'DuckDuckGo Search (Primary)',
                'Wikipedia API (Comprehensive)', 
                'DuckDuckGo Instant Answer',
                'PubMed Scientific Database',
                'Bing Search Engine',
                'Alternative Search Engines (Startpage, Ecosia)',
                'Yahoo/Yandex Search',
                'Google API Fallback',
                'Wikipedia Extended Search',
                'PubMed Extended Search',
                'Educational Sources (Universities)',
                'Medical & Clinical Databases'
            ]
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)