from flask import Flask, render_template, jsonify, request
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brain_region_assistant_langchain import UltraFastBrainAssistant

app = Flask(__name__)

# Initialize the brain assistant
assistant = UltraFastBrainAssistant(use_web_search=True)

@app.route('/')
def index():
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
        
        # Get brain region info
        is_valid, info = assistant.get_brain_region_info(region_name, mode)
        
        if is_valid:
            # Store current region for follow-up questions
            assistant.current_region = region_name
            return jsonify({
                'success': True,
                'message': info,
                'region': region_name
            })
        else:
            return jsonify({
                'success': False,
                'message': info
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/ask-question', methods=['POST'])
def ask_question():
    """Handle follow-up questions about current brain region"""
    try:
        data = request.json
        question = data.get('question')
        use_web = data.get('use_web', False)
        
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
        
        # Get answer
        answer = assistant.ask_question(question, use_web)
        
        return jsonify({
            'success': True,
            'message': answer,
            'region': assistant.current_region
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
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

if __name__ == '__main__':
    app.run(debug=True, port=5000)