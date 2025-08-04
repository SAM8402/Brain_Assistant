import os
import requests

class BrainRegionAssistant:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Try different model names based on availability
        self.model_names = ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-pro", "gemini-1.0-pro"]
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        self.current_region = None
        self.region_context = ""
        self.working_model = None
        
        # Find working model on initialization
        print("Initializing Brain Region Assistant...")
        self._find_working_model()
    
    def _find_working_model(self):
        """Find a working model on initialization"""
        headers = {"Content-Type": "application/json"}
        test_data = {
            "contents": [{
                "parts": [{"text": "test"}]
            }]
        }
        
        for model in self.model_names:
            url = f"{self.base_url.format(model=model)}?key={self.api_key}"
            try:
                response = requests.post(url, headers=headers, json=test_data, timeout=5)
                if response.status_code == 200:
                    self.working_model = model
                    print(f"Connected to {model} model.")
                    return
            except:
                continue
        
        print("Warning: Could not find a working model. Will retry on each request.")
    
    def _make_gemini_request(self, prompt: str) -> str:
        """Make a request to Gemini API using REST"""
        headers = {
            "Content-Type": "application/json",
        }
        
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1024,
            }
        }
        
        # Use cached working model if available
        models_to_try = [self.working_model] if self.working_model else self.model_names
        
        for model in models_to_try:
            url = f"{self.base_url.format(model=model)}?key={self.api_key}"
            
            try:
                response = requests.post(url, headers=headers, json=data, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    if "candidates" in result and len(result["candidates"]) > 0:
                        candidate = result["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            parts = candidate["content"]["parts"]
                            if len(parts) > 0 and "text" in parts[0]:
                                # Cache this working model
                                if not self.working_model:
                                    self.working_model = model
                                return parts[0]["text"]
                    return "No response generated."
                    
                elif response.status_code == 404 and not self.working_model:
                    continue
                elif response.status_code == 400:
                    return "Error: Invalid API key or request format."
                elif response.status_code == 403:
                    return "Error: API key doesn't have access to Gemini."
                else:
                    if not self.working_model:
                        continue
                    return f"Error: {response.status_code}"
                    
            except requests.exceptions.Timeout:
                return "Error: Request timed out. Please try again."
            except Exception as e:
                if not self.working_model:
                    continue
                return f"Error: {str(e)}"
        
        return "Error: Could not connect to any Gemini model."
    
    def fetch_brain_region_from_web(self, region_name: str) -> str:
        """Fetch brain region information from the web using Gemini"""
        prompt = f"""Provide a concise overview of the {region_name} brain region including:
        1. Location and structure
        2. Main functions
        3. Key connections
        4. Clinical significance
        Keep it under 200 words."""
        
        return self._make_gemini_request(prompt)
    
    def get_brain_region_info(self, region_name: str) -> str:
        """Get detailed information about a brain region using Gemini"""
        self.current_region = region_name
        
        # Fetch information from web using Gemini
        self.region_context = self.fetch_brain_region_from_web(region_name)
        return self.region_context
    
    def ask_question(self, question: str) -> str:
        """Ask a question about the current brain region"""
        if not self.current_region:
            return "Please specify a brain region first using 'region' command"
        
        prompt = f"""About the {self.current_region} brain region, answer: {question}
        Be concise and accurate."""
        
        return self._make_gemini_request(prompt)

def get_or_save_api_key():
    """Get API key from environment, file, or user input"""
    # Check environment variable first
    # api_key = os.environ.get("GOOGLE_API_KEY")
    api_key = "AIzaSyAlEwiCemb1kpclHSyb6z7RUgqSJoHUzvI"  # For testing purposes, replace with actual environment variable if needed
    if api_key:
        return api_key
    
    # Check if API key is saved in file
    api_key_file = os.path.expanduser("~/.gemini_api_key")
    if os.path.exists(api_key_file):
        try:
            with open(api_key_file, 'r') as f:
                api_key = f.read().strip()
                if api_key:
                    print("Using saved API key.")
                    return api_key
        except Exception:
            pass
    
    # Ask user for API key
    api_key = input("Please enter your Google Gemini API key: ").strip()
    
    # Ask if user wants to save it
    save_key = input("\nDo you want to save this API key for future use? (yes/no): ").strip().lower()
    if save_key in ['yes', 'y']:
        try:
            with open(api_key_file, 'w') as f:
                f.write(api_key)
            os.chmod(api_key_file, 0o600)  # Make file readable only by owner
            print(f"API key saved to {api_key_file}")
        except Exception as e:
            print(f"Warning: Could not save API key: {e}")
    
    return api_key

def main():
    # Get API key
    api_key = get_or_save_api_key()
    
    # Create assistant instance
    assistant = BrainRegionAssistant(api_key)
    
    print("\nBrain Region Assistant Ready!")
    print("=" * 50)
    print("Commands:")
    print("  region <name> - Get information about a brain region")
    print("  ask <question> - Ask a question about the current region")
    print("  quit - Exit the program")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            # Parse command
            parts = user_input.split(' ', 1)
            if len(parts) == 0:
                continue
                
            command = parts[0].lower()
            
            if command == 'region' and len(parts) > 1:
                region_name = parts[1]
                print(f"\nSearching for information about: {region_name}")
                print("-" * 40)
                info = assistant.get_brain_region_info(region_name)
                print(info)
                
            elif command == 'ask' and len(parts) > 1:
                question = parts[1]
                print(f"\nAnswering question about {assistant.current_region}...")
                print("-" * 40)
                answer = assistant.ask_question(question)
                print(answer)
                
            else:
                print("Invalid command. Use 'region <name>', 'ask <question>', or 'quit'")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    print("\nBrain Region Assistant - Powered by Google Gemini")
    print("="*50)
    print("This assistant uses Google Gemini to provide information about brain regions.")
    print("You can set the GOOGLE_API_KEY environment variable or enter it when prompted.")
    print("Get your API key from: https://makersuite.google.com/app/apikey")
    print("To remove saved API key, delete ~/.gemini_api_key")
    print("="*50)
    
    try:
        main()
    except Exception as e:
        print(f"\nError initializing assistant: {str(e)}")
        print("Please check your API key and internet connection.")