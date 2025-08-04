import os
import requests
import asyncio
import aiohttp
from typing import Dict, Tuple, Any, List, Optional
from functools import lru_cache
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

class GeminiLLM(LLM):
    """Simple LLM wrapper for Google Gemini without problematic caching"""
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.model_name = "gemini-1.5-flash"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 512,
                "topK": 10,
                "topP": 0.8
            }
        }
        
        try:
            response = requests.post(url, json=data, timeout=15, headers={"Content-Type": "application/json"})
            if response.status_code == 200:
                result = response.json()
                if "candidates" in result and result["candidates"]:
                    return result["candidates"][0]["content"]["parts"][0]["text"]
            return "Error generating response"
        except Exception as e:
            return f"Error: {str(e)}"
    
    @property
    def _llm_type(self) -> str:
        return "gemini"

class UltraFastBrainAssistant:
    def __init__(self, api_key: str, use_web_search: bool = True):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        self.current_region = None
        self.region_cache: Dict[str, str] = {}
        self.use_web_search = use_web_search
        
        # Initialize LangChain components if web search is enabled
        if self.use_web_search:
            try:
                self.llm = GeminiLLM(api_key=api_key)
                self.search = DuckDuckGoSearchRun()
                self.tools = [
                    Tool(
                        name="DuckDuckGoSearch",
                        func=self.search.run,
                        description="Search the web for current brain region information, neuroscience research, and medical information"
                    )
                ]
                self.agent = initialize_agent(
                    tools=self.tools,
                    llm=self.llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=False,
                    max_iterations=3,
                    handle_parsing_errors=True
                )
                print("⚡ Ask the Brain AI Assistant Ready! (Web Search Enabled)")
            except Exception as e:
                print(f"Warning: Web search failed to initialize: {e}")
                self.use_web_search = False
                print("⚡ Ask the Brain AI Assistant Ready! (Offline Mode)")
        else:
            print("⚡ Ask the Brain AI Assistant Ready! (Fast Mode)")
    
    async def _async_gemini_request(self, prompt: str) -> str:
        """Async request for maximum speed"""
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 256,
                "candidateCount": 1
            }
        }
        
        url = f"{self.base_url}?key={self.api_key}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, timeout=10, headers={"Content-Type": "application/json"}) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            print(f"Async request error: {e}")
        return "Error retrieving information"
    
    def _sync_gemini_request(self, prompt: str) -> str:
        """Synchronous request"""
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 512
            }
        }
        
        url = f"{self.base_url}?key={self.api_key}"
        
        try:
            response = requests.post(url, json=data, timeout=15, headers={"Content-Type": "application/json"})
            if response.status_code == 200:
                result = response.json()
                if "candidates" in result and result["candidates"]:
                    return result["candidates"][0]["content"]["parts"][0]["text"]
            return "Error generating response"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def validate_brain_region(self, region_name: str) -> bool:
        """Check if the input is a valid brain region"""
        
        # Simple heuristic check first for obvious non-brain terms
        non_brain_terms = {'apple', 'pizza', 'car', 'computer', 'table', 'phone', 'book', 'chair', 'door', 'window'}
        if region_name.lower() in non_brain_terms:
            return False
            
        validation_prompt = f"Is '{region_name}' a brain region, brain structure, or part of the brain? Answer only 'yes' or 'no'."
        
        # Use direct API call for validation
        data = {
            "contents": [{"parts": [{"text": validation_prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 10
            }
        }
        
        url = f"{self.base_url}?key={self.api_key}"
        
        try:
            response = requests.post(url, json=data, timeout=10, headers={"Content-Type": "application/json"})
            if response.status_code == 200:
                result = response.json()
                if "candidates" in result and result["candidates"]:
                    text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                    # Check for 'yes' at the beginning of the response
                    return text.lower().startswith('yes')
            # If API fails, do basic validation
            brain_terms = {'brain', 'cortex', 'lobe', 'hippocampus', 'amygdala', 'thalamus', 'cerebellum', 'stem', 'hemispheres'}
            return any(term in region_name.lower() for term in brain_terms)
        except Exception:
            # Fallback validation
            brain_terms = {'brain', 'cortex', 'lobe', 'hippocampus', 'amygdala', 'thalamus', 'cerebellum', 'stem', 'hemispheres'}
            return any(term in region_name.lower() for term in brain_terms)
    
    def get_brain_region_info(self, region_name: str, mode: str = "fast") -> Tuple[bool, str]:
        """Get brain region info with different modes"""
        
        # First validate if it's a brain region
        print("🧠 Validating brain region...")
        if not self.validate_brain_region(region_name):
            return (False, f"'{region_name}' is not a brain region. Please enter a valid brain region name.")
        
        self.current_region = region_name
        
        # Check cache
        cache_key = f"{region_name.lower()}_{mode}"
        if cache_key in self.region_cache:
            return (True, f"📋 (Cached)\n{self.region_cache[cache_key]}")
        
        try:
            if mode == "web" and self.use_web_search:
                # Use LangChain with web search
                print("🔍 Searching the web...")
                query = f"""Search for comprehensive information about the {region_name} brain region.
                Find and summarize:
                1. Anatomical location and structure
                2. Primary functions and roles
                3. Neural connections and pathways
                4. Clinical significance and related disorders
                
                Provide a concise but informative summary in about 150-200 words."""
                
                try:
                    result = self.agent.invoke({"input": query})
                    info = result.get("output", "No information found")
                except Exception as e:
                    print(f"Web search failed: {e}")
                    info = f"Web search failed. Using direct AI knowledge instead.\n\n"
                    prompt = f"Provide detailed information about the {region_name} brain region: location, functions, connections, clinical significance (200 words max)"
                    info += self._sync_gemini_request(prompt)
                
            elif mode == "ultra":
                # Use async for ultra-fast mode
                prompt = f"List key facts about {region_name} brain region: location, function, connections, clinical significance (max 100 words)"
                try:
                    loop = asyncio.get_event_loop()
                    info = loop.run_until_complete(self._async_gemini_request(prompt))
                except:
                    # Fallback to sync if async fails
                    info = self._sync_gemini_request(prompt)
                
            else:  # fast mode (default)
                # Use sync request
                prompt = f"""Provide a concise overview of the {region_name} brain region:
1. Location: Where in the brain?
2. Function: What does it do?
3. Connections: Key neural pathways
4. Clinical: Related conditions
Keep it under 200 words but be informative."""
                info = self._sync_gemini_request(prompt)
            
            # Cache the result
            if info and info != "Error" and not info.startswith("Error"):
                self.region_cache[cache_key] = info
                return (True, info)
            else:
                return (False, "Failed to retrieve information")
                
        except Exception as e:
            return (False, f"Error: {str(e)}")
    
    def ask_question(self, question: str, use_web: bool = False) -> str:
        """Ask questions with optional web search"""
        if not self.current_region:
            return "Please specify a brain region first using 'region <name>'"
        
        try:
            if use_web and self.use_web_search:
                query = f"Search for information to answer this specific question about the {self.current_region} brain region: {question}"
                try:
                    result = self.agent.invoke({"input": query})
                    return result.get("output", "Unable to find answer")
                except Exception as e:
                    # Fallback to direct AI if web search fails
                    prompt = f"About the {self.current_region} brain region, answer this question: {question}"
                    return f"Web search failed, using AI knowledge: {self._sync_gemini_request(prompt)}"
            else:
                prompt = f"About the {self.current_region} brain region, answer concisely: {question}"
                return self._sync_gemini_request(prompt)
        except Exception as e:
            return f"Error: {str(e)}"

def main():
    # Get API key
    api_key = "AIzaSyAlEwiCemb1kpclHSyb6z7RUgqSJoHUzvI"
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        api_key = input("Enter Google Gemini API key: ").strip()
    
    print("\n" + "="*60)
    print("⚡ Ask the Brain AI Assistant")
    print("="*60)
    
    # Ask user for mode preference
    print("\nChoose default mode:")
    print("1. Fast mode (AI knowledge only, fastest)")
    print("2. Web search mode (real-time data, more comprehensive)")
    mode_choice = input("Enter 1 or 2 (default=1): ").strip() or "1"
    
    use_web = mode_choice == "2"
    assistant = UltraFastBrainAssistant(api_key, use_web_search=use_web)
    
    print("\nCommands:")
    print("  region <name> - Get brain region info")
    print("  quit         - Exit")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            parts = user_input.split(' ', 2)
            if not parts:
                continue
            
            command = parts[0].lower()
            
            if command == 'region' and len(parts) > 1:
                region_name = " ".join(parts[1:])  # Handle multi-word regions
                
                # Ask for search mode after region is entered
                print(f"\nHow would you like to search for {region_name}?")
                print("1. Fast mode (AI knowledge)")
                print("2. Web search mode (real-time)")
                print("3. Ultra-fast mode (minimal)")
                
                mode_input = input("Enter choice (1/2/3, default=1): ").strip() or "1"
                
                mode_map = {"1": "fast", "2": "web", "3": "ultra"}
                mode = mode_map.get(mode_input, "fast")
                
                print(f"\nFetching: {region_name} (mode: {mode})")
                print("-" * 50)
                
                is_valid, info = assistant.get_brain_region_info(region_name, mode)
                print(info)
                
                if is_valid:
                    print("\n" + "-" * 50)
                    print(f"Do you have any questions about {region_name}?")
                    print("Type 'yes' or 'no':")
            
            elif command in ['no', 'n']:
                print("\n" + "="*60)
                print("Commands:")
                print("  region <name> - Get brain region info")
                print("  quit         - Exit")
                print("="*60)
            
            elif command in ['yes', 'y']:
                if assistant.current_region:
                    print(f"\nHow would you like to ask about {assistant.current_region}?")
                    print("1. Quick answer (AI knowledge)")
                    print("2. Search web for answer")
                    
                    ask_mode = input("Enter choice (1/2, default=1): ").strip() or "1"
                    
                    print(f"\nWhat is your question about {assistant.current_region}?")
                    question = input("Question: ").strip()
                    
                    if question:
                        use_web = ask_mode == "2"
                        if use_web:
                            print("🔍 Searching web for answer...")
                        
                        answer = assistant.ask_question(question, use_web)
                        print(f"\n{answer}")
                        
                        print("\n" + "-" * 50)
                        print("Do you have more questions? (yes/no):")
                else:
                    print("Please specify a brain region first using 'region <name>'")
            
            else:
                if assistant.current_region and user_input:
                    # If there's a current region and user types something else, treat it as a question
                    print(f"\nAssuming this is a question about {assistant.current_region}")
                    answer = assistant.ask_question(user_input, False)
                    print(f"\n{answer}")
                    print("\n" + "-" * 50)
                    print("Do you have more questions? (yes/no):")
                else:
                    print("Unknown command. Try: region <name> or quit")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")

if __name__ == "__main__":
    # Check dependencies
    try:
        import aiohttp
        from langchain_community.tools import DuckDuckGoSearchRun
        main()
    except ImportError:
        print("Installing required packages...")
        os.system("pip install aiohttp langchain langchain-community duckduckgo-search requests")
        print("Please restart the program.")
        exit()