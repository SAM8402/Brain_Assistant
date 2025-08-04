import os
import requests
import asyncio
import aiohttp
import openai
import time
from typing import Dict, Tuple, Any, List, Optional
from functools import lru_cache
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

# class GeminiLLM(LLM):
#     """Simple LLM wrapper for Google Gemini without problematic caching"""
#     
#     def __init__(self, api_key: str, **kwargs):
#         super().__init__(**kwargs)
#         self.api_key = api_key
#         self.model_name = "gemini-1.5-flash"
#     
#     def _call(
#         self,
#         prompt: str,
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> str:
#         url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
#         
#         data = {
#             "contents": [{"parts": [{"text": prompt}]}],
#             "generationConfig": {
#                 "temperature": 0.3,
#                 "maxOutputTokens": 512,
#                 "topK": 10,
#                 "topP": 0.8
#             }
#         }
#         
#         try:
#             response = requests.post(url, json=data, timeout=15, headers={"Content-Type": "application/json"})
#             if response.status_code == 200:
#                 result = response.json()
#                 if "candidates" in result and result["candidates"]:
#                     return result["candidates"][0]["content"]["parts"][0]["text"]
#             return "Error generating response"
#         except Exception as e:
#             return f"Error: {str(e)}"
#     
#     @property
#     def _llm_type(self) -> str:
#         return "gemini"

class LocalLlamaLLM(LLM):
    """LLM wrapper for local Llama model using OpenAI client"""
    
    client: openai.OpenAI = None
    model_name: str = "Llama-3.3-70b-instruct"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = openai.OpenAI(
            base_url="http://dgx5.humanbrain.in:8999/v1",
            api_key="dummy"
        )
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            messages = [{"role": "user", "content": prompt}]
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_tokens=1024,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    @property
    def _llm_type(self) -> str:
        return "local_llama"

class ImprovedWebSearchTool:
    """Improved web search tool with multiple fallback options"""
    
    def __init__(self, max_retries: int = 2, base_delay: int = 5):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.last_search_time = 0
        self.min_search_interval = 30  # Minimum seconds between searches
        
        # Try to initialize DuckDuckGo
        try:
            self.ddg_search = DuckDuckGoSearchRun(max_results=3)
            self.ddg_available = True
            print("✓ DuckDuckGo search initialized")
        except Exception as e:
            print(f"⚠ DuckDuckGo initialization failed: {e}")
            self.ddg_search = None
            self.ddg_available = False
    
    def _wait_for_rate_limit(self):
        """Ensure we don't hit rate limits"""
        import time
        current_time = time.time()
        time_since_last = current_time - self.last_search_time
        
        if time_since_last < self.min_search_interval:
            wait_time = self.min_search_interval - time_since_last
            print(f"⏱ Waiting {wait_time:.1f}s to avoid rate limits...")
            time.sleep(wait_time)
        
        self.last_search_time = time.time()
    
    def _try_ddg_search(self, query: str) -> Optional[str]:
        """Try DuckDuckGo search with retry logic"""
        import time
        if not self.ddg_available:
            return None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    # Exponential backoff
                    delay = self.base_delay * (2 ** (attempt - 1))
                    print(f"  Retrying in {delay}s... (attempt {attempt + 1})")
                    time.sleep(delay)
                
                self._wait_for_rate_limit()
                result = self.ddg_search.run(query)
                
                if result and len(result.strip()) > 20:  # Valid result
                    print("✓ DuckDuckGo search successful")
                    return result
                
            except Exception as e:
                error_msg = str(e).lower()
                if "ratelimit" in error_msg or "202" in error_msg:
                    print(f"⚠ Rate limited (attempt {attempt + 1})")
                    if attempt == self.max_retries:
                        print("✗ DuckDuckGo rate limit exceeded")
                else:
                    print(f"⚠ DuckDuckGo error: {e}")
                
                if attempt == self.max_retries:
                    break
        
        return None
    
    def _try_instant_answer_api(self, query: str) -> Optional[str]:
        """Try DuckDuckGo instant answer API as fallback"""
        try:
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                result_parts = []
                
                if data.get('Abstract'):
                    result_parts.append(f"Overview: {data['Abstract']}")
                
                if data.get('Definition'):
                    result_parts.append(f"Definition: {data['Definition']}")
                
                # Related topics
                if data.get('RelatedTopics'):
                    topics = []
                    for topic in data['RelatedTopics'][:2]:
                        if isinstance(topic, dict) and topic.get('Text'):
                            topics.append(topic['Text'][:100])
                    if topics:
                        result_parts.append(f"Related: {' | '.join(topics)}")
                
                if result_parts:
                    result = ' '.join(result_parts)
                    print("✓ Instant Answer API successful")
                    return result
            
        except Exception as e:
            print(f"⚠ Instant Answer API failed: {e}")
        
        return None
    
    def _generate_fallback_response(self, query: str, llm) -> str:
        """Generate response using local LLM when web search fails"""
        try:
            # Extract brain region from query
            region = "the specified brain region"
            query_lower = query.lower()
            
            brain_regions = [
                "hippocampus", "amygdala", "prefrontal cortex", "cerebellum", "cerebellar hemispheres",
                "thalamus", "hypothalamus", "brainstem", "parietal lobe",
                "temporal lobe", "frontal lobe", "occipital lobe", "insula", "corpus callosum"
            ]
            
            for brain_region in brain_regions:
                if brain_region in query_lower:
                    region = brain_region
                    break
            
            # Enhanced prompt for web search fallback - more detailed and research-focused
            fallback_prompt = f"""
            As a neuroscience expert, provide comprehensive information about {region}. Include:
            
            **ANATOMICAL DETAILS:**
            - Precise location and boundaries
            - Cellular organization and layers
            - Subregions and subdivisions
            
            **FUNCTIONAL ROLES:**
            - Primary functions and mechanisms
            - Behavioral contributions
            - Cognitive and motor roles
            
            **NEURAL CONNECTIVITY:**
            - Major input and output pathways
            - Network connections
            - Neurotransmitter systems involved
            
            **CLINICAL & RESEARCH INSIGHTS:**
            - Associated disorders and symptoms
            - Recent research findings
            - Therapeutic implications
            - Developmental aspects
            
            Query context: {query}
            
            Provide detailed, evidence-based information (300-400 words). Use scientific terminology and include specific details that demonstrate current understanding.
            """
            
            response = llm._call(fallback_prompt)
            return f"🌐 Enhanced AI Analysis (web search unavailable - using comprehensive neuroscience knowledge):\n\n{response}"
            
        except Exception as e:
            return f"Unable to retrieve information: {str(e)}"
    
    def search(self, query: str, llm=None) -> str:
        """Main search function with multiple fallbacks"""
        print(f"🔍 Searching: {query}")
        
        # Method 1: Try DuckDuckGo search with retries
        result = self._try_ddg_search(query)
        if result:
            return result
        
        # Method 2: Try instant answer API
        result = self._try_instant_answer_api(query)
        if result:
            return result
        
        # Method 3: Fallback to local AI knowledge
        print("⚠ All web search methods failed, using local AI knowledge")
        if llm:
            return self._generate_fallback_response(query, llm)
        else:
            return "Web search is currently unavailable. Please try again later or use fast mode for AI-based responses."

class UltraFastBrainAssistant:
    def __init__(self, use_web_search: bool = True):
        # self.api_key = api_key
        # self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        self.client = openai.OpenAI(
            base_url="http://dgx5.humanbrain.in:8999/v1",
            api_key="dummy"
        )
        self.model_name = "Llama-3.3-70b-instruct"
        self.current_region = None
        self.region_cache: Dict[str, str] = {}
        self.use_web_search = use_web_search
        
        # Initialize improved web search components if web search is enabled
        if self.use_web_search:
            try:
                self.llm = LocalLlamaLLM()
                self.web_search = ImprovedWebSearchTool()
                
                # Create a wrapper function for the search tool
                def search_wrapper(query: str) -> str:
                    return self.web_search.search(query, self.llm)
                
                self.tools = [
                    Tool(
                        name="WebSearch",
                        func=search_wrapper,
                        description="Search the web for current brain region information, neuroscience research, and medical information. Automatically handles rate limits and provides fallback responses."
                    )
                ]
                
                # Configure agent with better settings
                self.agent = initialize_agent(
                    tools=self.tools,
                    llm=self.llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=False,  # Set to True for debugging
                    max_iterations=3,  # Reduced from 5 to prevent long loops
                    handle_parsing_errors=True,
                    max_execution_time=60,  # Increased timeout
                    early_stopping_method="generate"  # Stop early if goal is reached
                )
                print("⚡ Brain AI Assistant Ready!")
            except Exception as e:
                print(f"Warning: Web search failed to initialize: {e}")
                import traceback
                traceback.print_exc()
                self.use_web_search = False
                print("⚡ Brain AI Assistant Ready! (Web search disabled)")
        else:
            print("⚡ Brain AI Assistant Ready!")
    
    # async def _async_gemini_request(self, prompt: str) -> str:
    #     """Async request for maximum speed"""
    #     data = {
    #         "contents": [{"parts": [{"text": prompt}]}],
    #         "generationConfig": {
    #             "temperature": 0.3,
    #             "maxOutputTokens": 256,
    #             "candidateCount": 1
    #         }
    #     }
    #     
    #     url = f"{self.base_url}?key={self.api_key}"
    #     
    #     try:
    #         async with aiohttp.ClientSession() as session:
    #             async with session.post(url, json=data, timeout=10, headers={"Content-Type": "application/json"}) as response:
    #                 if response.status == 200:
    #                     result = await response.json()
    #                     return result["candidates"][0]["content"]["parts"][0]["text"]
    #     except Exception as e:
    #         print(f"Async request error: {e}")
    #     return "Error retrieving information"
    
    async def _async_llama_request(self, prompt: str) -> str:
        """Async request for maximum speed using local Llama"""
        try:
            messages = [{"role": "user", "content": prompt}]
            # Note: The openai library doesn't support async for custom endpoints
            # So we'll use sync in an async wrapper
            completion = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0,
                    max_tokens=256,
                    stream=False
                )
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Async request error: {e}")
            return "Error retrieving information"
    
    # def _sync_gemini_request(self, prompt: str) -> str:
    #     """Synchronous request"""
    #     data = {
    #         "contents": [{"parts": [{"text": prompt}]}],
    #         "generationConfig": {
    #             "temperature": 0.3,
    #             "maxOutputTokens": 512
    #         }
    #     }
    #     
    #     url = f"{self.base_url}?key={self.api_key}"
    #     
    #     try:
    #         response = requests.post(url, json=data, timeout=15, headers={"Content-Type": "application/json"})
    #         if response.status_code == 200:
    #             result = response.json()
    #             if "candidates" in result and result["candidates"]:
    #                 return result["candidates"][0]["content"]["parts"][0]["text"]
    #         return "Error generating response"
    #     except Exception as e:
    #         return f"Error: {str(e)}"
    
    def _sync_llama_request(self, prompt: str) -> str:
        """Synchronous request using local Llama"""
        try:
            messages = [{"role": "user", "content": prompt}]
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_tokens=512,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def validate_brain_region(self, region_name: str) -> bool:
        """Check if the input is a valid brain region"""
        
        # Simple heuristic check first for obvious non-brain terms
        non_brain_terms = {'apple', 'pizza', 'car', 'computer', 'table', 'phone', 'book', 'chair', 'door', 'window'}
        if region_name.lower() in non_brain_terms:
            return False
            
        validation_prompt = f"Is '{region_name}' a brain region, brain structure, or part of the brain? Answer only 'yes' or 'no'."
        
        # # Use direct API call for validation
        # data = {
        #     "contents": [{"parts": [{"text": validation_prompt}]}],
        #     "generationConfig": {
        #         "temperature": 0.1,
        #         "maxOutputTokens": 10
        #     }
        # }
        # 
        # url = f"{self.base_url}?key={self.api_key}"
        
        try:
            messages = [{"role": "user", "content": validation_prompt}]
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=10,
                stream=False
            )
            text = completion.choices[0].message.content.strip()
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
                # Use enhanced web search with direct approach to avoid parsing issues
                print("🔍 Searching with enhanced web search...")
                query = f"Provide comprehensive information about the {region_name} brain region: anatomy, functions, neural connections, and clinical significance. Include recent research findings if available."
                
                # Use direct web search approach to avoid LangChain agent parsing issues
                info = self.web_search.search(query, self.llm)
                
            elif mode == "ultra":
                # Use async for ultra-fast mode
                prompt = f"List key facts about {region_name} brain region: location, function, connections, clinical significance (max 100 words)"
                try:
                    loop = asyncio.get_event_loop()
                    info = loop.run_until_complete(self._async_llama_request(prompt))
                except:
                    # Fallback to sync if async fails
                    info = self._sync_llama_request(prompt)
                
            else:  # fast mode (default)
                # Use sync request
                prompt = f"""Provide a concise overview of the {region_name} brain region:
1. Location: Where in the brain?
2. Function: What does it do?
3. Connections: Key neural pathways
4. Clinical: Related conditions
Keep it under 200 words but be informative."""
                info = self._sync_llama_request(prompt)
            
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
                query = f"Answer this specific question about the {self.current_region} brain region: {question}"
                try:
                    # Use direct web search for questions to avoid agent complexity
                    return self.web_search.search(query, self.llm)
                except Exception as e:
                    # Fallback to direct AI if web search fails
                    prompt = f"About the {self.current_region} brain region, answer this question: {question}"
                    return f"Web search unavailable, using AI knowledge: {self._sync_llama_request(prompt)}"
            else:
                prompt = f"About the {self.current_region} brain region, answer concisely: {question}"
                return self._sync_llama_request(prompt)
        except Exception as e:
            return f"Error: {str(e)}"

def main():
    # # Get API key
    # api_key = "AIzaSyAlEwiCemb1kpclHSyb6z7RUgqSJoHUzvI"
    # if not api_key:
    #     api_key = os.environ.get("GOOGLE_API_KEY", "")
    # if not api_key:
    #     api_key = input("Enter Google Gemini API key: ").strip()
    
    print("\n" + "="*60)
    print("⚡ Brain AI Assistant")
    print("="*60)
    
    # Always enable web search capability (users can choose per query)
    assistant = UltraFastBrainAssistant(use_web_search=True)
    
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
                print("1. Fast mode (concise overview)")
                print("2. Detailed mode (comprehensive analysis)")
                print("3. Ultra-fast mode (key facts only)")
                
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
                    print("2. Enhanced web search answer")
                    
                    ask_mode = input("Enter choice (1/2, default=1): ").strip() or "1"
                    
                    print(f"\nWhat is your question about {assistant.current_region}?")
                    question = input("Question: ").strip()
                    
                    if question:
                        use_web = ask_mode == "2"
                        if use_web:
                            print("🔍 Searching with enhanced web search...")
                        
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
        import openai
        import asyncio
        from langchain_community.tools import DuckDuckGoSearchRun
        main()
    except ImportError:
        print("Installing required packages...")
        os.system("pip install openai asyncio langchain langchain-community duckduckgo-search")
        print("Please restart the program.")
        exit()