// Global variables
let currentBrainRegion = null;
let isWaitingForResponse = false;

// DOM elements
const chatMessages = document.getElementById('chatMessages');
const questionInput = document.getElementById('questionInput');
const askBtn = document.getElementById('askBtn');
const customRegionInput = document.getElementById('customRegion');
const searchBtn = document.getElementById('searchBtn');
const currentRegionSpan = document.getElementById('currentRegion');
const useWebSearchCheckbox = document.getElementById('useWebSearch');
const chatbotSection = document.getElementById('chatbotSection');
const closeChat = document.getElementById('closeChat');
const brainSection = document.querySelector('.brain-section');

// Initialize event listeners
document.addEventListener('DOMContentLoaded', () => {
    // Brain region click handlers
    const brainRegions = document.querySelectorAll('.brain-region');
    brainRegions.forEach(region => {
        region.addEventListener('click', () => {
            const regionName = region.getAttribute('data-region');
            selectBrainRegion(regionName);
            highlightRegion(regionName);
        });
    });

    // Custom region search
    searchBtn.addEventListener('click', handleCustomRegionSearch);
    customRegionInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleCustomRegionSearch();
        }
    });

    // Question handling
    askBtn.addEventListener('click', handleAskQuestion);
    questionInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !askBtn.disabled) {
            handleAskQuestion();
        }
    });

    // Close chat button
    closeChat.addEventListener('click', () => {
        closeChatbot();
    });

    // Show/hide web search option when region is selected
    questionInput.addEventListener('input', () => {
        if (currentBrainRegion && questionInput.value.trim()) {
            document.querySelector('.web-search-toggle').style.display = 'block';
        }
    });
});

function handleCustomRegionSearch() {
    const regionName = customRegionInput.value.trim();
    if (regionName) {
        selectBrainRegion(regionName);
    }
}

function showChatbot() {
    chatbotSection.style.display = 'block';
    brainSection.classList.add('with-chat');
    brainSection.classList.remove('full-width');
    
    // Clear previous messages when opening for a new region
    chatMessages.innerHTML = '';
    
    // Add welcome message
    addMessage(`Welcome! I'm ready to help you learn about the ${currentBrainRegion}.`, 'bot');
}

function closeChatbot() {
    chatbotSection.style.display = 'none';
    brainSection.classList.remove('with-chat');
    brainSection.classList.add('full-width');
    currentBrainRegion = null;
    currentRegionSpan.textContent = '';
    askBtn.disabled = true;
    
    // Clear highlighted regions
    document.querySelectorAll('.brain-region').forEach(region => {
        region.classList.remove('selected');
    });
}

async function selectBrainRegion(regionName) {
    if (isWaitingForResponse) return;

    // Show chatbot
    showChatbot();
    
    // Update UI
    currentBrainRegion = regionName;
    currentRegionSpan.textContent = `Current: ${regionName}`;
    askBtn.disabled = false;
    document.querySelector('.web-search-toggle').style.display = 'none';
    
    // Add user message
    addMessage(`Tell me about the ${regionName}`, 'user');
    
    // Show loading
    const loadingMsg = addMessage('', 'bot');
    loadingMsg.innerHTML = '<div class="loading"></div> Analyzing brain region...';
    
    // Get selected mode
    const mode = document.querySelector('input[name="mode"]:checked').value;
    
    isWaitingForResponse = true;
    
    try {
        const response = await fetch('/api/brain-region', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                region: regionName,
                mode: mode
            })
        });
        
        const data = await response.json();
        
        // Remove loading message
        loadingMsg.remove();
        
        if (data.success) {
            addMessage(data.message, 'bot');
        } else {
            addMessage(`Error: ${data.message}`, 'bot');
            currentBrainRegion = null;
            currentRegionSpan.textContent = '';
            askBtn.disabled = true;
            
            // Close chatbot if invalid region
            setTimeout(() => {
                closeChatbot();
            }, 3000);
        }
    } catch (error) {
        loadingMsg.remove();
        addMessage(`Error: ${error.message}`, 'bot');
    } finally {
        isWaitingForResponse = false;
    }
}

async function handleAskQuestion() {
    const question = questionInput.value.trim();
    if (!question || !currentBrainRegion || isWaitingForResponse) return;
    
    // Add user message
    addMessage(question, 'user');
    
    // Clear input
    questionInput.value = '';
    
    // Show loading
    const loadingMsg = addMessage('', 'bot');
    loadingMsg.innerHTML = '<div class="loading"></div> Thinking...';
    
    const useWeb = useWebSearchCheckbox.checked;
    
    isWaitingForResponse = true;
    
    try {
        const response = await fetch('/api/ask-question', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                use_web: useWeb
            })
        });
        
        const data = await response.json();
        
        // Remove loading message
        loadingMsg.remove();
        
        if (data.success) {
            addMessage(data.message, 'bot');
        } else {
            addMessage(`Error: ${data.message}`, 'bot');
        }
    } catch (error) {
        loadingMsg.remove();
        addMessage(`Error: ${error.message}`, 'bot');
    } finally {
        isWaitingForResponse = false;
    }
}

function addMessage(text, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    const messageP = document.createElement('p');
    messageP.textContent = text;
    
    messageDiv.appendChild(messageP);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageP;
}

// Highlight selected region
function highlightRegion(regionName) {
    // Remove previous highlights
    document.querySelectorAll('.brain-region').forEach(region => {
        region.classList.remove('selected');
    });
    
    // Add highlight to selected region
    document.querySelectorAll('.brain-region').forEach(region => {
        if (region.getAttribute('data-region') === regionName) {
            region.classList.add('selected');
        }
    });
}