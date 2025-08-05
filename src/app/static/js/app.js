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
// Web search checkbox removed - automatically determined by mode
const chatbotSection = document.getElementById('chatbotSection');
const closeChat = document.getElementById('closeChat');
const brainSection = document.querySelector('.brain-section');
const historyBtn = document.getElementById('historyBtn');
const clearBtn = document.getElementById('clearBtn');
const historyModal = document.getElementById('historyModal');
const closeHistoryModal = document.getElementById('closeHistoryModal');
const closeHistoryBtn = document.getElementById('closeHistoryBtn');
const historyList = document.getElementById('historyList');
const historyStats = document.getElementById('historyStats');
const downloadHistory = document.getElementById('downloadHistory');

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

    // History button
    historyBtn.addEventListener('click', () => {
        showChatHistory();
    });

    // Clear history button
    clearBtn.addEventListener('click', () => {
        clearChatHistory();
    });

    // History modal close handlers
    closeHistoryModal.addEventListener('click', () => {
        historyModal.style.display = 'none';
    });
    
    closeHistoryBtn.addEventListener('click', () => {
        historyModal.style.display = 'none';
    });

    // Download history button
    downloadHistory.addEventListener('click', () => {
        downloadChatHistory();
    });

    // Close modal when clicking outside
    historyModal.addEventListener('click', (e) => {
        if (e.target === historyModal) {
            historyModal.style.display = 'none';
        }
    });

    // Enable/disable ask button based on input
    questionInput.addEventListener('input', () => {
        if (currentBrainRegion && questionInput.value.trim()) {
            askBtn.disabled = false;
        } else {
            askBtn.disabled = true;
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
    // Web search is now automatically determined by mode
    
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
    
    // Get selected mode to determine if web search should be used
    const mode = document.querySelector('input[name="mode"]:checked').value;
    const useWeb = (mode === 'web'); // Automatically use web for 'Detailed (Web)' mode
    
    isWaitingForResponse = true;
    
    try {
        const response = await fetch('/api/ask-question', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                use_web: useWeb,
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

// Chat History Functions
async function showChatHistory() {
    try {
        const response = await fetch('/api/chat-history');
        const data = await response.json();
        
        if (data.success) {
            displayChatHistory(data.chat_history, data.current_region, data.current_mode);
            historyModal.style.display = 'block';
        } else {
            alert('Error loading chat history: ' + data.message);
        }
    } catch (error) {
        alert('Error loading chat history: ' + error.message);
    }
}

function displayChatHistory(history, currentRegion, currentMode) {
    // Update stats
    const totalMessages = history.length;
    const userQuestions = history.filter(msg => msg.type === 'user_query').length;
    const regions = [...new Set(history.filter(msg => msg.region).map(msg => msg.region))];
    
    historyStats.innerHTML = `
        <strong>Chat Statistics:</strong> 
        ${totalMessages} total messages, 
        ${userQuestions} questions asked, 
        ${regions.length} regions explored
        ${currentRegion ? `<br><strong>Current:</strong> ${currentRegion} (${currentMode || 'fast'} mode)` : ''}
    `;
    
    // Clear previous history
    historyList.innerHTML = '';
    
    if (history.length === 0) {
        historyList.innerHTML = '<div class="no-history">No chat history yet. Start exploring brain regions!</div>';
        return;
    }
    
    // Group messages by region for better organization
    const groupedHistory = {};
    history.forEach((msg, index) => {
        const region = msg.region || 'General';
        if (!groupedHistory[region]) {
            groupedHistory[region] = [];
        }
        groupedHistory[region].push({...msg, index});
    });
    
    // Display grouped history
    Object.keys(groupedHistory).forEach(region => {
        const regionDiv = document.createElement('div');
        regionDiv.className = 'history-region-group';
        
        const regionHeader = document.createElement('h4');
        regionHeader.className = 'history-region-header';
        regionHeader.textContent = region;
        regionDiv.appendChild(regionHeader);
        
        groupedHistory[region].forEach(msg => {
            const historyItem = document.createElement('div');
            historyItem.className = `history-item ${msg.type}`;
            
            const timestamp = new Date(msg.timestamp).toLocaleString();
            const typeLabel = {
                'user_query': '❓ Question',
                'region_info': '🧠 Region Info', 
                'assistant_response': '🤖 Answer'
            }[msg.type] || '💬 Message';
            
            historyItem.innerHTML = `
                <div class="history-item-header">
                    <span class="history-type">${typeLabel}</span>
                    <span class="history-timestamp">${timestamp}</span>
                </div>
                <div class="history-content">${msg.content}</div>
            `;
            
            regionDiv.appendChild(historyItem);
        });
        
        historyList.appendChild(regionDiv);
    });
}

async function clearChatHistory() {
    if (!confirm('Are you sure you want to clear all chat history? This cannot be undone.')) {
        return;
    }
    
    try {
        const response = await fetch('/api/chat-history', {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (data.success) {
            alert('Chat history cleared successfully!');
            // Close chatbot and reset state
            closeChatbot();
            // Close history modal if open
            historyModal.style.display = 'none';
        } else {
            alert('Error clearing chat history: ' + data.message);
        }
    } catch (error) {
        alert('Error clearing chat history: ' + error.message);
    }
}

async function downloadChatHistory() {
    try {
        const response = await fetch('/api/chat-history');
        const data = await response.json();
        
        if (data.success) {
            const historyText = formatHistoryForDownload(data.chat_history);
            const blob = new Blob([historyText], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = `brain-assistant-history-${new Date().toISOString().split('T')[0]}.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } else {
            alert('Error downloading chat history: ' + data.message);
        }
    } catch (error) {
        alert('Error downloading chat history: ' + error.message);
    }
}

function formatHistoryForDownload(history) {
    let text = 'Brain Assistant Chat History\n';
    text += '================================\n';
    text += `Generated on: ${new Date().toLocaleString()}\n\n`;
    
    let currentRegion = '';
    
    history.forEach((msg, index) => {
        const timestamp = new Date(msg.timestamp).toLocaleString();
        
        // Add region header when it changes
        if (msg.region && msg.region !== currentRegion) {
            currentRegion = msg.region;
            text += `\n--- ${currentRegion.toUpperCase()} ---\n`;
        }
        
        const typeLabel = {
            'user_query': 'USER',
            'region_info': 'ASSISTANT (Region Info)',
            'assistant_response': 'ASSISTANT'
        }[msg.type] || 'MESSAGE';
        
        text += `[${timestamp}] ${typeLabel}:\n${msg.content}\n\n`;
    });
    
    return text;
}