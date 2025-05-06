/**
 * Chat handler for the XAI Chat Interface
 * Manages message display, user input, and interaction with the API
 */

class XAIChatHandler {
    constructor(config, apiService) {
        this.config = config;
        this.apiService = apiService;
        this.messagesContainer = document.getElementById('messages');
        this.userInput = document.getElementById('user-input');
        this.sendButton = document.getElementById('send-btn');
        this.clearButton = document.getElementById('clear-chat');
        this.processingIndicator = document.getElementById('processing-indicator');

        this.messageHistory = [];
        this.currentMessageId = null;

        // Initialize event listeners
        this.initEventListeners();
    }

    /**
     * Initialize event listeners for chat functionality
     */
    initEventListeners() {
        // Send message on button click
        this.sendButton.addEventListener('click', () => {
            this.sendMessage();
        });

        // Send message on Enter key (but allow Shift+Enter for new lines)
        this.userInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                this.sendMessage();
            }
        });

        // Clear chat on button click
        this.clearButton.addEventListener('click', () => {
            this.clearChat();
        });

        // Auto-resize textarea as user types
        this.userInput.addEventListener('input', () => {
            this.userInput.style.height = 'auto';
            this.userInput.style.height = (this.userInput.scrollHeight) + 'px';
        });
    }

    /**
     * Send user message to the API
     */
    async sendMessage() {
        const message = this.userInput.value.trim();

        if (!message) return;

        // Clear input and reset height
        this.userInput.value = '';
        this.userInput.style.height = 'auto';

        // Add user message to chat
        this.addUserMessage(message);

        // Show processing indicator
        this.showProcessingIndicator();

        try {
            // Call API service (or use simulation in development)
            let response;
            if (location.hostname === 'localhost' && !this.config.api.baseUrl.includes('localhost')) {
                // Use simulated response for local development without backend
                response = await this.apiService.simulateResponse(message);
            } else {
                // Call real API
                response = await this.apiService.sendMessage(message);
            }

            // Add AI response to chat
            this.addAIMessage(response);

            // Set current message ID for visualization
            this.currentMessageId = response.id;

            // Trigger visualization update if we have visualization data
            if (response.visualization) {
                const vizEvent = new CustomEvent('visualization-update', {
                    detail: {
                        data: response.visualization.data,
                        type: response.visualization.type || 'reasoning-tree'
                    }
                });
                document.dispatchEvent(vizEvent);
            }
        } catch (error) {
            // Add error message to chat
            this.addSystemMessage(`Error: ${error.message}`);
        } finally {
            // Hide processing indicator
            this.hideProcessingIndicator();
        }
    }

    /**
     * Add a user message to the chat
     * @param {string} message - User message text
     */
    addUserMessage(message) {
        const messageId = `msg-${Date.now()}`;
        const timestamp = new Date().toLocaleTimeString();

        const messageHtml = `
            <div class="chat-message user mb-4 new-message">
                <div class="flex items-start">
                    <div class="message-content bg-indigo-600 text-white rounded-lg p-3 ml-3 max-w-3xl">
                        <p>${this.formatMessage(message)}</p>
                    </div>
                    <div class="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center ml-3 flex-shrink-0">
                        <i class="fas fa-user"></i>
                    </div>
                </div>
                <div class="message-time text-right mr-12">${timestamp}</div>
            </div>
        `;

        this.messagesContainer.insertAdjacentHTML('beforeend', messageHtml);

        // Store in message history
        this.messageHistory.push({
            id: messageId,
            type: 'user',
            text: message,
            timestamp: new Date().toISOString()
        });

        // Scroll to bottom
        this.scrollToBottom();

        // Limit message history if needed
        this.limitMessageHistory();
    }

    /**
     * Add an AI message to the chat
     * @param {object} response - API response object
     */
    addAIMessage(response) {
        const timestamp = new Date().toLocaleTimeString();

        const messageHtml = `
            <div class="chat-message ai mb-4 new-message">
                <div class="flex items-start">
                    <div class="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center text-white mr-3 flex-shrink-0">
                        <i class="fas fa-robot"></i>
                    </div>
                    <div class="message-content bg-indigo-100 rounded-lg p-3 max-w-3xl">
                        <p>${this.formatMessage(response.text)}</p>
                    </div>
                </div>
                <div class="message-time ml-12">${timestamp}</div>
            </div>
        `;

        this.messagesContainer.insertAdjacentHTML('beforeend', messageHtml);

        // Store in message history
        this.messageHistory.push({
            id: response.id,
            type: 'ai',
            text: response.text,
            visualization: response.visualization,
            timestamp: new Date().toISOString()
        });

        // Scroll to bottom
        this.scrollToBottom();

        // Limit message history if needed
        this.limitMessageHistory();
    }

    /**
     * Add a system message to the chat
     * @param {string} message - System message text
     */
    addSystemMessage(message) {
        const timestamp = new Date().toLocaleTimeString();

        const messageHtml = `
            <div class="chat-message system mb-4 new-message">
                <div class="flex items-start">
                    <div class="w-8 h-8 rounded-full bg-gray-500 flex items-center justify-center text-white mr-3 flex-shrink-0">
                        <i class="fas fa-info"></i>
                    </div>
                    <div class="message-content bg-gray-100 rounded-lg p-3 max-w-3xl">
                        <p>${this.formatMessage(message)}</p>
                    </div>
                </div>
                <div class="message-time ml-12">${timestamp}</div>
            </div>
        `;

        this.messagesContainer.insertAdjacentHTML('beforeend', messageHtml);

        // Store in message history
        this.messageHistory.push({
            id: `system-${Date.now()}`,
            type: 'system',
            text: message,
            timestamp: new Date().toISOString()
        });

        // Scroll to bottom
        this.scrollToBottom();

        // Limit message history if needed
        this.limitMessageHistory();
    }

    /**
     * Format message text with HTML for display
     * @param {string} message - Message text
     * @returns {string} - Formatted HTML
     */
    formatMessage(message) {
        // Function to escape HTML to prevent XSS
        const escapeHtml = (text) => {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        };

        // Escape HTML
        let formattedMessage = escapeHtml(message);

        // Format code blocks (```code```)
        formattedMessage = formattedMessage.replace(
            /```(\w*)\n([\s\S]*?)```/g,
            '<pre><code class="language-$1">$2</code></pre>'
        );

        // Format inline code (`code`)
        formattedMessage = formattedMessage.replace(
            /`([^`]+)`/g,
            '<code>$1</code>'
        );

        // Convert URLs to links
        formattedMessage = formattedMessage.replace(
            /(https?:\/\/[^\s]+)/g,
            '<a href="$1" class="text-indigo-600 hover:underline" target="_blank">$1</a>'
        );

        // Convert line breaks to <br>
        formattedMessage = formattedMessage.replace(/\n/g, '<br>');

        return formattedMessage;
    }

    /**
     * Show processing indicator
     */
    showProcessingIndicator() {
        this.processingIndicator.classList.remove('hidden');
        this.sendButton.disabled = true;
        this.sendButton.classList.add('opacity-50');
    }

    /**
     * Hide processing indicator
     */
    hideProcessingIndicator() {
        this.processingIndicator.classList.add('hidden');
        this.sendButton.disabled = false;
        this.sendButton.classList.remove('opacity-50');
    }

    /**
     * Scroll messages container to bottom
     */
    scrollToBottom() {
        if (this.config.ui.autoScrollChat) {
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        }
    }

    /**
     * Limit message history to maximum number of messages
     */
    limitMessageHistory() {
        const maxMessages = this.config.ui.messageHistory;

        if (this.messageHistory.length > maxMessages) {
            // Remove oldest messages from history
            this.messageHistory = this.messageHistory.slice(-maxMessages);

            // Update the DOM to reflect message history
            while (this.messagesContainer.children.length > maxMessages) {
                this.messagesContainer.removeChild(this.messagesContainer.firstChild);
            }
        }
    }

    /**
     * Clear the chat history
     */
    clearChat() {
        // Clear messages container
        this.messagesContainer.innerHTML = '';

        // Clear message history
        this.messageHistory = [];

        // Add welcome message
        this.addSystemMessage("Welcome to XAI Chat! I'm here to demonstrate explainable AI reasoning. Ask me a question, and I'll show you my reasoning process.");
    }

    /**
     * Get the current message ID
     * @returns {string} - Current message ID
     */
    getCurrentMessageId() {
        return this.currentMessageId;
    }

    /**
     * Get the message history
     * @returns {Array} - Message history
     */
    getMessageHistory() {
        return this.messageHistory;
    }
}

// Create global chat handler instance
const chatHandler = new XAIChatHandler(XAIConfig, apiService);
