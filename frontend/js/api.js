/**
 * API service for the XAI Chat Interface
 * Handles communication with the backend server
 */

class XAIApiService {
    constructor(config) {
        this.baseUrl = config.api.baseUrl;
        this.endpoints = config.api.endpoints;
        this.timeout = config.api.timeout;
    }

    /**
     * Send a message to the chat API
     * @param {string} message - User message
     * @param {object} settings - Optional settings to use for this request
     * @returns {Promise} - Response from the server
     */
    async sendMessage(message, settings = null) {
        try {
            const endpoint = `${this.baseUrl}${this.endpoints.chat}`;

            const payload = {
                message: message,
                settings: settings || {
                    model: XAIConfig.model,
                    cgrt: XAIConfig.cgrt,
                    counterfactual: XAIConfig.counterfactual,
                    knowledgeGraph: XAIConfig.knowledgeGraph
                }
            };

            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), this.timeout);

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`API error: ${response.status} - ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error sending message:', error);

            if (error.name === 'AbortError') {
                throw new Error('Request timed out. The server took too long to respond.');
            }

            throw error;
        }
    }

    /**
     * Get visualization data
     * @param {string} vizType - Type of visualization
     * @param {string} messageId - ID of the message to visualize
     * @param {object} options - Optional visualization parameters
     * @returns {Promise} - Visualization data
     */
    async getVisualization(vizType, messageId, options = {}) {
        try {
            const endpoint = `${this.baseUrl}${this.endpoints.visualization}`;

            const payload = {
                vizType: vizType,
                messageId: messageId,
                options: {
                    importanceThreshold: XAIConfig.visualization.importanceThreshold,
                    ...options
                }
            };

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.status} - ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error getting visualization:', error);
            throw error;
        }
    }

    /**
     * Get counterfactual analysis
     * @param {string} messageId - ID of the message to analyze
     * @param {object} options - Optional counterfactual parameters
     * @returns {Promise} - Counterfactual analysis data
     */
    async getCounterfactual(messageId, options = {}) {
        try {
            const endpoint = `${this.baseUrl}${this.endpoints.counterfactual}`;

            const payload = {
                messageId: messageId,
                options: {
                    topKTokens: XAIConfig.counterfactual.topKTokens,
                    attentionThreshold: XAIConfig.counterfactual.attentionThreshold,
                    maxCounterfactuals: XAIConfig.counterfactual.maxCounterfactuals,
                    ...options
                }
            };

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.status} - ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error getting counterfactual analysis:', error);
            throw error;
        }
    }

    /**
     * Get knowledge graph validation
     * @param {string} messageId - ID of the message to validate
     * @param {object} options - Optional knowledge graph parameters
     * @returns {Promise} - Knowledge graph validation data
     */
    async getKnowledgeGraph(messageId, options = {}) {
        try {
            const endpoint = `${this.baseUrl}${this.endpoints.knowledgeGraph}`;

            const payload = {
                messageId: messageId,
                options: {
                    useLocalModel: XAIConfig.knowledgeGraph.useLocalModel,
                    similarityThreshold: XAIConfig.knowledgeGraph.similarityThreshold,
                    ...options
                }
            };

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.status} - ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error getting knowledge graph validation:', error);
            throw error;
        }
    }

    /**
     * Update system settings
     * @param {object} settings - New settings to apply
     * @returns {Promise} - Response from the server
     */
    async updateSettings(settings) {
        try {
            const endpoint = `${this.baseUrl}${this.endpoints.settings}`;

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(settings)
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.status} - ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error updating settings:', error);
            throw error;
        }
    }

    /**
     * Helper function to create a simulated response for development/testing
     * @param {string} message - User message
     * @returns {object} - Simulated response
     */
    simulateResponse(message) {
        // For development/testing when backend is not available
        return new Promise((resolve) => {
            setTimeout(() => {
                resolve({
                    id: `msg-${Date.now()}`,
                    text: `This is a simulated response to: "${message}"`,
                    visualization: {
                        type: 'reasoning-tree',
                        data: {
                            nodes: [
                                { id: 'root', text: 'Initial thought', importance: 0.8 },
                                { id: 'node1', text: 'Consideration 1', importance: 0.6 },
                                { id: 'node2', text: 'Consideration 2', importance: 0.7 },
                                { id: 'node3', text: 'Conclusion', importance: 0.9 }
                            ],
                            edges: [
                                { from: 'root', to: 'node1' },
                                { from: 'root', to: 'node2' },
                                { from: 'node1', to: 'node3' },
                                { from: 'node2', to: 'node3' }
                            ]
                        }
                    }
                });
            }, 1000);
        });
    }
}

// Create global API service instance
const apiService = new XAIApiService(XAIConfig);
