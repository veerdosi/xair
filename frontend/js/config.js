/**
 * Configuration for the XAI Chat Interface
 */

const XAIConfig = {
    // Default API settings
    api: {
        baseUrl: 'http://localhost:5000/api',
        endpoints: {
            chat: '/chat',
            visualization: '/visualization',
            counterfactual: '/counterfactual',
            knowledgeGraph: '/knowledge-graph',
            settings: '/settings'
        },
        timeout: 60000 // 60 seconds
    },

    // Default model settings
    model: {
        name: 'meta-llama/Llama-3.2-1B',
        maxTokens: 256,
        temperatures: [0.2, 0.7, 1.0],
        pathsPerTemp: 1
    },

    // CGRT settings
    cgrt: {
        klThreshold: 0.5,
        contextWindowSize: 5,
        compressionEnabled: true,
        propagationEnabled: true,
        propagationFactor: 0.5
    },

    // Counterfactual settings
    counterfactual: {
        topKTokens: 5,
        attentionThreshold: 0.3,
        maxCounterfactuals: 20
    },

    // Knowledge Graph settings
    knowledgeGraph: {
        useLocalModel: true,
        similarityThreshold: 0.6,
        skip: false
    },

    // Visualization settings
    visualization: {
        defaultType: 'reasoning-tree',
        importanceThreshold: 0.5,
        showNodeLabels: true,
        showEdgeLabels: false,
        highlightDivergence: true,
        nodeSizeFactor: 100.0,
        animationDuration: 500,
        colorScheme: 'indigo'
    },

    // UI settings
    ui: {
        darkMode: false,
        sidebarOpen: true,
        expandedViz: false,
        messageHistory: 50, // Maximum number of messages to keep in history
        typingIndicatorDelay: 300,
        autoScrollChat: true
    }
};

// Load settings from local storage if available
function loadSettings() {
    const savedSettings = localStorage.getItem('xai-settings');
    if (savedSettings) {
        try {
            const parsedSettings = JSON.parse(savedSettings);

            // Merge saved settings with defaults
            XAIConfig.model = { ...XAIConfig.model, ...parsedSettings.model };
            XAIConfig.cgrt = { ...XAIConfig.cgrt, ...parsedSettings.cgrt };
            XAIConfig.counterfactual = { ...XAIConfig.counterfactual, ...parsedSettings.counterfactual };
            XAIConfig.knowledgeGraph = { ...XAIConfig.knowledgeGraph, ...parsedSettings.knowledgeGraph };
            XAIConfig.visualization = { ...XAIConfig.visualization, ...parsedSettings.visualization };
            XAIConfig.ui = { ...XAIConfig.ui, ...parsedSettings.ui };

            console.log('Loaded settings from local storage');
        } catch (error) {
            console.error('Error loading settings from local storage:', error);
        }
    }
}

// Save settings to local storage
function saveSettings() {
    try {
        const settingsToSave = {
            model: XAIConfig.model,
            cgrt: XAIConfig.cgrt,
            counterfactual: XAIConfig.counterfactual,
            knowledgeGraph: XAIConfig.knowledgeGraph,
            visualization: XAIConfig.visualization,
            ui: XAIConfig.ui
        };

        localStorage.setItem('xai-settings', JSON.stringify(settingsToSave));
        console.log('Settings saved to local storage');
    } catch (error) {
        console.error('Error saving settings to local storage:', error);
    }
}

// Initialize settings on load
loadSettings();
