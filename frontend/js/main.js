/**
 * Main script for the XAI Chat Interface
 * Initializes all components and handles UI interactions
 */

document.addEventListener('DOMContentLoaded', () => {
    // Initialize visualization when data is available
    const vizContainer = document.getElementById('viz-content');
    const visualization = new XAIVisualization('viz-content', XAIConfig);

    // Handle visualization updates from chat
    document.addEventListener('visualization-update', (event) => {
        const { data, type } = event.detail;
        visualization.setData(data, type);

        // Make visualization visible on mobile
        const vizContainer = document.getElementById('visualization-container');
        if (vizContainer) {
            vizContainer.classList.remove('hidden');
            vizContainer.classList.add('md:block');
        }
    });

    // Handle visualization type changes
    const vizTypeSelect = document.getElementById('viz-type');
    if (vizTypeSelect) {
        vizTypeSelect.addEventListener('change', () => {
            const selectedType = vizTypeSelect.value;

            // Update config and current visualization
            XAIConfig.visualization.defaultType = selectedType;

            // Re-render current data with new type if available
            if (visualization.currentData) {
                visualization.setData(visualization.currentData, selectedType);
            }

            // Save settings
            saveSettings();
        });
    }

    // Handle importance threshold changes
    const importanceThreshold = document.getElementById('importance-threshold');
    if (importanceThreshold) {
        importanceThreshold.value = XAIConfig.visualization.importanceThreshold * 100;

        importanceThreshold.addEventListener('input', () => {
            const thresholdValue = importanceThreshold.value / 100;

            // Update config and current visualization
            XAIConfig.visualization.importanceThreshold = thresholdValue;
            visualization.setImportanceThreshold(thresholdValue);

            // Save settings
            saveSettings();
        });
    }

    // Handle path selection changes
    const pathSelection = document.getElementById('path-selection');
    if (pathSelection) {
        const pathCheckboxes = pathSelection.querySelectorAll('input[type="checkbox"]');

        pathCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                // Get selected paths
                const selectedPaths = Array.from(pathCheckboxes)
                    .filter(cb => cb.checked)
                    .map(cb => cb.id.replace('path-', ''));

                // Update visualization if data is available
                if (visualization.currentData && visualization.currentData.paths) {
                    // Filter paths based on selection
                    const filteredData = {
                        ...visualization.currentData,
                        paths: visualization.currentData.paths.filter(path =>
                            selectedPaths.includes(path.id.toString())
                        )
                    };

                    visualization.setData(filteredData);
                }
            });
        });
    }

    // Handle export visualization button
    const exportVizBtn = document.getElementById('export-viz');
    if (exportVizBtn) {
        exportVizBtn.addEventListener('click', () => {
            if (!visualization.svg) {
                alert('No visualization available to export');
                return;
            }

            try {
                // Get SVG content
                const svgContent = visualization.container.innerHTML;

                // Create a Blob with SVG content
                const blob = new Blob([svgContent], { type: 'image/svg+xml' });

                // Create download link
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = `xai-visualization-${new Date().toISOString().slice(0, 10)}.svg`;

                // Trigger download
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);

                // Cleanup
                URL.revokeObjectURL(url);
            } catch (error) {
                console.error('Error exporting visualization:', error);
                alert('Error exporting visualization');
            }
        });
    }

    // Handle settings modal
    const settingsBtn = document.getElementById('settings-btn');
    const settingsModal = document.getElementById('settings-modal');
    const closeSettings = document.getElementById('close-settings');
    const saveSettingsBtn = document.getElementById('save-settings');

    if (settingsBtn && settingsModal && closeSettings && saveSettingsBtn) {
        // Initialize settings form with current values
        initializeSettingsForm();

        // Open settings modal
        settingsBtn.addEventListener('click', () => {
            settingsModal.classList.remove('hidden');
        });

        // Close settings modal
        closeSettings.addEventListener('click', () => {
            settingsModal.classList.add('hidden');
        });

        // Save settings
        saveSettingsBtn.addEventListener('click', () => {
            // Update config from form values
            updateConfigFromForm();

            // Save settings to local storage
            saveSettings();

            // Update API settings on server if connected
            try {
                apiService.updateSettings({
                    model: XAIConfig.model,
                    cgrt: XAIConfig.cgrt,
                    counterfactual: XAIConfig.counterfactual,
                    knowledgeGraph: XAIConfig.knowledgeGraph
                }).catch(error => {
                    console.error('Error updating settings on server:', error);
                });
            } catch (error) {
                console.error('Error updating settings:', error);
            }

            // Close modal
            settingsModal.classList.add('hidden');
        });

        // Close modal on click outside
        settingsModal.addEventListener('click', (event) => {
            if (event.target === settingsModal) {
                settingsModal.classList.add('hidden');
            }
        });
    }

    // Handle help modal
    const helpBtn = document.getElementById('help-btn');
    const helpModal = document.getElementById('help-modal');
    const closeHelp = document.getElementById('close-help');

    if (helpBtn && helpModal && closeHelp) {
        // Open help modal
        helpBtn.addEventListener('click', () => {
            helpModal.classList.remove('hidden');
        });

        // Close help modal
        closeHelp.addEventListener('click', () => {
            helpModal.classList.add('hidden');
        });

        // Close modal on click outside
        helpModal.addEventListener('click', (event) => {
            if (event.target === helpModal) {
                helpModal.classList.add('hidden');
            }
        });
    }

    // Handle sidebar toggle
    const toggleSidebarBtn = document.getElementById('toggle-sidebar');
    const sidebar = document.getElementById('sidebar');

    if (toggleSidebarBtn && sidebar) {
        toggleSidebarBtn.addEventListener('click', () => {
            sidebar.classList.toggle('hidden');
        });
    }

    // Handle expand visualization button
    const expandVizBtn = document.getElementById('expand-viz');
    const chatContainer = document.getElementById('chat-container');
    const visualizationContainer = document.getElementById('visualization-container');

    if (expandVizBtn && chatContainer && visualizationContainer) {
        expandVizBtn.addEventListener('click', () => {
            const isExpanded = visualizationContainer.classList.contains('md:w-4/5');

            if (isExpanded) {
                // Restore default layout
                visualizationContainer.classList.remove('md:w-4/5');
                visualizationContainer.classList.add('md:w-1/2', 'lg:w-3/5');
                chatContainer.classList.remove('md:w-1/5');
                chatContainer.classList.add('md:w-1/2', 'lg:w-2/5');
                expandVizBtn.innerHTML = '<i class="fas fa-expand-alt"></i>';
            } else {
                // Expand visualization
                visualizationContainer.classList.remove('md:w-1/2', 'lg:w-3/5');
                visualizationContainer.classList.add('md:w-4/5');
                chatContainer.classList.remove('md:w-1/2', 'lg:w-2/5');
                chatContainer.classList.add('md:w-1/5');
                expandVizBtn.innerHTML = '<i class="fas fa-compress-alt"></i>';
            }

            // Update config
            XAIConfig.ui.expandedViz = !isExpanded;
            saveSettings();

            // Re-render visualization to fit new container size
            if (visualization.currentData) {
                visualization.render();
            }
        });
    }

    // Handle node details panel
    const closeNodeDetails = document.getElementById('close-node-details');
    if (closeNodeDetails) {
        closeNodeDetails.addEventListener('click', () => {
            visualization.closeNodeDetails();
        });
    }

    // Initialize UI based on saved settings
    initializeUI();

    // Function to initialize settings form with current values
    function initializeSettingsForm() {
        // Model settings
        document.getElementById('model-selection').value = XAIConfig.model.name;
        document.getElementById('max-tokens').value = XAIConfig.model.maxTokens;

        // CGRT settings
        document.getElementById('temperature-values').value = XAIConfig.model.temperatures.join(',');
        document.getElementById('paths-per-temp').value = XAIConfig.model.pathsPerTemp;

        // Counterfactual settings
        document.getElementById('counterfactual-tokens').value = XAIConfig.counterfactual.topKTokens;
        document.getElementById('attention-threshold').value = XAIConfig.counterfactual.attentionThreshold;
    }

    // Function to update config from form values
    function updateConfigFromForm() {
        // Model settings
        XAIConfig.model.name = document.getElementById('model-selection').value;
        XAIConfig.model.maxTokens = parseInt(document.getElementById('max-tokens').value, 10);

        // CGRT settings
        const temperatureValues = document.getElementById('temperature-values').value
            .split(',')
            .map(temp => parseFloat(temp.trim()))
            .filter(temp => !isNaN(temp));

        XAIConfig.model.temperatures = temperatureValues.length > 0 ? temperatureValues : [0.2, 0.7, 1.0];
        XAIConfig.model.pathsPerTemp = parseInt(document.getElementById('paths-per-temp').value, 10);

        // Counterfactual settings
        XAIConfig.counterfactual.topKTokens = parseInt(document.getElementById('counterfactual-tokens').value, 10);
        XAIConfig.counterfactual.attentionThreshold = parseFloat(document.getElementById('attention-threshold').value);
    }

    // Function to initialize UI based on saved settings
    function initializeUI() {
        // Set visualization type selector to saved value
        if (vizTypeSelect) {
            vizTypeSelect.value = XAIConfig.visualization.defaultType;
        }

        // Set importance threshold slider to saved value
        if (importanceThreshold) {
            importanceThreshold.value = XAIConfig.visualization.importanceThreshold * 100;
        }

        // Apply expanded visualization state if saved
        if (XAIConfig.ui.expandedViz && expandVizBtn && chatContainer && visualizationContainer) {
            visualizationContainer.classList.remove('md:w-1/2', 'lg:w-3/5');
            visualizationContainer.classList.add('md:w-4/5');
            chatContainer.classList.remove('md:w-1/2', 'lg:w-2/5');
            chatContainer.classList.add('md:w-1/5');
            expandVizBtn.innerHTML = '<i class="fas fa-compress-alt"></i>';
        }
    }
});
