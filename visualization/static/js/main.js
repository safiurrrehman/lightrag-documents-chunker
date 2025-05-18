/**
 * Main JavaScript file for RAG Visualization
 * 
 * This file contains common functions used across the visualization interface.
 */

// Initialize tooltips and popovers when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize Bootstrap popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
});

/**
 * Format a number with commas as thousands separators
 * 
 * @param {number} num - The number to format
 * @returns {string} - Formatted number string
 */
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

/**
 * Truncate a string to a specified length and add ellipsis if needed
 * 
 * @param {string} str - The string to truncate
 * @param {number} maxLength - Maximum length before truncation
 * @returns {string} - Truncated string
 */
function truncateString(str, maxLength) {
    if (str.length <= maxLength) return str;
    return str.substring(0, maxLength) + '...';
}

/**
 * Create a color scale based on a value between 0 and 1
 * 
 * @param {number} value - Value between 0 and 1
 * @returns {string} - RGB color string
 */
function getColorScale(value) {
    // Ensure value is between 0 and 1
    value = Math.max(0, Math.min(1, value));
    
    // Red to Yellow to Green color scale
    let r, g, b = 0;
    
    if (value < 0.5) {
        // Red to Yellow (value: 0 to 0.5)
        r = 255;
        g = Math.round(255 * (value * 2));
    } else {
        // Yellow to Green (value: 0.5 to 1)
        r = Math.round(255 * (1 - (value - 0.5) * 2));
        g = 255;
    }
    
    return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Show a notification message
 * 
 * @param {string} message - Message to display
 * @param {string} type - Message type (success, info, warning, error)
 * @param {number} duration - Duration in milliseconds
 */
function showNotification(message, type = 'info', duration = 3000) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} notification-toast`;
    notification.innerHTML = message;
    notification.style.position = 'fixed';
    notification.style.top = '20px';
    notification.style.right = '20px';
    notification.style.zIndex = '9999';
    notification.style.minWidth = '300px';
    notification.style.boxShadow = '0 4px 8px rgba(0,0,0,0.1)';
    notification.style.transition = 'opacity 0.3s ease-in-out';
    
    // Add to document
    document.body.appendChild(notification);
    
    // Remove after duration
    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, duration);
}

/**
 * Format a date string
 * 
 * @param {string} dateString - ISO date string
 * @returns {string} - Formatted date string
 */
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

/**
 * Create a download link for JSON data
 * 
 * @param {Object} data - JSON data to download
 * @param {string} filename - Name of the file to download
 * @returns {string} - URL for the download link
 */
function createJsonDownloadLink(data, filename) {
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.className = 'btn btn-sm btn-secondary';
    link.innerHTML = '<i class="fas fa-download me-1"></i> Download JSON';
    
    return link;
}

/**
 * Debounce function to limit how often a function can be called
 * 
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} - Debounced function
 */
function debounce(func, wait) {
    let timeout;
    return function(...args) {
        const context = this;
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(context, args), wait);
    };
}
