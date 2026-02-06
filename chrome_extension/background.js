// URL of your running Flask application's prediction endpoint
const PREDICTION_API_URL = "http://127.0.0.1:5000/api/predict"; // Use the correct API endpoint // Or use your machine's local IP if accessing from another device

// Listener for when a page navigation is fully committed
// This runs AFTER the page starts loading but before all sub-resources finish
chrome.webNavigation.onCommitted.addListener((details) => {
    // We only care about the main page frame, not iframes etc.
    // and only when the transition type is typical navigation (link click, typed address, bookmark)
    if (details.frameId === 0 && ["link", "typed", "auto_bookmark", "generated"].includes(details.transitionType)) {
        const urlToCheck = details.url;

        // Basic check: Ignore chrome internal pages, local files, etc.
        if (!urlToCheck || !urlToCheck.startsWith('http')) {
            console.log("Ignoring non-http(s) URL:", urlToCheck);
            return;
        }

        console.log("Detected navigation to:", urlToCheck);
        checkUrlForPhishing(urlToCheck, details.tabId);
    }
}, { url: [{ schemes: ["http", "https"] }] }); // Only listen for http and https URLs


// Function to send URL to backend and handle response
async function checkUrlForPhishing(url, tabId) {
    console.log("Sending URL to backend:", url);
    try {
        // Use the fetch API to POST the URL to your Flask app
        const response = await fetch(PREDICTION_API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded', // Flask expects form data
            },
            // Encode the URL properly for form submission
            body: new URLSearchParams({ 'url': url })
        });

        // Check if the request itself failed (e.g., Flask server not running)
        if (!response.ok) {
            console.error(`Backend request failed with status: ${response.status}`);
            // Decide what to do - maybe alert the user the checker isn't working?
            // For now, we'll just log it and allow navigation.
            return;
        }

        // --- IMPORTANT: Need Flask app to return JSON ---
        // We need to modify app.py to return JSON instead of rendering HTML directly for the extension
        const data = await response.json(); // Assumes Flask returns JSON like {"prediction": 1} or {"prediction": 0}

        console.log("Received prediction from backend:", data);

        // Check the prediction result
        if (data && data.prediction === 1) { // If prediction is 1 (Phishing)
            console.warn("Phishing detected! Redirecting tab:", tabId);
            // Redirect the user to our local warning page
            const warningUrl = chrome.runtime.getURL('warning.html') + '?url=' + encodeURIComponent(url);
            chrome.tabs.update(tabId, { url: warningUrl });
        } else {
            console.log("URL seems legitimate or prediction failed safely.");
        }

    } catch (error) {
        console.error("Error checking URL:", error);
        // Handle errors (e.g., Flask server down, network issue)
        // For safety in case of error, maybe allow navigation but log?
    }
}

// Optional: Log when the extension is installed or updated
chrome.runtime.onInstalled.addListener(() => {
    console.log("Phishing Detector Extension Installed/Updated.");
});