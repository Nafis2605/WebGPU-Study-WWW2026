chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "monitor") {
      // Perform your page-specific tasks here
  
      // Example: Change some configuration on the page
      document.body.style.backgroundColor = "lightblue";
  
      // Monitor resource usage
      const resources = window.performance.getEntriesByType("resource");
      let resourceData = resources.map(resource => ({
        name: resource.name,
        type: resource.initiatorType,
        duration: resource.duration
      }));
  
      // Send data back to the background script
      chrome.runtime.sendMessage({ action: 'logData', data: resourceData });
    }
  });
  