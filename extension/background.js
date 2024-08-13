chrome.runtime.onInstalled.addListener(() => {
    console.log("Extension installed.");
  });
  
  chrome.webNavigation.onCompleted.addListener((details) => {
    chrome.scripting.executeScript({
      target: { tabId: details.tabId },
      files: ["content.js"]
    });
  }, { url: [{ urlMatches: '<all_urls>' }] });
  
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'logData') {
      fetch('http://localhost:3000/log', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(message.data)
      })
      .then(response => response.json())
      .then(data => {
        console.log('Data logged successfully:', data);
      })
      .catch(error => {
        console.error('Error logging data:', error);
      });
    }
  });
  