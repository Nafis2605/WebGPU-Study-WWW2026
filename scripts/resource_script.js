// Get all resource entries
let resourceEntries = performance.getEntriesByType("resource");

// Process and log specific data
resourceEntries.forEach(resource => {
  console.log(`Resource: ${resource.name}`);
  console.log(`Type: ${resource.initiatorType}`);
  console.log(`Start Time: ${resource.startTime.toFixed(2)} ms`);
  console.log(`Duration: ${resource.duration.toFixed(2)} ms`);
  console.log(`Transfer Size: ${resource.transferSize} bytes`);
  console.log('---');
});

// Optionally, summarize data if needed
let totalResources = resourceEntries.length;
let totalDuration = resourceEntries.reduce((sum, resource) => sum + resource.duration, 0);
let totalSize = resourceEntries.reduce((sum, resource) => sum + resource.transferSize, 0);

console.log(`Total Number of Resources: ${totalResources}`);
console.log(`Total Duration: ${totalDuration.toFixed(2)} ms`);
console.log(`Total Transfer Size: ${totalSize} bytes`);


//Filtering Examples
let scriptResources = resourceEntries.filter(resource => resource.initiatorType === 'script');
scriptResources.forEach(script => {
  console.log(`Script: ${script.name}, Duration: ${script.duration.toFixed(2)} ms`);
});

//Summary
let totalScriptsSize = scriptResources.reduce((total, script) => total + script.transferSize, 0);
console.log(`Total Size of Script Resources: ${totalScriptsSize} bytes`);
