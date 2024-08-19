import {MnistData} from './data_mnist.js';

//tf.setBackend('webgl').then(() => {
  //console.log('WebGL backend set');
  //run(); // Start your main function after setting the backend
//});

// Function to measure the loading time of data and model
async function measureLoadTime(func, name) {
    const startTime = performance.now();
    await func();
    const endTime = performance.now();
    console.log(`${name} loading time: ${endTime - startTime} ms`);
  }

// Function to log GPU and performance usage, including training and inference times
async function logPerformanceMetrics(model, data) {
    console.log("Starting performance profiling...");
    
    // Measure warm-up time
    const warmUpStart = performance.now();
    const testBatch = data.nextTestBatch(1);
    model.predict(testBatch.xs.reshape([1, 28, 28, 1]));
    const warmUpEnd = performance.now();
    console.log(`Warm-up time: ${warmUpEnd - warmUpStart} ms`);
  
    // Measure training time
    const trainingStart = performance.now();
    await train(model, data);
    const trainingEnd = performance.now();
    console.log(`Training time: ${trainingEnd - trainingStart} ms`);
  
    // Profiling during training
    const profile = await tf.profile(async () => {
      model.predict(testBatch.xs.reshape([1, 28, 28, 1]));
    });
  
    const kernelMs = profile.kernels
      .map(kernel => kernel.kernelMs)
      .filter(time => !isNaN(time))
      .reduce((a, b) => a + b, 0);
    console.log(`Total kernel time: ${kernelMs} ms`);
  
    if (profile.peakBytes) {
      console.log(`GPU peak bytes allocated: ${profile.peakBytes}`);
    } else {
      console.log('GPU peak bytes information is not available in this environment.');
    }
  
    if (profile.tensorMap) {
      console.log("Disposing tensors from the profile...");
      for (const tensorInfo of Object.values(profile.tensorMap)) {
        if (tensorInfo.tensor) {
          tensorInfo.tensor.dispose();
        }
      }
    } else {
      console.log("No tensor data found in the profile.");
    }
  
    // Measure inference time
    const inferenceStart = performance.now();
    const predictions = model.predict(testBatch.xs.reshape([1, 28, 28, 1]));
    const inferenceEnd = performance.now();
    console.log(`Inference time: ${inferenceEnd - inferenceStart} ms`);

    // Use await to handle tensor data
    const predictionData = await predictions.data();
    console.log(predictionData);

  }
  

async function showExamples(data) {
  // Create a container in the visor
  const surface =
    tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});  

  // Get the examples
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];
  
  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });
    
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';
    //await tf.browser.draw(imageTensor, canvas);
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
}

async function run() {
  tf.wasm.setThreadsCount(4);
  // Set the backend to WASM
  await tf.setBackend('wasm');
  console.log('WASM backend is being used');
  console.log('Thread count:', tf.wasm.getThreadsCount());
  
  // Define data and model variables in a scope accessible to all functions
  const data = new MnistData();
  const model = getModel();

  // Measure the loading time of the data
  await measureLoadTime(async () => {
    await data.load();
    await showExamples(data);
  }, "MNIST data");

  // Measure the loading time of the model
  await measureLoadTime(async () => {
    tfvis.show.modelSummary({ name: 'Model Architecture', tab: 'Model' }, model);
  }, "Model");

  // Log performance metrics including warm-up, training, and inference times
  await logPerformanceMetrics(model, data);

  // Display accuracy and confusion matrix
  await showAccuracy(model, data);
  await showConfusion(model, data);
}
  
  document.addEventListener('DOMContentLoaded', run);

function getModel() {
    const model = tf.sequential();
    
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;
    
    // Flatten the input from 28x28 to 784 units
    model.add(tf.layers.flatten({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS]
    }));
  
    // Add a dense layer with 128 units
    model.add(tf.layers.dense({
      units: 128,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
    
    // Add another dense layer with 64 units
    model.add(tf.layers.dense({
      units: 64,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
    
    // Output layer with 10 units, one for each digit class
    const NUM_OUTPUT_CLASSES = 10;
    model.add(tf.layers.dense({
      units: NUM_OUTPUT_CLASSES,
      kernelInitializer: 'varianceScaling',
      activation: 'softmax'
    }));
    
    // Compile the model with the adam optimizer and categorical crossentropy loss
    const optimizer = tf.train.adam();
    model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
  
    return model;
  }

  async function train(model, data) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
      name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
    
    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 5500;
    const TEST_DATA_SIZE = 1000;
  
    const [trainXs, trainYs] = tf.tidy(() => {
      const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
      return [
        d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
        d.labels
      ];
    });
  
    const [testXs, testYs] = tf.tidy(() => {
      const d = data.nextTestBatch(TEST_DATA_SIZE);
      return [
        d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
        d.labels
      ];
    });
  
    return model.fit(trainXs, trainYs, {
      batchSize: BATCH_SIZE,
      validationData: [testXs, testYs],
      epochs: 10,
      shuffle: true,
      callbacks: fitCallbacks
    });
  }

  const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

  function doPrediction(model, data, testDataSize = 500) {
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const testData = data.nextTestBatch(testDataSize);
    const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
    const labels = testData.labels.argMax(-1); // Ensure labels are 1D
    const preds = model.predict(testxs).argMax(-1); // Ensure predictions are 1D
  
    testxs.dispose();
    return Promise.all([preds.data(), labels.data()]);
  }


async function showAccuracy(model, data) {
  const [predsArray, labelsArray] = await doPrediction(model, data);

  // Convert the arrays back to tensors
  const preds = tf.tensor1d(predsArray, 'int32');
  const labels = tf.tensor1d(labelsArray, 'int32');

  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const container = { name: 'Accuracy', tab: 'Evaluation' };
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

  preds.dispose();
  labels.dispose();
}

async function showConfusion(model, data) {
  const [predsArray, labelsArray] = await doPrediction(model, data);

  // Convert the arrays back to 1D tensors
  const preds = tf.tensor1d(predsArray, 'int32');
  const labels = tf.tensor1d(labelsArray, 'int32');

  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = { name: 'Confusion Matrix', tab: 'Evaluation' };
  tfvis.render.confusionMatrix(container, { values: confusionMatrix, tickLabels: classNames });

  preds.dispose();
  labels.dispose();
}
