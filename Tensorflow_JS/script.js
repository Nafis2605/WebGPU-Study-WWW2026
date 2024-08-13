import {MnistData} from './data.js';

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
    model.predict(testBatch.xs.reshape([1, 28, 28, 1]));
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
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
}

async function run() {
  await tf.setBackend('wasm');
  console.log('WebGPU backend is being used');
  
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
    
    // In the first layer of our convolutional neural network we have 
    // to specify the input shape. Then we specify some parameters for 
    // the convolution operation that takes place in this layer.
    model.add(tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
  
    // The MaxPooling layer acts as a sort of downsampling using max values
    // in a region instead of averaging.  
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    
    // Repeat another conv2d + maxPooling stack. 
    // Note that we have more filters in the convolution.
    model.add(tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    
    // Now we flatten the output from the 2D filters into a 1D vector to prepare
    // it for input into our last layer. This is common practice when feeding
    // higher dimensional data to a final classification output layer.
    model.add(tf.layers.flatten());
  
    // Our last layer is a dense layer which has 10 output units, one for each
    // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
    const NUM_OUTPUT_CLASSES = 10;
    model.add(tf.layers.dense({
      units: NUM_OUTPUT_CLASSES,
      kernelInitializer: 'varianceScaling',
      activation: 'softmax'
    }));
  
    
    // Choose an optimizer, loss function and accuracy metric,
    // then compile and return the model
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
  const labels = testData.labels.argMax(-1);
  const preds = model.predict(testxs).argMax(-1);

  testxs.dispose();
  return Promise.all([preds.data(), labels.data()]);
}


async function showAccuracy(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const container = {name: 'Accuracy', tab: 'Evaluation'};
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

  labels.dispose();
}

async function showConfusion(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = {name: 'Confusion Matrix', tab: 'Evaluation'};
  tfvis.render.confusionMatrix(container, {values: confusionMatrix, tickLabels: classNames});

  labels.dispose();
}
