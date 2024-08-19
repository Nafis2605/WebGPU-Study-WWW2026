import {Cifar10Data} from './data_cifar10.js';

async function measureLoadTime(func, name) {
    const startTime = performance.now();
    await func();
    const endTime = performance.now();
    console.log(`${name} loading time: ${endTime - startTime} ms`);
}

async function logPerformanceMetrics(model, data) {
    console.log("Starting performance profiling...");
    
    const warmUpStart = performance.now();
    const testBatch = data.nextTestBatch(1);
    model.predict(testBatch.xs.reshape([1, 32, 32, 3]));
    const warmUpEnd = performance.now();
    console.log(`Warm-up time: ${warmUpEnd - warmUpStart} ms`);
  
    const trainingStart = performance.now();
    await train(model, data);
    const trainingEnd = performance.now();
    console.log(`Training time: ${trainingEnd - trainingStart} ms`);
  
    const profile = await tf.profile(async () => {
        model.predict(testBatch.xs.reshape([1, 32, 32, 3]));
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
  
    const inferenceStart = performance.now();
    const predictions = model.predict(testBatch.xs.reshape([1, 32, 32, 3]));
    const inferenceEnd = performance.now();
    console.log(`Inference time: ${inferenceEnd - inferenceStart} ms`);
  
    const predictionData = await predictions.data();
    console.log(predictionData);
}

async function showExamples(data) {
    const surface = tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});  
    const examples = data.nextTestBatch(20);
    const numExamples = examples.xs.shape[0];
  
    for (let i = 0; i < numExamples; i++) {
        const imageTensor = tf.tidy(() => {
            return examples.xs
                .slice([i, 0], [1, examples.xs.shape[1]])
                .reshape([32, 32, 3]);
        });
    
        const canvas = document.createElement('canvas');
        canvas.width = 32;
        canvas.height = 32;
        canvas.style = 'margin: 4px;';
        await tf.browser.toPixels(imageTensor, canvas);
        //await tf.browser.draw(imageTensor, canvas);
        surface.drawArea.appendChild(canvas);

        imageTensor.dispose();
    }
}

async function run() {
    await tf.setBackend('wasm');
    console.log('WASM backend is being used');
  
    const data = new Cifar10Data();
    const model = getModel();

    await measureLoadTime(async () => {
        await data.load();
        await showExamples(data);
    }, "CIFAR-10 data");

    await measureLoadTime(async () => {
        tfvis.show.modelSummary({ name: 'Model Architecture', tab: 'Model' }, model);
    }, "Model");

    await logPerformanceMetrics(model, data);

    await showAccuracy(model, data);
    await showConfusion(model, data);
}

document.addEventListener('DOMContentLoaded', run);

function getModel() {
    const model = tf.sequential();
    
    const IMAGE_SIZE = 32 * 32 * 3; // Flattened image size (32x32 with 3 channels)

    model.add(tf.layers.flatten({
        inputShape: [32, 32, 3]
    }));

    model.add(tf.layers.dense({
        units: 1024,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));

    model.add(tf.layers.dense({
        units: 512,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));

    model.add(tf.layers.dense({
        units: 256,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));

    model.add(tf.layers.dense({
        units: 128,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));

    const NUM_OUTPUT_CLASSES = 10;
    model.add(tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        activation: 'softmax',
        kernelInitializer: 'varianceScaling'
    }));

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
    const TRAIN_DATA_SIZE = 40000;
    const TEST_DATA_SIZE = 10000;
  
    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
        return [
            d.xs.reshape([TRAIN_DATA_SIZE, 32, 32, 3]),
            d.labels
        ];
    });
  
    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(TEST_DATA_SIZE);
        return [
            d.xs.reshape([TEST_DATA_SIZE, 32, 32, 3]),
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

const classNames = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck'];

function doPrediction(model, data, testDataSize = 500) {
    const IMAGE_WIDTH = 32;
    const IMAGE_HEIGHT = 32;
    const testData = data.nextTestBatch(testDataSize);
    const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 3]);
    const labels = testData.labels.argMax(-1);
    const preds = model.predict(testxs).argMax(-1);
  
    testxs.dispose();
    return Promise.all([preds.data(), labels.data()]);
}

async function showAccuracy(model, data) {
    const [predsArray, labelsArray] = await doPrediction(model, data);

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

    const preds = tf.tensor1d(predsArray, 'int32');
    const labels = tf.tensor1d(labelsArray, 'int32');

    const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
    const container = { name: 'Confusion Matrix', tab: 'Evaluation' };
    tfvis.render.confusionMatrix(container, { values: confusionMatrix, tickLabels: classNames });

    preds.dispose();
    labels.dispose();
}
