let inputs = {
  data: [{
      input: [0, 0],
      label: [0]
    },
    {
      input: [1, 0],
      label: [1]
    },
    {
      input: [0, 1],
      label: [1]
    },
    {
      input: [1, 1],
      label: [0]
    }
  ]
}
let inputLayer;


function loadMNIST(callback) {
  let mnist = {};
  let files = {
    train_images: 'data/train-images-idx3-ubyte',
    train_labels: 'data/train-labels-idx1-ubyte',
    test_images: 'data/t10k-images.idx3-ubyte',
    test_labels: 'data/t10k-labels.idx1-ubyte'
  };
  return Promise.all(Object.keys(files).map(async file => {
      mnist[file] = await loadFile(files[file])
    }))
    .then(() => callback(mnist));
}

async function loadFile(file) {
  let buffer = await fetch(file).then(r => r.arrayBuffer());
  let headerCount = 4;
  let headerView = new DataView(buffer, 0, 4 * headerCount);
  let headers = new Array(headerCount).fill().map((_, i) => headerView.getUint32(4 * i, false));

  // Get file type from the magic number
  let type, dataLength;
  if (headers[0] == 2049) {
    type = 'label';
    dataLength = 1;
    headerCount = 2;
  } else if (headers[0] == 2051) {
    type = 'image';
    dataLength = headers[2] * headers[3];
  } else {
    throw new Error("Unknown file type " + headers[0])
  }

  let data = new Uint8Array(buffer, headerCount * 4);
  if (type == 'image') {
    dataArr = [];
    for (let i = 0; i < headers[1]; i++) {
      dataArr.push(data.subarray(dataLength * i, dataLength * (i + 1)));
    }
    return dataArr;
  }
  return data;
}

let mnist;
let outputLayer;
let nn;

function setup() {
  createCanvas(600, 600);
  background(153);
  line(0, 0, width, height);
  line(1, 1, 10, 10);
  angleMode(DEGREES);
  // put setup code here
  // console.log(inputs);
  /*
  inputLayer = new Layer("Input", 784);
  let prevLayer = inputLayer;
  let nextLayer = inputLayer;
  let currentLayer = inputLayer;
  for (let i = 0; i < 2; i++) {
    let layer = new Layer("Hidden", 64);
    layer.setPrevLayer(currentLayer);
    currentLayer.setNextLayer(layer);
    currentLayer = layer;
  }
  outputLayer = new Layer("Output", 10);
  outputLayer.setPrevLayer(currentLayer);
  currentLayer.setNextLayer(outputLayer);
  */
  //for (let i = 0; i < 1; i++) {
  //console.log(inputLayer);

  //let data = random(inputs.data);
  //let data = inputs.data[1];
  //console.log(data.input + " " + data.label);
  //inputLayer.train(data.input, data.label, outputLayer);
  //}
  let hidden = {
    neuronInLayer: [64, 64],
    layerCount: 2
  }
  nn = new NeuralNetwork(784, hidden, 10);
  console.log("Done");
  //for (let i = 0; i < 10; i++) {
  //console.log(inputLayer);

  //let data = random(inputs.data);
  //let data = inputs.data[1];
  //console.log(data.input + " " + data.label);
  //nn.trainNetwork(data.input, data.label);
  //}
  //let t = inputLayer.feedFowrard(inputs.data[0].input);
  //console.log(inputLayer.feedFowrard(inputs.data[0].input))
  //console.log(t);
  loadMNIST(function(data) {
    mnist = data;
    console.log(mnist);
  });
  //console.log(traningdata);

}

let inputsTrain = [];
let testInputs = [];
let train_index = 0;
let epoch = 0;

function findMax(arr) {
  let highest = arr[0];
  let index = 0;
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > highest) {
      highest = arr[i];
      index = i;
    }
  }
  return index;
}

function draw() {
  if (mnist) {
    for (let numOfTimes = 0; numOfTimes < 20; numOfTimes++) {
      for (let i = 0; i < 784; i++) {
        let bright = mnist.train_images[train_index][i];
        inputsTrain[i] = bright / 255;
      }
      let label = mnist.train_labels[train_index];
      let targets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
      targets[label] = 1;

      // console.log(inputs);
      // console.log(targets);

      //console.log(train_index);
      nn.trainNetwork(inputsTrain, targets);
      //inputLayer.train(inputsTrain, targets, outputLayer);
      //nn.train(inputs, targets);
      //console.log(train_index);
      train_index = (train_index + 1);
      //if (train_index == 100) {
      if (train_index >= mnist.train_labels.length) {
        console.log("EPOCH... " + epoch + " & testing");
        epoch++;
        let sumRight = 0;
        //mnist.test_labels.length
        for (let i = 0; i < mnist.test_labels.length; i++) {
          for (let j = 0; j < 784; j++) {
            let bright = mnist.test_images[i][j];
            testInputs[j] = bright / 255;
          }
          let resp = nn.getResponse(testInputs)
          let guess = findMax(resp);
          if (guess == mnist.test_labels[i]) {
            sumRight++;
          }
          //console.log("Guess :" + findMax(guess));
          if (i % 1000 == 0 && i != 0) {
            //console.log(resp);
            console.log((sumRight / i) * 100 + "% Right");
          }
        }
        //noLoop();
        if (epoch == 30) {
          console.log("DONE");
          saveJSON(nn.serialize(), 'nn.json');
          noLoop();
        }
        train_index = 0;
      }
    }
  }

  // put drawing code here
}
