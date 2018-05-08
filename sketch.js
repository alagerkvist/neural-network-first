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

/*
Idea and loadMNIST is from
https://github.com/CodingTrain/Toy-Neural-Network-JS
*/
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
let trainPoints = [];
let testPoints = [];

function setup() {
  createCanvas(1220, 600);
  background(153);
  noStroke();
  fill(255);
  textSize(12);
  for (let i = 0; i < width; i++) {
    text(i, i * 20, height);
  }
  for (let i = 0; i < 0.5; i += 0.01) {
    if (i == 0) {
      text(nf(i, 1, 0), 0, height - (i * 1200));
    } else {
      text(nf(i, 1, 2), 0, height - (i * 1200));
    }

  }
  let hidden = {
    neuronInLayer: [64]
  }
  nn = new NeuralNetwork(784, hidden, 10);
  console.log("Done");

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
let trainRight = 0;

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
      nn.trainNetwork(inputsTrain, targets);

      train_index = (train_index + 1);
      //if (train_index == 100) {
      if (train_index >= mnist.train_labels.length) {
        let percentTrain = trainRight / mnist.train_labels.length;
        percentTrain = 1 - percentTrain;
        trainRight = 0;
        console.log(percentTrain);
        console.log("EPOCH... " + epoch + " & testing");

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
            //console.log("Rest " + resp + " " + guess);
            console.log((sumRight / i) * 100 + "% Right");
          }


        }
        translate(0, height);
        strokeWeight(4);
        stroke(255, 0, 0);

        let trainPos = createVector(epoch * 20, -percentTrain * height * 2);
        trainPoints.push(trainPos);
        let testDataPercent = 1 - (sumRight / mnist.test_labels.length);
        let testPos = createVector(epoch * 20, -testDataPercent * height * 2);
        testPoints.push(testPos);
        if (trainPoints.length == 1) {
          point(trainPos.x, trainPos.y);
          stroke(0, 0, 255);
          point(testPos.x, testPos.y);
        } else {
          line(trainPoints[epoch - 1].x, trainPoints[epoch - 1].y, trainPos.x, trainPos.y);
          stroke(0, 0, 255);
          line(testPoints[epoch - 1].x, testPoints[epoch - 1].y, testPos.x, testPos.y);
        }
        epoch++;
        train_index = 0;
      }
    }
  }
}
