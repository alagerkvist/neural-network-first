class NeuralNetwork {

  constructor(numOfInputs, hiddenLayer, numOfOutput) {

    this.inputLayer = new Layer("Input", numOfInputs);
    //inputLayer = new Layer("Input", 784);
    let currentLayer = this.inputLayer;
    console.log(hiddenLayer.neuronInLayer.length);
    for (let i = 0; i < hiddenLayer.neuronInLayer.length; i++) {
      let layer = new Layer("Hidden", hiddenLayer.neuronInLayer[i]);
      layer.setPrevLayer(currentLayer);
      currentLayer.setNextLayer(layer);
      currentLayer = layer;
    }
    this.outputLayer = new Layer("Output", numOfOutput);
    this.outputLayer.setPrevLayer(currentLayer);
    currentLayer.setNextLayer(this.outputLayer);
  }

  getResponse(inputs) {
    return this.inputLayer.feedFowrard(inputs);
  }

  trainNetwork(inputs, targets) {
    this.inputLayer.train(inputs, targets, this.outputLayer);
  }

  serialize() {
    let seen = [];
    return JSON.stringify(this, function(key, val) {
      if (val != null && typeof val == "object") {
        if (seen.indexOf(val) >= 0) {
          return;
        }
        seen.push(val);
      }
      return val;
    });
  }

  deserialize() {

  }

}
