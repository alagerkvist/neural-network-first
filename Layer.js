class Layer {

  constructor(type, neurons) {
    this.neurons = Array.from({
      length: neurons
    }, () => new Neuron());
    this.type = type;
    this.nextLayer = null;
    this.prevLayer = null;
    this.nextInputs = [];
    this.learningRate = 0.1;
  }

  setNextLayer(layer) {
    this.nextLayer = layer;
  }

  setPrevLayer(layer) {
    this.prevLayer = layer;
    for (let neuron of this.neurons) {
      neuron.initWeigths(layer.neurons.length);
    }
  }

  feedFowrard(inputs) {
    this.nextInputs = inputs;
    if (this.type != "Input") {
      this.nextInputs = [];
      for (let neuron of this.neurons) {
        let output = neuron.calc(inputs);
        this.nextInputs.push(output);
        //console.log(output);
      }
    }
    if (this.nextLayer != null) {
      return this.nextLayer.feedFowrard(this.nextInputs);
    } else {
      return this.nextInputs;
    }

  }



  backProp(output, label) {
    //console.log("?" + this.type);
    if (this.type == "Output") {
      let derSig = [];

      for (let i = 0; i < this.neurons.length; i++) {
        //Can add output - label here
        let error = output[i] - label[i];
        //console.log(error);
        let t = output[i] * (1 - output[i]);
        let tot = t * error;
        //console.log(tot);
        derSig.push(tot);
        //derSig.push(derSigmoid(output[i]) * error);
        for (let j = 0; j < this.neurons[i].weights.length; j++) {
          let outputFromPrev = this.prevLayer.nextInputs[j];
          let delta = outputFromPrev * derSig[i] * -this.learningRate;
          //console.log(this.neurons[i].weights[j]);
          this.neurons[i].weights[j] += delta;
          //console.log(this.neurons[i].weights[j]);
          //console.log(delta);
        }
        this.neurons[i].bias += tot * -this.learningRate;
      }
      this.prevLayer.backProp(derSig);
    } else if (this.type == "Hidden") {
      //console.log(this.nextInputs);
      let backOut = [];
      for (let i = 0; i < this.neurons.length; i++) {

        let derSig = this.nextInputs[i] * (1 - this.nextInputs[i]);
        //let derSig = derSigmoid(this.nextInputs[i]);
        let sum = 0;

        for (let j = 0; j < this.nextLayer.neurons.length; j++) {
          sum += output[j] * this.nextLayer.neurons[j].weights[i];
        }

        for (let j = 0; j < this.neurons[i].weights.length; j++) {
          let outputFromPrev = this.prevLayer.nextInputs[j];
          let delta = outputFromPrev * sum * derSig * -this.learningRate;
          backOut.push(sum * derSig);
          this.neurons[i].weights[j] += delta;
        }
        this.neurons[i].bias += derSig * sum * -this.learningRate;
      }
      if (this.prevLayer.type == "Hidden") {
        this.prevLayer.backProp(backOut);
      }
    }
  }

  train(inputs, label, outputLayer) {
    let output = this.feedFowrard(inputs);

    if (findMax(output) == findMax(label)) {
      trainRight++;
      //console.log(this);
    }
    //console.log("Error: " + output);
    //Switch to outputLayer
    outputLayer.backProp(output, label);
  }

}
