function sigmoid(x) {
  return 1 / (1 + exp(-x));
}

class Neuron {

  constructor() {
    this.weights = null;
    this.bias = random(-1, 1);
  }

  initWeigths(weights) {
    this.weights = Array.from({
      length: weights
    }, () => random(-1, 1));
  }

  calc(inputs) {
    let sum = 0;
    for (let i = 0; i < inputs.length; i++) {
      sum += inputs[i] * this.weights[i];
    }

    return sigmoid(sum + this.bias);
  }

}
