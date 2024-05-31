public class NeuralNetwork {
    Layer[] layers;
    public NeuralNetwork(int[] layerSizes) {
        layers = new Layer[layerSizes.length - 1];
        for(int i = 0; i < layers.length; i++)
            layers[i] = new Layer(layerSizes[i], layerSizes[i + 1]);
    }

    public double[] calcOutput(double[] inputs) {
        for(Layer layer : layers) {
            inputs = layer.calcOutput(inputs);
        }
        return inputs;
    }

    public int getClassification(double[] inputs) {
        double[] outputs = calcOutput(inputs);

        int indexOfMax = 0;
        double max = outputs[indexOfMax];
        for(int i = 1; i < outputs.length; i++)
            if(outputs[i] > max) {
                max = outputs[i];
                indexOfMax = i;
            }

        return indexOfMax;
    }

    public double calcSingleCost(DataPoint dataPoint) {
        double[] outputs = calcOutput(dataPoint.getInputs());
        Layer outputLayer = layers[layers.length - 1];
        double cost = 0;

        for(int i = 0; i < outputs.length; i++)
            cost += outputLayer.calcLayerCost(outputs[i], dataPoint.getExpectedOutputs()[i]);
        
        return cost;
    }

    public double calcMultipleCost(DataPoint[] data) {
        double totalCost = 0;

        for(DataPoint dataPoint : data)
            totalCost += calcSingleCost(dataPoint);

        return totalCost / data.length;
    }

    public void applyAllGradients(double learnRate) {
        for(Layer layer : layers)
            layer.applyGradients(learnRate);
    }

    public void updateAllGradients(DataPoint dataPoint) {
        calcOutput(dataPoint.getInputs());

        Layer outputLayer = layers[layers.length - 1];
        double[] nodeValues = outputLayer.calcOutputLayerNodeValues(dataPoint.getExpectedOutputs());

        outputLayer.updateGradients(nodeValues);

        for(int i = layers.length - 2; i >= 0; i--) {
            Layer hiddenLayer = layers[i];
            nodeValues = hiddenLayer.calcHiddenLayerNodeValues(layers[i + 1], nodeValues);

            hiddenLayer.updateGradients(nodeValues);
        }
    }

    public void clearAllGradients() {
        for(Layer layer : layers)
            layer.clearGradients();
    }
    public void learn(DataPoint[] trainingData, double learnRate) {
        for(DataPoint datapoint : trainingData) {
            updateAllGradients(datapoint);
        }
        applyAllGradients(learnRate / trainingData.length);
        clearAllGradients();
    }

    @Override
    public String toString() {
        return "";
    }
}
