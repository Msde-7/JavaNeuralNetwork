import java.util.Arrays;
import java.util.Random;

public class Layer {
    private double[][] weights, weightGradients;
    private double[] biases, biasGradients;
    private int numIn, numOut;

    private double[] activations;
    private double[] weightedInputs;
    private double[] inputs;

    private final static double LEAKY_RELU_X = 0.01;

    public Layer(int numIn, int numOut) {
        this.weights = new double[numIn][numOut];
        this.weightGradients = new double[numIn][numOut];

        this.biases = new double[numOut];
        this.biasGradients = new double[numOut];

        this.numIn = numIn;
        this.numOut = numOut;

        randomizeWeights(new Random());
    }

    public void randomizeWeights(Random rand) {
        for (int i = 0; i < weights.length; i++)
        {
            for(int j = 0; j < weights[0].length; j++)
            weights[i][j] = randomInNormalDistribution(rand, 0, 1) / Math.sqrt(numIn);
        }


    }

    private double randomInNormalDistribution(Random rand, double mean, double standardDeviation)
    {
        double x1 = 1 - rand.nextDouble();
        double x2 = 1 - rand.nextDouble();

        double y1 = Math.sqrt(-2.0 * Math.log(x1)) * Math.cos(2.0 * Math.PI * x2);
        return y1 * standardDeviation + mean;
    }

    public double[] calcOutput(double[] inputs) {
        this.inputs = inputs;
        this.weightedInputs = new double[numOut];
        double[] activations = new double[numOut];

        for (int i = 0; i < numOut; i++) {
            double curNum = biases[i];
            for (int j = 0; j < numIn; j++)
                curNum += inputs[j] * weights[j][i];

            //Activation
            weightedInputs[i] = curNum;
            activations[i] = calcReLU(curNum);
        }

        this.activations = activations;
        return activations;
    }


    public void applyGradients(double learnRate) {
        for (int i = 0; i < numOut; i++) {
            biases[i] -= biasGradients[i] * learnRate;
            for (int j = 0; j < numOut; j++) {
                weights[j][i] -= weightGradients[j][i] * learnRate;
            }
        }

    }

    public double[] getBiases() {
        return biases;
    }

    public double[][] getWeights() {
        return weights;
    }

    public static double calcReLU(double weightedInput) {
        //Leaky ReLU
        return Math.max(LEAKY_RELU_X * weightedInput, weightedInput);
        //return 1 / (1 + Math.exp(-weightedInput));
    }

    public static double calcReLUDerivative(double weightedInput) {
        if(weightedInput < 0)
            return LEAKY_RELU_X;
        else
            return 1;


        //double sig = calcReLU(weightedInput);
        //return sig * (1 - sig);
    }

    public double calcLayerCost(double outputGiven, double outputExpected) {
        return Math.pow(outputExpected - outputGiven, 2);
    }

    public double calcLayerCostDerivative(double outputActivation, double expectedOutput) {
        return 2 * (outputActivation - expectedOutput);
    }

    public int getNumIn() {
        return numIn;
    }

    public int getNumOut() {
        return numOut;
    }

    public double[][] getWeightGradients() {
        return weightGradients;
    }

    public double[] getBiasGradients() {
        return biasGradients;
    }

    public double[] calcOutputLayerNodeValues(double[] expectedOutputs) {
        //System.out.println("d " + Arrays.toString(activations));
        //System.out.println("c " + Arrays.toString(weightedInputs));
        double[] nodeValues = new double[expectedOutputs.length];

        for (int i = 0; i < nodeValues.length; i++) {
            double dCost = calcLayerCostDerivative(activations[i], expectedOutputs[i]);
            double dActivation = calcReLUDerivative(weightedInputs[i]);
            nodeValues[i] = dActivation * dCost;
        }

        //System.out.println(Arrays.toString(nodeValues));

        return nodeValues;
    }

    public void updateGradients(double[] nodeValues) {
        for (int i = 0; i < numOut; i++) {
            for (int j = 0; j < numIn; j++) {
                double derivativeCostWrtWeight = inputs[j] * nodeValues[i];
                weightGradients[j][i] += derivativeCostWrtWeight;
            }
            double derivativeCostWrtBias = nodeValues[i];
            biasGradients[i] += derivativeCostWrtBias;
        }
        //System.out.println(Arrays.deepToString(weightGradients));
    }

    public double[] getActivations() {
        return activations;
    }

    public double[] getWeightedInputs() {
        return weightedInputs;
    }

    public double[] calcHiddenLayerNodeValues(Layer oldLayer, double[] oldNodeValues) {
        double[] newNodeValues = new double[numOut];
        for(int i = 0; i < newNodeValues.length; i++) {
            double newNodeValue = 0;
            for(int j = 0; j < oldNodeValues.length; j++) {
                double weightedInputDerivative = oldLayer.weights[i][j];
                newNodeValue += weightedInputDerivative * oldNodeValues[j];
            }
            newNodeValue *= calcReLUDerivative(weightedInputs[i]);
            newNodeValues[i] = newNodeValue;
        }

        return newNodeValues;
    }

    public void clearGradients() {
        for(int i = 0; i < weightGradients.length; i++)
            for(int j = 0; j < weightGradients[0].length; j++)
                weightGradients[i][j] = 0;
        for(int i = 0; i < biasGradients.length; i++)
            biasGradients[i] = 0;
    }
}
