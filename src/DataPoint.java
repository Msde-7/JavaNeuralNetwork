public class DataPoint {
    private double[] inputs;
    private double[] expectedOutputs;
    private int expectedOutput;

    public DataPoint(double[] inputs, double[] expectedOutputs) {
        this.inputs = inputs;
        this.expectedOutputs = expectedOutputs;
    }

    public DataPoint (double[] inputs, int expectedOutput) {
        this.inputs = inputs;
        this.expectedOutputs = new double[10];
        this.expectedOutput = expectedOutput;
        this.expectedOutputs[expectedOutput] = 1;
    }

    public double[] getExpectedOutputs() {
        return expectedOutputs;
    }

    public int getExpectedOutput() {
        return expectedOutput;
    }

    public double[] getInputs() {
        return inputs;
    }
}
