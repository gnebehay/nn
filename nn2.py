import numpy as np

class Layer:
    NextLayer = None
    PrevLayer = None

    InputCardinality = None
    OutputCardinality = None

    InputValues = None
    OutputValues = None

    ID = None
    
    def __pow__(self, n):
        self.NextLayer = n
        n.PrevLayer = self
        return self

    def Init(self):
        if self.NextLayer is not None:
            self.NextLayer.ID = self.ID + 1
            self.NextLayer.Init()

    def Calculate(self):
        self.OutputValues = self.InputValues

    def Evaluate(self, Input):
        self.InputValues = Input
        self.Calculate()
        assert self.OutputValues.shape[0] == self.OutputCardinality, "Data Cardinality does not match for {} Layer #{}".format(type(self).__name__, self.ID)
        return self.NextLayer.Evaluate(self.OutputValues)

    def Train(self, Input, Target, Rate):
        Output = self.Evaluate(Input)
        Difference = Target - Output
        self.Learn(Difference, Rate)

    def Learn(self, Difference, Factor):
        self.NextLayer.Learn(Difference, Factor)

    def CalculateGradient(self, NextGradient):
        self.PrevLayer.CalculateGradient(NextGradient)
        

class Input(Layer):
    def __init__(self, Inputs):
        self.InputCardinality = Inputs
        self.OutputCardinality = self.InputCardinality
        self.ID = 1

    def Calculate(self):
        self.OutputValues = np.atleast_2d(self.InputValues)

    def CalculateGradient(self, NextGradient):
        pass;

       
class Output(Layer):
    def __init__(self, Outputs):
        self.OutputCardinality = Outputs
        self.InputCardinality = self.OutputCardinality

    def Evaluate(self, Input):
        self.InputValues = Input
        self.Calculate()
        return self.OutputValues

    def Learn(self, Difference, Factor):
        Error = np.average(Difference ** 2)
        #print(Error)
        self.CalculateGradient(Difference)


class Straight(Layer):
    def Init(self):
        if self.PrevLayer.OutputCardinality is not None:
            self.InputCardinality = self.PrevLayer.OutputCardinality
            self.OutputCardinality = self.InputCardinality

            super().Init()

        else:
            super().Init()
            self.OutputCardinality = self.NextLayer.InputCardinality
            self.InputCardinality = self.OutputCardinality

        assert self.InputCardinality == self.PrevLayer.OutputCardinality, "InputCardinalitiy does not match for {} Layer #{}".format(type(self).__name__, self.ID)
        assert self.OutputCardinality == self.NextLayer.InputCardinality, "OutputCardinality does not match for {} Layer #{}".format(type(self).__name__, self.ID)
        

class Bias(Layer):
    def Init(self):
        if self.PrevLayer.OutputCardinality is not None:
            self.InputCardinality = self.PrevLayer.OutputCardinality
            self.OutputCardinality = self.InputCardinality + 1

            super().Init()

        else:
            super().Init()
            self.OutputCardinality = self.NextLayer.InputCardinality
            self.InputCardinality = self.OutputCardinality - 1
        
        assert self.InputCardinality == self.PrevLayer.OutputCardinality, "InputCardinalitiy does not match for {} Layer #{}".format(type(self).__name__, self.ID)
        assert self.OutputCardinality == self.NextLayer.InputCardinality, "OutputCardinality does not match for {} Layer #{}".format(type(self).__name__, self.ID)

    def Calculate(self):
        self.OutputValues = np.vstack((self.InputValues, np.ones((1, self.InputValues.shape[1]))))

    def CalculateGradient(self, NextGradient):
        self.PrevLayer.CalculateGradient(NextGradient[:self.InputValues.shape[0],:])


class Weight(Layer):

    Parameters = None
    tmpGradients = None

    def __init__(self, Outputs):
        self.OutputCardinality = Outputs
        
    def Init(self):
        self.InputCardinality = self.PrevLayer.OutputCardinality
        super().Init()
        self.Parameters = np.random.rand(self.InputCardinality, self.OutputCardinality)

    def Calculate(self):
        self.OutputValues = self.InputValues.T.dot(self.Parameters).T

    def Learn(self, Difference, Factor):
        super().Learn(Difference, Factor)
        self.Parameters = self.Parameters + Factor * self.tmpGradients

    def CalculateGradient(self, NextGradient):
        self.tmpGradients = self.InputValues.dot(NextGradient.T)
        self.PrevLayer.CalculateGradient(self.Parameters.dot(NextGradient))


class Tanh(Straight):
    def Calculate(self):
        self.OutputValues = np.tanh(self.InputValues)

    def CalculateGradient(self, NextGradient):
        self.PrevLayer.CalculateGradient(NextGradient * (1 - self.OutputValues ** 2))

