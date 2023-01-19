
class solution:
    def __init__(self):
        self.best = 0
        self.bestIndividual=[]
        self.convergence1 = []
        self.convergence2 = []
        self.convergence3 = []

        self.optimizer=""
        self.objfname=""
        self.lb=0
        self.ub=0

        self.weights=[]
        self.biases=[]
        self.hiddenNode=0

        self.trainAcc=None
        self.testAcc=None

        self.beta=None
        self.F_measure = None
        self.recall = None
        self.precision = None
        



        
