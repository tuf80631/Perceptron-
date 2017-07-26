import numpy as np  
 import matplotlib.pyplot as plt  
 import random  
 def generateData(N):  
     # generates N random points (x,y)  
      x = np.random.uniform(low=-1,high=1,size=N)  
      y = np.random.uniform(low=-1,high=1,size=N)  
      array = np.zeros((N, 2))  
      Ylabel = np.zeros(N)  
      # first random point for line to pass through  
      randomSingleX = np.random.uniform(low=-1,high=1,size=1)  
      randomSingleY = np.random.uniform(low=-1,high=1,size=1)  
      # second random point for line to pass through  
      randomSingleSecondX = np.random.uniform(low=-1,high=1,size=1)  
      randomSingleSecondY = np.random.uniform(low=-1,high=1,size=1)  
      # slope of line   
      m = (randomSingleSecondY - randomSingleY)/(randomSingleSecondX - randomSingleX)  
   # b intercept of line  
      b = randomSingleSecondY - m*randomSingleSecondX  
      ## store into training points into array  
      i = 0  
      for item in x:  
           array[i][0]=item  
           i+=1  
      i = 0  
      for item in y:  
           array[i][1]=item  
           i+=1  
      i = 0  
      ## calculate labels for the training examples  
      for item in array:  
           if item[1]>(item[0]*m+b):  
                Ylabel[i] = 1;  
           else:   
                Ylabel[i] = -1;  
           i+=1  
      return (array,Ylabel) # array = X (Nx2), Ylabel = Y (Nx1)  
 ## PLA random picks for each iteration       
 def pla(X,Y,w0):  
      w = w0  
      b = w0[2]  
      iteration = 0  
      s = set() #set to hold index values of X  
      m = set() #set to hold discarded index values of X  
      d = 0  
      #initalize set to store all index of s.   
      while (d<len(X)):  
           s.add(d)  
           d+=1  
      while (didConverge(X,Y,w,b)==False):  
           i = 0  
           while(len(s)!=0):  
                randomIndexList = random.sample(s,1)  
                randomIndex = randomIndexList[0]  
                s.remove(randomIndex)  
                m.add(randomIndex)  
                a = X[randomIndex][0]*w[0]+X[randomIndex][1]*w[1] + b  
                if (Y[randomIndex]*a)<=0:  
                     w[0]=w[0]+Y[randomIndex]*X[randomIndex][0]  
                     w[1]=w[1]+Y[randomIndex]*X[randomIndex][1]  
                     b = b + Y[randomIndex]  
                     iteration +=1  
                i+=1  
           s = m #m should contain all indexes so set s back to m.  
           m = set() #set m back to empty   
      w[2]=b  
      return (w,iteration)       
 ## finds optimal weights for lease squared regression. The third weight in w is the bias.   
 def pseudoinverse(X,Y):  
      newX = np.zeros((len(X), 3)) # initialize a new [Nx3] matrix to store the 1's bias   
      i=0  
      ## Process X into a [Nx3] matrix where the third column is 1's that handle the bias  
      for item in X:  
           newX[i][0]=item[0]  
           newX[i][1]=item[1]  
           newX[i][2]=1  
           i+=1  
      XTransposeX = np.dot(np.transpose(newX),newX)   
      inverseOfXTransposeX = np.linalg.inv(XTransposeX)  
      pseudoInverse = np.dot(inverseOfXTransposeX,np.transpose(newX))  
      w = np.dot(pseudoInverse,Y)  
      return w   
 ## checks all points in X to see if they are classified correctly or not with current weights and bias  
 def didConverge(X,Y,w,b):  
      converge = True  
      i = 0  
      for item in X:  
           if (((item[0]*w[0]+item[1]*w[1])+b)*Y[i])<=0:  
                converge = False  
           i+=1  
      return converge  
 # Runs 100 PLA test with and without PseudoInverse for N  
 def runPLAtests(nTest, N):  
      sumIterationPLA = 0  
      sumIterationPLAwithRegression = 0  
      averageIterationPLA = 0  
      averageIterationPLAwithRegression = 0  
      iterationsWithoutRegression = np.zeros(nTest)  
      iterationsWithRegression = np.zeros(nTest)  
      i = 0  
      while (i<nTest):  
           (X,Y) = generateData(N)  
           w0 = np.zeros((3))  
           (a,b) = pla(X,Y,w0) #a is weight and b is iterations  
           sumIterationPLA = sumIterationPLA + b  
           iterationsWithoutRegression[i]=b  
           (d,e) = pla(X,Y,pseudoinverse(X,Y)) #d is weight and e is iterations  
           iterationsWithRegression[i]=e   
           sumIterationPLAwithRegression = sumIterationPLAwithRegression + e  
           i+=1  
      print("iterationsWithoutRegression: ", N)  
      print(iterationsWithoutRegression)  
      print("iterationsWithRegression: ", N)  
      print(iterationsWithRegression)  
      #store the iteration results into txt file  
      nameOfFileWithoutRegression = str(N) + "iterationsWithoutRegression"  
      nameOfFileWithRegression = str(N) + "iterationsWithRegression"  
      np.savetxt(nameOfFileWithoutRegression, iterationsWithoutRegression)  
      np.savetxt(nameOfFileWithRegression, iterationsWithRegression)  
      averageIterationPLA = sumIterationPLA/nTest  
      averageIterationPLAwithRegression = sumIterationPLAwithRegression/nTest  
      return (averageIterationPLA, averageIterationPLAwithRegression)  
 #method to run all 6 experiments       
 def run6Test():  
      #10  
      print("N = 10")  
      (averageIterationPLA, averageIterationPLAwithRegression) = runPLAtests(100, 10)  
      print("PLA without Regression Average Iteration:", averageIterationPLA)  
      print("PLA with Regression Average Iteration:", averageIterationPLAwithRegression)  
      #50  
      print("N = 50")  
      (averageIterationPLA, averageIterationPLAwithRegression) = runPLAtests(100, 50)  
      print("PLA without Regression Average Iteration:", averageIterationPLA)  
      print("PLA with Regression Average Iteration:", averageIterationPLAwithRegression)  
      #100  
      print("N = 100")  
      (averageIterationPLA, averageIterationPLAwithRegression) = runPLAtests(100, 100)  
      print("PLA without Regression Average Iteration:", averageIterationPLA)  
      print("PLA with Regression Average Iteration:", averageIterationPLAwithRegression)  
      #200  
      print("N = 200")  
      (averageIterationPLA, averageIterationPLAwithRegression) = runPLAtests(100, 200)  
      print("PLA without Regression Average Iteration:", averageIterationPLA)  
      print("PLA with Regression Average Iteration:", averageIterationPLAwithRegression)  
      #500  
      print("N = 500")  
      (averageIterationPLA, averageIterationPLAwithRegression) = runPLAtests(100, 500)  
      print("PLA without Regression Average Iteration:", averageIterationPLA)  
      print("PLA with Regression Average Iteration:", averageIterationPLAwithRegression)  
      #1000  
      print("N = 1000")  
      (averageIterationPLA, averageIterationPLAwithRegression) = runPLAtests(100, 1000)  
      print("PLA without Regression Average Iteration:", averageIterationPLA)  
      print("PLA with Regression Average Iteration:", averageIterationPLAwithRegression)  
 #plots the current training example and line generated by the weights.  
 def plotGraph(plt,X,Y,w,b):  
      countPositive = 0  
      countNegative = 0  
      for item in Y:  
           if item == 1:  
                countPositive+=1  
           else:  
                countNegative+=1  
      X1Positive = np.zeros(countPositive)  
      X2Positive = np.zeros(countPositive)  
      X1Negative = np.zeros(countNegative)  
      X2Negative = np.zeros(countNegative)  
      i=0  
      j=0  
      k=0  
      for item in X:  
           if (Y[k]==1):  
                X1Positive[i]=item[0]  
                X2Positive[i]=item[1]  
                i+=1  
           if (Y[k]==-1):  
                X1Negative[j]=item[0]  
                X2Negative[j]=item[1]  
                j+=1  
           k+=1  
      fig = plt.figure(figsize=(5,5))  
      plt.xlim(-1.5,1.5)  
      plt.ylim(-1.5,1.5)  
      plt.scatter(X1Positive,X2Positive, color = 'red')  
      plt.scatter(X1Negative,X2Negative, color = 'blue')  
      l = np.linspace(-2,2)  
      plt.plot(l, -w[0]/w[1]*l-b/w[1], color = 'green')  
      plt.show()  
 #single test run for both PLAs  
 def testSampleCaseBothPLAs(N):  
      (X,Y) = generateData(N)  
      w0 = np.zeros((3))  
      np.savetxt('X.txt', X)  
      np.savetxt('Y.txt', Y)  
      w = pseudoinverse(X,Y)  
      np.savetxt('w.txt', w)  
      (w1,iterationR) = pla(X,Y,pseudoinverse(X,Y))  
      (w2,iterationNoR) = pla(X,Y,w0)  
      print("with Regression", (w1,iterationR,b1))  
      print("without Regression", (w2,iterationNoR,b2))  
      plotGraph(plt,X,Y,w1,b1)  
 #single test run for PLA with Pseudoinverse  
 def onlyWithRegression(N):  
      (X,Y) = generateData(N)  
      w = pseudoinverse(X,Y)  
      print(w)  
      plotGraph(plt,X,Y,w,w[2])  
      (w1,iterationR) = pla(X,Y,pseudoinverse(X,Y))  
      print("with Regression", (w1,iterationR))  
      plotGraph(plt,X,Y,w1,w1[2])  
 #single test run for PLA without Pseudoinverse  
 def withoutRegression(N):  
      (X,Y) = generateData(N)  
      w0 = np.zeros((3))  
      w = pseudoinverse(X,Y)  
      (w1,iterationR) = pla(X,Y,w0)  
      print("without", (w1,iterationR))  
      plotGraph(plt,X,Y,w1,w1[2])       
 def main():   
   run6Test() # runs 6 experiments for N = {10, 50, 100, 200, 500, 1000} to find average iterations for PLA and PLA with Regression  
 if __name__ == "__main__":  
   main()   