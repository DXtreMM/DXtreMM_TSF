import tensorflow as tf
import numpy as np
import pandas as pd
import random
import functools
from itertools import product
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Bidirectional, BatchNormalization, GRU, Lambda, Conv1D, MaxPooling1D, Flatten, TimeDistributed, Dropout, RepeatVector

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy, CategoricalCrossentropy, MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

class vieextExtremeGpd:

    def __init__(self, threshold, lag):

        # Save Parameters
        self.lthreshold = threshold[0]
        self.rthreshold = threshold[1]
        self.lag = lag
        
        # Right Extreme Model
        self.modelLExtreme = vieextExtremeGpd.getExtremeLModel(lag)
        # Left Extreme Model
        self.modelRExtreme = vieextExtremeGpd.getExtremeRModel(lag)
        
        # Normal Model
        self.modelNormal = vieextExtremeGpd.getNormalModel(lag)

        # GPD Distribution
        self.gpd = None

    def train(
            self,
            timeSeries,
            gpdParamRanges,
            numPsoParticles,
            numPsoIterations,
            numExtremeIterations,
            numNormalIterations
    ):
        trainSummary = dict()
            
        # Estimate GPD Parameters
        trainSummary['gpd-convergence'] = \
            self.estimateParams(timeSeries, gpdParamRanges, numPsoParticles, numPsoIterations)

        # Train Extreme Model
        trainSummary['loss-extreme'] = \
            self.trainExtremeModel(timeSeries, numExtremeIterations)

        # Train Normal Model
        trainSummary['loss-normal'] = \
            self.trainNormalModel(timeSeries, numNormalIterations)

        # Train Ensemble Model
        # trainSummary['loss-classifier'] = \
            # self.trainClassifierModel(timeSeries, numClassifierIterations)

        return trainSummary

    def predict(self, timeSeries,train_low, test_high, run_count, file, getAllOutputs=False):

        # Build Input
        inputData = []
        predOutputs=[]

        for i in range(train_low+self.lag, test_high):
            inputData.append(timeSeries[i - self.lag: i])
        inputData = np.array(inputData)

        # Predict using Extreme Model
        # inputData = inputData.reshape((inputData.shape[0],inputData.shape[1])) ## lstm predict
        predRExtreme = self.gpdR.computeQuantile(self.modelRExtreme.predict(inputData)) \
            + self.rthreshold
        # predRExtreme=self.modelRExtreme.predict(inputData)
        predRExtreme = np.squeeze(predRExtreme, axis=1)
        
        predLExtreme = self.gpdL.computeQuantile(self.modelLExtreme.predict(inputData)) \
            - self.lthreshold
        # predLExtreme =self.modelLExtreme.predict(inputData)
        predLExtreme = np.squeeze(predLExtreme, axis=1)

        # Predict using Normal Model
        predNormal = self.modelNormal.predict(inputData)
        predNormal = np.squeeze(predNormal, axis=1)

        # Output
        #predOutputs = extremeProb * predExtreme + (1 - extremeProb) * predNormal

        ##for forked output
        predOutputs = np.zeros(predNormal.shape)
        
        file_name=str('VD_AE_extended/'+file+'/_risk_test_wholeVIE{}.npy'.format(run_count))
        whole_result = np.load(file_name)
        
        result=np.argmax(whole_result, axis=1)
        
        for i in range(predOutputs.shape[0]):
            if (result[i]==0):
                predOutputs[i] = predNormal[i]
            elif (result[i]==1):
                predOutputs[i] = predRExtreme[i] 
            elif (result[i]==2):
                predOutputs[i] = -predLExtreme[i] 
            
        if not getAllOutputs:
            return predOutputs
        else:
            return predOutputs, predNormal, predLExtreme, predRExtreme

    def estimateParams(
            self,
            timeSeries,
            gpdParamRanges,
            numPsoParticles,
            numPsoIterations
    ):

        # Compute Exceedances Series
        exceedSeries = timeSeries[timeSeries > self.rthreshold] - self.rthreshold

        # Compute GPD Parameters by performing ML Estimation using PSO
        paramsr, _, maxLogLikelihoodValuesr = GpdEstimate.psoMethod(
            exceedSeries,
            Pso.computeInitialPos(gpdParamRanges, numPsoParticles),
            numIterations=numPsoIterations
        )

        # Create the GPD Distribution Object
        self.gpdR = GeneralizedParetoDistribution(*paramsr)
        
        # Compute Exceedances Series
        exceedSeries = -timeSeries[-timeSeries > -self.lthreshold] + self.lthreshold

        # Compute GPD Parameters by performing ML Estimation using PSO
        paramsl, _, maxLogLikelihoodValuesl = GpdEstimate.psoMethod(
            exceedSeries,
            Pso.computeInitialPos(gpdParamRanges, numPsoParticles),
            numIterations=numPsoIterations
        )
        self.gpdL = GeneralizedParetoDistribution(*paramsl)

        return [maxLogLikelihoodValuesl,maxLogLikelihoodValuesr]

    def trainExtremeModel(self, timeSeries, numExtremeIterations):

        inputData = []
        outputData = []
        
        for i in range(self.lag, timeSeries.shape[0]):

            if -timeSeries[i] > -self.lthreshold:
                inputData.append(-timeSeries[i - self.lag: i])

                exceedance = -timeSeries[i] + self.lthreshold
                outputData.append(self.gpdL.cdf(exceedance))
                # outputData.append(-timeSeries[i])

        inputData = np.array(inputData)
        # inputData = inputData.reshape((inputData.shape[0],inputData.shape[1],1))             ##LSTM
        outputData = np.expand_dims(np.array(outputData), axis=1)
        
        # self.modelLExtreme = OracleExtremeGpd.getExtremeLModel(self,self.lag)
        
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        history = self.modelLExtreme.fit(
            inputData, outputData, validation_split=0.2, epochs=numExtremeIterations,
            verbose=0, callbacks=[es]
        )
        
        left_loss_history=history.history['loss']
        
        inputData = []
        outputData = []
        
        for i in range(self.lag, timeSeries.shape[0]):

            if timeSeries[i] > self.rthreshold:
                inputData.append(timeSeries[i - self.lag: i])

                exceedance = timeSeries[i] - self.rthreshold
                outputData.append(self.gpdR.cdf(exceedance))
                # outputData.append(timeSeries[i])

        inputData = np.array(inputData)
        # inputData = inputData.reshape((inputData.shape[0],inputData.shape[1],1))                ##LSTM
        outputData = np.expand_dims(np.array(outputData), axis=1)
        
        # self.modelRExtreme = OracleExtremeGpd.getExtremeRModel(self,self.lag)
        
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        history = self.modelRExtreme.fit(
            inputData, outputData, validation_split=0.2, epochs=numExtremeIterations,
            verbose=0, callbacks=[es]
        )
        right_loss_history=history.history['loss']
        
        return [left_loss_history,right_loss_history]

    def trainNormalModel(self, timeSeries, numNormalIterations):

        inputData = []
        outputData = []

        for i in range(self.lag, timeSeries.shape[0]):

            if timeSeries[i] <= self.rthreshold and timeSeries[i] >= self.lthreshold:
                inputData.append(timeSeries[i - self.lag: i])
                outputData.append(timeSeries[i])

        inputData = np.array(inputData)
        # inputData = inputData.reshape((inputData.shape[0],inputData.shape[1],1))            ##LSTM
        outputData = np.expand_dims(np.array(outputData), axis=1)
        
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        history = self.modelNormal.fit(
            inputData, outputData, validation_split=0.2, epochs=numNormalIterations,
            verbose=0, callbacks=[es]
        )

        return history.history['loss']



    @staticmethod
    def getExtremeLModel(lag):
        modelLExtreme = Sequential([
            BatchNormalization(),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='sigmoid')
            # Lambda(lambda x:self.gpdL.computeQuantile(x)- self.lthreshold)
        ])
        modelLExtreme.build(input_shape=(None, lag))
        
        # modelLExtreme = Sequential([
        #     LSTM(32, activation='relu', input_shape=(lag, 1)),
        #     # LSTM(16, activation='relu'),
        #     Dense(1, activation='sigmoid')
        # ])
        # modelLExtreme.build() #LSTM
 
        modelLExtreme.compile(
            optimizer=Adam(
                ExponentialDecay(
                    # 0.02, 50, 0.9
                    1e-3, 50, 0.99
                )
            ),
            # loss=tf.losses.MeanSquaredError()
            loss=qloss
        )

        return modelLExtreme
        
    def getExtremeRModel(lag):
      
        modelRExtreme = Sequential([
            BatchNormalization(),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='sigmoid')
            # Lambda(lambda x:self.gpdR.computeQuantile(x)+ self.rthreshold)
        ])
        modelRExtreme.build(input_shape=(None, lag))
        
        # modelRExtreme = Sequential([
        #     LSTM(32, activation='relu', input_shape=(lag, 1)),
        #     # LSTM(16, activation='relu'),
        #     Dense(1, activation='sigmoid')
        # ])
        # modelRExtreme.build() #LSTM
 
        modelRExtreme.compile(
            optimizer=Adam(
                ExponentialDecay(
                    # 0.02, 50, 0.9
                    1e-3, 50, 0.99
                )
            ),
            # loss=tf.losses.MeanSquaredError()
            loss=qloss
        )

        return modelRExtreme

    @staticmethod
    def getNormalModel(lag):
        modelNormal = Sequential([
            # BatchNormalization(),
            Dense(64, activation='relu'),
            # BatchNormalization(),
            Dense(128, activation='relu'),
            # BatchNormalization(),
            Dense(64, activation='relu'),
            # BatchNormalization(),
            Dense(32, activation='relu'),
            # BatchNormalization(),
            Dense(16, activation='relu'),
            # BatchNormalization(),
            Dense(1, activation='linear'),
        ])
        modelNormal.build(input_shape=(None, lag))

        # modelNormal = Sequential([
        #     LSTM(32, activation='relu', input_shape=(lag, 1)),
        #     # LSTM(32, activation='relu'),
        #     Dense(1, activation='linear')
        # ])
        # modelNormal.build() #LSTM
        
        modelNormal.compile(
            optimizer=Adam(
                ExponentialDecay(
                    # 0.02, 50, 0.9
                    1e-3, 50, 0.99
                )
            ),
            loss=tf.losses.MeanSquaredError()
            # loss=qloss
        )

        return modelNormal

def qloss(y_true, y_pred):
    # Pinball loss for multiple quantiles
    qs = [0.350]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = tf.cast(y_true, tf.float32) - y_pred
    v = tf.maximum(q*e, (q-1.0)*e)
    return K.sum(v) 



class GeneralizedParetoDistribution:

    def __init__(self, shapeParam, scaleParam):
        """
        Creates instance of the Generalized Pareto Distribution
        based on the shape and scale parameters provided

        :param shapeParam: shape parameter of the distribution
        :param scaleParam: scale parameter of the distribution
        """

        self.shapeParam = shapeParam
        self.scaleParam = scaleParam

    def computeQuantile(self, p):
        """
        Compute the p-quantile of this distribution

        :param p: CDF probability
        :return: the point z such that CDF(z) = p, i.e. the
        p-quantile of this distribution
        """

        if self.shapeParam != 0:
            return self.scaleParam * ((1 - p) ** (-self.shapeParam) - 1) / self.shapeParam

        else:
            return - self.scaleParam * np.log(1 - p)

    def pdf(self, x):
        """
        Compute PDF for all values in the input

        :param x: scalar or a numpy array of any shape
        :return: scalar value if x is scalar, or numpy array of shape
        same as x if x is a numpy array. This is the PDF at every point in x
        """

        if self.shapeParam != 0:
            return (1 + self.shapeParam * x / self.scaleParam) ** (-1 / self.shapeParam - 1) \
                / self.scaleParam

        else:
            return np.exp(-x / self.scaleParam) / self.scaleParam

    def cdf(self, x):
        """
        Compute CDF for all values in the input

        :param x: scalar or a numpy array of any shape
        :return: scalar value if x is scalar, or numpy array of shape
        same as x if x is a numpy array. This is the CDF at every point in x
        """

        if self.shapeParam != 0:
            return 1 - (1 + self.shapeParam * x / self.scaleParam) ** (-1 / self.shapeParam)

        else:
            return 1 - np.exp(-x / self.scaleParam)

    @staticmethod
    def logLikelihood(shapeParam, scaleParam, data):
        """
        Computes log likelihood of the data given the parameters

        :param shapeParam: shape parameter of the distribution
        :param scaleParam: scale parameter of the distribution
        :param data: data whose log likelihood is to be computed
        :return: log likelihood of the data given the parameters
        """

        if scaleParam <= 0:
            return None

        n = data.shape[0]

        if shapeParam == 0:
            if np.any(data < 0):
                return None

            return -n * np.log(scaleParam) - np.sum(data, axis=0) / scaleParam

        logArg = 1 + shapeParam * data / scaleParam
        if np.any(logArg <= 0):
            return None

        return -n * np.log(scaleParam) \
               - (1 / shapeParam + 1) * np.sum(np.log(logArg), axis=0)

    @staticmethod
    def computeNegLogLikelihoodGrad(shapeParam, scaleParam, data):
        """
        Computes the gradient of the log likelihood of the GPD distribution
        with respect to the shape and scale parameters

        :param shapeParam: shape parameter
        :param scaleParam: scale parameter
        :param data: the data, a numpy array of shape (n,)
        :return: (derivative with respect to shape parameter,
            derivative with respect to scale parameter)
        """

        n = data.shape[0]

        if shapeParam == 0:
            shapeGrad, scaleGrad = 0, n / scaleParam \
                                   - np.sum(data, axis=0) / (scaleParam * scaleParam)

        else:
            logArg = 1 + shapeParam * data / scaleParam
            shapeGrad = \
                -np.sum(np.log(logArg), axis=0) / np.square(shapeParam) \
                + (1 + 1 / shapeParam) * np.sum(data / logArg, axis=0) / scaleParam

            scaleGrad = n / scaleParam \
                - ((1 + 1 / shapeParam)
                   * shapeParam
                   * (1 / np.square(scaleParam))
                   * np.sum(data / logArg, axis=0))

        return shapeGrad, scaleGrad

class GpdEstimate:

    @staticmethod
    def psoMethod(
            data,
            initialPos,
            inertiaCoeff=1,
            inertiaDamp=0.99,
            personalCoeff=2,
            socialCoeff=2,
            numIterations=20
    ):
        """
        PSO method for maximum likelihood estimation of GPD's parameters

        :param data: the data, it is a numpy array of shape (n,)
        :param initialPos: initial positions of the particles in the
        parameter space, it is a numpy array of shape (numParticles, 2)
        :param inertiaCoeff: coefficient used for updating the velocity
        based on previous velocity
        :param inertiaDamp: used for damping inertia coefficient after
        every iteration.
        :param personalCoeff: coefficient used for updating the velocity
        based on personal best
        :param socialCoeff: coefficient used for updating the velocity
        based on global best
        :param numIterations: number of iterations to be performed
        :return: (estimated parameters,
            maximum value of the log likelihood,
            global maximum likelihood over each iteration),
        where the estimated parameters is a numpy array of shape (2,)
        containing the shape and scale parameters in that order
        """

        def minFunc(param):
            """ The function to minimize - negative log likelihood
             of the data given the parameters """

            logLikelihood = GeneralizedParetoDistribution\
                .logLikelihood(param[0], param[1], data)

            return -logLikelihood if logLikelihood is not None else np.inf

        params, bestCost, bestCosts = Pso.pso(
            minFunc,
            initialPos,
            inertiaCoeff,
            inertiaDamp,
            personalCoeff,
            socialCoeff,
            numIterations
        )

        return params, -bestCost, np.array(list(map(lambda x: -x, list(bestCosts))))

class Pso:

    @staticmethod
    def computeInitialPos(
            paramRange,
            numParticles
    ):
        """
        Compute initial positions of particles by sampling for each
        component of each particle from the uniform distribution
        with end points specified for each component as an argument
        taken by this function.

        :param paramRange: list of 2-tuples (low, high) of length equal
        to the dimension of the parameter space. There should be a
        tuple for every dimension. Hence, for dimension d, the tuple
        (low, high) says that each particle position's dth component must
        be sampled from uniform(low, high).
        :param numParticles: number of particles
        :return: initial position matrix of shape (numParticles, dimension)
        containing initial position vectors for each particle.
        """

        dim = len(paramRange)
        initialPos = np.zeros((numParticles, dim))

        for d in range(dim):
            low, high = paramRange[d]
            initialPos[:, d] = np.random.uniform(low, high, (numParticles,))

        return initialPos

    @staticmethod
    def pso(
            minFunc,
            initialPos,
            inertiaCoeff=1,
            inertiaDamp=0.99,
            personalCoeff=2,
            socialCoeff=2,
            numIterations=20
    ):
        """
        Particle Swarm Optimization algorithm

        :param minFunc: function which is to be minimized. It must
        accept a numpy array of shape (dim,) where dim is the dimension
        of the parameter space
        :param initialPos: initial positions for each of the particles.
        It should be a numpy array of shape (numParticles, dim)
        :param inertiaCoeff: coefficient used for updating the velocity
        based on previous velocity
        :param inertiaDamp: used for damping inertia coefficient after
        every iteration.
        :param personalCoeff: coefficient used for updating the velocity
        based on personal best
        :param socialCoeff: coefficient used for updating the velocity
        based on global best
        :param numIterations: number of iterations to be performed
        :return: (optimized parameters,
            optimal value of the function,
            global best cost values at each iteration),
        where the optimized parameters is a numpy array of shape (dim,)
        and optimal value is the value of the function achieved by these
        parameters
        """

        numParticles, dim = initialPos.shape

        pos = initialPos.copy()
        vel = np.zeros(pos.shape)

        bestPos = initialPos.copy()
        bestCosts = np.zeros((numParticles,))
        bestParticle = None

        for i in range(numParticles):
            bestCosts[i] = minFunc(pos[i])
            bestParticle = i if bestParticle is None \
                or bestCosts[i] < bestCosts[bestParticle] else bestParticle

        iterBestCosts = np.zeros((numIterations,))
        for iterNum in range(numIterations):

            vel = inertiaCoeff * vel \
                + personalCoeff * np.random.rand(numParticles, dim) * (bestPos - pos) \
                + (socialCoeff * np.random.rand(numParticles, dim)
                    * (np.expand_dims(bestPos[bestParticle], axis=0) - pos))

            pos = pos + vel

            for i in range(numParticles):
                currValue = minFunc(pos[i])

                if currValue < bestCosts[i]:
                    bestCosts[i] = currValue
                    bestPos[i, :] = pos[i, :]

                    if currValue < bestCosts[bestParticle]:
                        bestParticle = i

            iterBestCosts[iterNum] = bestCosts[bestParticle]
            inertiaCoeff *= inertiaDamp

        return bestPos[bestParticle], bestCosts[bestParticle], iterBestCosts

