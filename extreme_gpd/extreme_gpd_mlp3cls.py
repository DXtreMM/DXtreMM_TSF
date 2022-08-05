import tensorflow as tf
import numpy as np
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

class ExtremeGpd:

    def __init__(self, threshold, lag):

        # Save Parameters
        self.lthreshold = threshold[0]
        self.rthreshold = threshold[1]
        self.lag = lag
        
        # Right Extreme Model
        self.modelLExtreme = ExtremeGpd.getExtremeLModel(lag)
        # Left Extreme Model
        self.modelRExtreme = ExtremeGpd.getExtremeRModel(lag)
        
        # Normal Model
        self.modelNormal = ExtremeGpd.getNormalModel(lag)

        # MLP Ensemble Classifier
        self.modelDetect = ExtremeGpd.getClassifierModel(lag)

        # GPD Distribution
        self.gpd = None

    def train(
            self,
            timeSeries,
            gpdParamRanges,
            numPsoParticles,
            numPsoIterations,
            numExtremeIterations,
            numNormalIterations,
            numClassifierIterations
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
        trainSummary['loss-classifier'] = \
            self.trainClassifierModel(timeSeries, numClassifierIterations)

        return trainSummary

    def predict(self, timeSeries, getAllOutputs=False):

        # Build Input
        inputData = []
        predOutputs=[]
        for i in range(self.lag, timeSeries.shape[0]+1):
            inputData.append(timeSeries[i - self.lag: i])

        inputData = np.array(inputData)
        # extremeProb = self.modelDetect.predict(inputData) ##MLP
        
        # Probability of being extreme
        inputData1 = inputData.reshape((inputData.shape[0],inputData.shape[1],1)) ## CNN and LSTM
        extremeProb = self.modelDetect.predict(inputData1)
        
        # subsequences=2
        # timesteps=inputData.shape[1]//subsequences
        # inputData1 = inputData.reshape((inputData.shape[0], subsequences, timesteps, 1)) ##CNN-LSTM subseq
        # extremeProb = self.modelDetect.predict(inputData1) 
        
        # = np.squeeze(extremeProb, axis=1) 
        
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

        ##for forked putput
        predOutputs = np.zeros(predNormal.shape)
        for i in range(predOutputs.shape[0]):
            if np.argmax(extremeProb[i])==0:
                predOutputs[i] = predNormal[i] 
            elif np.argmax(extremeProb[i])==1:
                predOutputs[i] = predRExtreme[i] 
            elif np.argmax(extremeProb[i])==2:
                predOutputs[i] = -predLExtreme[i] 
            
        if not getAllOutputs:
            return predOutputs
        else:
            return predOutputs, extremeProb, predNormal, predLExtreme, predRExtreme

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

            if -timeSeries[i] >= -self.lthreshold:
                inputData.append(-timeSeries[i - self.lag: i])

                exceedance = -timeSeries[i] + self.lthreshold
                outputData.append(self.gpdL.cdf(exceedance))
                # outputData.append(-timeSeries[i])

        inputData = np.array(inputData)
        # inputData = inputData.reshape((inputData.shape[0],inputData.shape[1],1))             ##LSTM
        outputData = np.expand_dims(np.array(outputData), axis=1)
        
        # self.modelLExtreme = ExtremeGpd.getExtremeLModel(self,self.lag)
        
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        history = self.modelLExtreme.fit(
            inputData, outputData, validation_split=0.2, epochs=numExtremeIterations,
            verbose=0, callbacks=[es]
        )
        
        left_loss_history=history.history['loss']
        
        inputData = []
        outputData = []
        
        for i in range(self.lag, timeSeries.shape[0]):

            if timeSeries[i] >= self.rthreshold:
                inputData.append(timeSeries[i - self.lag: i])

                exceedance = timeSeries[i] - self.rthreshold
                outputData.append(self.gpdR.cdf(exceedance))
                # outputData.append(timeSeries[i])

        inputData = np.array(inputData)
        # inputData = inputData.reshape((inputData.shape[0],inputData.shape[1],1))                ##LSTM
        outputData = np.expand_dims(np.array(outputData), axis=1)
        
        # self.modelRExtreme = ExtremeGpd.getExtremeRModel(self,self.lag)
        
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

            if timeSeries[i] <= self.rthreshold and timeSeries[i] > self.lthreshold:
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

    def trainClassifierModel(self, timeSeries, numClassifierIterations):
        
        '''
        pos_idx = []
        neg_idx = []
        
        for i in range(0, timeSeries.shape[0] - self.lag):
            if timeSeries[i + self.lag] > self.rthreshold:
                pos_idx.append(i)
                
            elif -timeSeries[i + self.lag] < -self.lthreshold:
                negpos_idx.append(i)
                
            else:
                neg_idx.append(i)
                
        take_neg = [True for i in range(len(neg_idx))]
        
        for i in range(len(take_neg)):
            
            low, high = 0, len(pos_idx+negpos_idx) - 1
            max_idx = -1
            
            while low < high:
                
                mid = (low + high) // 2
                
                if pos_idx[mid] <= neg_idx[i]:
                    max_idx = max(max_idx, pos_idx[mid])
                    low = mid + 1
                    
                else:
                    high = mid - 1
                    
            take_neg[i] = not (max_idx >= 0 and max_idx + self.lag >= i)
            
        inputData = []
        outputData = []
        
        for idx in (pos_idx):
            inputData.append(timeSeries[idx: idx + self.lag])
            outputData.append(1.0)
            
        for i in range(len(neg_idx)):
            if take_neg[i]:
                idx = neg_idx[i]
                inputData.append(timeSeries[idx: idx + self.lag])
                outputData.append(0.0)
                
        inputData = np.array(inputData) 
        outputData = np.array(outputData)[:, np.newaxis]
        
        indices = np.arange(inputData.shape[0])
        np.random.shuffle(indices)
        
        inputData = inputData[indices]
        outputData = outputData[indices]
        
        '''
        inputData = []
        outputData = []

        for i in range(self.lag, timeSeries.shape[0]):

            inputData.append(timeSeries[i - self.lag: i])
    
            if timeSeries[i] > self.rthreshold:
                outputData.append(1.0)
            elif -timeSeries[i] > -self.lthreshold:
                outputData.append(2.0)
            else:
                outputData.append(0.0)
        
        rext=len(np.where(outputData==1.0))
        lext=len(np.where(outputData==2.0))
        normal=len(np.where(outputData==0.0))
        total=len(outputData)
        
        inputData = np.array(inputData)
        inputData = inputData.reshape((inputData.shape[0],inputData.shape[1],1))            ##CNNand LSTM
        
        # subsequences = 2
        # timesteps = inputData.shape[1]//subsequences
        # inputData = inputData.reshape((inputData.shape[0], subsequences, timesteps, 1)) ##CNN -LSTM subseq
        outputData1 = np.array(outputData)
        onehot_encoder = OneHotEncoder(sparse=False)
        outputData=onehot_encoder.fit_transform(outputData1.reshape(len(outputData1), 1))
        
        optimizer = Adam(
            ExponentialDecay(
                1e-3, 50, 0.99
            )
        )
        
        cce=CategoricalCrossentropy()
        def w_categorical_crossentropy(y_true, y_pred, weights):
            nb_cl = len(weights)
            final_mask = tf.zeros_like(y_pred[:, 0])
            y_pred_max = tf.reduce_max(y_pred, axis=1)
            y_pred_max = tf.reshape(y_pred_max, (tf.shape(y_pred)[0], 1))
            y_pred_max_mat = tf.equal(y_pred, y_pred_max)
            for c_p, c_t in product(range(nb_cl), range(nb_cl)):
                final_mask += (tf.cast(weights[c_t, c_p],tf.float32) * tf.cast(y_pred_max_mat[:, c_p],tf.float32) * tf.cast(y_true[:, c_t],tf.float32))
            # return cce(y_true, y_pred) * final_mask
            return cce(y_true, y_pred) * tf.pow(final_mask, (1-y_pred_max))
            # return cce(y_true, y_pred) * tf.pow((1-y_pred_max),2) * final_mask
            
        w_array = np.ones((3,3))
        w_array[1, 0] = np.log(total/rext)
        w_array[2, 0] = np.log(total/lext)
        w_array[1, 2] = 2*np.log(total/rext)
        w_array[2, 1] = 2*np.log(total/lext)
        
        # w_array[1, 0] = w_array[2, 0] = 40.0
        # w_array[1, 2] = w_array[2, 1] = 80.0
        
        # tot=rext+lext+normal
        # freq_r=rext/tot
        # freq_l=lext/tot
        # freq_n=normal/tot
        
        # w_array = np.ones((3,3))
        # w_array[0, 0]=w_array[0, 1]=w_array[0, 2]=freq_n
        # w_array[1, 0]=w_array[1, 1]=w_array[1, 2]=freq_r
        # w_array[2, 0]=w_array[2, 1]=w_array[2, 2]=freq_l

        ncce = functools.partial(w_categorical_crossentropy, weights=w_array)
        
        #self.modelDetect.compile(optimizer, Evl.evl)
        self.modelDetect.compile(optimizer,loss=ncce)
        
        es = EarlyStopping(monitor='loss',mode='min', verbose=1, patience=5)
        
        history = self.modelDetect.fit(
            inputData, outputData, epochs=numClassifierIterations,
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
            BatchNormalization(),
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
            # loss=tf.losses.MeanSquaredError()
            loss=qloss
        )

        return modelNormal

    @staticmethod
    def getClassifierModel(lag):
        # model = Sequential([
        #     BatchNormalization(),
        #     Dense(64, activation='gelu'),
        #     # BatchNormalization(),
        #     Dense(128, activation='gelu'),
        #     # BatchNormalization(),
        #     Dense(64, activation='gelu'),
        #     # BatchNormalization(),
        #     Dense(32, activation='gelu'),
        #     # BatchNormalization(),
        #     Dense(16, activation='gelu'),
        #     # BatchNormalization(),
        #     Dense(3, activation='softmax')
        # ])
        # model.build(input_shape=(None,lag)) ##MLP
        
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu',input_shape=(lag, 1)),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            # Flatten(),
            # Conv1D(filters=32, kernel_size=3, activation='relu'),
            # MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(200,activation='relu'),
            BatchNormalization(),
            Dense(64,activation='relu'),
            BatchNormalization(),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax')
        ])
        model.build() #CNN
        
        # model = Sequential([
        #     LSTM(32, activation='relu', input_shape=(lag, 1)),
        #     Dense(32, activation='relu'),
        #     Dense(16, activation='relu'),
        #     # Dropout(0.4),
        #     Dense(3, activation='softmax')
        # ])
        # model.build() #LSTM
        
        # model = Sequential([
        #     Conv1D(filters=64, kernel_size=3, activation='relu',input_shape=(lag, 1)),
        #     Conv1D(filters=64, kernel_size=3, activation='relu'),
        #     BatchNormalization(),
        #     MaxPooling1D(pool_size=2),
        #     # Flatten(),
        #     # Conv1D(filters=32, kernel_size=3, activation='relu'),
        #     # MaxPooling1D(pool_size=2),
        #     Flatten(),
        #     Dense(200,activation='relu'),
        #     BatchNormalization(),
        #     # Dropout(0.3),
        #     RepeatVector(1),
        #     LSTM(32, activation='relu'),
        #     BatchNormalization(),
        #     Dense(16,activation='relu'),
        #     Dense(3, activation='softmax')
        # ])
        # model.build() #CNN-LSTM

        return model
        
def qloss(y_true, y_pred):
    # Pinball loss for multiple quantiles
    qs = [0.250]
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

class Evl:
    
    # evl loss
    EPS = 1e-8
    EXTREME_INDEX = 2.0
    
    @staticmethod
    def evl(y_true, y_pred):
            
        extreme_prop = tf.math.reduce_sum(y_true) \
            / tf.cast(tf.size(y_true), tf.float32)
        
        normal_prop = 1. - extreme_prop
        
        extreme_term = -normal_prop \
            * tf.pow(1 - y_pred / Evl.EXTREME_INDEX, Evl.EXTREME_INDEX) \
            * y_true * tf.math.log(y_pred + Evl.EPS)
            
        normal_term = -extreme_prop \
            * tf.pow(1 - (1 - y_pred) / Evl.EXTREME_INDEX, Evl.EXTREME_INDEX) \
            * (1 - y_true) * tf.math.log(1 - y_pred + Evl.EPS)
            
        return tf.reduce_mean(extreme_term + normal_term)

    '''
    #new (square) loss
    EPS =1e-8
    EXTREME_INDEX = 2.0

    @staticmethod
    def evl(y_true, y_pred):
            
        extreme_prop = tf.math.reduce_sum(y_true) \
            / tf.cast(tf.size(y_true), tf.float64)
        
        normal_prop = 1. - extreme_prop
        
        extreme_term = 1/extreme_prop \
            * tf.pow(1 - y_pred , Evl.EXTREME_INDEX) \
            * y_true 
            
        normal_term = 1/normal_prop \
            * tf.pow(1 - (1 - y_pred), Evl.EXTREME_INDEX) \
            * (1 - y_true)
            
        return tf.reduce_mean(extreme_term + normal_term)
    
    ''' 
    '''
    # sig loss
    def evl(y_true,y_pred):

        extreme_prop = tf.math.reduce_sum(y_true) \
            / tf.cast(tf.size(y_true), tf.float64)
        
        normal_prop = 1. - extreme_prop
        
        extreme_term = (1/extreme_prop) \
            * 1/(1+tf.math.exp(-15*(-y_pred+0.5))) \
            * y_true 
            
        normal_term = 1/normal_prop \
            * 1/(1+tf.math.exp(15*(-y_pred+0.5))) \
            * (1 - y_true)
            
        return tf.reduce_mean(extreme_term + normal_term)
    
    '''
    '''
    #focal loss
    EPS = 1e-8
    EXTREME_INDEX = 2.0

    @staticmethod
    def evl(y_true, y_pred):
            
        extreme_prop = tf.math.reduce_sum(y_true) \
            / tf.cast(tf.size(y_true), tf.float64)
        
        normal_prop = 1. - extreme_prop
        
        extreme_term = -normal_prop \
            * tf.pow(1 - y_pred , Evl.EXTREME_INDEX) \
            * y_true * tf.math.log(y_pred + Evl.EPS)
            
        normal_term = -extreme_prop \
            * tf.pow(1 - (1 - y_pred), Evl.EXTREME_INDEX) \
            * (1 - y_true) * tf.math.log(1 - y_pred + Evl.EPS)
            
        return tf.reduce_mean(extreme_term + normal_term)
    '''
    '''
    #ambiguous loss fn
    phi=-50
    def evl(y_true, y_pred):
           loss=(1/Evl.phi)*(tf.math.log(tf.math.exp(Evl.phi*y_pred)+tf.math.exp(Evl.phi*(1-y_pred))))
           return tf.reduce_mean(loss)

    #absolute error
    def evl(y_true,y_pred):
         loss=tf.math.abs(y_true-y_pred)
         return tf.reduce_mean(loss)
    '''
