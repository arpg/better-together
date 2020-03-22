"""Python implementation of the persistence filter algorithm."""
from numpy import log, exp, log1p
import numpy as np
import scipy.integrate as integrate
import sys
# need to download and import https://github.com/david-m-rosen/persistence_filter for the utility functions
#sys.path.append('../../persistence_filter/python/')
from persistence_filter_utils import logsum, logdiff, log_general_purpose_survival_function

# code is similar to that of Rosen to make it easy to compare the two
# checkout his work at https://github.com/david-m-rosen/persistence_filter which our work builds upon


class JCF:
    """Implementation of Joint Clique Filter"""

    def __init__(self, log_survival_function, num_features=2, intialization_time=0.0):
        self.log_survival_function = log_survival_function
        self.initalization_time = [intialization_time for i in range(num_features)]
        self.last_observation_time = [initalization_time for i in range(num_features)]

        self.log_likelihood = [0.0 for i in range(num_features)]

        self.clique_likelihood = np.sum(self.log_likelihood)

        self.log_clique_lower_evidence_sum = None

        self.log_clique_evidence = 0.0

        self.shifted_log_surivial_function = lambda t, i: self.log_survival_function(t - self.last_observation_time[i])

        self.shifted_logdF = lambda t1, t0, i: logdiff(self.shifted_log_surivial_function(t0, i), self.shifted_log_surivial_function(t1, i)) if t1 - t0 != 0 else 0.0

    def update(self, detector_output, observation_time, P_M, P_F):
        list_of_detected_features = []
        for i in range(len(self.log_likelihood)):
            if(detector_output[i] == 1):
                list_of_detected_features.append(i)

        clique_detected = False
        if(len(list_of_detected_features)):
            clique_detected = True

        for i in range(len(self.log_likelihood)):
            self.log_likelihood[i] = self.log_likelihood[i] + (log(1.0 - P_M) if detector_output[i] else log(P_M))
            if i in list_of_detected_features:
                self._last_observation_time[i] = observation_time
        
        if self.log_clique_lower_evidence_sum is not None:
            self.log_clique_lower_evidence_sum = logsum(self.log_clique_lower_evidence_sum, np.sum([el for el in self.log_likelihood]) + 
                                                self.shifted_logdF(observation_time, max(self.last_observation_time), np.asarray(self.last_observation_time).argmax())) + np.sum([(log(P_F) if el else log(1 - P_F)) for el in detector_output])
        else:
            self._log_clique_lower_evidence_sum = np.sum([(log(P_F) if el else log(1 - P_F)) for el in detector_output]) + log1p(-exp(self.shifted_log_survival_function(observation_time, 0)))
    
    

        self.clique_likelihood = np.sum(self.log_likelihood)

        self.log_clique_evidence = logsum(self.log_clique_lower_evidence_sum, self.clique_likelihood + self.shifted_log_survival_function(self.last_observation_time[np.asarray(self.last_observation_time).argmax()], np.asarray(self.last_observation_time).argmax()))

    def predict(self, predicition_time):
        return np.exp(self.clique_likelihood - self.log_clique_evidence + self.shifted_log_survival_function(prediction_time, np.asarray(self.last_observation_time).argmax()))

    # Stuff from the david-m-rosen/persistance-filter class
    @property
    def log_survival_function(self):
        return self._log
    
    @property
    def shifted_log_survival_function(self):
        return self._shifted_log_survival_function
      
    @property
    def last_observation_time(self):
        return self._last_observation_time
    
    @property
    def initialization_time(self):
        return self._initialization_time
    
