import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import special_ortho_group
from mpl_toolkits.mplot3d import Axes3D

# Start code from https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


# end code from https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python

class clique_simulator():
    def __init__(self, features_per_clique_mean, features_per_clique_varience, integration_function, observation_range = 25.0, P_Miss_detection=0.03, P_False_detection=0.01, confidence_range_ratio=0.5):
        # Preselected Feature Cliques
        self.number_of_cliques = 9
        self.clique_centers = [[0,-10,np.random.randn()],[0, 20, np.random.randn()],[ 0, 50, np.random.randn()],[30, 20, np.random.randn()],[-30, 20, np.random.randn()], [20, 0, np.random.randn()],[-20, 0, np.random.randn()],[20, 40, np.random.randn()],[-20, 40, np.random.randn()]]
        self.clique_num_features = np.random.randint(features_per_clique_mean - features_per_clique_varience, features_per_clique_mean + features_per_clique_varience, self.number_of_cliques)
        self.clique_features = [[special_ortho_group.rvs(3) @ np.array([np.random.randn() + 3,0,0]) for i in range(self.clique_num_features[c])] for c in range(self.number_of_cliques)]
        # all features always persist unless a index of this corrisponding to a clique feature is turned to False
        self.clique_feature_persistance = [[True for j in range(self.clique_num_features[i])] for i in range(self.number_of_cliques)]
        
        self.pose =  np.array([[1.0,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        self.pose_list = [self.pose]
        #should take just current pose and time
        # integration_function(pose_so3, time) -> pose_so3
        self.integration_function = integration_function
        
        self.time = 0
        self.timestep = 1.0
        
        self.observation_range = observation_range
        self.confidence_range = observation_range * confidence_range_ratio
        self.camera_fov = 55.0 * np.pi / 180.0
        self.max_observation_angle =  35.0 * np.pi / 180.0
        self.lidar_fov = 30.0* np.pi/180.0
        
        self.miss_detection_probability = P_Miss_detection
        self.false_detection_probability = P_False_detection
        self.miss_detection_probability_function = lambda d: 1 - np.exp(-1.0/self.confidence_range * d)
        
        self.sensor_noise_variance = 0.01
        
        self.range_cache = [[0.0 for i in range(self.clique_num_features[c])] for c in range(self.number_of_cliques)]
    def display_all_cliques(self, ax):
        for i in range(self.number_of_cliques):
            clique_temp = np.array(self.clique_features[i]) 
            ax.scatter(self.clique_centers[i][0] + clique_temp[:,0], self.clique_centers[i][1] + clique_temp[:,1], self.clique_centers[i][2] + clique_temp[:,2])
    
    def step_one_timestep(self):
        self.pose = self.integration_function(self.pose, self.time)
        self.pose_list.append(self.pose)
        self.time += self.timestep
    def display_trajectory(self, ax):
        temp = np.array(sim.pose_list)[:, :3, 3]
        ax.plot(temp[:, 0], temp[:, 1],temp[:, 2])
    
    def display_pose(self, ax, pose = None):
        if pose is None:
            pose = self.pose
        base = pose @ np.array([0.0,0,0,1])
        tip_x = pose @ np.array([5.0,0,0,0])
        tip_y = pose @ np.array([0,5.0,0,0])
        tip_z = pose @ np.array([0,0,5.0,0])
        #print(tip)
        ax.quiver(base[0], base[1], base[2], tip_x[0], tip_x[1], tip_x[2], color='r')
        ax.quiver(base[0], base[1], base[2], tip_y[0], tip_y[1], tip_y[2], color='g')
        ax.quiver(base[0], base[1], base[2], tip_z[0], tip_z[1], tip_z[2], color='y')
        
        #print('length of unit vector', np.linalg.norm(tip - base))
    def observable_features_range(self, pose=None, ax=None):
        if pose is None:
            pose = self.pose
        
            
        observed_feats = []
        for c in range(self.number_of_cliques):
            for i in range(self.clique_num_features[c]):
                point_range = np.linalg.norm(self.clique_features[c][i] + self.clique_centers[c] - pose[:3,3])
                self.range_cache[c][i] = point_range
                if point_range < self.observation_range:
                    observation_vector = np.array(self.clique_features[c][i] + np.random.randn(3) * np.sqrt(self.sensor_noise_variance) + self.clique_centers[c] - pose[:3,3])
                    if ax is not None:
                        test_inv_obs = np.linalg.inv(sim.pose[:3,:3]) @ observation_vector
                        ax.quiver(pose[0,3], pose[1,3], pose[2,3], observation_vector[0], observation_vector[1], observation_vector[2])
                        ax.quiver(0, 0, 0, test_inv_obs[0], test_inv_obs[1], test_inv_obs[2])
                    
                    observation_vector = observation_vector / np.linalg.norm(observation_vector)
                    normal_vector = np.array(self.clique_features[c][i])
                    normal_vector = normal_vector / np.linalg.norm(normal_vector)
                    transformed_observation_vector = observation_vector
                    transformed_origin_vector_x = pose @ np.array([1.0,0,0,0])
                    transformed_origin_vector_y = pose @ np.array([0,1.0,0,0])
                    transformed_origin_vector_z = pose @ np.array([0,0,1.0,0])
                    #print((transformed_origin_vector[[False, True, True]]), unit_vector(transformed_observation_vector[[False, True, True]]))
                    #print('x transformed',transformed_origin_vector_x, self.pose @ np.array([1.0,0,0,0]))
                    
                    angles = angle_between(observation_vector, normal_vector)
                    sensor_angles = (angle_between(transformed_observation_vector, transformed_origin_vector_x[:3]),
                                     angle_between(transformed_observation_vector, transformed_origin_vector_y[:3]),
                                     angle_between(transformed_observation_vector, transformed_origin_vector_z[:3]))
                    detection_var = False
                    if(self.clique_feature_persistance[c][i]):
                        detection_var = True if np.random.random() > self.miss_detection_probability_function(point_range) else False
                    else:
                        detection_var = True if np.random.random() > self.false_detection_probability else False
                    
                    observed_feats.append({
                        "clique_id": c,
                        "feature_id": i,
                        "observation_angle": angles,
                        "sensor_angle": sensor_angles,
                        "detection": detection_var,
                        "local_measurement": np.linalg.inv(sim.pose[:3,:3]) @ observation_vector,
                        "range": point_range
                    })
        return observed_feats
    def get_ranges_of_clique(self, clique_id):
        return self.range_cache[clique_id]
        
    def observable_features_camera(self,pose=None, ax=None):
        if pose is None:
            pose = self.pose
        cleaned_observed_feats = []
        #print(pose)
        observed_feats = self.observable_features_range(pose, ax)
        for el in observed_feats:
            
            if abs(el['sensor_angle'][0]) < self.camera_fov:
                if abs(el['observation_angle']) > self.max_observation_angle:
                #if np.pi - abs(el['observation_angle']) < self.max_observation_angle:
                #print('sensor angle',(el['sensor_angle'][0])* 180.0 / np.pi)
                    cleaned_observed_feats.append(el)
                else:
                    el['detection'] = False
                    cleaned_observed_feats.append(el)
        return cleaned_observed_feats
    
    def observable_features_lidar(self, pose=None, ax=None):
        if pose is None:
            pose = self.pose
        cleaned_observed_feats = []
        #print(pose)
        observed_feats = self.observable_features_range(pose, ax)
        for el in observed_feats:
            
            if abs(el['sensor_angle'][2]) < self.lidar_fov + np.pi/2.0 and abs(el['sensor_angle'][2]) > np.pi/2.0 - self.lidar_fov:
                if abs(el['observation_angle']) > self.max_observation_angle:
                    #if np.pi - abs(el['observation_angle']) < self.max_observation_angle:
                    #print('sensor angle',(el['sensor_angle'][0])* 180.0 / np.pi)
                    cleaned_observed_feats.append(el)
                else:
                    # in fov but not within angle of observation
                    el['detection'] = False
                    cleaned_observed_feats.append(el)
        return cleaned_observed_feats
    
    def display_set_of_features(self, ax, features):
        feature_points = [[] for i in range(self.number_of_cliques)]
        for el in features:
            feature_points[el['clique_id']].append(self.clique_features[el['clique_id']][el['feature_id']])
        feature_points = [np.array(el) for el in feature_points]
        #print(feature_points)
        for i in range(self.number_of_cliques):
            if feature_points[i].size != 0:
                ax.scatter(self.clique_centers[i][0] + feature_points[i][:,0], self.clique_centers[i][1] + feature_points[i][:,1], self.clique_centers[i][2] + feature_points[i][:,2])
