
####Colaborated with Saranya Kadiyala(002822614) on analysis.py

import bagpy
from bagpy import bagreader
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

b = bagreader(r'path-to-file.bag')
print(f'\n{b.topic_table}\n')
csvfiles = []
for t in b.topics:
    data = b.message_by_topic(t)
    csvfiles.append(data)

print(f'Length of csv file is : {len(csvfiles[0][0][0])}')
print(f'Data in csv file is:{csvfiles[0]}')


# Read data from CSV files
df_open = pd.read_csv('/content/open_area_wns.csv')  
df_occluded = pd.read_csv('/content/Occluded_SL_f.csv')  
df_moving = pd.read_csv('/content/Straight_line_3wons.csv') 

# Extract columns for stationary conditions
easting_open = df_open['utm_easting']
northing_open = df_open['utm_northing']
altitude_open = df_open['altitude']
time_open = df_open['Time']

easting_occluded = df_occluded['utm_easting']
northing_occluded = df_occluded['utm_northing']
altitude_occluded = df_occluded['altitude']
time_occluded = df_occluded['Time']

# Extract columns for moving condition
easting_moving = df_moving['utm_easting']
northing_moving = df_moving['utm_northing']
altitude_moving = df_moving['altitude']
time_moving = df_moving['Time']

# Normalize data to the range [0, 1]
scaler = MinMaxScaler()

easting_open_normalized = scaler.fit_transform(easting_open.values.reshape(-1, 1))
northing_open_normalized = scaler.fit_transform(northing_open.values.reshape(-1, 1))
altitude_open_normalized = scaler.fit_transform(altitude_open.values.reshape(-1, 1))

easting_occluded_normalized = scaler.fit_transform(easting_occluded.values.reshape(-1, 1))
northing_occluded_normalized = scaler.fit_transform(northing_occluded.values.reshape(-1, 1))
altitude_occluded_normalized = scaler.fit_transform(altitude_occluded.values.reshape(-1, 1))

easting_moving_normalized = scaler.fit_transform(easting_moving.values.reshape(-1, 1))
northing_moving_normalized = scaler.fit_transform(northing_moving.values.reshape(-1, 1))
altitude_moving_normalized = scaler.fit_transform(altitude_moving.values.reshape(-1, 1))

# Calculate centroids for stationary conditions
centroid_open = np.array([np.mean(easting_open_normalized), np.mean(northing_open_normalized)])
centroid_occluded = np.array([np.mean(easting_occluded_normalized), np.mean(northing_occluded_normalized)])

# Subtract centroid from each data point for stationary conditions
easting_open_centered = easting_open_normalized - centroid_open[0]
northing_open_centered = northing_open_normalized - centroid_open[1]
easting_occluded_centered = easting_occluded_normalized - centroid_occluded[0]
northing_occluded_centered = northing_occluded_normalized - centroid_occluded[1]

# Plot normalized stationary northing vs. easting scatterplots
plt.figure(figsize=(10, 6))
plt.scatter(easting_open_normalized, northing_open_normalized, label='Open', marker='o')
plt.scatter(easting_occluded_normalized, northing_occluded_normalized, label='Occluded', marker='x')
plt.xlabel('Normalized Easting')
plt.ylabel('Normalized Northing')
plt.title('Normalized Stationary Northing vs. Easting Scatterplot')
plt.legend()
plt.show()

# Plot after subtracting the centroid from each normalized data point for stationary conditions
plt.figure(figsize=(10, 6))
plt.scatter(easting_open_centered, northing_open_centered, label='Open', marker='o')
plt.scatter(easting_occluded_centered, northing_occluded_centered, label='Occluded', marker='x')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Centered Normalized Easting')
plt.ylabel('Centered Normalized Northing')
plt.title('Stationary Northing vs. Easting After Centering (Normalized)')
plt.legend()
plt.text(-0.2, -0.2, f'Centroid Open: {centroid_open}', color='blue')
plt.text(-0.2, -0.4, f'Centroid Occluded: {centroid_occluded}', color='orange')
plt.show()

# Plot normalized moving northing vs. easting scatterplot with line of best fit
plt.figure(figsize=(10, 6))
plt.scatter(easting_moving, northing_moving, label='Moving', marker='o')
plt.xlabel('Moving Easting')
plt.ylabel('Moving Northing')
plt.title('Moving Northing vs. Easting Scatterplot')
plt.legend()

z_open = np.polyfit(easting_moving, northing_moving, 1)
p_open = np.poly1d(z_open)
plt.plot(easting_moving, p_open(easting_moving), "r--", label = "Line of best fit(Open)")

plt.show()


# Plot stationary altitude vs. time plot
plt.figure(figsize=(10, 6))
plt.plot(time_open, altitude_open, label='Open', marker='o')
plt.plot(time_occluded, altitude_occluded, label='Open', marker='x')
plt.xlabel('Time')
plt.ylabel('Altitude Stationary')
plt.title('Stationary Altitude vs. Time Plot')
plt.legend()
plt.show()

# Plot moving altitude vs. time plot
plt.figure(figsize=(10, 6))
plt.plot(time_moving, altitude_moving, label='Moving', marker='o')
plt.xlabel('Time')
plt.ylabel('Moving Altitude')
plt.title('Moving Altitude vs. Time Plot')
plt.legend()
plt.show()

#histogram Calculate the centroid
centroid_open = [df_open['utm_easting'].mean(), df_open['utm_northing'].mean()]
centroid_occluded = [df_occluded['utm_easting'].mean(), df_occluded['utm_northing'].mean()]

# Calculate Euclidean distance for each point to the centroid
open_distances = np.linalg.norm(df_open[['utm_easting', 'utm_northing']] - centroid_open, axis=1)
occluded_distances = np.linalg.norm(df_occluded[['utm_easting', 'utm_northing']] - centroid_occluded, axis=1)

# Plot histograms for Euclidean distances
plt.figure()
plt.hist(open_distances, bins=30, alpha=0.5, label='Open', color='Yellow')
#plt.hist(occluded_distances, bins=30, alpha=0.5, label='Occluded')
plt.legend()
plt.title('Euclidean Distances from Centroid')
plt.show()

# Plot histograms for Euclidean distances
plt.figure()
plt.hist(occluded_distances, bins=30, alpha=0.5, label='Occluded', color='cyan')
plt.legend()
plt.title('Euclidean Distances from Centroid')
plt.show()