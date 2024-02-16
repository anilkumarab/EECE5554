#lab2
import csv
import rosbag
import matplotlib.pyplot as plt
import numpy as np

#function to convert .bag to .csv
def bag_t_csv(ipbagfiles, opcsvfiles):
    for ipbagfile, opcsvfile in zip(ipbagfiles, opcsvfiles):
        with rosbag.Bag(ipbagfile, 'r') as bag:
            with open(opcsvfile, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['Time', 'Latitude', 'Longitude', 'Altitude', 'UTM Easting', 'UTM Northing', 'Zone', 'Letter', 'HDOP', 'Fix Quality', 'GNGGA Read'])

                for topic, msg, timestamp in bag.read_messages(topics=['/gps']):
                    # Extract data from the message
                    time = msg.header.stamp
                    latitude = msg.latitude
                    longitude = msg.longitude
                    altitude = msg.altitude
                    utm_easting = msg.utm_easting
                    utm_northing = msg.utm_northing
                    zone = msg.zone
                    letter = msg.letter
                    hdop = msg.hdop
                    fix_quality = msg.fix_quality
                    gngga_read = msg.gngga_read

                    # Write the data to the CSV file
                    csvwriter.writerow([time, latitude, longitude, altitude, utm_easting, utm_northing, zone, letter, hdop, fix_quality, gngga_read])

if __name__ == "__main__":
    ipbagfiles = ['/home/abhi/Documents/RSN_allfiles/lab2/lab2-master/dataset/occludedRTK_4.bag', '/home/abhi/Documents/RSN_allfiles/lab2/lab2-master/dataset/openRTK_4.bag', '/home/abhi/Documents/RSN_allfiles/lab2/lab2-master/dataset/walkingRTK_4.bag']
    opcsvfiles = ['/home/abhi/Documents/RSN_allfiles/lab2/lab2-master/dataset/occludedRTK_4.csv', '/home/abhi/Documents/RSN_allfiles/lab2/lab2-master/dataset/openRTK_4.csv', '/home/abhi/Documents/RSN_allfiles/lab2/lab2-master/dataset/walkingRTK_4.csv']
    bag_t_csv(ipbagfiles, opcsvfiles)


def plot_scatter(csv_file):
    eastings = []
    northings = []
    altitude = []
    time = []
    with open(csv_file, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            eastings.append(float(row['UTM Easting']))
            northings.append(float(row['UTM Northing']))
            altitude.append(float(row['Altitude']))
            time.append(float(row['Time']))


    return eastings, northings, altitude, time

def centroid(eastings, northings):
    centroid_easting = np.mean(eastings)
    centroid_northing = np.mean(northings)
    return centroid_easting, centroid_northing

def euclid_dist(point, centroid):
    return np.sqrt((point[0] - centroid[0])**2 + (point[1] - centroid[1])**2)

if __name__ == "__main__":

# data from occludedRTK.csv
  occludedRTK_csv = '/content/occludedRTK_4.csv'
  occludedRTK_eastings, occludedRTK_northings, occludedRTK_alt, occludedRTK_time = plot_scatter(occludedRTK_csv)
  print("Number of data points in occludedRTK_eastings:", len(occludedRTK_eastings))
  print("Number of data points in occludedRTK_northings:", len(occludedRTK_northings))


#data from openRTk.csv
  openRTK_csv = '/content/openRTK_4.csv'
  openRTK_eastings, openRTK_northings, openRTK_alt, openRTK_time = plot_scatter(openRTK_csv)
  print("Number of data points in openRTK_eastings:", len(openRTK_eastings))
  print("Number of data points in openRTK_northings:", len(openRTK_northings))


 #Data from walkingRTK.csv
  walkingRTK_csv = '/content/walkingRTK_4.csv'
  walkingRTk_eastings, walkingRTk_northings, walkingRTK_alt, walkingRTK_time = plot_scatter(walkingRTK_csv)

  print("Number of data points in walkingRTK_alt:", len(walkingRTK_alt))
  print("Number of data points in walkingRTK_time:", len(walkingRTK_time))


#Plot-1
  occludedRTK_centroid = centroid(occludedRTK_eastings, occludedRTK_northings)
  easting_occl_cent = occludedRTK_eastings - occludedRTK_centroid[0]
  northing_occl_cent = occludedRTK_northings - occludedRTK_centroid[1]

  plt.scatter(easting_occl_cent, northing_occl_cent, label='OccludedRTK', color = 'red')
  plt.xlabel('UTM Easting in meters')
  plt.ylabel('UTM Northing in meters')
  plt.title('Scatter Plot(stationary) Northings vs Eastings')
  plt.legend()
  plt.grid(True)
  plt.show()

  openRTK_centroid = centroid(openRTK_eastings, openRTK_northings)
  easting_open_cent = openRTK_eastings - openRTK_centroid[0]
  northing_open_cent = openRTK_northings - openRTK_centroid[1]

  plt.scatter(easting_open_cent, northing_open_cent, label='openRTK', color = 'blue')
  plt.xlabel('UTM Easting in meters')
  plt.ylabel('UTM Northing in meters')
  plt.title('Scatter Plot(stationary) Northings vs Eastings')
  plt.legend()
  plt.grid(True)
  plt.show()


#2
# altitude vs time plot

  plt.scatter(occludedRTK_alt, occludedRTK_time, label = 'OccludedRTK')
  plt.xlabel('Time in seconds')
  plt.ylabel('Altitude in meters')
  plt.title('Altitude vs Time')
  plt.legend()
  plt.grid(True)
  plt.show()

  plt.scatter(openRTK_alt, openRTK_time, label = 'openRTK')
  plt.xlabel('Time in seconds')
  plt.ylabel('Altitude in meters')
  plt.title('Altitude vs Time')
  plt.legend()
  plt.grid(True)
  plt.show()

# 3
# Plot histograms of Euclidean distances for both occulded and open
# Calculate centroid
  occludedRTK_centroid = centroid(occludedRTK_eastings, occludedRTK_northings)
  print(occludedRTK_centroid)
  openRTK_centroid = centroid(openRTK_eastings, openRTK_northings)
  print(openRTK_centroid)

# Euclidean distances from each point to the centroid
  occludedRTK_distances = [euclid_dist((e, n), occludedRTK_centroid) for e, n in zip(occludedRTK_eastings, occludedRTK_northings)]
  print(np.mean(occludedRTK_distances))
  openRTK_distances = [euclid_dist((e, n), openRTK_centroid) for e, n in zip(openRTK_eastings, openRTK_northings)]
  print(np.mean(openRTK_distances))

# Plot histograms of Euclidean distances
  plt.figure(figsize=(10, 8))

  plt.hist(occludedRTK_distances, bins=20, color='red', alpha=0.7, label='OccludedRTK')
  plt.title('Histogram of Euclidean Distances from Centroid - OccludedRTK')
  plt.xlabel('Euclidean Distance')
  plt.ylabel('Frequency')
  plt.legend()

  plt.figure(figsize=(10, 8))

  plt.hist(openRTK_distances, bins=20, color='blue', alpha=0.7, label='OpenRTK')
  plt.title('Histogram of Euclidean Distances from Centroid - OpenRTK')
  plt.xlabel('Euclidean Distance')
  plt.ylabel('Frequency')
  plt.legend()

  plt.tight_layout()
  plt.show()

#4 Moving (walking) data northing vs. easting scatterplot with line of best fit
  plt.scatter(walkingRTk_eastings, walkingRTk_northings, label = 'WalkingRTK(moving)', color = 'yellow')
  plt.xlabel('UTM Easting in meters')
  plt.ylabel('UTM Northing in meters')
  plt.title('Scatter Plot(moving) Northings vs Eastings')

  slp, y_int = np.polyfit(walkingRTk_eastings, walkingRTk_northings, 1)
  plt.plot(walkingRTk_eastings, slp * np.array(walkingRTk_eastings) + y_int, color='black', linestyle='-', label='Line of Best Fit')

  plt.legend()
  plt.show()

#Calculating RMSE error
  #predicted northing coordinates using the line of best fit equation
  predicted_northings = slp * np.array(walkingRTk_eastings) + y_int

  # Calculate difference between observed and predicted northing coordinates)
  diff_obv_prd = np.array(walkingRTk_northings) - predicted_northings

  # Squaring the difference
  sqr_diff = diff_obv_prd ** 2

  # mean squared error 
  mse = np.mean(sqr_diff)

  # root mean squared error (RMSE)
  rmse = np.sqrt(mse)

  print("Root Mean Squared Error (RMSE):", rmse)



#5 Moving (walking) data altitude vs. time plot
  plt.plot(walkingRTK_alt[0:68], walkingRTK_time[0:68], label = 'walkingRTK', color = 'hotpink')
  plt.xlabel('Time in meters')
  plt.ylabel('Altitude in meters')
  plt.title('Altitude vs Time')
  plt.legend()
  plt.grid(True)
  plt.show()

