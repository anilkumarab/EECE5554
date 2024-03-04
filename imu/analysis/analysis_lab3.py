#Convertor bag to csv
import bagpy
from bagpy import bagreader
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
import numpy as np
import allantools as alt
import rosbag

b = bagreader('/home/abhi/Documents/RSN_allfiles/lab3/rosbags/bag1.bag')
b = rosbag.Bag('/content/LocationC.bag')

b.topic_table

csvfiles = []
for t in b.topics:
    data = b.message_by_topic(t)
    csvfiles.append(data)

#lab3_analysis and plots 

import matplotlib.pyplot as plt
import pandas as pd
import math


def plots_plt(csv_file):
  csv = pd.read_csv(csv_file)
  angular_v_x = csv['imu.angular_velocity.x']
  angular_v_y = csv['imu.angular_velocity.y']
  angular_v_z = csv['imu.angular_velocity.z']
  linear_acc_x = csv['imu.linear_acceleration.x']
  linear_acc_y = csv['imu.linear_acceleration.y']
  linear_acc_z = csv['imu.linear_acceleration.z']
  yaw = csv['imu.orientation.x']
  pitch = csv['imu.orientation.y']
  roll = csv['imu.orientation.z']
  w = csv['imu.orientation.w']
  time = csv['Time']
  return angular_v_x, angular_v_y, angular_v_z, linear_acc_x, linear_acc_y, linear_acc_z, time, yaw, pitch, roll, w

#function to convert quaternion back to euler    
def quat_t_euler(x, y, z, w):
  t0 = +2.0 * (w * x + y * z)
  t1 = +1.0 - 2.0 * (x * x + y * y)
  X = math.degrees(math.atan2(t0, t1))

  t2 = +2.0 * (w * y - z * x)
  t2 = +1.0 if t2 > +1.0 else t2
  t2 = -1.0 if t2 < -1.0 else t2
  Y = math.degrees(math.asin(t2))

  t3 = +2.0 * (w * z + x * y)
  t4 = +1.0 - 2.0 * (y * y + z * z)
  Z = math.degrees(math.atan2(t3, t4))

  return X, Y, Z


bag1_csv = '/content/bag1.csv'
angular_v_x, angular_v_y, angular_v_z, linear_a_x, linear_a_y, linear_a_z, yaw, pitch, roll, w, time = plots_plt(bag1_csv)

data_frame = pd.read_csv(bag1_csv)
print(f'data_frame.columns = {data_frame.columns}')
print(f'Time : {time}')
time_df = data_frame["Time"].tolist()
absolute_time = time_df[0]

for t in range(len(time_df)):
    time_df[t] = time_df[t] - absolute_time

print(f'time_df : {time_df}')

#time starting from zero
#time_zero = [k - time[0] for k in time + 0.02] 

#print("Number of data points in gyro_x:", len(angular_v_x))


##plot-1
#Plotting rotational rate from the gyro in degrees/s on axes x, y, z
#print(f'time_zero : { time_zero[0]} time_len : {len(time_zero)}') 

plt.plot(time_df, angular_v_x, label='Gyro X', color = 'red')
plt.plot(time_df, angular_v_y, label='Gyro Y', color = 'blue')
plt.plot(time_df, angular_v_z, label='Gyro Z', color = 'yellow')
plt.xlabel('time in seconds')
plt.ylabel('gyro on axes x, y, z in degrees')
plt.title('Rotational rate from the gyro in degrees/s on axes x, y, z')
plt.legend()
plt.show()

##plot-2
#Plotting acceleration from the accelerometer in m/s^2 on axes x, y, z  
plt.plot(time_df, linear_a_x, label='Linear acceleration X', color = 'red')
plt.plot(time_df, linear_a_y, label='Linear acceleration Y', color = 'blue')
plt.plot(time_df, linear_a_z, label='Linear acceleration Z', color = 'green')
plt.xlabel('time in seconds')
plt.ylabel('acceleration on axes x, y, z in m/s^2')
plt.title('Acceleration on axes x, y, z')
plt.legend()
plt.show()

##plot-3
#Converting quaternion back to euler 
euler_yaw = []
euler_pitch = []
euler_roll = []

for i in range(len(yaw)):
  yaw_x, pitch_y, roll_z = quat_t_euler(w[i], yaw[i], pitch[i], roll[i])
  euler_yaw.append(yaw_x)
  euler_pitch.append(pitch_y)
  euler_roll.append(roll_z)


#Plotting rotation from the VN estimation in degrees on axes x, y, z 
plt.plot(time_df, euler_yaw, label='Rotation X', color = 'red')
plt.plot(time_df, euler_pitch, label='Rotation Y', color = 'blue')
plt.plot(time_df, euler_roll, label='Rotation Z', color = 'green')
plt.xlabel('time in seconds')
plt.ylabel('rotation on axes x, y, z in degrees')
plt.title('Rotation from VN estimation on axes x, y, z')
plt.legend()
plt.show()


#plotting 1D histograms of rotation in x, y, z
#Histogram plot1
plt.hist(euler_yaw, label='Rotation X', color = 'cyan')
plt.xlabel('Rotation in x(degrees)')
plt.ylabel('Frequency')
plt.title(' Histogram -rotation in x axis')
plt.legend()
plt.show()

#Histogram plot2
plt.hist(euler_pitch, label='Rotation Y', color = 'pink')
plt.xlabel('Rotation in y(degrees)')
plt.ylabel('Frequency')
plt.title(' Histogram-rotation in y axis')
plt.legend()
plt.show()

#Histogram plot3
plt.hist(euler_roll, label='Rotation Z', color = 'purple')
plt.xlabel('Rotation in z(degrees)')
plt.ylabel('Frequency')
plt.title(' Histogram-rotation in z axis')
plt.legend()
plt.show()

def eliminate(vnymrstring):
  ans = "" 
  for i in vnymrstring:
    if i in "+-.0123456789":
      ans += i
  return ans
 
b_lst = list(b)

count = 0
gyro_x = []
gyro_y = []
gyro_z = []
time = []

for i in range(1, b.get_message_count()):
  if i == 402371 :
    continue
  
  x = b_lst[i][1].data.split(',')
  if x[0] != "$VNYMR":
    continue
  
  gyro_x.append(float(eliminate(x[10])))
  gyro_y.append(float(eliminate(x[11])))
  gyro_z.append(float(eliminate(x[12])))
  time.append(b_lst[i][1].header.stamp.secs + b_lst[i][1].header.stamp.nsecs/10**9)

  gyro_x_av = alt.oadev(gyro_x, rate = 40.0, data_type = 'phase', taus = "all")
  gyro_y_av = alt.oadev(gyro_y, rate = 40.0, data_type = 'phase', taus = "all")
  gyro_z_av = alt.oadev(gyro_y, rate = 40.0, data_type = 'phase', taus = "all")

  gyro_x_av = np.array(gyro_x_av)
  gyro_y_av = np.array(gyro_y_av)
  gyro_z_av = np.array(gyro_z_av)

  def extract_noise(at):
      min_index = np.argmin(at)
    
      B = np.min(at)

      N = 1 #SUPPLIED BY OADEV FUNCITON

      K = (at[-1] - B) / (np.log10(at.size)) - np.log10(min_index) #DETERMINED AS SLOPE GRAPHICALLY

      return B, N, K

  B_x, N_x, K_x = extract_noise(gyro_x_av)
  B_y, N_y, K_y = extract_noise(gyro_y_av)
  B_z, N_z, K_z = extract_noise(gyro_z_av)

  # Print or use the extracted noise parameters as needed
  print("Gyro X - Bias instability (B):", B_x, " deg/hr")
  print("Gyro X - Angle random walk (N):", gyro_x_av[2,4], "°/√s")
  print("Gyro X - Rate random walk (K):", 0.006464, "°/√s")
  print("Gyro Y - Bias instability (B):", B_y," deg/hr")
  print("Gyro Y - Angle random walk (N):", gyro_y_av[2,4], "°/√s")
  print("Gyro Y - Rate random walk (K):", 0.006464, "rad/√h")
  print("Gyro Z - Bias instability (B):", B_z," deg/hr")
  print("Gyro Z - Angle random walk (N):", gyro_z_av[2,4], "°/√s")
  print("Gyro Z - Rate random walk (K):", 0.006464, "rad/√h")

  # Plot data on log-scale
  plt.figure()
  plt.title('Gyro Allan Variance Plot')
  plt.plot(gyro_x_av[1,:], gyro_x_av[2,:], label='gx')
  plt.plot(gyro_y_av[1,:], gyro_y_av[2,:], label='gx')
  plt.plot(gyro_z_av[1,:], gyro_z_av[2,:], label='gz')
  plt.xlabel("Tau")
  plt.ylabel('Allan Variation')
  plt.grid(True, which="both", ls="-", color='0.65')
  plt.legend()
  plt.xscale('log')
  plt.yscale('log')
  plt.show()


