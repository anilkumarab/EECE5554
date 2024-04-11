#lab_5 analysis
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.integrate import cumtrapz
from scipy.signal import butter, filtfilt
from scipy import signal

imudata_circle_file = "/content/circle_drive_imu.csv"
gpsdata_circle_file = "/content/circle_drive_gps.csv"
imudata_longdrive_file = "/content/longdrive_imu.csv"
gpsdata_longdrive_file = "/content/longdrive_gps.csv"

#driving in circle
df_imucircle = pd.read_csv(imudata_circle_file)
df_gpscircle = pd.read_csv(gpsdata_circle_file)
#long drive(ld)
df_imuld = pd.read_csv(imudata_longdrive_file)
df_gpsld = pd.read_csv(gpsdata_longdrive_file)

#GPS data going in circles
utm_northing = df_gpscircle['utm_northing']
utm_easting = df_gpscircle['utm_easting']
#GPS data long drive
utm_northing_ld = df_gpsld['utm_northing']
utm_easting_ld = df_gpsld['utm_easting']

#IMU data going in circles
magnetometer_x = df_imucircle["mag_field.magnetic_field.x"]
magnetometer_y = df_imucircle["mag_field.magnetic_field.y"]
#IMU data long drive
magnetometer_x_ld = df_imuld["mag_field.magnetic_field.x"]
magnetometer_y_ld = df_imuld["mag_field.magnetic_field.y"]


##--PLOT-1 A plot showing the magnetometer data before and after the correction in your report.
##------------------------------------CIRCLE DATA PLOTTING------------------------------------##
#plotting before calibration(IMU)
plt.figure(figsize=(8, 6))
plt.scatter(magnetometer_x, magnetometer_y, marker='.', color='green', label='IMU Data')
plt.xlabel('Magnetometer X')
plt.ylabel('Magnetometer Y')
plt.title('IMU Data Plot')
plt.legend()
plt.grid(True)
plt.show()

#hard iron calibration(IMU)
#mean

mean_magx = np.mean(magnetometer_x)
mean_magy = np.mean(magnetometer_y)
#subtracting mean from all points
#IMU
####################changes made here
split = 10
tot_len = len(magnetometer_y)
start_pt = int(2*tot_len/split)

magnetometer_x = magnetometer_x[start_pt:]
magnetometer_y = magnetometer_y[start_pt:]

magnetometer_x = list(magnetometer_x)
magnetometer_y = list(magnetometer_y)

mean_magx = np.mean(magnetometer_x)
mean_magy = np.mean(magnetometer_y)

####################
magx_hardiron = magnetometer_x - mean_magx
magy_hardiron = magnetometer_y - mean_magy

plt.figure(figsize=(8, 6))
plt.scatter(magx_hardiron, magy_hardiron, marker='.', color='green', label='IMU Data')
plt.xlabel('Magnetometer X')
plt.ylabel('Magnetometer Y')
plt.title('IMU Data Plot after Hard Iron Calibration')
plt.legend()
plt.grid(True)
plt.axis("equal")

#soft iron calibration
theta = -np.pi/12
step = len(magnetometer_x)

#print(step)
dist = np.sqrt(np.array(magnetometer_x-mean_magx)**2 + np.array(magnetometer_y-mean_magy)**2)
r = np.mean(dist)
t = np.linspace(0, 2*np.pi, 100)

def calc_anb() :
    sort_dist = np.sort(dist)
    a = np.mean(sort_dist[:int(len(dist)*0.5)])
    b = np.mean(sort_dist[-1*int(len(dist)*0.5):])

    return a, b

a, b = calc_anb()
print(a, b, r)

magx_softiron = []
magy_softiron = []
for i in range(step):
  x_t, y_t = np.matmul([[(a+b)*np.cos(theta)/(2*a), -1*(a+b)*np.sin(theta)/(2*a)],
                        [(a+b)*np.sin(theta)/(2*b), (a+b)*np.cos(theta)/(2*b)]],
                       [magnetometer_x[i]-mean_magx, magnetometer_y[i]-mean_magy])
  magx_softiron.append(x_t)
  magy_softiron.append(y_t)


plt.figure(figsize = (8, 6))
plt.plot(r*np.cos(t), r*np.sin(t), color = "r")
plt.scatter(magx_softiron, magy_softiron, label = 'Soft Iron calibrated data', marker = '.', color = 'green')
plt.title("Magnetometer data after soft iron calibration")
plt.xlabel("Magnetic data North(Gauss)")
plt.ylabel("Magnetic data East(Gauss)")
plt.grid(True)
plt.legend()
plt.axis("equal")
plt.show()
##-------------------------------------------------------------------------------------##

## PLOT2:The magnetometer yaw estimation before and after hard and soft iron calibration vs. time
##-----------------------------------LONG DRIVE data-----------------------------------##

#Calculate the yaw angle from the magnetometer calibration & plot the raw magnetometer yaw with the corrected yaw for comparison.
#Going forward, only use the corrected magnetometer yaw.

raw_yaw = np.arctan2(magnetometer_y - mean_magy, magnetometer_x - mean_magx)
raw_yaw_deg = np.degrees(raw_yaw)

# Calculating the corrected magnetometer yaw angle
corr_yaw = np.arctan2(magy_softiron, magx_softiron)
corr_yaw_deg = np.degrees(corr_yaw)

# Plotting raw magnetometer yaw vs. corrected yaw for comparison
plt.figure(figsize=(8, 6))
plt.plot(raw_yaw_deg, label='Raw Magnetometer Yaw')
plt.plot(corr_yaw_deg, label='Corrected Magnetometer Yaw')
plt.xlabel('Sample')
plt.ylabel('Yaw Angle (degrees)')
plt.title('Raw vs. Corrected Magnetometer Yaw')
plt.legend()
plt.grid(True)
plt.show()

#The magnetometer yaw estimation before and after hard and soft iron calibration vs. time

raw_yaw = np.arctan2(magnetometer_y_ld, magnetometer_x_ld) ## raw yaw calculation before calibration
raw_yaw_deg = np.degrees(raw_yaw)

magnetometer_x_ld = list(magnetometer_x_ld)
magnetometer_y_ld = list(magnetometer_y_ld)

magxld_mean = np.mean(magnetometer_x_ld)
magyld_mean = np.mean(magnetometer_y_ld)

magxld_hardiron = magnetometer_x_ld - magxld_mean
magyld_hardiron = magnetometer_y_ld - magyld_mean

corr_yaw_hardiron = np.arctan2(magyld_hardiron, magxld_hardiron)## corrected Hard iron calibration
corr_yaw_deg_hardiron = np.degrees(corr_yaw_hardiron)

##----------------------------------------------------------------------------------##

#soft iron calibration
step = len(magnetometer_x_ld)

#print(step)
dist = np.sqrt(np.array(magnetometer_x_ld-magxld_mean)**2 + np.array(magnetometer_y_ld-magyld_mean)**2)
r = np.mean(dist)
t = np.linspace(0, 2*np.pi, 100)

def calc_anb() :
    sort_dist = np.sort(dist)
    a = np.mean(sort_dist[:int(len(dist)*0.5)])
    b = np.mean(sort_dist[-1*int(len(dist)*0.5):])

    return a, b

a, b = calc_anb()
print(a, b, r)

magx_softiron_ld = []
magy_softiron_ld = []
for i in range(step):
  x_t, y_t = np.matmul([[(a+b)*np.cos(theta)/(2*a), -1*(a+b)*np.sin(theta)/(2*a)],
                        [(a+b)*np.sin(theta)/(2*b), (a+b)*np.cos(theta)/(2*b)]],
                       [magnetometer_x_ld[i]-magxld_mean, magnetometer_y_ld[i]-magyld_mean])
  magx_softiron_ld.append(x_t)
  magy_softiron_ld.append(y_t)


corr_yaw_softiron = np.arctan2(magy_softiron_ld, magx_softiron_ld) ## corrected soft iron yaw
corr_yaw_deg_softiron = np.degrees(corr_yaw_softiron)

##----------------------------------------------------------------------------------##

plt.figure(figsize=(14, 8))
plt.plot(df_imuld["Time"], raw_yaw_deg, label='Raw Magnetometer Yaw')
plt.plot(df_imuld["Time"], corr_yaw_deg_hardiron, label='Corrected Yaw (Hard Iron Calibration)')
plt.plot(df_imuld["Time"], corr_yaw_deg_softiron, label='Corrected Yaw (Soft Iron Calibration)')
plt.xlabel('Time')
plt.ylabel('Yaw Angle (degrees)')
plt.title('Magnetometer Yaw Estimation Before and After Calibration vs. Time')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Plot of gyro yaw estimation vs. time

yaw_rate = df_imuld["imu.angular_velocity.z"]
time = df_imuld["Time"]

cumulative_yaw = cumtrapz(yaw_rate, x=time, initial=0)

plt.figure(figsize=(10, 6))
plt.plot(time, cumulative_yaw, color = 'yellow' )
plt.xlabel('Time (s)')
plt.ylabel('Cumulative Yaw Angle (degrees)')
plt.title('Cumulative Yaw Angle from Gyro Sensor')
plt.grid(True)
plt.show()

#Low pass filter of magnetometer data, high pass filter of gyro data, complementary filter output, and IMU heading estimate as 4 subplots on one plot

#mag_heading calculation
magnetometer_x_ld = list(magnetometer_x_ld)
magnetometer_y_ld = list(magnetometer_y_ld)
mag_heading = np.arctan2(magnetometer_y_ld, magnetometer_x_ld)
mag_heading = np.degrees(mag_heading)

#gyro_heading
rotational_rate_x = np.array(df_imuld['imu.angular_velocity.x'])
rotational_rate_y = np.array(df_imuld['imu.angular_velocity.y'])
rotational_rate_z = np.array(df_imuld['imu.angular_velocity.z'])
rr_x = cumtrapz(rotational_rate_x, time, initial=0)
rr_y = cumtrapz(rotational_rate_y, time, initial=0)
rr_z = cumtrapz(rotational_rate_z, time, initial=0)

gyro_heading = np.arctan2(rr_x, rr_y)
gyro_heading = np.degrees(gyro_heading)

#gyro_heading = np.degrees(rot_z)
gyro_heading = gyro_heading+120 #Calibration offset
for i in range(len(gyro_heading)): #Wrap angles
    if gyro_heading[i] > 180:
        gyro_heading[i] -= 360
    if gyro_heading[i] < -180:
        gyro_heading[i] += 360



lowpassfreq_c = 0.2
highpassfreq_c = 0.3
nyquist_freq = 0.5*40
order = 5
b, a = signal.butter(order, lowpassfreq_c / nyquist_freq, 'low')
c, d = signal.butter(order, highpassfreq_c / nyquist_freq, 'high')
LowPassFreq_mag = signal.lfilter(b,a,mag_heading)
HighPassFreq_gyro = signal.lfilter(c,d,gyro_heading)

alpha = 0.8

complementary_filter = [alpha * yaw_cal + (1-alpha) * yaw_est for yaw_cal, yaw_est in zip(LowPassFreq_mag, HighPassFreq_gyro)]

plt.figure
plt.subplot(4,1,1)
plt.plot(time, LowPassFreq_mag, label='LowPass-Filter of Magnetometer data')
plt.title('Filter Results')
plt.xlabel('Time(s)')
plt.ylabel('Heading(deg)')
plt.grid(True)
plt.legend()

plt.subplot(4,1,2)
plt.plot(time, HighPassFreq_gyro, label='HighPass-Filter of Gyro data')
plt.xlabel('Time(s)')
plt.ylabel('Heading(deg)')
plt.grid(True)
plt.legend()

plt.subplot(4,1,3)
plt.plot(time, complementary_filter, label='Complementary filter output')
plt.xlabel('Time(s)')
plt.ylabel('Heading(deg)')
plt.grid(True)
plt.legend()
plt.show()

plt.subplot(4,1,4)
#plt.plot(time, comp_filter, label='Complementary Filter / Sensor Fusion')
plt.plot(time, gyro_heading, label='Gyro Yaw Heading')
plt.xlabel('Time (s)')
plt.ylabel('Heading (deg)')
plt.grid(True)
plt.legend()
plt.show()


#Plot of forward velocity from accelerometer before and after any adjustments

linear_acceleration_x = df_imuld["imu.linear_acceleration.x"]
ld_time_imu = df_imuld["Time"]


# Adjustments:
# 1. Low-pass filter to reduce noise
def low_pass_filter(data, cutoff_freq, fs):
    # Calculate the filter coefficients
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    # Apply the filter
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Define the cutoff frequency and sampling frequency for the low-pass filter
cutoff_frequency = 2.0  
sampling_frequency = 50  

# Apply low-pass filter to accelerometer data
filtered_acceleration_x = low_pass_filter(linear_acceleration_x, cutoff_frequency, sampling_frequency)

# 2. Integrate to obtain forward velocity
velocity_x = cumtrapz(filtered_acceleration_x, x=ld_time_imu, initial=0)

# Plotting the raw and adjusted accelerometer data for comparison
plt.figure(figsize=(10, 6))
plt.plot(ld_time_imu, linear_acceleration_x, label='Raw Accelerometer Data')
plt.plot(ld_time_imu, filtered_acceleration_x, label='Filtered Accelerometer Data')
plt.xlabel('Time')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Raw vs Filtered Accelerometer Data')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the adjusted forward velocity from accelerometer data
plt.figure(figsize=(10, 6))
plt.plot(ld_time_imu, velocity_x, label='Adjusted Forward Accelerometer Data (Velocity)')
plt.xlabel('Time')
plt.ylabel('Velocity (m/s)')
plt.title('Adjusted Forward Accelerometer Data Plot')
plt.legend()
plt.grid(True)
plt.show()

#Plot of forward velocity from gps

d_utm_easting = np.diff(df_gpsld["utm_easting"])
d_utm_northing = np.diff(df_gpsld["utm_northing"])

d_time = np.diff(df_gpsld["Time"])

forward_velocity = np.sqrt(d_utm_easting**2 + d_utm_northing**2) / d_time

# Plotting forward velocity from GPS data
plt.figure(figsize=(12, 5))
plt.plot(df_gpsld['Time'].values[1:], forward_velocity, label='Forward Velocity', color='red')
plt.xlabel('Time')
plt.ylabel('Forward Velocity (m/s)')
plt.title('Forward Velocity from GPS Data')
plt.legend()
plt.grid(True)
plt.show()

#Plot of estimated trajectory from GPS and from IMU velocity/yaw data (2 subplots)

##Estimated trajectory from GPS
centered_northing = utm_northing_ld #- utm_northing_ld[0]
centered_easting = utm_easting_ld #- utm_easting_ld[0]

latitude = df_gpsld["latitude"]
longitude = df_gpsld["longitude"]

centered_latitude = latitude - latitude[0]
centered_longitude = longitude - longitude[0]


##Estimated trajectory from IMU velocity/yaw data

linear_acceleration_x = np.array(df_imuld["imu.linear_acceleration.x"])
linear_acceleration_y = np.array(df_imuld["imu.linear_acceleration.y"])

linear_acceleration_x = linear_acceleration_x * np.sin(df_imuld['imu.orientation.x'])
linear_acceleration_y = linear_acceleration_y * np.cos(df_imuld['imu.orientation.x'])

px = cumtrapz(linear_acceleration_x, ld_time_imu, initial=0)+df_gpsld['utm_easting'].iloc[0]
py = cumtrapz(linear_acceleration_y, ld_time_imu, initial=0)+df_gpsld['utm_northing'].iloc[0]

lpf_freq_c = 0.1
hpf_freq_c = 0.1
nyquist_freq = 0.5*40
order = 5
b, a = signal.butter(order, lpf_freq_c / nyquist_freq, 'low')
c, d = signal.butter(order, hpf_freq_c / nyquist_freq, 'high')

filtered_acceleration_x = signal.lfilter(b, a, linear_acceleration_x)
filtered_acceleration_y = signal.lfilter(b, a, linear_acceleration_y)


velocity_x = cumtrapz(filtered_acceleration_x, time, initial = 0)
velocity_y = cumtrapz(filtered_acceleration_y, time, initial = 0)

displacement_x = cumtrapz(velocity_x, initial=0)
displacement_y = cumtrapz(velocity_y, initial=0)

imu_velocity = np.sqrt(velocity_x**2 + velocity_y**2)


plt.figure(figsize=(8, 6))
#plt.scatter(displacement_x, displacement_y, marker='.', color='orange', label='IMU data')
plt.scatter(centered_easting, centered_northing, marker='.', color='blue', label='GPS Data')
plt.xlabel('GPS lat/velocity_x ')
plt.ylabel('GPS long/velocity_y')
plt.title('GPS Data Plot/IMU velocity plot')
plt.legend()
plt.grid(True)
plt.show()

#Part 2

utm_northing = df_gpsld['utm_northing'].values[:]
utm_easting = df_gpsld['utm_easting'].values[:]
time_gps = df_gpsld['Time'].values[:]

accel_x = np.array(df_imuld["imu.linear_acceleration.x"])

time = ld_time_imu.values[:]
time = np.array(time) - time[0]

time_gps = np.array(time_gps) - time_gps[0]

gps_velocity = [np.sqrt((utm_northing[i+1]-utm_northing[i])**2 + (utm_easting[i+1]-utm_easting[i])**2)/(time_gps[i+1] - time_gps[i]) for i in range(len(df_gpsld['Time'].values[:])-1)]

modified_accel_x = np.array(accel_x)

forward_vel = cumtrapz(np.array(accel_x) - np.median(accel_x))
plt.plot((np.array(time[:-1])-time[0]), (forward_vel)/40)
plt.plot(np.array(time_gps[:-1])-time_gps[0], np.array(gps_velocity))
plt.xlabel("Forward velocity/GPS velocity(m/s^2)")
plt.ylabel('Time(s)')
plt.legend(["forward_velocity", "GPS_velocity"])
plt.show()


plt.plot(np.array(gps_velocity))
plt.plot(np.array(gps_velocity) == 0)

bin_lst = np.array(gps_velocity) == 0

lst = []

prev = False
last_value = 0
lst.append(0)
for i in range(len(gps_velocity)) :
  if(bin_lst[i] == True and prev == False and i - last_value > 18) :
    lst.append(i)
    prev = True
    last_value = i

  if(bin_lst[i] == False and prev == True) :
    prev = False

lst.append(len(gps_velocity)-1)

range_list = lst
modified_accel_x = np.array(accel_x)

for i in range(len(range_list)-1) :
    st_pt = range_list[i]
    pt = range_list[i+1]
    print(st_pt, pt)

    net_zero_acc = accel_x[int(st_pt*len(time)/701):int(pt*len(time)/701)]
    modified_accel_x[int(st_pt*len(time)/701):int(pt*len(time)/701)] = net_zero_acc - np.mean(net_zero_acc)


forward_vel = cumtrapz(np.array(modified_accel_x))
forward_vel = np.array(forward_vel)
forward_vel[forward_vel<0] = 0

plt.plot((np.array(time[:-1])-time[0]), (forward_vel)/40)
plt.plot(np.array(time_gps[:-1])-time_gps[0], np.array(gps_velocity))
plt.xlabel('Forward velocity/GPS velocity(m/s^2)')
plt.ylabel('Time(s)')
plt.title('IMU velocity plot after adjustment')
plt.legend(["forward_velocity", "GPS_velocity"])
plt.show()


plt.show()
plt.plot((np.array(gps_velocity) == 0))
#plt.legend(["Adjusted_forward_velocity", "GPS_velocity"])
plt.show()

v_n = cumtrapz(np.array(forward_vel*np.sin(np.array(yaw)[:-1]*np.pi/180)))
v_e = cumtrapz(np.array(forward_vel*np.cos(np.array(yaw)[:-1]*np.pi/180)))

plt.plot(v_n/1600, v_e/1600)
plt.plot(np.array(utm_northing)-utm_northing[0], np.array(utm_easting)-utm_easting[0])
plt.xlabel("UTM Northing/v_n")
plt.ylabel("UTM Easting/v_e")
plt.legend(["Estimated trajectory", "GPS track"])
plt.show()

plt.plot(np.array(utm_easting)-utm_easting[0], np.array(utm_northing)-utm_northing[0])

degree = -215
theta1 = yaw[0]*np.pi/180 + degree*np.pi/180
theta2 = np.arctan2(utm_northing[1]-utm_northing[0], utm_easting[1]-utm_easting[0])

rotation_matrix = [[np.cos(theta1-theta2), -1*np.sin(theta1-theta2)], [np.sin(theta1-theta2), np.cos(theta1-theta2)]]

v_n_rotated = []
v_e_rotated = []
for i in range(len(yaw)-2) :
    x_, y_ = np.matmul(rotation_matrix, [v_e[i], v_n[i]])
    v_e_rotated.append(x_)
    v_n_rotated.append(y_)

plt.plot(np.array(v_n_rotated)/1600, np.array(v_e_rotated)/1600)
plt.plot(np.array(utm_easting)-utm_easting[0], np.array(utm_northing)-utm_northing[0])
plt.legend(["Estimated trajectory (after adjustment)", "GPS track"])
plt.show()

range_list = [0, 125, 190, 265, 370, 420, 500, 670, 740, 828]

for i in range(len(range_list)-1) :
    st_pt = range_list[i]
    pt = range_list[i+1]

    net_zero_acc = accel_x[int(st_pt*len(time)/828):int(pt*len(time)/828)]
    modified_accel_x[int(st_pt*len(time)/828):int(pt*len(time)/828)] = net_zero_acc - np.mean(net_zero_acc)

forward_vel = cumtrapz(np.array(accel_x) - np.median(accel_x))
plt.plot((np.array(time[:-1])-time[0]), (forward_vel)/40)
plt.plot(np.array(time_gps[:-1])-time_gps[0], np.array(gps_velocity))
plt.legend(["forward_velocity", "GPS_velocity"])
plt.show()

plt.plot(time, modified_accel_x)
plt.plot(time, accel_x)
plt.legend(["Adjusted_linear_acceleration_x", "linear_acceleration_x"])
plt.show()

forward_vel = cumtrapz(np.array(modified_accel_x))
forward_vel = np.array(forward_vel)
forward_vel[forward_vel<0] = 0

plt.plot((np.array(time[:-1])-time[0]), (forward_vel)/40)
plt.plot(np.array(time_gps[:-1])-time_gps[0], np.array(gps_velocity))
plt.legend(["Adjusted_forward_velocity", "GPS_velocity"])
plt.show()