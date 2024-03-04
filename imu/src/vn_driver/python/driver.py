#!/usr/bin/env python3
import serial
import rospy
import sys
import numpy as np
from vn_driver.msg import Vectornav
from std_msgs.msg import Header
import time
#Publisher node 
rospy.init_node('nv_driver', anonymous=True)
pub = rospy.Publisher('imu', Vectornav, queue_size=10)
Vectornav_msg = Vectornav()

#vnymrRead = '$VNYMR,+164.618,+022.062,-003.757,-00.3611,-00.0797,+00.2916,+03.553,+00.595,-08.826,+00.004000,-00.000843,+00.000141*64'

def VNYMR_String(stringfromport):
    if '$VNYMR' in stringfromport:
        print('Great Success!')
    else:
        print('No VNYMR string!')

def ReadFromPort(serialPortAddr, serial_baud):
    serialPort = serial.Serial(serialPortAddr, serial_baud)
     #This line opens the port, do not modify
    vnymrRead = serialPort.readline().decode('utf-8', errors = 'replace') #Replace this line with a 1-line code to read from the serial port
    output_freq = b'$VNWRG,07,40*XX\n'
    serialPort.write(output_freq)
    # print(vnymrRead)
    serialPort.close() #Do not modify
    return vnymrRead

def convert_to_quaternion(roll, pitch, yaw):
        #Converting euler angles(roll, pitch, yaw) to a quaternion
        quatern_x = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        quatern_y = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        quatern_z = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        quatern_w = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return [quatern_x, quatern_y, quatern_z, quatern_w]

#print(sys.argv)
serialPortAddr = rospy.get_param("~port", sys.argv[1]) #You will need to change this to the emulator or GPS puck port
#serialPortAddr = rospy.get_param("~port", '/dev/pts/0')
serial_baud = rospy.get_param('~baudrate', 115200)

while True:
    vnymrRead = str(ReadFromPort(serialPortAddr, serial_baud))
    vnymrsplit = vnymrRead.split(',')#Splitting the string
    
    if vnymrsplit[0] != "$VNYMR":
        # print(f' vnymrsplit[0] : { vnymrsplit[0] }             *****       Bad Boy is back')
        # time.sleep(3)
        continue
    # for i in range(len(vnymrsplit)):
    #     if vnymrsplit[i] != 13:
    #         continue
    # try:
    #     # Assuming the data needs to be converted to specific data types, handle conversions here
    #     # Example: Convert string values to floats if needed
    #     values = [float(v) for v in vnymrsplit[1:]]
    # except ValueError as e:
    #     print("Error: Failed to convert data to expected format:", e)
    #     break

    #print("\n", vnymrsplit, "\n")

    yaw = float(vnymrsplit[1])
    pitch = float(vnymrsplit[2])
    roll = float(vnymrsplit[3])

    # print("Roll, Pitch, Yaw: ", roll, pitch, yaw)
    #values(converted from Euler --> Quaternion)
    quaternions = convert_to_quaternion(roll,pitch,yaw)

    #Magnetometer data along three axes X,Y & Z
    mag_x = float(vnymrsplit[4])
    mag_y = float(vnymrsplit[5])
    mag_z = float(vnymrsplit[6])

        #liner acceleration along X,Y & Z - accelerometer data
    acel_x = float(vnymrsplit[7])
    acel_y = float(vnymrsplit[8])
    acel_z = float(vnymrsplit[9])

    #angular rate along X,Y & Z - gyroscope data
    gyro_x = float(vnymrsplit[10])
    gyro_y = float(vnymrsplit[11])
    gyro_z = str(vnymrsplit[12])
    gyro_z = gyro_z.split('*')
    gyro_z = float(gyro_z[0])

    #  #If the coordinates arent received, stop the code
    # if vnymrsplit[2]=='':
    #     print("Data not being received")
    #     break


    #Publishing data to Vectornav.msg

    Vectornav_msg.header = Header(frame_id ='imu1_frame', stamp=rospy.Time.now())
    Vectornav_msg.imu.orientation.x = quaternions[0]
    Vectornav_msg.imu.orientation.y = quaternions[1]
    Vectornav_msg.imu.orientation.z = quaternions[2]
    Vectornav_msg.imu.orientation.w = quaternions[3]
    Vectornav_msg.imu.angular_velocity.x = gyro_x
    Vectornav_msg.imu.angular_velocity.y = gyro_y
    Vectornav_msg.imu.angular_velocity.z = gyro_z
    Vectornav_msg.imu.linear_acceleration.x = acel_x
    Vectornav_msg.imu.linear_acceleration.y = acel_y
    Vectornav_msg.imu.linear_acceleration.z = acel_z
    Vectornav_msg.mag_field.magnetic_field.x = mag_x
    Vectornav_msg.mag_field.magnetic_field.y = mag_y
    Vectornav_msg.mag_field.magnetic_field.z = mag_z
    Vectornav_msg.String = vnymrRead
    # Vectornav_msg.raw_data = str(vnymrRead)

    # publish_time = rospy.Time.now()
    # rospy.loginfo("Data published at timestamp: %s", publish_time)

    # # Calculate the time difference between successive publications
    # if 'previous_publish_time' in locals():
    #     time_difference = (publish_time - previous_publish_time).to_sec()
    #     rospy.loginfo("Time difference since last publication: %.5f seconds", time_difference)

    #     previous_publish_time = publish_time 
    #     print(f"time : previous_publish_time" )
    # rospy.ROSInterruptException
    rospy.loginfo(Vectornav_msg)
    rospy.loginfo("   ")
    #rate = rospy.Rate(40)
    pub.publish(Vectornav_msg)
    #rate.sleep()








































    

