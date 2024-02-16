#!/usr/bin/env python3
import utm
import time
import serial
import rospy
import sys
from gps_driver.msg import Customrtk
from std_msgs.msg import Header

#Publisher node
rospy.init_node('gps_driver_node', anonymous=True)
pub = rospy.Publisher('gps', Customrtk, queue_size=10)
custom_rtk_msg = Customrtk()


def isGNGGAinString(stringreadfromport):
    if '$GNGGA' in stringreadfromport:
        print('Great Success!')
    else:
        print('GNGGA not found in string')

def lat_deg_minutes_to_deg_decimal(latitude):
    deg = int(latitude[:2])
    mins = float(latitude[2:])
    degDec = mins / 60
    return deg + degDec

def lon_deg_minutes_to_deg_decimal(longitude):
    Longitude_1 =str((longitude).split(".",1)[0])
    print(f'longitude string: {longitude}')
    print(f'longitude string: {Longitude_1}')
    if(len(Longitude_1)) > 4:
      deg = int(longitude[:3])
      mins = float(longitude[3:])
    else:
      deg = int(longitude[:2])
      mins = float(longitude[2:])
    
    degDec = mins / 60
    return deg + degDec

   
def convert_coordinates_acc_directions(Latitude, LatitudeDir, Longitude, LongitudeDir):
     Latitude = -Latitude if LatitudeDir == 'S' else Latitude

     Longitude = -Longitude if LongitudeDir == 'W'  else Longitude

     return Latitude, Longitude

def convertToUTM(converted_latitude, converted_longitude):
    UTMVals = utm.from_latlon(converted_latitude, converted_longitude)
    UTMEasting = float(UTMVals[0])
    UTMNorthing = float(UTMVals[1])
    UTMZone = int(UTMVals[2])
    UTMLetter  = str(UTMVals[3]) #Again, replace these with values from UTMVals
    print(UTMVals)
    return [UTMEasting, UTMNorthing, UTMZone, UTMLetter]

def UTCtoUTCEpoch(UTC):
     UTC = int(UTC)
     UTCinSecs = (UTC // 10000) * 3600 + int((UTC % 10000) // 100) * 60 + (UTC % 100)
     TimeSinceEpoch = time.time()
     TimeSinceEpochBOD = TimeSinceEpoch - TimeSinceEpoch % 86400 
     CurrentTime = TimeSinceEpochBOD + UTCinSecs
     CurrentTimeSec = int(CurrentTime)
     CurrentTimeNsec = 0
     print(CurrentTime)
     return [CurrentTimeSec, CurrentTimeNsec]


'''def UTCtoUTCEpoch(UTC):
    UTCinSecs = (int(UTC // 10000) * 3600 + int((UTC % 10000) // 100) * 60 + UTC % 100)
    TimeSinceEpoch = time.time()
    TimeSinceEpochBOD = TimeSinceEpoch - UTCinSecs 
    CurrentTime = TimeSinceEpochBOD + UTCinSecs
    CurrentTimeSec = int(CurrentTime)
    CurrentTimeNsec = float((CurrentTime - CurrentTimeSec) * 1e9)
    print(CurrentTime)
    return [CurrentTimeSec, CurrentTimeNsec]'''

def ReadFromSerial(serialPortAddr, serial_baud):
    serialPort = serial.Serial(serialPortAddr, serial_baud) #This line opens the port, do not modify
    gnggaRead = serialPort.readline() #Replace this line with a 1-line code to read from the serial port
    #print(gpggaRead)
    serialPort.close() #Do not modify
    return gnggaRead


print(sys.argv)
serialPortAddr = rospy.get_param("~port", sys.argv[1]) #You will need to change this to the emulator or GPS puck port
#serialPortAddr = rospy.get_param("~port", '/dev/pts/0')
serial_baud = rospy.get_param('~baudrate',4800)
while True:
    gnggaRead = ReadFromSerial(serialPortAddr, serial_baud)
    gnggaRead = str(gnggaRead, "ISO-8859-1")

    gnggasplit = gnggaRead.split(',')
    print("R", gnggasplit)

    if gnggasplit[0] != "$GNGGA" :
        continue

    print("\n", gnggasplit, "\n")

    UTC = float(gnggasplit[1])  
    Latitude = float(gnggasplit[2])  
    LatitudeDir = str(gnggasplit[3])  
    Longitude = float(gnggasplit[4])  
    LongitudeDir = str(gnggasplit[5])  
    Fix_Quality = int(gnggasplit[6])
    HDOP = float(gnggasplit[8])
    

    latitude_dddd_dd = lat_deg_minutes_to_deg_decimal((str(Latitude)))
    longitude_dddd_dd = lon_deg_minutes_to_deg_decimal((str(Longitude)))

    converted_latitude, converted_longitude = convert_coordinates_acc_directions(latitude_dddd_dd, LatitudeDir, longitude_dddd_dd, LongitudeDir)
    print(converted_longitude)
    convertToUTM(converted_latitude, converted_longitude)

    CurrentTime = UTCtoUTCEpoch(UTC)

    # time.sleep(1)

    #Populate custom_gps_message with data

    custom_rtk_msg.header = Header(frame_id='GPS1_Frame', stamp=rospy.Time(CurrentTime[0], CurrentTime[1]))
    custom_rtk_msg.latitude = converted_latitude
    custom_rtk_msg.longitude = converted_longitude
    custom_rtk_msg.altitude = float(gnggasplit[9])  # Replace with your altitude data
    custom_rtk_msg.utm_easting, custom_rtk_msg.utm_northing, custom_rtk_msg.zone, custom_rtk_msg.letter = convertToUTM(converted_latitude, converted_longitude)
    custom_rtk_msg.fix_quality = Fix_Quality
    custom_rtk_msg.hdop = HDOP
    custom_rtk_msg.gngga_read = gnggaRead

    rospy.loginfo(custom_rtk_msg)
    rospy.loginfo("   ")

    pub.publish(custom_rtk_msg)
    
    print(latitude_dddd_dd)
    print(longitude_dddd_dd)