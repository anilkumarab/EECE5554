import utm
import time
import serial
import rospy
import sys

def isGPGGAinString(stringreadfromport):
    if '$GPGGA' in stringreadfromport:
        print('Great Success!')
    else:
        print('GPGGA not found in string')

def deg_minutes_to_deg_decimal(ddmm_mm):
    degree = ddmm_mm/100
    minutes = ddmm_mm%100
    deg_dec = degree + minutes / 60.0
    return deg_dec

def convert_coordinates_acc_directions(Latitude, LatitudeDir, Longitude, LongitudeDir):
     Latitude = -Latitude if LatitudeDir == 'S' else Latitude

     Longitude = -Longitude if LongitudeDir == 'W'  else Longitude

     return Latitude, Longitude

def convertToUTM(converted_latitude, converted_longitude):
    UTMVals = utm.from_latlon(converted_latitude, converted_longitude)
    UTMEasting, UTMNorthing, UTMZone, UTMLetter  = UTMVals #Again, replace these with values from UTMVals
    print(UTMVals)
    return [UTMEasting, UTMNorthing, UTMZone, UTMLetter]

def UTCtoUTCEpoch(UTC):
    UTCinSecs = (int(UTC // 10000) * 3600 + int((UTC % 10000) // 100) * 60 + UTC % 100)
    TimeSinceEpoch = time.time()
    TimeSinceEpochBOD = TimeSinceEpoch - TimeSinceEpoch % 86400 
    CurrentTime = TimeSinceEpochBOD + UTCinSecs
    CurrentTimeSec = int(CurrentTime)
    CurrentTimeNsec = int((CurrentTime - CurrentTimeSec) * 1e9)
    print(CurrentTime)
    return [CurrentTimeSec, CurrentTimeNsec]

def ReadFromSerial(serialPortAddr):
    serialPort = serial.Serial(serialPortAddr) #This line opens the port, do not modify
    gpggaRead = serialPort.readline() #Replace this line with a 1-line code to read from the serial port
    #print(gpggaRead)
    serialPort.close() #Do not modify
    return gpggaRead
print(sys.argv)
serialPortAddr = rospy.get_param("~port",sys.argv[1]) #You will need to change this to the emulator or GPS puck port
#serialPortAddr = rospy.get_param("~port", '/dev/pts/0')
while True:
    gpggaRead = ReadFromSerial(serialPortAddr).decode('utf-8')

    gpggasplit = gpggaRead.split(',')

    if gpggasplit[0] != '$GPGGA' :
        continue

    print("\n", gpggasplit, "\n")

    UTC = float(gpggasplit[1])  
    Latitude = float(gpggasplit[2])  
    LatitudeDir = str(gpggasplit[3])  
    Longitude = float(gpggasplit[4])  
    LongitudeDir = str(gpggasplit[5])  
    HDOP = float(gpggasplit[8])


    latitude_dddd_dd = deg_minutes_to_deg_decimal((Latitude))
    longitude_dddd_dd = deg_minutes_to_deg_decimal((Longitude))

    converted_latitude, converted_longitude = convert_coordinates_acc_directions(latitude_dddd_dd, LatitudeDir, longitude_dddd_dd, LongitudeDir)

    convertToUTM(converted_latitude, converted_longitude)

    CurrentTime = UTCtoUTCEpoch(UTC)

    time.sleep(1)


stringreadfromport = '$GPGGA,202530.00,5109.0262,N,11401.8407,W,5,40,0.5,1097.36,M,-17.00,M,18,TSTR*61'

isGPGGAinString("$GPGGA,202530.00,5109.0262,N,11401.8407,W,5,40,0.5,1097.36,M,-17.00,M,18,TSTR*61")

gpggaread = '$GPGGA,202530.00,5109.0262,N,11401.8407,W,5,40,0.5,1097.36,M,-17.00,M,18,TSTR*61'

print(gpggasplit)


 


print("UTC:", UTC)
print("Latitude:", Latitude)
print("LatitudeDir:", LatitudeDir)
print("Longitude:", Longitude)
print("LongitudeDir:", LongitudeDir)
print("HDOP:", HDOP)



latitude_ddmm_mm = 5109.0262
longitude_ddmm_mm = 11401.8407



print("Latitude in DD.dddd:", latitude_to_dd_dddd)
print("Longitude in DD.dddd:", longitude_to_ddmm_mm)



latitude_dddd_dd = 51.240698666666674
longitude_dddd_dd = 114.04908533333335
LatitudeDir = 'N'
LongitudeDir = 'W'
 
print("Converted Latitude:", converted_latitude)
print("Converted Longitude:", converted_longitude)












 




    
    
    
