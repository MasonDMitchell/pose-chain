import serial

def read(loc='/dev/ttyACM0'):
    try:
        ser = serial.Serial(loc)
    except:
        print("Can't connect to Serial")

    line = ser.readline()
    line = str(line)[2:-5].split(',')
    try:
        line = list(map(int, line)) 
    except:
        line = [0,0,0]

    if len(line) != 3:
        line = [0,0,0]

    return line
