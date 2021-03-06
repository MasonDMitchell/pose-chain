import streamlit as st
from scipy.fft import fft
import serial
import numpy as np
from training import input_data
from straight_filter import Filter
from read_serial import read
import plotly.express as px
import time
import matplotlib.pyplot as plt
from magpylib.source.magnet import Box
import magpylib as magpy

mode = st.selectbox("How do you want to run the filter?",['Training Mode','Live Mode'])
st.sidebar.markdown("# Filter Parameters#")
if mode == 'Live Mode':
    N = st.sidebar.number_input("Number of Pairs",min_value=1,max_value=10,value=1)
particles = st.sidebar.number_input("Number of Particles",min_value = 3,max_value=10000,value=1000)
axis_noise = st.sidebar.number_input("Axis Noise",min_value=0.,max_value=1.,value=.1)
segment_length = st.sidebar.number_input("Segment Length",min_value=1., max_value=300.,value=86.711098)
sensor_segment_length = st.sidebar.number_input("Sensor Segment Length",min_value=1., max_value=300.,value=13.288902)
if mode == 'Live Mode':
    sensor_pos = []
    sensor_angle = []
    sensor_axis = []
    magnet_pos = []
    magnet_angle = []
    magnet_axis = []
    for i in range(N):
        sensor_pos.append(list(map(float,st.sidebar.text_input("Init Sensor Pos " +str(i),value=str((segment_length+sensor_segment_length)*i) + ', 0.0, 0.0').split(','))))
        sensor_angle.append(float(st.sidebar.text_input("Init Sensor Angle " +str(i),value=0)))
        sensor_axis.append(list(map(float,st.sidebar.text_input("Init Sensor Axis " +str(i), value = '0.0, 0.0, 0.0').split(','))))
        magnet_pos.append(list(map(float,st.sidebar.text_input("Init Magnet Pos " +str(i),value=str(segment_length*(i+1)) + ', 0.0, 0.0').split(','))))
        magnet_angle.append(float(st.sidebar.text_input("Init Magnet Angle " +str(i),value=0)))
        magnet_axis.append(list(map(float,st.sidebar.text_input("Init Magnet Axis " +str(i), value = '0.0, 0.0, 0.0').split(','))))

if mode == "Training Mode":
    filename="data/processed.csv"
    filename = st.text_input("Input filename",value="data/processed.csv")
    N, sensor_data, sensor_pos, sensor_angle, sensor_axis, magnet_pos, magnet_angle, magnet_axis = input_data(filename)
    st.sidebar.markdown("#### Number of Pairs: " + str(N))
    graph_pose_error = st.checkbox("Graph Pose Error Data")


graph_sensor = st.checkbox("Graph Live Sensor Data")
graph_pose = st.checkbox("Graph Live Pose Data")


if mode == 'Live Mode':
    zero_data = st.checkbox("Zero Filter")

start = st.button("Run Filter",key="Start Button")

if start == True:
    x = Filter(N,particles)

    x.axis_noise = axis_noise
    x.segment_length = segment_length
    x.sensor_segment_length = sensor_segment_length
 
    x.create_particles(sensor_pos[0:N],sensor_angle[0:N],sensor_axis[0:N],magnet_pos[0:N],magnet_angle[0:N],magnet_axis[0:N])

    #Initialize screen
    iteration = st.text("Iteration")
    pos = st.text("")
    

    ser = serial.Serial('/dev/ttyACM0')
    time.sleep(2)
    if mode == "Live Mode":
        if zero_data == True:

            good_data = False
            while good_data != True: 
                serial_zero_data = ser.readline()
                serial_zero_data = str(serial_zero_data)[2:-5].split(',')
                try:
                    serial_zero_data = list(map(float,serial_zero_data))
                except:
                    serial_zero_data = [0,0,0]
                if len(serial_zero_data) != 3:
                    serial_zero_data = [0,0,0]
            
                if isinstance(serial_zero_data[0],float):
                    good_data = True

            serial_zero_data = np.array(serial_zero_data)/10000
        
            print(serial_zero_data)

    if graph_sensor == True: 
        #Plot filter sensor data
        best_data = []
        data_plot = st.line_chart([0])

        if mode == "Training Mode":
            #Plot real sensor data
            real_data_plot = st.line_chart([0])
            #Plot error sensor data
            data_difference = []
            error_data_plot = st.line_chart([0])

    if graph_pose == True:
        pos_plot = st.pyplot()
    
    if mode == "Training Mode":
        if graph_pose_error == True:
            pos_difference = []
            error_plot = st.line_chart([0])

    i=0

    while(True):

        iteration.text("Iteration " + str(i))    
        
        if mode == "Training Mode":
            x.sensor_data = sensor_data[i]
        else:
            serial_data = ser.readline()
            serial_data = str(serial_data)[2:-5].split(',')
            try:
                serial_data = list(map(float,serial_data))
            except:
                serial_data = [0,0,0]
            if len(serial_data) != 3:
                serial_data = [0,0,0]

            serial_data = np.array(serial_data)/10000
            if zero_data == True:
                serial_data = np.subtract(serial_data,serial_zero_data)

            x.sensor_data = serial_data

        x.compute_pose()
        x.compute_flux()
        x.reweigh()
        x.predict()

        x.resample()
        x.update()

        pos.text(x.best_pos)
    
        if graph_sensor == True:
            #Save best filter data
            best_data.append(x.sensor_data)
            #Update filter data plot
            if i % 10 == 0 and i > 100:
                data_plot.line_chart(best_data[-100:-1])
            if mode == "Training Mode":
                #Plot real sensor data
                real_data_plot.line_chart(sensor_data[0:i])
                #Calculate and plot error
                data_difference.append(np.subtract(sensor_data[i],x.best_data[0]))
                error_data_plot.line_chart(data_difference)
        
        if graph_pose == True:
            magnet = Box(mag=[0,0,1],dim=[6.35,6.35,6.35],pos=x.best_pos[0],angle=x.best_angle[0],axis=x.best_axis[0])
            magpy.displaySystem(magnet,suppress=True) 
            pos_plot.pyplot()
            plt.close()
            
        if mode == "Training Mode":
            if graph_pose_error == True:
                pos_difference.append(np.linalg.norm(np.subtract(x.best_pos[0],magnet_pos[i])))
                error_plot.line_chart(pos_difference)

        i = i+1 
