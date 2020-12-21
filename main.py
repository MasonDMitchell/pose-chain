import streamlit as st
import serial
import numpy as np
from training import input_data
from straight_filter import Filter
from read_serial import read
import plotly.express as px
import altair as alt
import time
import matplotlib.pyplot as plt
from magpylib.source.magnet import Box
from tools import createChain,noise,noise_rejection
import magpylib as magpy
import pandas as pd

mode = st.selectbox("How do you want to run the filter?",['Training Mode','Live Mode'])
st.sidebar.markdown("# Filter Parameters#")
if mode == 'Live Mode':
    N = st.sidebar.number_input("Number of Pairs",min_value=1,max_value=10,value=1)
particles = st.sidebar.number_input("Number of Particles",min_value = 3,max_value=10000,value=1000)
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
graph_params = st.checkbox("Graph Live Parameters")


if mode == 'Live Mode':
    zero_data = st.checkbox("Zero Filter")

start = st.button("Run Filter",key="Start Button")
print("Pairs:",N)
chain = createChain(particles,N,0,0,14,0)
if start == True:
    x = Filter(chain,noise)
 
    #Initialize screen
    iteration = st.text("Iteration:")
    bend_angle = st.text("Bend Angle:")
    bend_direction = st.text("Bend Direction:")
    pos = st.text("")

    simulation = [0,0,0]
    
    if mode == "Live Mode":
        ser = serial.Serial('/dev/ttyACM0')
        time.sleep(2)
        if zero_data == True:

            good_data = False
            index = 0
            while good_data != True: 
                serial_zero_data = ser.readline()
                serial_zero_data = str(serial_zero_data)[2:-5].split(',')
                try:
                    serial_zero_data = list(map(float,serial_zero_data))

                    temp = serial_zero_data[0]
                    serial_zero_data[0] = serial_zero_data[2]
                    serial_zero_data[2] = temp
                except:
                    serial_zero_data = [0,0,0]
                if len(serial_zero_data) != 3:
                    serial_zero_data = [0,0,0]
            
                if isinstance(serial_zero_data[0],float):
                    good_data = True

            serial_zero_data = np.array(serial_zero_data)/1000
            print(serial_zero_data)
            
            simulation = [-2.22,0,0]
            #simulation = [-8.4699,0,0]
            
            serial_zero_data = serial_zero_data - simulation


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

    if graph_params == True:
        params_plot = st.pyplot()
    
    if mode == "Training Mode":
        if graph_pose_error == True:
            pos_difference = []
            error_plot = st.line_chart([0])

    i=0

    serial_data_median = np.tile(np.array(simulation),5)
    serial_data_median = np.reshape(serial_data_median,(5,3))
    bend_a = 0
    bend_d = 0
    bend_params = [[0,0],[0,0],[0,0],[0,0],[0,0]]

    while(True):

        iteration.text("Iteration:" + str(i))    
        bend_angle.text("Bend Angle:" + str(bend_a))
        bend_direction.text("Bend Direction:" + str(bend_d))
        
        if mode == "Training Mode":
            x.sensor_data = sensor_data[i]
        else:
            serial_data = ser.readline()
            serial_data = str(serial_data)[2:-5].split(',')
            try:
                serial_data = list(map(float,serial_data))
                temp = serial_data[0]
                serial_data[0] = serial_data[2]
                serial_data[2] = temp
            except:
                serial_data = [0,0,0]
            if len(serial_data) != 3:
                serial_data = [0,0,0]

            serial_data = np.array(serial_data)/1000
            if zero_data == True:
                serial_data = np.subtract(serial_data,serial_zero_data)

            #Median Filter! Yay
            serial_data_median = np.insert(serial_data_median,0,serial_data,axis=0)
            
            serial_data_median = serial_data_median[:-1]
            sorted_0 = np.sort(serial_data_median[:,0])
            sorted_1 = np.sort(serial_data_median[:,1])
            sorted_2 = np.sort(serial_data_median[:,2])
            serial_data_median_sorted = [sorted_0[2],sorted_1[2],sorted_2[2]]

            x.sensor_data = serial_data_median_sorted
        
        #Filter stuff

        x.compute_flux()
        x.reweigh()
        x.resample()
        pos, bend_a,bend_d = x.predict()
        x.update()
 
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

        if graph_params == True:
            bend_pos = [np.cos(bend_d)*bend_a/1.57,np.sin(bend_d)*bend_a/1.57]
            bend_params.append(bend_pos)
            
            #plt.plot(np.array(bend_params)[-10:-1][:,0],np.array(bend_params)[-10:-1][:,1])
            #plt.xlim(-1.57,1.57)
            #plt.ylim(-1.57,1.57)
            if i % 10 == 0:
                source = pd.DataFrame({
                    'x': np.array(bend_params)[-10:-1][:,0],
                    'f(x)': np.array(bend_params)[-10:-1][:,1]
                    })
                c= alt.Chart(source).mark_line().encode(
                    alt.X('x',scale=alt.Scale(domain=[-1.57,1.57])),
                    alt.Y('f(x)',scale=alt.Scale(domain=[-1.57,1.57]))
                    )
                params_plot.altair_chart(c)

            #params_plot.pyplot()
            #plt.close()
            
        if mode == "Training Mode":
            if graph_pose_error == True:
                pos_difference.append(np.linalg.norm(np.subtract(x.best_pos[0],magnet_pos[i])))
                error_plot.line_chart(pos_difference)

        i = i+1 
