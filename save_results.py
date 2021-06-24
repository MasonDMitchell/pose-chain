from read_sensor import sensor_read,sensor_run
import pandas as pd
import numpy as np
file1 = 'bend_4_try_2'
file2 = 'bend_test_4_try_2'

bend_dir,bend_ang,time = sensor_run(file2)
time = np.array(time) -time[0]
df = pd.DataFrame(bend_dir,columns=['bend_dir'])

df['bend_ang'] = bend_ang
df['time'] = time
df.to_csv('./filter_readings/' + file1 + '.csv')
