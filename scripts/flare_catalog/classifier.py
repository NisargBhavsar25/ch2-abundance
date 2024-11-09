import sys

import numpy as np
import pandas as pd

arg = sys.argv[1]

classfile = open(f"./SuryaDrishti/classnames/classnames/classnames{arg}.txt", 'r')
classfile = classfile.readlines()

xsm_catalog = pd.read_csv('./SuryaDrishti/catalog/SD-CH2.csv')

outputfile = open(f"./classifier_output/output{arg}.csv", 'w')

outputfile.write('class_file_name,')
for i in xsm_catalog.columns:
    outputfile.write(i + ',')
outputfile.write('\n')

linklen = len("20190914T144415606_20190914T144423606")

i = 0
for class_link in classfile:
    class_link = class_link.strip()
    if len(class_link) != linklen:
        continue
    st, et = class_link.split('_')
    sy, stm = st.split('T')
    ey, etm = et.split('T')

    stm = int(stm[0:2])*3600000 + int(stm[2:4]) * \
        60000 + int(stm[4:6])*1000 + int(stm[6:])
    etm = int(etm[0:2])*3600000 + int(etm[2:4]) * \
        60000 + int(etm[4:6])*1000 + int(etm[6:])

    stm = stm/1000
    etm = etm/1000

    sy = np.int64(sy)
    loca = xsm_catalog.loc[(xsm_catalog['date'] == sy) & (
        xsm_catalog['post_fit_start_time'] <= stm) & (xsm_catalog['post_fit_end_time'] >= etm)]

    if len(loca) != 0:
        outputfile.write(class_link[-59:-11] + ',')
        data = ""
        for i in loca.iloc[0]:
            data += str(i) + ','
        outputfile.write(data)
        print(data)
        outputfile.write('\n')

    i += 1
    if i % 10000 == 0:
        print(i)

outputfile.close()
