import neo
import os
import hsd


file_name = "rgb20190528_180044_6684_json/20190528_180044_6684.hsd"
with open(file_name, 'rb') as f:
    data = f.read()
hsdinput = hsd.load(file_name)
print(hsdinput)
