import csv

with open('C:/Users/kk/Desktop/color_mapping.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

color_mapping = {}    
for i in range(len(data)):
    temp1 = data[i][1].split("(")[1].split(")")[0].split(",")[0]
    temp2 = data[i][1].split("(")[1].split(")")[0].split(",")[1].split(" ")[1]
    temp3 = data[i][1].split("(")[1].split(")")[0].split(",")[2].split(" ")[1]
    key = temp1 + temp2 + temp3   
    value = int(data[i][0])
    

    if not (key in color_mapping):
        color_mapping[key] = value
