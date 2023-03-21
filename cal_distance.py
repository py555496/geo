from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r * 1000
import pandas as pd

def read_csv_file(file_path):
    tiananmen_loc = [116.403972, 39.915118]
    df = pd.read_csv(file_path)
    locs = list(zip(df['x'], df['y']))
    dis_arr = []
    for g in locs:
        dis = haversine(tiananmen_loc[0], tiananmen_loc[1], g[0], g[1])
        dis_arr.append(dis)
    df['distance'] = dis_arr
    df.to_csv('output.csv', index=False)
read_csv_file('./all_data.csv')