import csv
import random

##############
#### Constants for environment
###############
TIME_FACTOR = 400
STATION_PROCESSING_TIME = 20
V_MAX = 20
ORDER_TIME_BONUS = 300



class env:
    ###############################################################################################
    ########### Init takes 4 files, last one is optional ##########################################
    ########### if 4th file is given, an order list is loaded instead of created randomly #########
    ########### file1: table with distances between stations measured in tiles
    ########### file2: table with name (function) of station and its number
    ########### file3: file to control how many open orders a station can have, typcially ranging from 0-4
    ########### file4: optional - file with an order list
    
    def __init__(self, file1, file2, file3, file4 = None):
        with open(file1, newline='') as csvfile:
            data = list(csv.reader(csvfile, delimiter = ";"))
        self.ttable = data
        with open(file2, newline='') as csvfile:
            data = list(csv.reader(csvfile, delimiter = ";"))
        self.station_dict = dict(data)
        with open(file3, newline='') as csvfile:
            data = list(csv.reader(csvfile, delimiter = ";"))
        self.order_data = data
        if file4 is not None:
            self.load_order_list(file4)
        else:
            self.act_order_list = self.create_order_list(self.order_data) #self.init_order_list
        self.V_MAX = V_MAX

    
    ############ reset means: create new order list, reset V_MAX although it should not have changed ################
    def reset(self):
        self.act_order_list = self.create_order_list(self.order_data)
        self.V_MAX = V_MAX

    ####### function which takes starting station and final station and returns the distance in tiles #######################
    def get_driving_distance(self, start, stop):
        if isinstance(start, str):
            start_num = int(self.station_dict[start])
        else:
            start_num = start
        if isinstance(stop, str):
            stop_num = int(self.station_dict[stop])
        else:
            stop_num = stop
        dist = self.ttable[start_num + 1][stop_num + 1]
        return int(dist)

    ######### function which takes an agv object and the destination station and returns the driving time ##################
    ######### the starting position is inferred from the current position of the agv object ################################
    ######### the true velocity (load taken into account) is also inferred from the agv object #############################
    def get_driving_time(self, agv, goal):
        t = self.get_driving_distance(agv.act_stat, goal) / agv.get_act_vel() * TIME_FACTOR + STATION_PROCESSING_TIME
        return t

    ###### create order list #########################################################
    ###### the time is hard-coded here, not a really flexible approach ###############
    def create_order_list(self, data):
        order_list = [[0,0], [0,0]]
        for stat in data[2:]:
            num  = random.randrange(int(stat[1]), int(stat[2]))
            time = random.randrange(350, 1200)
            if num == 0:
                order_list.append([0,0])
            else:
                order_list.append([num, time])
        return order_list

    ####### function which updates order list when orders are fulfilled  #################
    ####### if order is fulfilled only partially, grant an order_time_bonus -> order is not that urgent, anymore ########
    def update_order_list(self, stat, change):
        self.act_order_list[stat][0] -= change
        if self.act_order_list[stat][0] > 0:  # Es ist noch nicht alles da, trotzdem Auftrag nicht mehr so dringend
            self.act_order_list[stat][1] += ORDER_TIME_BONUS
    

    def save_order_list(self, order_list_file):
        with open(order_list_file, "w") as f:
            wr = csv.writer(f)
            wr.writerows(self.act_order_list)

    def load_order_list(self, order_list_file):
        with open(order_list_file, "r", newline = '\n') as f:
            csv_reader = csv.reader(f)
            list_of_csv = list(csv_reader)
        self.act_order_list = [[int(x[0]), int(x[1])] for x in list_of_csv]