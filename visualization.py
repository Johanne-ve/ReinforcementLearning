### ROUTING - visualization
# general strategy: Drive to next higher or next lower station number
# exception 1: shortcut betweeen station 6/7 and 10/11
# exception 2: final goal is storage 0 or storage 1
#PIXELANZAHL = 32

STATION_COUNT = 13

import csv

class visualization:
    def __init__(self, gm, stations_positions):
        self.gm = gm
        with open(stations_positions, newline='') as csvfile:
            self.data = list(csv.reader(csvfile, delimiter = ";"))
        self.data[0][0] = '0'
        self.driving_dir_dict = {'x': lambda x: [x[0] + 1, x[1]], '-x': lambda x: [x[0] - 1, x[1]],
                                'y': lambda x: [x[0], x[1] + 1], '-y': lambda x: [x[0], x[1] - 1]}
        self.station_positions = self.extract_station_positions_from_data(self.data)
        self.station_iso_positions = [self.get_iso_view_coords_from_row_col(self.station_positions[x]) for x in range(STATION_COUNT)]
        final_handling_pos_dir = [self.get_handling_pos_dir(x) for x in range(STATION_COUNT)]
        self.final_handling_iso_pos_stat = [self.driving_dir_dict[final_handling_pos_dir[x]](self.station_iso_positions[x]) for x in range(STATION_COUNT)]
        #print(self.station_positions)
        self.agv_act_iso_pos = self.final_handling_iso_pos_stat[self.gm.agv.act_stat]

    def reset(self):
        self.agv_act_iso_pos = self.final_handling_iso_pos_stat[self.gm.agv.act_stat]

        

    def extract_station_positions_from_data(self, data):
        return (dict(tuple(zip([int(x[0]) for x in data], [[int(x[1]), int(x[2])] for x in data]))))


    def get_iso_view_coords_from_station(self, stat_num):
        stat_coords = self.station_positions[stat_num]
        return self.get_iso_view_coords_from_row_col(stat_coords)

    def get_iso_view_coords_from_row_col(self, row_col):
        yiso = 10 - row_col[1]
        xiso = row_col[0]
        return [xiso, yiso]

    def get_agv_next_iso_coords(self, final_goal):
        next_stop = self.next_stop(final_goal)
        return next_stop
        
    def next_stop(self, final_goal):
        if self.gm.avg.act_stat in [6, 7]:  # use shortcut?
            if final_goal == 10:
                next_stop = 10
            elif final_goal >= 11:
                next_stop = 11
        elif self.gm.avg.act_stat in [10, 11]: # use shortcut?
            if final_goal in [1, 7]:
                next_stop = final_goal
            elif final_goal <= 6:
                next_stop = 6
        elif final_goal == 0:  # goal is storage 0?
            if self.gm.avg.act_stat == 3:
                next_stop = 0
            elif self.gm.avg.act_stat == 2:
                next_stop = 3
        elif final_goal == 1:   # goal is storage 1?
            if self.gm.avg.act_stat == 8:
                next_stop = 1
            elif self.gm.avg.act_stat in [6,7]:
                next_stop = 1
        elif self.gm.avg.act_stat == 0:  # starting point is storage 0 -> first step drive to 3
            next_stop = 3
        elif self.gm.avg.act_stat == 1:   # starting point is storage 1 -> set next_stop to 7 (avg will not(!) move):
            next_stop = 7
        elif self.gm.avg.act_stat < final_goal:   # no special case: if goal station number is less than actual station number
            next_stop = self.gm.avg.act_stat - 1
        else:   # no special case and goal station number is higher than actual station number
            next_stop = self.gm.avg.act_stat + 1
        return next_stop

    
    
    def next_iso_coords(self, final_goal):
        #final_iso_coords_station = self.get_iso_view_coords_from_station(final_goal)
        #final_handling_pos_dir = self.get_handling_pos_dir(final_goal)
        final_iso_coords = self.final_handling_iso_pos_stat[final_goal]
        
        if self.agv_act_iso_pos[0] == 6:  # general direction most likely in y-direction
            if (final_goal == 1) and (self.agv_act_iso_pos[1] == 8):   # AGV an "Abzweig" zu Stationen 2-4, möchte aber zu lager 1
                driving_direction = '-y'
            elif (final_goal <= 4) and (self.agv_act_iso_pos[1] == 8): # AGV an "Abzweig" zu Stationen 2-4
                driving_direction = '-x'
            elif self.agv_act_iso_pos[1] == 6 and final_goal >= 10:  # AGV an Abkürzungsweg
                driving_direction = 'x'
            elif self.agv_act_iso_pos[1] == 2 and final_goal >= 9:  # AGV rechts außen
                driving_direction = 'x'
            else: 
                if final_iso_coords[1] == self.agv_act_iso_pos[1]:
                    if final_goal in [11, 12]:  # AGV an Stat 6 oder 5
                        driving_direction = '-y'
                    else:  # Kann nur an Station 7 stehen
                        driving_direction = 'y'
                        
                elif final_iso_coords[1] > self.agv_act_iso_pos[1]:
                    driving_direction = 'y'
                else:
                    driving_direction = '-y'
        elif self.agv_act_iso_pos[0] == 10:   # In unterer Reihe wird in +/- y-Richtung gefahren, außer an Abkürzung ggf. und ganz außen
            if self.agv_act_iso_pos[1] == 6:
                if final_goal <= 8: # Nimm Abkürzung 
                    driving_direction = '-x'
                elif final_goal <= 10:
                    driving_direction = '-y'
                else:
                    driving_direction = 'y'
            elif (self.agv_act_iso_pos[1] == 2) and final_goal <= 9:  # AGV steht ganz außen und muss jetzt ggf. in die x-Richtung fahren
                driving_direction = '-x'
            else:
                if self.agv_act_iso_pos[1] == 3 and final_goal in [8, 9]:
                    driving_direction = '-y'
                elif (final_goal <= 5) and self.agv_act_iso_pos[1] > 6:#
                    driving_direction = '-y'
                elif final_iso_coords[1] >= self.agv_act_iso_pos[1]:
                    driving_direction = 'y'
                else:
                    driving_direction = '-y'
            if (self.agv_act_iso_pos[1] >= 7) and (final_goal <= 6):
                driving_direction = '-y'
        elif self.agv_act_iso_pos[0] == 3:  # Wenn AGV auf der Achse zum Lager 0 steht
            if final_goal == 0:
                driving_direction = '-y'
            elif self.agv_act_iso_pos[1] == 7:
                driving_direction = 'y'
            elif final_goal == 2 and (self.agv_act_iso_pos[1] == 8):
                driving_direction = '-x'
            else:
                driving_direction = 'x'
        elif self.agv_act_iso_pos[0] < final_iso_coords[0]:
            driving_direction = 'x'
        else:
            driving_direction = '-x'

        return self.driving_dir_dict[driving_direction](self.agv_act_iso_pos), driving_direction
            

    def get_handling_pos_dir(self, final_goal):
        if final_goal in [1, 10, 11, 12]:
            handling_dir = 'x'
        elif final_goal in [5,6,7,8]:
            handling_dir = '-x'
        elif final_goal == 0:
            handling_dir = 'y'
        else:
            handling_dir = '-y'
        return handling_dir
            

    def next_coords_center(self, final_goal):
        return self.get_coords_center_from_position(self.next_stop(final_goal))
        
        
