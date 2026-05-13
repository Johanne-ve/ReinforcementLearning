###########
### Constants ####
V_MAX = 20
AGV_CAP = 4

class agv:
    def __init__(self):
        self.capacity = AGV_CAP
        self.start_stat = 1  #### lager B  [7, 6]
        self.act_stat = self.start_stat
        self.load = [0, 0, 0, 0]
        self.goal_station_num = 1
        self.V_MAX = V_MAX
        self.act_vel = self.V_MAX

    def reset(self):
        self.act_stat = self.start_stat
        self.load = [0, 0, 0, 0]
        self.act_vel = self.V_MAX

    def unload(self, amount = -1):  ## -1 heißt alles abladen, sonst eine Menge erforderlich
        unloaded = 0
        for pos in range(self.capacity):
            if self.load[pos] == self.act_stat:
                self.load[pos] = 0      
                unloaded += 1
                #print('unloaded')
            if unloaded == amount:
                break

    def force_unload_all(self):
        self.load = [0]*self.capacity

    def free_cap(self):
        return self.load.count(0)
    
    
    def load_goods(self, what, amount):
        free_cap = self.free_cap()        
        loaded = 0
        for pos in range(self.capacity):
            if amount > loaded:
                if self.load[pos] == 0:
                    self.load[pos] = what
                    loaded += 1
            else:
                break              
        return amount - loaded

    def get_act_vel(self):
        self.act_vel = self.V_MAX*self.get_act_vel_factor()
        return self.act_vel

    def get_act_vel_factor(self):
        return (1 - 0.05*(4 - self.load.count(0)))

    def get_rid_of_empty_boxes(self):
        for n in range(self.capacity):
            if self.load[n] > 50:
                self.load[n] = 0