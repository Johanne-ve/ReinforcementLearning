#############################################
############################################
#### Fixed Variables - do NOT touch #######
####################################################
#####################################################

STATION_COUNT = 13
TIME_URGENT_RED = 320
TIME_URGENT_ORANGE = 640
TIME_URGENT_YELLOW = 960
STATION_PROCESSING_TIME = 20
### Points
ORDER_OVERDUE_MINUS = - 0.2 
ORDER_OVERDUE_MINUS_EVERY_SECONDS = 20
POINTS_ORDER_FULFILLED = 1
TIME_FACTOR = 400

# Ladezustand + act_stat + ENV: order list
state_size_agv = 4+13+4+13
state_size_storage = 4*10 + 1
#state_size = state_size_storage

#### Action size
# Fahre zu einer der 13 Stationen
action_size_agv = 13
### Lade auf für eine der 11 Stationen
action_size_storage = 11
###################################################
###################################################
###### End of fixed Variables ######################
#####################################################
######################################################



############################################
###### Rewards for storage training ########
###### Please change as you please #########
###########################################
REWARD_IMPOSSIBLE_ACTION = -1.0
REWARD_COMPLETE_LOAD = 0.5
REWARD_PARTIAL_LOAD = -0.9
REWARD_EMPTY_AGV_POSITIONS_AWAY = [0.2, -0.1, -0.3, -0.5, -0.8]
REWARD_URGENT_LOAD = 0.35
REWARD_DRIVING_DISTANCE = -0.04
MAX_TIME_ONE_GAME = 3600*2
############################################
############################################


import numpy as np

class game_mechanics:
    def __init__(self, logic, env, agv):
        self.logic = logic
        self.env = env
        self.agv = agv
        self.finished = False
        self.points = 0
        self.delay_minus = 0
        self.demand_range_storage = range(10)
        self.urgency_range_storage = range(10,20)
        self.on_AGV_for_range_storage = range(20,30)
        self.OHE_mask_range_storage = range(30,41)
        self.act_state_storage = [0]*state_size_storage

        self.load_range_agv = range(self.agv.capacity)
        self.station_range_agv = range(self.agv.capacity, self.agv.capacity+STATION_COUNT)
        self.test_stat_order_range_agv = range(self.agv.capacity + STATION_COUNT, self.agv.capacity + STATION_COUNT + 4)
        self.OHE_mask_range_agv = range(self.agv.capacity + STATION_COUNT + 4, self.agv.capacity + STATION_COUNT + 4 + STATION_COUNT)
        self.act_state_agv = [0]*state_size_agv

        
        self.update_state_vec_agv() 
        self.update_state_vec_storage()
        self.loading = False
        self.total_time = 0
        self.total_reward = 0.
        self.active_agent = 0  # Lager entscheidet zuerst
        reward = 0.

    def reset(self):
        self.finished = False
        self.points = 0
        self.delay_minus = 0
        self.agv.reset()
        self.env.reset()
        self.act_state_storage = [0]*state_size_storage
        self.act_state_agv = [0]*state_size_agv
        self.update_state_vec_agv()
        self.update_state_vec_storage()
        self.loading = False
        self.total_time = 0
        self.total_reward = 0
        self.active_agent = 0  # Lager entscheidet zuerst
        reward = 0

    def update_state_vec_storage(self):      
        #self.act_state[self.agv.capacity + self.agv.act_stat] = 1  # agv startet an Lager B
        self.change_demand_state()
        self.change_urgency_state()
        self.change_on_AGV_for_state()   
        self.change_OHE_mask_storage()

    def update_state_vec_agv(self):      
        #self.act_state[self.agv.capacity + self.agv.act_stat] = 1  # agv startet an Lager B
        self.change_state_load()
        self.change_state_stat()
        self.change_test_stat_order()   
        self.change_OHE_mask_agv()

    def change_demand_state(self):       
        self.act_state_storage[0:10] = [0]*10
        i = 0
        for elem in self.env.act_order_list[2:-1]:
            self.act_state_storage[i] = elem[0]
            i += 1        
            
    def change_urgency_state(self):
        i = 0
        for n in self.urgency_range_storage:
            act_order = self.env.act_order_list[i]
            if act_order[0] > 0:
                if act_order[1] < TIME_URGENT_RED:
                    self.act_state_storage[n] = 3
                elif act_order[1] < TIME_URGENT_ORANGE:
                    self.act_state_storage[n] = 2
                elif act_order[1] < TIME_URGENT_YELLOW:
                    self.act_state_storage[n] = 1
                else:
                    self.act_state_storage[n] = 0
            i += 1
        
        
    def change_on_AGV_for_state(self): 
        on_AGV_for_vec = [0]*10
        for good in self.agv.load:
            if good <= 50:
                on_AGV_for_vec[good - 2] = 1
        i = 0
        for n in self.on_AGV_for_range_storage:
            self.act_state_storage[n] = on_AGV_for_vec[i]
            i += 1
    
    def change_OHE_mask_storage(self):
        station_mask = [0]*action_size_storage    # Maske für Aktionen: 0: Abfahren,  1-10 fahre Station 2-11 an ergibt 11(!) Einträge
        station_mask[0] = 1  # Es ist prinzipiell immer möglich, ohne Waren abzufahren - aber sinnfrei wenn Teststat keine Aufträge!?
        agv_free_cap = self.agv.load.count(0)
        if (agv_free_cap == self.agv.capacity) and (self.env.act_order_list[12][0] == 0):  # Wenn AGV nichts geladen hat und Teststation keine Aufträge -> schicke nicht(!) los
            station_mask[0] = 0
        if  agv_free_cap > 0:   # AGV ist nicht voll  
            ### Es ist prinzipiell sinnvoll Waren für Stationen zu laden, die einen Auftrag haben
            good = 2
            for order in self.env.act_order_list[2:-1]:
                act_demand = order[0] - self.agv.load.count(good)   # aktuell Anforderung: Anforderung minus AGV schon geladen
                if act_demand > 0:
                    station_mask[good - 1] = 1
                good += 1
        i = 0
        for n in self.OHE_mask_range_storage:
            self.act_state_storage[n] = station_mask[i]
            i += 1

    def change_state_load(self):  # OHE encoded load state: all 0 -> 0, 1 0 0 0 one good, 0 0 0 1 four goods, e.g.
        occupied_spaces = self.agv.capacity - self.agv.load.count(0)
        self.act_state_agv[0:4] = [0]*4
        if occupied_spaces > 0:
            self.act_state_agv[occupied_spaces - 1] = 1
        
            
    def change_state_stat(self):
        for n in self.station_range_agv:
            self.act_state_agv[n] = 0
        self.act_state_agv[self.station_range_agv[0] + self.agv.act_stat] = 1
        
    def change_test_stat_order(self): 
        test_stat_order_vec = [0]*4
        test_stat_order = self.env.act_order_list[12][0]
        if test_stat_order > 0:
            test_stat_order_vec[test_stat_order - 1] = 1
        i = 0
        for n in self.test_stat_order_range_agv:
            self.act_state_agv[n] = test_stat_order_vec[i]
            i += 1

    def change_OHE_mask_agv(self):
        station_mask = [0]*STATION_COUNT
        agv_free_cap = self.agv.load.count(0)
        if  agv_free_cap == self.agv.capacity:   # AGV ist leer -> Lager oder Teststation sinnvoll
            station_mask[0] = 1  # Lager auf jeden Fall, Teststation über eigene Abfrage
            station_mask[1] = 1

        if (agv_free_cap >= self.env.act_order_list[12][0]) and (self.env.act_order_list[12][0] > 0):  # wenn AGV genügend Platz hat um Leergut von Teststation aufzunehmen und dort Aufträge offen
            station_mask[12] = 1
        
        ### Es ist prinzipiell sinnvoll Stationen anzufahren, für die das AGV Waren geladen hat
        for i in range(len(self.agv.load)):
            what = self.agv.load[i]
            if what > 50:  # Das AGV hat Leergut geladen - prinzipiell könnte man also das Lager anfahren
                station_mask[0] = 1
                station_mask[1] = 1
            elif what > 0:
                station_mask[what] = 1
        i = 0
        for n in self.OHE_mask_range_agv:
            self.act_state_agv[n] = station_mask[i]
            i += 1
        #print(f'change OHE mask agv: station_mask last 13 entries: {station_mask[-13:]}')
        #print(f'now act_state_agv is: {self.act_state_agv}')
        #print(f'agv load: {self.agv.load}')
  

    def drive_to(self, new_stat):
        old_order_list = [x for x in self.env.act_order_list]
        delta_time, return_code = self.logic.driving_order(self.agv, self.env, new_stat)
        self.act_order_list_delta_time(delta_time)  # Wichtig dass dies vor(!) Updaten des state-vec passiert, damit Zeit an der angefahrenen Station ggf. berücksichtigt wird
        return delta_time, return_code, old_order_list[new_stat]  # Melde zurück, wie der Eintrag der order_list nun aussieht (für Belohnung etc. ob noch Zeit war)


    def act_order_list_delta_time(self, delta_time):
        self.total_time += delta_time
        for n in range(len(self.env.act_order_list)):
            if (self.agv.act_stat != n) and (self.env.act_order_list[n][0] > 0): # Wenn noch Aufträge an Station offen zählt Zeit runter; steht AGV an Station wurde Auftrag (wahrscheinlich) eben erst erfüllt und Zeit zählt auch wenn Null
                self.env.act_order_list[n][1] -= delta_time


    def test_finished(self):
        open_orders = sum(i[0] for i in self.env.act_order_list)
        number_waste_on_agv = sum([i > 50 for i in self.agv.load])
        return (open_orders == 0) and (number_waste_on_agv == 0)

    def load_goods(self, what, amount = 1):
        #print(f'loading {amount} goods for station {what}..')
        if self.agv.act_stat > 1:   # agv nicht am Lager!
            return -1   # Code für kein Aufladen möglich, da am flaschen Ort -> darf hier nicht passieren!
        if self.agv.load.count(0) == 0:  # Das sollte nie passieren, da vorher schon das AGV aktiviert wird
            return -2
        elif amount > self.agv.load.count(0):   # Falls nicht alles aufs agv passt
            self.agv.load_goods(what, self.agv.load.count(0)) # Lade auf was geht
            return -3    # Code für Teilladung
        self.agv.load_goods(what, amount)
        self.change_state_load()
        if not(self.loading):  # Wenn neuer Ladevorgang Zeit hochzählen, ansonsten nicht (Ladevorgang am Lager kostet nicht mehr Zeit)
            self.act_order_list_delta_time(STATION_PROCESSING_TIME)
        return int(what)

    def force_unload_all(self):
        if self.agv.act_stat > 1:
            return -1   # Code für kein Abladen möglich, da nicht am Lager
        else:
            self.agv.force_unload_all()
            self.act_order_list_delta_time(STATION_PROCESSING_TIME)
            return 0    

    def game_step(self, storage_agent = None):
        action = -1
        self.update_state_vec_storage()
        action = self.choose_action(storage_agent)
        res = self.do_action(action)
        return res, action

    def choose_action(self, agent):
        act_state = self.act_state_storage

        return agent.choose_action(np.reshape(act_state, [1, len(act_state)]))
        
    def do_action(self, action):   # Lager: 0: done, 1-10: Aufladen für Station 2-11
        if action == 0:   # storage is done
            res = -9
            self.loading = False    
            self.active_agent = 1   # agv ab jetzt am Zug
        else:  # action = 1-10 -> Aufladen für Stationen 2 - 11 (STATION_COUNT beinhaltet 2x Lager!)
            #print(f'load good for station {action+1}')
            res = self.load_goods(action + 1, self.env.act_order_list[action + 1][0] - self.agv.load.count(action +1))  # Für Lager wird nicht aufgeladen, daher action = 1 -> Aufladen für Station 2
            self.loading = True           
                
        self.update_state_vec_storage()
        
        return res
    
        
    def award_points(self, res):   # Punkte fürs Spielen! Nicht (direkt) dasselbe wie reward fürs RL!

        #### Minuspunkte wegen rückständiger Aufträge
        delta_time = [x[1] for x in self.env.act_order_list]  # Punktabzug für unerfüllte Aufträge mit negativer Zeit
        delay_minus_points = sum([(x / ORDER_OVERDUE_MINUS_EVERY_SECONDS) % 1 * ORDER_OVERDUE_MINUS for x in delta_time])
        new_minus = delay_minus_points - self.delay_minus 
        self.points += new_minus
        self.delay_minus = delay_minus_points + self.delay_minus

        #### Punkte fürs Aufträge erfüllen
        if isinstance(res, list):   # Gibt es nur, wenn gefahren ist - Auftrag könnte erfüllt sein
            num_order_fulfilled = res[2][0] - self.env.act_order_list[self.agv.act_pos][0]     # Alter Eintrag des Auftrags - neuer -> Anzahl erfüllt
            self.points += num_order_fulfilled * POINTS_ORDER_FULFILLED
    

    def run_game(self, storage_agent):
        memory_vec = []
        res = -99
        dist = 0
        action = 0
        #print(gm.act_state)
        while not(self.finished):
            #### Es erhält einen Malus für Distanzen zwischen Stationen geladener Waren
            #### Es erhält Belohnung für erfüllte  Aufträge (soll dafür sorgen, dass AGV meist voll gefüllt wird)
            #### Es erhält Malus für Aufräge, die überfällig sind

            old_action = action
            old_state = [x for x in self.act_state_storage]
            state = np.reshape(old_state, [1, storage_agent.state_size])

            ##### Erhalte nächste Aktion, merke Resultat
            res, action = self.game_step(storage_agent)
            ######
            reward = 0
            if res == -3:
                reward += REWARD_PARTIAL_LOAD
            elif (res == -9):  # Wenn Lager ein nicht volles AGV wegschicken möchte
                reward += REWARD_EMPTY_AGV_POSITIONS_AWAY[self.agv.load.count(0)]
                
            elif res > 0:
                reward += REWARD_COMPLETE_LOAD
                if self.env.act_order_list[action + 1][1] < TIME_URGENT_ORANGE:
                    reward += REWARD_URGENT_LOAD
                if self.env.act_order_list[action + 1][1] < TIME_URGENT_RED: # Wenn sogar ganz dringend, verdopple den reward fürs Aufladen
                    reward += REWARD_URGENT_LOAD
            if old_action > 0:  # Es wurde etwas aufgeladen
                dist = self.env.get_driving_distance(old_action + 1, action + 1)
                reward += REWARD_DRIVING_DISTANCE*dist
            
            if (self.agv.load.count(0) == 0) or (res == - 9):  # Wenn AGV voll oder Lager AGV wegschickt -> Zustand resetten
                action = 0
                
                for good in self.agv.load:
                    self.env.act_order_list[good][0] = 0
                    if good >= 10:
                        self.env.act_order_list[12][0] = 0
                for n in range(2, len(self.env.act_order_list)):
                    if self.env.act_order_list[n][0] > 0:
                        self.env.act_order_list[n][1] -= dist/ self.agv.get_act_vel() * TIME_FACTOR + STATION_PROCESSING_TIME*2
                self.total_time += dist/ self.agv.get_act_vel() * TIME_FACTOR + STATION_PROCESSING_TIME*2
                dist = 0
                for n in range(self.agv.capacity):
                    self.agv.load[n] = 0   # Lade alle Waren ab
                self.update_state_vec_storage()

            storage_agent.remember(old_state, action, reward, self.act_state_storage, self.finished)            
            self.total_reward += reward
            self.finished = self.test_finished()
            if (self.total_time > MAX_TIME_ONE_GAME) or self.finished:
                return self.total_reward, self.total_time

            
            
        return self.total_reward, self.total_time
