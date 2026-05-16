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
state_size = state_size_agv
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
###### Rewards for agv training ########
###### Please change as you please #########
###########################################

REWARD_TIME_ELAPSED = -0.005           # ← 5× stärker (vorher -0.001)
REWARD_NOT_FITTING_STATION = -0.5
REWARD_TEST_STATION_NOT_ENOUGH_agv_CAP = -1.0
REWARD_EMPTY_BOXES_RETURN = 0
REWARD_RETURN_TO_STORAGE_WITH_GOODS = -0.5
REWARD_RETURN_TO_STORAGE_EMPTY = 0.2
REWARD_AGV_GOOD_0 = -0.3
REWARD_IMPOSSIBLE_ACTION = -1.0
REWARD_UNLOAD_GOOD = 0.1               # ← runter (vorher 0.5)

MAX_TIME_ONE_GAME = 3600*2
############################################
############################################


import numpy as np
import random

class game_mechanics:
    def __init__(self, logic, env, agv):
        self.logic = logic
        self.env = env
        self.agv = agv
        self.finished = False
        self.points = 0
        self.delay_minus = 0
        self.load_range = range(self.agv.capacity)
        self.station_range = range(self.agv.capacity, self.agv.capacity+STATION_COUNT)
        self.test_stat_order_range = range(self.agv.capacity + STATION_COUNT, self.agv.capacity + STATION_COUNT + 4)
        self.OHE_mask_range = range(self.agv.capacity + STATION_COUNT + 4, self.agv.capacity + STATION_COUNT + 4 + STATION_COUNT)
        self.act_state = [0]*state_size
        self.update_state_vec() 
        self.loading = False
        self.total_time = 0
        self.total_reward = 0.
        self.active_agent = 0  # Lager entscheidet zuerst
        reward = 0.

    def reset(self):
        self.finished = False
        self.points = 0
        self.delay_minus = 0
        #self.agv.reset()
        #self.env.reset()
        self.act_state = [0]*state_size
        self.update_state_vec()
        self.loading = False
        self.total_time = 0
        self.total_reward = 0.
        self.active_agent = 0  # Lager entscheidet zuerst
        reward = 0.

    def update_state_vec(self):      
        #self.act_state[self.agv.capacity + self.agv.act_stat] = 1  # agv startet an Lager B
        self.change_state_load()
        self.change_state_stat()
        self.change_test_stat_order()   
        self.change_OHE_mask()

    def change_state_load(self):
        occupied_spaces = self.agv.capacity - self.agv.load.count(0)
        self.act_state[0:4] = [0]*4
        if occupied_spaces > 0:
            self.act_state[occupied_spaces - 1] = 1
        
            
    def change_state_stat(self):
        for n in self.station_range:
            self.act_state[n] = 0
        self.act_state[self.station_range[0] + self.agv.act_stat] = 1
        
    def change_test_stat_order(self): 
        test_stat_order_vec = [0]*4
        test_stat_order = self.env.act_order_list[12][0]
        if test_stat_order > 0:
            test_stat_order_vec[test_stat_order - 1] = 1
        i = 0
        for n in self.test_stat_order_range:
            self.act_state[n] = test_stat_order_vec[i]
            i += 1
    
    def change_OHE_mask(self):
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
        for n in self.OHE_mask_range:
            self.act_state[n] = station_mask[i]
            i += 1
        

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
        if self.agv.act_stat > 1:   # agv nicht am Lager!
            return -1   # Code für kein Aufladen möglich, da am flaschen Ort
        if amount > self.agv.load.count(0):   # Falls nicht alles aufs agv passt
            return -2    # Code für zu voll
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

    def game_step(self, agv_agent = None):
        action = -1
        if (self.agv.load.count(0) == self.agv.capacity) and (self.agv.act_stat in [0,1]):
            self.active_agent = 0

        if self.agv.load.count(0) == 0:  # AGV ist voll
            self.active_agent = 1

        if self.active_agent == 0:   # Lager ist am Zug
            if self.agv.load.count(0) > 0:  # Wenn agv noch nicht voll
                action = self.agv_training_storage_decision()
                if action == -99:   # Es kann nichts mehr aufgeladen werden
                    self.active_agent = 1
                    self.loading = False
            else:  # AGV ist voll, Lager ist am Zug -> schicke weg von Lager
                self.active_agent = 1

        if self.active_agent == 1:
            action = self.choose_action(agv_agent)

        if action == -1:
            print('something went wrong....')
            
        res = self.do_action(action)
        return res, action

    def choose_action(self, agent):
        return agent.choose_action(np.reshape(self.act_state, [1, len(self.act_state)]))
        
    def do_action(self, action):   # Lager: 0: done, 1-10: Aufladen für Station 2-11
        if self.active_agent == 0:
            if action == 0:   # storage is done
                res = 0
                self.loading = False    
                self.active_agent = 1   # agv ab jetzt am Zug
            else:  # action = 1-10 -> Aufladen für Stationen 2 - 11 (STATION_COUNT beinhaltet 2x Lager!)
                res = self.load_goods(action + 1, 1)  # Für Lager wird nicht aufgeladen, daher action = 1 -> Aufladen für Station 2
                self.loading = True           
        else:    # AGV entscheidet
            res = self.drive_to(action)
                
        self.update_state_vec()
        if (res == -2) or (res == 50) or (res == -5) : # agv ist nach einem Fahrbefehl am Lager angekommen und ggf. abgeladen worden
            self.active_agent = 0  # Lager ist am Zug
        if self.agv.load.count(0) == 4 and self.agv.act_stat in [0,1]:  # AGV steht leer an Lager (zusätzlich zu oben wegen Problemen)
            self.active_agent = 0
        return res

    def agv_training_storage_decision(self):
        if random.random() < 0.999:
            good_vec = [x for x in range(2, STATION_COUNT-1)]
            random.shuffle(good_vec)
            for good in good_vec:
                if (self.env.act_order_list[good][0] - self.agv.load.count(good)) > 0:
                    return good - 1
            return -99
        else:
            return 0
    
        
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
    

    def run_game(self, agent):
        #print(gm.act_state)

        total_tracker = {k: 0 for k in ['impossible_action', 'test_station_not_enough_cap',
                                     'not_fitting_station', 'return_storage_with_goods',
                                     'empty_boxes_return', 'return_storage_empty',
                                     'agv_good_0', 'unload_good', 'time_elapsed']}

        while not(self.finished):
            old_state = [x for x in self.act_state]
            state = np.reshape(old_state, [1, agent.state_size])
            res, action = self.game_step(agent)
            next_state = np.reshape(self.act_state, [1, agent.state_size])
            self.award_points(res)

            reward, tracker = self.get_rewards(res)  # ← tracker empfangen
            for k in total_tracker:
                total_tracker[k] += tracker[k]       # ← akkumulieren
            self.total_reward += reward

            if self.active_agent == 1:
                if not((np.array_equal(state[0], next_state[0])) and (reward == 0)):
                    agent.remember(state[0], action, reward, next_state[0], self.finished)
                if (self.total_time > MAX_TIME_ONE_GAME) or self.finished:
                    return self.total_reward, self.total_time, total_tracker
            #if np.abs(reward) > 5:
            #    print('Something strange happened...')

            self.finished = self.test_finished()
        return self.total_reward, self.total_time, total_tracker


    def get_rewards(self, res):
        reward = 0
        tracker = {
            'impossible_action': 0,
            'test_station_not_enough_cap': 0,
            'not_fitting_station': 0,
            'return_storage_with_goods': 0,
            'empty_boxes_return': 0,
            'return_storage_empty': 0,
            'agv_good_0': 0,
            'unload_good': 0,
            'time_elapsed': 0
        }

        if (isinstance(res, int)):
            if res == -1:
                reward += REWARD_IMPOSSIBLE_ACTION
                tracker['impossible_action'] += REWARD_IMPOSSIBLE_ACTION  # ← NEU
            elif res == -2:
                reward += REWARD_TEST_STATION_NOT_ENOUGH_agv_CAP
                tracker['test_station_not_enough_cap'] += REWARD_TEST_STATION_NOT_ENOUGH_agv_CAP  # ← NEU
            elif res == -3:
                reward += REWARD_AGV_GOOD_0
                tracker['agv_good_0'] += REWARD_AGV_GOOD_0  # ← NEU

        else:
            res_code = res[1]
            if res_code == -60:
                reward += REWARD_NOT_FITTING_STATION
                tracker['not_fitting_station'] += REWARD_NOT_FITTING_STATION  # ← NEU
            elif res_code == -5:
                reward += REWARD_RETURN_TO_STORAGE_WITH_GOODS
                tracker['return_storage_with_goods'] += REWARD_RETURN_TO_STORAGE_WITH_GOODS  # ← NEU
            elif res_code == 50:
                reward += REWARD_EMPTY_BOXES_RETURN
                tracker['empty_boxes_return'] += REWARD_EMPTY_BOXES_RETURN  # ← NEU
            elif res_code == -1:
                reward += REWARD_NOT_FITTING_STATION
                tracker['not_fitting_station'] += REWARD_NOT_FITTING_STATION  # ← NEU
            elif res_code == -2:
                if self.agv.load.count(0) == self.agv.capacity:
                    reward += REWARD_RETURN_TO_STORAGE_EMPTY
                    tracker['return_storage_empty'] += REWARD_RETURN_TO_STORAGE_EMPTY  # ← NEU
            elif res_code > 0:
                reward += REWARD_TEST_STATION_NOT_ENOUGH_agv_CAP
                tracker['test_station_not_enough_cap'] += REWARD_TEST_STATION_NOT_ENOUGH_agv_CAP  # ← NEU
            elif res_code == 0:
                reward += REWARD_UNLOAD_GOOD
                tracker['unload_good'] += REWARD_UNLOAD_GOOD  # ← NEU

            reward += REWARD_TIME_ELAPSED * res[0]
            tracker['time_elapsed'] += REWARD_TIME_ELAPSED * res[0]  # ← NEU

        return reward, tracker