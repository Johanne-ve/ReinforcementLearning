class logic:
    def __init__(self):
        self.is_initialized = True
    
    def arrival_action(self, agv, env, agent = None):
        if agv.act_stat >=2:  # nicht am Lager
            if agv.act_stat != 12:  # nicht an Teststation
                # versuche Auftrag zu erfüllen
                match_count = agv.load.count(agv.act_stat)
                to_unload = min(match_count, env.act_order_list[agv.act_stat][0])
                if to_unload == 0:
                    return -1   # Hier kein Auftrag zu erfüllen
                #print(f'match count: {match_count}')
                #print(f'to unload: {to_unload}')
                env.update_order_list(agv.act_stat, to_unload)
                agv.unload(to_unload) 
                if agv.act_stat == int(env.station_dict['Lack']):  # An der Lackiererei muss sofort wieder die Menge (leer) aufgeladen werden, die abgeladen wurde
                    agv.load_goods(agv.act_stat + 50, to_unload)
                return 0
            else:   # an Teststation
                to_load = env.act_order_list[agv.act_stat][0]    # Wie viel muss aufgeladen werden?
                if to_load == 0:
                    return -60
                #print(f'to_load: {to_load}')
                cap_too_low_by = agv.load_goods(agv.act_stat + 50, to_load)
                #print(f'cap_too_low_by: {cap_too_low_by}')
                loaded = to_load - cap_too_low_by
                #print(f'loaded: {loaded}')
                env.update_order_list(agv.act_stat, loaded)
                return int(cap_too_low_by)
        else:   ### agv kommt am Lager an
            empty_boxes = len([n for n in agv.load if n > 50])
            if empty_boxes > 0:  # Gibt es Leerbehälter zum Abladen?
                agv.get_rid_of_empty_boxes()
                return 50
            elif agv.load.count(0) < 4:   # AGV ist mit Ware (nicht Leergut) am Lager angekommen
                return -5
            else:
                return -2
            

    def driving_order(self, agv, env, goal):  # Ändere Position agv, gebe Fahrzeit + Stationsreaktionszeit zurück, rufe arrival_action auf
        time_elapsed = env.get_driving_time(agv, goal)
        if agv.act_stat == goal:   # Fahrauftrag lautet zu der Station zu fahren, an der AGV schon steht -> ist Quatsch
            return 0, - 1
        agv.act_stat = goal
        arr_action_return = self.arrival_action(agv, env)
        #print(f'time elapsed: {time_elapsed}')
        return time_elapsed, arr_action_return