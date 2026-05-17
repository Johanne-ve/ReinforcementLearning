"""
Testskript für trainierte AGV-Modelle
Führt Testspiele durch ohne Training (epsilon=0)
"""

import numpy as np
import csv

# Imports der Projektdateien
import factory_environment as env
import agv
import logic
import game_mechanics_4_agv_training as gm
import RL_Agent

# AGV-spezifische Konstanten
action_size_agv = 13
GAMES_PER_EPISODE = 8

# Hyperparameter für Agent (werden beim Laden überschrieben)
GAMMA = 0.95
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99
LEARNING_RATE = 0.0005

def test_model(model_path, num_games=20, verbose=True):
    """
    Testet ein einzelnes Modell
    
    Args:
        model_path: Pfad zur .keras Datei
        num_games: Anzahl Testspiele
        verbose: Ob Einzelergebnisse ausgegeben werden
    
    Returns:
        dict mit Statistiken
    """
    # Umgebung initialisieren
    my_env = env.env('ttable.csv', 'station_number.csv', 'env_control_order.csv')
    my_agv = agv.agv()
    my_logic = logic.logic()
    game_mech = gm.game_mechanics(my_logic, my_env, my_agv)
    
    # Modell laden mit epsilon=0 (komplett greedy, keine Exploration)
    test_agent = RL_Agent.DQNAgent(
        len(game_mech.act_state), 
        action_size_agv, 
        GAMMA, 
        0.0,  # epsilon = 0
        EPSILON_MIN, 
        EPSILON_DECAY, 
        LEARNING_RATE, 
        model_path
    )
    test_agent.epsilon = 0.0  # sicherstellen
    
    # Testspiele durchführen
    rewards = []
    times = []
    points = []
    trackers = []
    
    for i in range(num_games):
        my_env.reset()
        my_agv.reset()
        game_mech.reset()
        
        total_reward, total_time, tracker = game_mech.run_game(test_agent)
        
        rewards.append(total_reward)
        times.append(total_time)
        points.append(game_mech.points)
        trackers.append(tracker)
        
        if verbose:
            print(f"  Spiel {i+1:2d}: Reward {total_reward:+7.2f} | Zeit {total_time:6.0f}s | Punkte {game_mech.points:7.1f}")
    
    # Statistiken berechnen
    stats = {
        'model': model_path,
        'num_games': num_games,
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'mean_points': np.mean(points),
        'std_points': np.std(points),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        # Durchschnittliche Reward-Komponenten
        'avg_unload_good': np.mean([t['unload_good'] for t in trackers]),
        'avg_time_elapsed': np.mean([t['time_elapsed'] for t in trackers]),
        'avg_not_fitting': np.mean([t['not_fitting_station'] for t in trackers]),
        'avg_impossible': np.mean([t['impossible_action'] for t in trackers]),
    }
    
    return stats


def compare_models(model_paths, num_games=20):
    """
    Vergleicht mehrere Modelle
    
    Args:
        model_paths: Liste von Pfaden zu .keras Dateien
        num_games: Anzahl Testspiele pro Modell
    """
    print("="*80)
    print(f"MODELLVERGLEICH — {num_games} Testspiele pro Modell")
    print("="*80)
    
    all_stats = []
    
    for i, model_path in enumerate(model_paths, 1):
        print(f"\n[{i}/{len(model_paths)}] Teste: {model_path}")
        print("-"*80)
        
        stats = test_model(model_path, num_games=num_games, verbose=False)
        all_stats.append(stats)
        
        print(f"  Mean Reward:  {stats['mean_reward']:+7.2f} ± {stats['std_reward']:.2f}")
        print(f"  Mean Time:    {stats['mean_time']:7.0f}s ± {stats['std_time']:.0f}s")
        print(f"  Mean Punkte:  {stats['mean_points']:7.1f} ± {stats['std_points']:.1f}")
        print(f"  Range:        {stats['min_reward']:+7.2f} bis {stats['max_reward']:+7.2f}")
    
    # Ranking nach Mean Reward
    print("\n" + "="*80)
    print("RANKING (nach Mean Reward):")
    print("="*80)
    
    sorted_stats = sorted(all_stats, key=lambda x: x['mean_reward'], reverse=True)
    
    for rank, stats in enumerate(sorted_stats, 1):
        model_name = stats['model'].split('/')[-1]  # nur Dateiname
        print(f"{rank}. {model_name:30s} | Reward: {stats['mean_reward']:+7.2f} | "
              f"Zeit: {stats['mean_time']:6.0f}s | Punkte: {stats['mean_points']:7.1f}")
    
    print("\n" + "="*80)
    print("BESTE MODELL-DETAILS:")
    print("="*80)
    best = sorted_stats[0]
    print(f"Modell:           {best['model']}")
    print(f"Mean Reward:      {best['mean_reward']:+.2f} ± {best['std_reward']:.2f}")
    print(f"Mean Zeit:        {best['mean_time']:.0f}s ± {best['std_time']:.0f}s")
    print(f"Mean Punkte:      {best['mean_points']:.1f} ± {best['std_points']:.1f}")
    print(f"\nReward-Komponenten:")
    print(f"  unload_good:      {best['avg_unload_good']:+.2f}")
    print(f"  time_elapsed:     {best['avg_time_elapsed']:+.2f}")
    print(f"  not_fitting:      {best['avg_not_fitting']:+.2f}")
    print(f"  impossible:       {best['avg_impossible']:+.2f}")


if __name__ == "__main__":
    # VARIANTE 2: Mehrere Modelle vergleichen (auskommentieren wenn gewünscht)
 
    print("\n\n### MODELLVERGLEICH ###\n")
    models_to_compare = [
        './models/agv_V8.keras',
        './models/agv_V7.keras',
        #'./models/agv_V1.keras',
        './models/agv_V2.keras',
        './models/agv_final_ep40.keras',
        './models/agv_V7Basis_ep100.keras',
        #'./models/agv_V1.keras',
    ]
    compare_models(models_to_compare, num_games=20)

'''
🥇 1 agv_V7Basis_ep100.keras-8.48-7.84 bis -9.23 (±0.70)
🥈 2 agv_V2.keras-8.73-7.28 bis -9.58 (±1.15) ⚠️
🥉 3 agv_V7.keras-8.78-7.94 bis -9.42 (±0.74)
   4 agv_V8.keras-8.79-7.79 bis -9.80 (±1.01)
   5 agv_final_ep40.keras-9.03-8.29 bis -10.28 (±0.99)'''