 import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

temp = ctrl.Antecedent(np.arange(0, 101, 1), 'temperature')
pressure = ctrl.Antecedent(np.arange(0, 11, 0.1), 'pressure')

hot_valve = ctrl.Consequent(np.arange(-90, 91, 1), 'hot_valve')
cold_valve = ctrl.Consequent(np.arange(-90, 91, 1), 'cold_valve')

temp['cold'] = fuzz.trimf(temp.universe, [0, 0, 25])
temp['cool'] = fuzz.trimf(temp.universe, [0, 25, 50])
temp['warm'] = fuzz.trimf(temp.universe, [25, 50, 75])
temp['not_hot'] = fuzz.trimf(temp.universe, [50, 75, 100])
temp['hot'] = fuzz.trimf(temp.universe, [75, 100, 100])

pressure['weak'] = fuzz.trimf(pressure.universe, [0, 0, 5])
pressure['med'] = fuzz.trimf(pressure.universe, [0, 5, 10])
pressure['strong'] = fuzz.trimf(pressure.universe, [5, 10, 10])

for valve in (hot_valve, cold_valve):
    valve['L_large'] = fuzz.trimf(valve.universe, [-90, -90, -45])
    valve['L_med'] = fuzz.trimf(valve.universe, [-90, -45, 0])
    valve['L_small'] = fuzz.trimf(valve.universe, [-45, -20, 0])
    valve['Z'] = fuzz.trimf(valve.universe, [-20, 0, 20])
    valve['R_small'] = fuzz.trimf(valve.universe, [0, 20, 45])
    valve['R_med'] = fuzz.trimf(valve.universe, [0, 45, 90])
    valve['R_large'] = fuzz.trimf(valve.universe, [45, 90, 90])

rules = [
    ctrl.Rule(temp['hot'] & pressure['strong'], (hot_valve['L_med'], cold_valve['R_med'])),
    ctrl.Rule(temp['hot'] & pressure['med'], (hot_valve['Z'], cold_valve['R_med'])),
    ctrl.Rule(temp['not_hot'] & pressure['strong'], (hot_valve['L_small'], cold_valve['Z'])),
    ctrl.Rule(temp['not_hot'] & pressure['weak'], (hot_valve['R_small'], cold_valve['R_small'])),
    ctrl.Rule(temp['warm'] & pressure['med'], (hot_valve['Z'], cold_valve['Z'])),
    ctrl.Rule(temp['cool'] & pressure['strong'], (hot_valve['R_med'], cold_valve['L_med'])),
    ctrl.Rule(temp['cool'] & pressure['med'], (hot_valve['R_med'], cold_valve['L_small'])),
    ctrl.Rule(temp['cold'] & pressure['weak'], (hot_valve['R_large'], cold_valve['Z'])),
    ctrl.Rule(temp['cold'] & pressure['strong'], (hot_valve['L_med'], cold_valve['R_med'])),
    ctrl.Rule(temp['warm'] & pressure['strong'], (hot_valve['L_small'], cold_valve['L_small'])),
    ctrl.Rule(temp['warm'] & pressure['weak'], (hot_valve['R_small'], cold_valve['R_small']))
]

mixer_ctrl = ctrl.ControlSystem(rules)
mixer_sim = ctrl.ControlSystemSimulation(mixer_ctrl)

mixer_sim.input['temperature'] = 30
mixer_sim.input['pressure'] = 3

mixer_sim.compute()

print(f"Кут повороту крану гарячої води: {mixer_sim.output['hot_valve']:.2f} градусів")
print(f"Кут повороту крану холодної води: {mixer_sim.output['cold_valve']:.2f} градусів")