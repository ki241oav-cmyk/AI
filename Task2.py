import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

temp = ctrl.Antecedent(np.arange(10, 41, 1), 'temperature')
speed = ctrl.Antecedent(np.arange(-2, 2.1, 0.1), 'speed')
regulator = ctrl.Consequent(np.arange(-90, 91, 1), 'regulator')

temp['v_cold'] = fuzz.trimf(temp.universe, [10, 10, 18])
temp['cold'] = fuzz.trimf(temp.universe, [15, 20, 23])
temp['normal'] = fuzz.trimf(temp.universe, [21, 23, 25])
temp['warm'] = fuzz.trimf(temp.universe, [24, 27, 32])
temp['v_warm'] = fuzz.trimf(temp.universe, [30, 40, 40])

speed['negative'] = fuzz.trimf(speed.universe, [-2, -2, 0])
speed['zero'] = fuzz.trimf(speed.universe, [-0.5, 0, 0.5])
speed['positive'] = fuzz.trimf(speed.universe, [0, 2, 2])

regulator['L_large'] = fuzz.trimf(regulator.universe, [-90, -90, -45])
regulator['L_small'] = fuzz.trimf(regulator.universe, [-45, -25, 0])
regulator['off'] = fuzz.trimf(regulator.universe, [-10, 0, 10])
regulator['R_small'] = fuzz.trimf(regulator.universe, [0, 25, 45])
regulator['R_large'] = fuzz.trimf(regulator.universe, [45, 90, 90])

rules = [
    ctrl.Rule(temp['v_warm'] & speed['positive'], regulator['L_large']),
    ctrl.Rule(temp['v_warm'] & speed['negative'], regulator['L_small']),
    ctrl.Rule(temp['warm'] & speed['positive'], regulator['L_large']),
    ctrl.Rule(temp['warm'] & speed['negative'], regulator['off']),
    ctrl.Rule(temp['v_cold'] & speed['negative'], regulator['R_large']),
    ctrl.Rule(temp['v_cold'] & speed['positive'], regulator['R_small']),
    ctrl.Rule(temp['cold'] & speed['negative'], regulator['R_large']),
    ctrl.Rule(temp['cold'] & speed['positive'], regulator['off']),
    ctrl.Rule(temp['v_warm'] & speed['zero'], regulator['L_large']),
    ctrl.Rule(temp['warm'] & speed['zero'], regulator['L_small']),
    ctrl.Rule(temp['v_cold'] & speed['zero'], regulator['R_large']),
    ctrl.Rule(temp['cold'] & speed['zero'], regulator['R_small']),
    ctrl.Rule(temp['normal'] & speed['positive'], regulator['L_small']),
    ctrl.Rule(temp['normal'] & speed['negative'], regulator['R_small']),
    ctrl.Rule(temp['normal'] & speed['zero'], regulator['off'])
]

ac_system = ctrl.ControlSystem(rules)
ac_sim = ctrl.ControlSystemSimulation(ac_system)

ac_sim.input['temperature'] = 28
ac_sim.input['speed'] = 0.2
ac_sim.compute()

print(f"Положення регулятора: {ac_sim.output['regulator']:.2f}")

temp.view()
plt.title('Membership Functions: Temperature')
plt.show()

speed.view()
plt.title('Membership Functions: Speed')
plt.show()

regulator.view(sim=ac_sim)
plt.title('Regulator Output')
plt.show()