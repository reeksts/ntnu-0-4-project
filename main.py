import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

dataset = pd.read_excel('input_data.xlsx', sheet_name='Sheet1', index_col=0)

H1 = 2.153
H1B = 0.185
H2 = 2.163
H2T = 0.181

KAIR = 0.024
KWAT = 0.6
KICE = 2.24


def adj_temp(x, loc):
    if loc=='top1':
        return 1.00400 * x + 0.27879
    elif loc=='top2':
        return 1.00373 * x + 0.26542
    elif loc=='bot1':
        return 1.00568 * x + 0.28725
    elif loc=='bot2':
        return 1.00230 * x + 0.27976


def calculate_th_con_from_test(temps, name):
    print(name)

    # Step 1 - correct temperatures
    t1 = adj_temp(temps[0], 'top1')
    t2 = adj_temp(temps[1], 'top2')
    b1 = adj_temp(temps[2], 'bot1')
    b2 = adj_temp(temps[3], 'bot2')

    # Step 1 - calculate gradients
    grad_top = (t1 - t2) / H1
    grad_bot = (b1 - b2) / H2

    # Step 3 - adjust thermal conductivity of glass
    avg_top = (t1 + t2) / 2
    th_top = 0.001733 * avg_top + 1.04
    avg_bot = (b1 + b2) / 2
    th_bot = 0.001733 * avg_bot + 1.04

    # Step 4 - calculate flux value
    flux_top = grad_top * th_top
    flux_bot = grad_bot * th_bot
    flux_avg = np.mean([flux_top, flux_bot])

    # Step 5 - extrapolate to calculate surface temperatures
    surf_temp_top = t2 - grad_top * H1B
    surf_temp_bot = b1 + grad_bot * H2T

    # Step 6 - calculate sample temperature gradient
    grad_sample = (surf_temp_top - surf_temp_bot) / 7.5

    # Step 7 - calculate sample thermal conductivity
    sample_th = flux_avg / grad_sample

    print(sample_th)
    print('\n')

    return sample_th

def calculate_th_con_from_model(ks, n):
    # Step 1 - calculate k dry
    kdry = ks**((1-n)**0.59)*KAIR**(n**0.73)

    # Step 2 - calculate k sat
    ksat_u = ks**(1-n)*KWAT**n
    ksat_f = ks**(1-n)*KICE**n

    # Step ?? - calculate Sr frozen?

    # Step 3 - calculate k norm
    kr_u_list = []
    kr_f_list = []
    for i in np.arange(0, 1.01, 0.01):
        kr_u = (4.7*i) / (1+3.7*i)
        kr_f = (1.8*i) / (1+0.8*i)
        kr_u_list.append(kr_u)
        kr_f_list.append(kr_f)

    # Step 4 - calculate k
    ku_list = []
    kf_list = []
    for i in kr_u_list:
        ku = (ksat_u - kdry) * i + kdry
        ku_list.append(ku)

    for i in kr_f_list:
        kf = (ksat_f - kdry) * i + kdry
        kf_list.append(kf)

    return ku_list, kf_list

# Test value lists
Lorenskog_test_uf = []
Lorenskog_test_f = []
Tau_test_uf = []
Tau_test_f = []
Sarpsborg_test_uf = []
Sarpsborg_test_f = []
Vassfjell_test_uf = []
Vassfjell_test_f = []
Limestone_test_uf = []
Limestone_test_f = []

# Moisture value lists
Lorenskog_moisture = [dataset.loc['Lørenskog I uf']['w%'],
                      dataset.loc['Lørenskog II uf']['w%'],
                      dataset.loc['Lørenskog III uf']['w%']]
Tau_moisture = [dataset.loc['Tau I uf']['w%'],
                      dataset.loc['Tau II uf']['w%'],
                      dataset.loc['Tau III uf']['w%']]
Sarpsborg_moisture = [dataset.loc['Sarpsborg I uf']['w%'],
                      dataset.loc['Sarpsborg II uf']['w%'],
                      dataset.loc['Sarpsborg III uf']['w%']]
Limestone_moisture = [dataset.loc['Limestone I uf']['w%'],
                      dataset.loc['Limestone II uf']['w%'],
                      dataset.loc['Limestone III uf']['w%']]
Vassfjell_moisture = [dataset.loc['Vassfjell I uf']['w%'],
                      dataset.loc['Vassfjell II uf']['w%'],
                      dataset.loc['Vassfjell III uf']['w%']]

# Sr value lists
Lorenskog_Sr = [dataset.loc['Lørenskog I uf']['Sr'],
                dataset.loc['Lørenskog II uf']['Sr'],
                dataset.loc['Lørenskog III uf']['Sr']]
Tau_Sr = [dataset.loc['Tau I uf']['Sr'],
          dataset.loc['Tau II uf']['Sr'],
          dataset.loc['Tau III uf']['Sr']]
Sarpsborg_Sr = [dataset.loc['Sarpsborg I uf']['Sr'],
                dataset.loc['Sarpsborg II uf']['Sr'],
                dataset.loc['Sarpsborg III uf']['Sr']]
Limestone_Sr = [dataset.loc['Limestone I uf']['Sr'],
                dataset.loc['Limestone II uf']['Sr'],
                dataset.loc['Limestone III uf']['Sr']]
Vassfjell_Sr = [dataset.loc['Vassfjell I uf']['Sr'],
                dataset.loc['Vassfjell II uf']['Sr'],
                dataset.loc['Vassfjell III uf']['Sr']]


for mat in ['Lørenskog I uf', 'Lørenskog II uf', 'Lørenskog III uf']:
    th = calculate_th_con_from_test([dataset.loc[mat]['Top I'],
                                     dataset.loc[mat]['Top II'],
                                     dataset.loc[mat]['Bottom I'],
                                     dataset.loc[mat]['Bottom II']],
                                    'Lørenskog unfrozen:')
    Lorenskog_test_uf.append(th)

for mat in ['Lørenskog I f', 'Lørenskog II f', 'Lørenskog III f']:
    th = calculate_th_con_from_test([dataset.loc[mat]['Top I'],
                                     dataset.loc[mat]['Top II'],
                                     dataset.loc[mat]['Bottom I'],
                                     dataset.loc[mat]['Bottom II']],
                                    'Lørenskog frozen')
    Lorenskog_test_f.append(th)

for mat in ['Tau I uf', 'Tau II uf', 'Tau III uf']:
    th = calculate_th_con_from_test([dataset.loc[mat]['Top I'],
                                     dataset.loc[mat]['Top II'],
                                     dataset.loc[mat]['Bottom I'],
                                     dataset.loc[mat]['Bottom II']],
                                    'Tau unfrozen:')
    Tau_test_uf.append(th)

for mat in ['Tau I f', 'Tau II f', 'Tau III f']:
    th = calculate_th_con_from_test([dataset.loc[mat]['Top I'],
                                     dataset.loc[mat]['Top II'],
                                     dataset.loc[mat]['Bottom I'],
                                     dataset.loc[mat]['Bottom II']],
                                    'Tau frozen:')
    Tau_test_f.append(th)

for mat in ['Sarpsborg I uf', 'Sarpsborg II uf', 'Sarpsborg III uf']:
    th = calculate_th_con_from_test([dataset.loc[mat]['Top I'],
                                     dataset.loc[mat]['Top II'],
                                     dataset.loc[mat]['Bottom I'],
                                     dataset.loc[mat]['Bottom II']],
                                    'Sarpsborg unfrozen:')
    Sarpsborg_test_uf.append(th)

for mat in ['Sarpsborg I f', 'Sarpsborg II f', 'Sarpsborg III f']:
    th = calculate_th_con_from_test([dataset.loc[mat]['Top I'],
                                     dataset.loc[mat]['Top II'],
                                     dataset.loc[mat]['Bottom I'],
                                     dataset.loc[mat]['Bottom II']],
                                    'Sarpsborg frozen:')
    Sarpsborg_test_f.append(th)

for mat in ['Limestone I uf', 'Limestone II uf', 'Limestone III uf']:
    th = calculate_th_con_from_test([dataset.loc[mat]['Top I'],
                                     dataset.loc[mat]['Top II'],
                                     dataset.loc[mat]['Bottom I'],
                                     dataset.loc[mat]['Bottom II']],
                                    'Limestone unfrozen:')
    Limestone_test_uf.append(th)

for mat in ['Limestone I f', 'Limestone II f', 'Limestone III f']:
    th = calculate_th_con_from_test([dataset.loc[mat]['Top I'],
                                     dataset.loc[mat]['Top II'],
                                     dataset.loc[mat]['Bottom I'],
                                     dataset.loc[mat]['Bottom II']],
                                    'Limestone frozen:')
    Limestone_test_f.append(th)

for mat in ['Vassfjell I uf', 'Vassfjell II uf', 'Vassfjell III uf']:
    th = calculate_th_con_from_test([dataset.loc[mat]['Top I'],
                                     dataset.loc[mat]['Top II'],
                                     dataset.loc[mat]['Bottom I'],
                                     dataset.loc[mat]['Bottom II']],
                                    'Vassfjell unfrozen:')
    Vassfjell_test_uf.append(th)

for mat in ['Vassfjell I f', 'Vassfjell II f', 'Vassfjell III f']:
    th = calculate_th_con_from_test([dataset.loc[mat]['Top I'],
                                     dataset.loc[mat]['Top II'],
                                     dataset.loc[mat]['Bottom I'],
                                     dataset.loc[mat]['Bottom II']],
                                    'Vassfjell frozen:')
    Vassfjell_test_f.append(th)

# Model prediction:
Lorenskog_model_uf, Lorenskog_model_f = calculate_th_con_from_model(dataset.loc['Lørenskog I f']['ks'],
                                                                    dataset.loc['Lørenskog I f']['n'])
Tau_model_uf, Tau_model_f = calculate_th_con_from_model(dataset.loc['Tau I f']['ks'],
                                                        dataset.loc['Tau I f']['n'])
Sarpsborg_model_uf, Sarpsborg_model_f = calculate_th_con_from_model(dataset.loc['Sarpsborg I f']['ks'],
                                                                    dataset.loc['Sarpsborg I f']['n'])
Limestone_model_uf, Limestone_model_f = calculate_th_con_from_model(dataset.loc['Limestone I f']['ks'],
                                                                    dataset.loc['Limestone I f']['n'])
Vassfjell_model_uf, Vassfjell_model_f = calculate_th_con_from_model(dataset.loc['Vassfjell I f']['ks'],
                                                                    dataset.loc['Vassfjell I f']['n'])

def figure_unfrozen():
    # Initializer
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))

    # Plot data
    ax.plot(Lorenskog_Sr, Lorenskog_test_uf,
            label='Lørenskog',
            marker='o',
            ms=6,
            linestyle='None',
            markerfacecolor='tab:blue')
    ax.plot(Tau_Sr, Tau_test_uf,
            label='Tau',
            marker='o',
            ms=6,
            linestyle='None',
            markerfacecolor='tab:orange')
    ax.plot(Sarpsborg_Sr,
            Sarpsborg_test_uf,
            label='Sarpsborg',
            marker='o',
            ms=6,
            linestyle='None',
            markerfacecolor='tab:green')
    ax.plot(Limestone_Sr,
            Limestone_test_uf,
            label='Tromsdalen',
            marker='o',
            ms=6,
            linestyle='None',
            markerfacecolor='tab:red')
    ax.plot(Vassfjell_Sr,
            Vassfjell_test_uf,
            label='Vassfjell',
            marker='o',
            ms=6,
            linestyle='None',
            markerfacecolor='tab:purple')


    ax.plot(np.arange(0, 1.01, 0.01),
            Lorenskog_model_uf,
            color='tab:blue',
            linestyle='--')
    ax.plot(np.arange(0, 1.01, 0.01),
            Tau_model_uf,
            color='tab:orange',
            linestyle='--')
    ax.plot(np.arange(0, 1.01, 0.01),
            Sarpsborg_model_uf,
            color='tab:green',
            linestyle='--')
    ax.plot(np.arange(0, 1.01, 0.01),
            Limestone_model_uf,
            color='tab:red',
            linestyle='--')
    ax.plot(np.arange(0, 1.01, 0.01),
            Vassfjell_model_uf,
            color='tab:purple',
            linestyle='--')

    # adjust design
    ax.set_xlabel('Degree of saturation', size=14)
    ax.set_ylabel('Thermal conductivity, W/mK', size=14)
    ax.legend(loc='lower right', prop={'size': 14})
    ax.set_xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks = [0, 0.5, 1, 1.5, 2.0, 2.5]
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 2.5)

    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(1.5)
    ax.tick_params(axis='both', direction='in', width=1.5, right=True, top=True, labelsize=14, size=5, pad=8)
    #ax.xaxis.set_minor_locator(AutoMinorLocator())
    #ax.tick_params(axis='x', which='minor', direction='in', right=True, top=True, color='r', size=10)

    plt.savefig('unfrozen.jpg', dpi=300)

    plt.show()

def figure_frozen():
    # Initializer
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))

    # Plot data
    ax.plot(Lorenskog_Sr,
            Lorenskog_test_f,
            label='Lørenskog',
            marker='o',
            ms=6,
            linestyle='None',
            markerfacecolor='tab:blue')
    ax.plot(Tau_Sr,
            Tau_test_f,
            label='Tau',
            marker='o',
            ms=6,
            linestyle='None',
            markerfacecolor='tab:orange')
    ax.plot(Sarpsborg_Sr,
            Sarpsborg_test_f,
            label='Sarpsborg',
            marker='o',
            ms=6,
            linestyle='None',
            markerfacecolor='tab:green')
    ax.plot(Limestone_Sr,
            Limestone_test_f,
            label='Tromsdalen',
            marker='o',
            ms=6,
            linestyle='None',
            markerfacecolor='tab:red')
    ax.plot(Vassfjell_Sr,
            Vassfjell_test_f,
            label='Vassfjell',
            marker='o',
            ms=6,
            linestyle='None',
            markerfacecolor='tab:purple')

    ax.plot(np.arange(0, 1.01, 0.01),
            Lorenskog_model_f,
            color='tab:blue',
            linestyle='--')
    ax.plot(np.arange(0, 1.01, 0.01),
            Tau_model_f,
            color='tab:orange',
            linestyle='--')
    ax.plot(np.arange(0, 1.01, 0.01),
            Sarpsborg_model_f,
            color='tab:green',
            linestyle='--')
    ax.plot(np.arange(0, 1.01, 0.01),
            Limestone_model_f,
            color='tab:red',
            linestyle='--')
    ax.plot(np.arange(0, 1.01, 0.01),
            Vassfjell_model_f,
            color='tab:purple',
            linestyle='--')

    # adjust design
    ax.set_xlabel('Degree of saturation', size=14)
    ax.set_ylabel('Thermal conductivity, W/mK', size=14)
    ax.legend(loc='lower right', prop={'size': 14})
    ax.set_xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks = [0, 0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3.5)

    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(1.5)
    ax.tick_params(axis='both', direction='in', width=1.5, right=True, top=True, labelsize=14, size=5, pad=8)
    #ax.xaxis.set_minor_locator(AutoMinorLocator())
    #ax.tick_params(axis='x', which='minor', direction='in', right=True, top=True, color='r', size=10)

    plt.savefig('frozen.jpg', dpi=300)

    plt.show()


figure_unfrozen()
figure_frozen()