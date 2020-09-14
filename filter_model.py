import os
import sys
import glob
import time
import numpy as np
import pandas as pd
import subprocess as sp
from Q_Velest import simple_model
from heapq import merge
from gmt_layout import color_cycle

out_dir = os.path.join(os.path.dirname(os.getcwd()), 'plot_result')
output = os.path.join(os.path.dirname(os.getcwd()), 'output')

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
if not os.path.exists(output):
    os.makedirs(output)

Adjust = sorted(glob.glob("*HypoAdjustment-all*"))

init_model = sorted(glob.glob("*ModelInitial*"))
fnal_model = sorted(glob.glob("*ModelFinal*"))
init_cat = sorted(glob.glob("*CatalogInitial*"))
fnal_cat = sorted(glob.glob("*CatalogFinal*"))

if len(init_model) != len(fnal_model):
    sys.exit("Number of initial model doesn't match with final model")
elif len(init_cat) != len(fnal_cat):
    sys.exit("Number of initial catalog doesn't match with final catalog")
elif len(fnal_model) != len(fnal_cat):
    sys.exit("Number of model doesn't match with catalog")
elif len(fnal_model) != len(Adjust):
    sys.exit("Number of model doesn't match with adjustment data")

mod_rms = pd.DataFrame(columns=['Model', 'Iterasi', 'Min_RMS'])

for i, ha in zip(range(len(Adjust)), Adjust):

    if os.stat(ha).st_size == 0:
    
        sys.exit("Adjustment data is empty\n")
        
    data = pd.read_csv(ha, header=None, delim_whitespace=True)
    
    mod_rms.loc[i] = [os.path.basename(ha).replace('-HypoAdjustment-all.dat', ''), data[5].idxmin(), data[5].min()]
    
print(mod_rms.sort_values('Min_RMS'))

threshold = float(input('\nSet minimum RMS threshold \n'))

drop_rows = mod_rms[mod_rms.Min_RMS > threshold]
accp_rows = mod_rms[mod_rms.Min_RMS <= threshold]
dropped_model = drop_rows['Model'].to_list()
accepted_model = accp_rows['Model'].to_list()

print(f'\n{len(accepted_model)} accepted model:\n{accp_rows}\n')

plot = open('Accepted-Model.bat', 'w')
plot.write('@echo off\n')
plot.write('echo ^>^>^>Plot All Final Model and Accepted Model . . .& echo.\n')
plot.write('set ps=Accepted_Model.ps\n')
plot.write('set out=../plot_result/Accepted_Model.png\n')
plot.write('set J1=8/-15\n')
plot.write('set R_nhyp=-4/62/0/80\n')
plot.write('set R_nhyp2=0/80/-4/62\n')

plot.write('set R_mod=3/9/-4/62\n')
plot.write('\npsbasemap -JX%J1% -R%R_mod% -BWnSe -Byafg+l"Depth (km)" -Bxaf+l"Velocity (km/s)" '
           '--FONT_ANNOT_PRIMARY=10 --FONT_LABEL=13 -K > %ps%\n')
           
for i, dm in zip(range(len(dropped_model)), dropped_model):
    
    plot.write(f'gawk "{{print $2, $1}}" {dm}-ModelFinal.dat | psxy -J -R -W1,gray -O -K >> %ps%\n')
    
acc_model = pd.DataFrame(columns=['depth'])
list_acc_model = []

for i, am in zip(range(len(accepted_model)), accepted_model):

    plot.write(f'gawk "{{print $2, $1}}" {am}-ModelFinal.dat | psxy -J -R -W1,black -O -K >> %ps%\n')

    if os.stat(f'{am}-ModelFinal.dat').st_size == 0:

        sys.exit("Model data is empty\n")

    data = pd.read_csv(f'{am}-ModelFinal.dat', header=None, delim_whitespace=True)

    sampling = 0.5

    arr_top = np.arange(data[0].min(), data[0].max() + sampling, sampling)
    arr_bot = np.arange(data[0].min() + sampling, data[0].max(), sampling)
    list_dep = list(merge(arr_top, arr_bot))
    list_vel = np.zeros(len(list_dep))
    index_layer = np.arange(0, len(list_dep))

    acc_model = acc_model.reindex(acc_model.index.union(index_layer))
    acc_model['depth'] = list_dep
    acc_model[f'vel_{am}'] = list_vel
    list_acc_model.append(f'vel_{am}')

    k = 0
    for j in range(len(acc_model['depth'])):

        if acc_model['depth'][j] == data[0][k]:
            acc_model[f'vel_{am}'][j] = data[1][k]
            k += 1

        elif acc_model['depth'][j] < data[0][k]:
            acc_model[f'vel_{am}'][j] = data[1][k]

        elif acc_model['depth'][j] > data[0][k]:
            sys.exit('Sampling is too large or maksimum depth first model too small.\n'
                     'Please check sampling value than cover all layer depth value')

acc_model['mean_vel'] = acc_model[list_acc_model].mean(axis=1)

list_dep = acc_model['depth'].to_list()
mean_vel = acc_model['mean_vel'].to_list()
list_dam = np.zeros(len(mean_vel)).tolist()

# SET SIMPLE VELOCITY MODEL THEN WRITE TO NLLOC VELEST FORMAT
simple_v, simple_d = simple_model(mean_vel, list_dep)
vpvs = 1.75

mod_nlloc = open(os.path.join(output, 'nlloc_mean_1Dmodel.dat'), 'w')
mod_nlloc.write(
    '# model layers (LAYER depth, Vp_top, Vp_grad, Vs_top, Vs_grad, p_top, p_grad)')

for v, d in zip(simple_v, simple_d):
    l = f'\nLAYER {d:5.1f} {v:5.2f} 0.00   {v/vpvs:5.2f}  0.00  2.7 0.0'
    mod_nlloc.write(l)

mod_nlloc.close()

simple_v, simple_d, simple_dam = simple_model(mean_vel, list_dep, list_dam)

modout = open(os.path.join(output, 'velest_mean_1Dmodel.dat'), 'w')
modout.write(' Output model:\n')
modout.write(str(len(simple_v)) + '\n')

for vl, dp, dm in zip(simple_v, simple_d, simple_dam):
    l = f'{vl:5.2f}     {dp:7.2f}  {dm:7.3f}\n'
    modout.write(l)

modout.close()

plot.write(f'gawk "{{print $1, $2}}" {os.path.join(output, "velest_mean_1Dmodel.dat")} | psxy -J -R -hi2 -W2,blue -O -K >> %ps%\n')
    
plot.write('\npsbasemap -JX4/11 -R0.5/9/-0.5/21 -BWnSe -O -K >> %ps%\n')
plot.write(f'echo 1 1 > leg.tmp\n')
plot.write(f'echo 3.4 1 >> leg.tmp\n')
plot.write(f'echo 4 1 Mean 1D Model | '
           f'pstext -J -R -F+f10p,Helvetica,black+jLM -N -O -K >> %ps%\n')
plot.write(f'psxy leg.tmp -J -R -W2,blue -N -O -K >> %ps%\n')
plot.write(f'echo 1 2 > leg.tmp\n')
plot.write(f'echo 3.4 2 >> leg.tmp\n')
plot.write(f'echo 4 2 Accepted Model | '
           f'pstext -J -R -F+f10p,Helvetica,black+jLM -N -O -K >> %ps%\n')
plot.write(f'psxy leg.tmp -J -R -W1,black -N -O -K >> %ps%\n')
plot.write(f'echo 1 3 > leg.tmp\n')
plot.write(f'echo 3.4 3 >> leg.tmp\n')
plot.write(f'echo 4 3 Final Model | '
           f'pstext -J -R -F+f10p,Helvetica,black+jLM -N -O -K >> %ps%\n')
plot.write(f'psxy leg.tmp -J -R -W1,gray -N -O -K >> %ps%\n')

clr = color_cycle

for i, am in zip(range(len(accepted_model)), accepted_model):

    color, transp = clr(i, start_transp=50, grad=10)
    
    if i == 0:
        plot.write(f'gawk "{{print $3}}" {am}-CatalogInitial.dat | pshistogram -JX%J1% -R%R_nhyp% -A -W1.01+l -Ggray '
                   f'-t30 --FONT_ANNOT_PRIMARY=10 --FONT_LABEL=13 -L -O -K -X8.5 >> %ps%\n')
    plot.write(f'gawk "{{print $3}}" {am}-CatalogFinal.dat | pshistogram -JX%J1% -R%R_nhyp% -A -W1.01+l -G{color} '
               f'-t30 --FONT_ANNOT_PRIMARY=10 --FONT_LABEL=13 -L -O -K >> %ps%\n')

plot.write('psbasemap -JX%J1% -R%R_nhyp2% -BwnSE -Byafg+l"Depth (km)" -Bxaf+l"n events of accepted models" '
           '--FONT_ANNOT_PRIMARY=10 --FONT_LABEL=13 -O -K >> %ps%\n')

plot.write('\npsbasemap -JX4/11 -R0.5/9/-0.5/21 -BWnSe -O -K -X3.5 >> %ps%\n')

if len(accepted_model) <= 20:

    plot.write(f'echo 3 2 | '
               f'psxy -J -R -SB0.2b1 -Ggray -N -O -K >> %ps%\n')
    plot.write(f'echo 3.5 2 Initial Hypocenter | '
               f'pstext -J -R -F+f9p,Helvetica,black+jLM -N -O -K >> %ps%\n')
               
else:

    plot.write(f'echo 3 2 | '
               f'psxy -J -R -SB0.15b1 -Ggray -N -O -K >> %ps%\n')
    plot.write(f'echo 3.5 2 Initial Hypocenter | '
               f'pstext -J -R -F+f7p,Helvetica,black+jLM -N -O -K >> %ps%\n')

for i, am in zip(range(len(accepted_model)), accepted_model):

    color, transp = clr(i, start_transp=50, grad=10)

    if len(accepted_model) <= 20:

        plot.write(f'echo 3 {len(accepted_model) + 2 - i} | '
                   f'psxy -J -R -SB0.2b1 -G{color} -t{transp} -N -O -K >> %ps%\n')
        if i == len(accepted_model) - 1:
            plot.write(f'echo 3.5 {len(accepted_model) + 2 - i} {am} | '
                       f'pstext -J -R -F+f9p,Helvetica,black+jLM -N -O >> %ps%\n')
        else:
            plot.write(f'echo 3.5 {len(accepted_model) + 2 - i} {am} | '
                       f'pstext -J -R -F+f9p,Helvetica,black+jLM -N -O -K >> %ps%\n')

    else:

        plot.write(f'echo 3 {22 - i * 20 / len(accepted_model)} | '
                   f'psxy -J -R -SB0.15b1 -G{color} -t{transp} -N -O -K >> %ps%\n')
        if i == len(accepted_model) - 1:
            plot.write(f'echo 3.5  {22 - i * 20 / len(accepted_model)} {am} | '
                       f'pstext -J -R -F+f7p,Helvetica,black+jLM -N -O >> %ps%\n')
        else:
            plot.write(f'echo 3.5  {22 - i * 20 / len(accepted_model)} {am} | '
                       f'pstext -J -R -F+f7p,Helvetica,black+jLM -N -O -K >> %ps%\n')

plot.write('\npsconvert %ps% -Tg -E512 -A0.2 -P -F%out%\n')
plot.write('del gmt.* *.tmp *.cpt *.ps\n')
plot.close()

time.sleep(1)
# print(f"\nPlotting All Final Model and Accepted Model . . .\n")
p = sp.Popen(['Accepted-Model.bat'], shell=True)
p.communicate()
time.sleep(2)