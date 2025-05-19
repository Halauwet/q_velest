import os
import sys
import time
import shutil
import platform
import subprocess
import hashlib
import statistics
import numpy as np
import subprocess as sp
from velest_rw import ids, ReadMod, ReadCNV, ReadSta, ReadVelestMain, ReadVelestOptmVel, \
    ReadVelestVar, ReadVelestVel, CNV_EvtCount
from cryptography.fernet import Fernet
from datetime import datetime as dt
from gmt_layout import color_cycle
from matplotlib import pyplot as plt

"""
===========================================
Velest processing routine by @eqhalauwet
==========================================

Python module for reading velest output, ploting and exporting to GMT input.

Written By, eQ Halauwet BMKG-PGR IX Ambon.
yehezkiel.halauwet@bmkg.go.id


Notes:

1. Read velest mainprint to see adjustmet hypocenter, velocity model, and RMS every iteration

Logs:

2017-Sep: Added _check_header line to automatic check data format from few Seiscomp3 version (see Notes).
2019-Oct: Major change: store readed data in dictionary format.
2020-May: Correction: select phase only without 'X' residual (unused phase on routine processing).
2020-Jul: Major change added Run_Velest() and Run_VelestSet() to run Velest recursively
2020-Jul: RunVelestSet() added recursive routine to adjust input velocity layer if velest is hang (solution not stable)

"""


FERNET_KEY = b'vH_oNg_JBd66-4Fz51rTxotxpbC-3Hhe50mg_geEAMI='  # <<< replace this securely
HMAC_SECRET = b'Qsecret_for_digital_signature'  # Must match in validation

fernet = Fernet(FERNET_KEY)

def get_machine_uuid():
    system = platform.system()
    try:
        if system == "Windows":
            output = subprocess.check_output("wmic csproduct get uuid", shell=True)
            uuid = output.decode().splitlines()[2].strip()
            return uuid

        elif system == "Linux":
            # Read UUID from DMI path
            try:
                result = subprocess.run(
                    ['sudo', 'cat', '/sys/class/dmi/id/product_uuid'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                return result.stdout.strip()
            except subprocess.CalledProcessError as e:
                return f"Command failed: {e}"

        elif system == "Darwin":
            # macOS: Read IOPlatformUUID
            output = subprocess.check_output(
                "ioreg -rd1 -c IOPlatformExpertDevice", shell=True
            ).decode()
            for line in output.splitlines():
                if "IOPlatformUUID" in line:
                    return line.split('=')[-1].strip().strip('"')
        else:
            return None
    except Exception as e:
        print(f"Error retrieving UUID: {e}")
        return None


def is_license_valid(machine_uuid):
    authorized_hash = ["5d25467892a1cfab0cdaf7f9e149efa4f479d9c60af46733311e3267a9cf947e",
                       "be11461055a5482143235980e03877fd4ceaa3e60af327dcf0b6e5b2ef9406f8"]
    hashed_uuid = hashlib.sha256(machine_uuid.encode()).hexdigest()

    return hashed_uuid in authorized_hash


def plot_model(vel_type='P'):
    # vel_type = 'P'

    my_dir = os.getcwd()
    mod_dir = os.path.join('input', 'mod')

    inpmod = []
    mod_nm = []
    for m in os.listdir(os.path.join(my_dir, mod_dir)):
        if os.path.isfile(os.path.join(my_dir, mod_dir, m)):
            inpmod.append(os.path.join(mod_dir, m))
            mod_nm.append(m[:-4])

    fig1, ax1 = plt.subplots()
    for mod, md_nm in zip(inpmod, mod_nm):
        plt.figure()
        plt.ion()
        plt.show()
        md = ReadMod(mod)
        x = md[vel_type]['vel']
        y = [d * -1 for d in md['P']['dep']]
        ax1.plot(x, y)
        plt.plot(x, y)
        plt.title(md_nm)
        plt.draw()
        plt.pause(0.001)


def velest2gmt(model_nm, init_pha, final_pha, main_out, itrmax, invratio, plot=False, gmtplot=False):
    """
    output gmt plot file in batch. Only works on gmt for windows
    :param model_nm: model root name for output
    :param init_pha: initial phase.cnv data
    :type init_pha: list
    :param final_pha: final phase.cnv data
    :type final_pha: list
    :param main_out: mainprint data
    :type main_out: list
    :param plot: option to plot model (on python)
    :param gmtplot: option to plot all gmt output
    """
    # init_pha = ['input/phase.cnv', 'output/grad1_1/finalhypo01_grad1_1.cnv']
    # final_pha = ['output/grad1_1/finalhypo01_grad1_1.cnv', 'output/grad1_1/finalhypo02_grad1_1.cnv']
    # main_out = ['output/grad1_1/mainprint01_grad1_1.out', 'output/grad1_1/mainprint02_grad1_1.out']
    # model_nm = 'grad1_1'

    # Cumulative every set output
    final_data = {}
    velestdata = {}
    init_data = {}
    optimum_data = {}

    if len(main_out) > 1:
        flag_iset = True
    else:
        flag_iset = False
    iset = 0
    allitt = 0

    max_vel = 0
    min_vel = 5
    # max_dvel = 0
    # min_dvel = 5
    max_hyp = 0
    max_veladj = 0
    min_veladj = 5
    dep = []
    sta_dly = []

    itr_all = []
    adj_ot_all = []
    adj_lon_all = []
    adj_lat_all = []
    adj_dep_all = []
    rms_all = []

    ids = ''
    station = 'input/station.dat'
    outvel = 'gmt'
    plot_result = 'plot_result'

    bmkg_sta = ReadSta(station)

    if not os.path.exists(outvel):
        os.makedirs(outvel)
    if not os.path.exists(plot_result):
        os.makedirs(plot_result)

    if len(main_out) == len(init_pha) and len(init_pha) == len(final_pha):

        for i in range(len(main_out)):
            iset = i + 1

            if flag_iset:
                outdir = outvel + '/Iteration_Set-' + str(iset)
                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                ffinal = open(outdir + '/' + model_nm + '-CatalogFinal-iSet_' + str(iset) + '.dat', 'w')
                fvar = open(outdir + '/Plot-' + model_nm + '.bat', 'w')
                finit = open(outdir + '/' + model_nm + '-CatalogInitial-iSet_' + str(iset) + '.dat', 'w')

            velestdata, init_data, final_data, optimum_data, next_set,  ids = ReadVelestMain(main_out[i])
            
            i_dat = ReadCNV(init_pha[i])
            f_dat = ReadCNV(final_pha[i])

            _catalog = ''

            if iset == 1:
                f_init = open(outvel + '/' + model_nm + '-CatalogInitial.dat', 'w')
            f_final = open(outvel + '/' + model_nm + '-CatalogFinal.dat', 'w')

            for evt in sorted(i_dat):
                i_hyp = i_dat[evt]
                _catalog = (f"{i_hyp['lon']:.4f} {i_hyp['lat']:.4f} {i_hyp['dep']:.2f} {i_hyp['mag']:.2f} "
                            f"{i_hyp['gap']} {i_hyp['rms']:.2f} "
                            f"{i_hyp['erlon']} {i_hyp['erlat']} {i_hyp['errv']} {i_hyp['nevt']}\n")
                if flag_iset:
                    finit.write(_catalog)
                if iset == 1:
                    f_init.write(_catalog)

            for evt in sorted(f_dat):
                f_hyp = f_dat[evt]
                _catalog = (f"{f_hyp['lon']:.4f} {f_hyp['lat']:.4f} {f_hyp['dep']:.2f} {f_hyp['mag']:.2f} "
                            f"{f_hyp['gap']} {f_hyp['rms']} "
                            f"{f_hyp['erlon']} {f_hyp['erlat']} {f_hyp['errv']} {f_hyp['nevt']}\n")
                f_final.write(_catalog)
                if flag_iset:
                    ffinal.write(_catalog)

            if flag_iset:
                finit.close()
                ffinal.close()
            if iset == 1:
                f_init.close()
            f_final.close()

            itr = []
            adj_ot = []
            adj_lon = []
            adj_lat = []
            adj_dep = []
            rms = []
            adj_vel = []
            vel = []
            dep = []
            max_vel = 0
            min_vel = 5
            max_dvel = 0
            min_dvel = 5
            # output initial model and adjustment iteration 0
            ini = init_data
            
            if iset == 1:
                vel_arr = ini['vel_mod']['vel']
                dep_arr = ini['vel_mod']['dep']
            else:
                vel_arr, dep_arr = adjust_layer(ini['vel_mod']['vel'], ini['vel_mod']['dep'],
                                                step=2, dev=0.05, plot=False, flag_v=False)
            dep_arr = np.array(dep_arr)
            vel_arr = np.array(vel_arr)
            damp_arr = np.array(ini['vel_mod']['damp'])
            model_adj = np.vstack((dep_arr, vel_arr, damp_arr)).T
            result = np.asarray(model_adj)
            if flag_iset:
                np.savetxt(outdir + '/' + model_nm + '-ModelInitial-iSet_' + str(iset) +
                           '.dat', result, fmt='%.6f', delimiter=" ")
            if iset == 1:
                np.savetxt(outvel + '/' + model_nm + '-ModelInitial.dat', result, fmt='%.6f', delimiter=" ")

            itr.append(0)
            adj_ot.append(0)
            adj_dep.append(0)
            adj_lon.append(0)
            adj_lat.append(0)
            rms.append(ini['rms_res'])
            if iset == 1:
                itr_all.append(0)
                adj_ot_all.append(0)
                adj_dep_all.append(0)
                adj_lon_all.append(0)
                adj_lat_all.append(0)
                rms_all.append(ini['rms_res'])
            fin = final_data
            hyp = fin['ray_stat']['nhyp']
            if max(hyp) > max_hyp:
                max_hyp = max(hyp)
            sta_dly = fin['stn_stat']['delay']
            hyp_arr = np.array(fin['ray_stat']['nhyp'])
            sta_nm_arr = np.array(fin['stn_stat']['stn'])
            sta_ph_arr = np.array(fin['stn_stat']['pha'])
            sta_nobs_arr = np.array(fin['stn_stat']['nobs'])
            sta_res_arr = np.array(fin['stn_stat']['avres'])
            sta_dly_arr = np.array(fin['stn_stat']['delay'])

            sta_lon = []
            sta_lat = []

            for st in sta_nm_arr:
                try:
                    sta_lon.append(bmkg_sta[st]['lon'])
                    sta_lat.append(bmkg_sta[st]['lat'])
                except KeyError:
                    print(f'Station {st} not found in the station list!')

            sta_lon_arr = np.array(sta_lon)
            sta_lat_arr = np.array(sta_lat)
            sta_stats = np.vstack((sta_nm_arr, sta_ph_arr, sta_lon_arr,
                                   sta_lat_arr, sta_dly_arr, sta_res_arr, sta_nobs_arr)).T

            result = np.asarray(sta_stats)

            if flag_iset:
                np.savetxt(outdir + '/' + model_nm + '-StationStatistic-iSet_' + str(iset) +
                           '.dat', result, fmt='%s', delimiter=" ")
            if iset == len(main_out):
                np.savetxt(outvel + '/' + model_nm + '-StationStatistic-final.dat', result, fmt='%s', delimiter=" ")

            if plot:
                fig1, (ax1, ax2) = plt.subplots(1, 2, sharey='row')
                plt.subplots_adjust(wspace=0, hspace=0)
                
            # count number of invert velmod
            # lenveldata = 0
            
            # for itt in velestdata:
                # d = velestdata[itt]
                # if 'vel_mod' in d:
                    # lenveldata += 1

            # minima_rms = 3
            # minima_itr = 1
            j = 0
            for itt in velestdata:
                d = velestdata[itt]
                # if 'vel_mod' not in d:
                    # continue  # TODO: debugging filter iteration with invertratio != 1
                j += 1
                allitt += 1
                adj = d['adjustment']['avr']
                # if 'min_rms' not in itt:
                itr.append(j)
                rms.append(d['rms_res'])
                adj_ot.append(adj['ot'])
                adj_lon.append(adj['x'])
                adj_lat.append(adj['y'])
                adj_dep.append(adj['z'])

                itr_all.append(allitt)
                rms_all.append(d['rms_res'])
                adj_ot_all.append(adj['ot'])
                adj_lon_all.append(adj['x'])
                adj_lat_all.append(adj['y'])
                adj_dep_all.append(adj['z'])
                   
                if 'vel_mod' in d:
                
                    dep = d['vel_mod']['dep']
                    vel.append(d['vel_mod']['vel'])
                    adj_vel.append(d['vel_mod']['dvel'])

                    dep_arr = np.array(dep)
                    dep_array = dep_arr * -1
                    dep_array = dep_array.T

                    vel_arr2 = vel_arr

                    vel_arr, dep_arr = adjust_layer(d['vel_mod']['vel'], d['vel_mod']['dep'],
                                                    step=2, dev=0.05, plot=False, flag_v=False)

                    vel_arr = np.array(vel_arr)
                    vel_mod = vel_arr.T

                    # dvel_arr = np.array(d['vel_mod']['dvel'])
                    # RECALCULATE ADJUSTMENT VELOCITY AFTER READJUST LAYER
                    # if 'min_rms' in itt:
                    #     dvel_arr = np.zeros(len(vel_arr))
                    # else:
                    dvel_arr = vel_arr - vel_arr2
                    vel_adj = dvel_arr.T
                    mx_vel = max(vel_arr)
                    mn_vel = min(vel_arr)
                    mx_dvel = max(dvel_arr)
                    mn_dvel = min(dvel_arr)

                    if mx_vel > max_vel:
                        max_vel = mx_vel
                    if mn_vel < min_vel:
                        min_vel = mn_vel
                    if mx_dvel > max_dvel:
                        max_dvel = mx_dvel
                    if mn_dvel < min_dvel:
                        min_dvel = mn_dvel

                    if mx_dvel > max_veladj:
                        max_veladj = mx_dvel
                    if mn_dvel < min_veladj:
                        min_veladj = mn_dvel

                    if plot:
                        ax1.plot(vel_mod, dep_array, label=itt)
                        # ax1.legend(fancybox=True, framealpha=0.5)
                        ax2.barh(dep_array, vel_adj, alpha=0.5, label=itt)
                        ax2.legend(fancybox=True, framealpha=0.5)

                    model_adj = np.vstack((dep_arr, vel_arr, dvel_arr, hyp_arr)).T
                    result = np.asarray(model_adj)
                    if j == invratio:
                        minrms_result = result
                        last_rms = d['rms_res']
                    elif d['rms_res'] < last_rms:
                        minrms_result = result
                        last_rms = d['rms_res']

                    if flag_iset:
                        np.savetxt(outdir + '/' + model_nm + '-ModelAdjustment-iSet_' + str(iset) + '_' + str(j) + '.dat',
                                   result, fmt='%.6f', delimiter=" ")

                    np.savetxt(outvel + '/' + model_nm + '-ModelAdjustment-iter' + str(allitt) + '.dat',
                               result, fmt='%.6f', delimiter=" ")

                if iset == len(main_out) and j == len(velestdata):
                    np.savetxt(outvel + '/' + model_nm + '-ModelFinal.dat',
                               minrms_result, fmt='%.6f', delimiter=" ")
                # else:

            if plot:
                fig1.suptitle('Velocity Adjustment - Iteration Set ' + str(iset), fontsize=12)
                ax1.grid(alpha=0.5, axis='y')
                ax2.grid(alpha=0.5, axis='y')
                ax1.set_ylabel('Depth (km)')
                ax1.set_xlabel('Velocity (km/s)')
                ax2.spines['left'].set_position('zero')
                xmin, xmax = ax2.get_xlim()
                xmin = -(max(abs(xmax), abs(xmin)))
                xmax = (max(abs(xmax), abs(xmin)))
                ax2.set_xlim([xmin, xmax])
                ax2.set_xlabel('Vel adjustment (km/s)')
                plt.show()

                if flag_iset:
                    plt.savefig(outdir + '/' + model_nm + '-ModelAdjustment-iSet_' + str(iset))
                    plt.close()

                fig2 = plt.figure(constrained_layout=True)
                gs = fig2.add_gridspec(3, 3)
                axs1 = fig2.add_subplot(gs[0, 0])
                axs2 = fig2.add_subplot(gs[0, 1])
                axs3 = fig2.add_subplot(gs[1, 0])
                axs4 = fig2.add_subplot(gs[1, 1])
                axs5 = fig2.add_subplot(gs[0:, 2])
                fig2.suptitle('Hypocenter Adjustment and RMS - Iteration Set ' + str(iset), fontsize=12)
                axs1.plot(itr, adj_ot)
                # min_ot, max_ot = axs1.get_xlim()
                axs1.ticklabel_format(style='sci', axis='y', scilimits=(0, 1), useMathText=True)
                axs1.set_xlabel('Iteration')
                axs1.set_ylabel('Origin Time adjustment (s)')
                axs2.plot(itr, adj_dep)
                # min_dp, max_dp = axs2.get_xlim()
                axs2.ticklabel_format(style='sci', axis='y', scilimits=(0, 1), useMathText=True)
                axs2.set_xlabel('Iteration')
                axs2.set_ylabel('Depth adjustment (km)')
                axs3.plot(itr, adj_lon)
                # min_ln, max_ln = axs3.get_xlim()
                axs3.ticklabel_format(style='sci', axis='y', scilimits=(0, 1), useMathText=True)
                axs3.set_xlabel('Iteration')
                axs3.set_ylabel('Longitude adjustment (km)')
                axs4.plot(itr, adj_lat)
                # min_lt, max_lt = axs4.get_xlim()
                axs4.ticklabel_format(style='sci', axis='y', scilimits=(0, 1), useMathText=True)
                axs4.set_xlabel('Iteration')
                axs4.set_ylabel('Latitude adjustment (km)')
                axs5.plot(itr, rms)
                # min_rms, max_rms = axs5.get_xlim()
                axs5.set_xlabel('Iteration')
                axs5.set_ylabel('RMS residual (s)')
                if flag_iset:
                    plt.savefig(outdir + '/' + model_nm + '-HypoAdjustment-iSet_' + str(iset))
                    plt.close()

            itr_arr = np.array(itr)
            adj_ot_arr = np.array(adj_ot)
            adj_dep_arr = np.array(adj_dep)
            adj_lon_arr = np.array(adj_lon)
            adj_lat_arr = np.array(adj_lat)
            rms_arr = np.array(rms)
            hypo_adj = np.vstack((itr_arr, adj_ot_arr, adj_dep_arr, adj_lon_arr, adj_lat_arr, rms_arr)).T
            result = np.asarray(hypo_adj)

            if flag_iset:
                np.savetxt(outdir + '/' + model_nm + '-HypoAdjustment-iSet_' + str(iset) +
                           '.dat', result, fmt='%.6f', delimiter=" ")

            # set scale limit for gmt
            min_it = min(itr)
            if min_it < 0:
                min_x = min_it
            else:
                min_x = 0
            max_it = max(itr)
            max_x = max_it + 1

            min_ot = (min(adj_ot) * 1000 - np.ceil((max(adj_ot) - min(adj_ot)) * 100)) / 1000
            max_ot = (max(adj_ot) * 1000 + np.ceil((max(adj_ot) - min(adj_ot)) * 100)) / 1000
            min_ot = -(max(abs(min_ot), abs(max_ot)))
            max_ot = (max(abs(min_ot), abs(max_ot)))

            if float(max_ot) == 0.0:
                max_ot = 0.002
                min_ot = -0.002

            min_dp = (min(adj_dep) * 1000 - np.ceil((max(adj_dep) - min(adj_dep)) * 100)) / 1000
            max_dp = (max(adj_dep) * 1000 + np.ceil((max(adj_dep) - min(adj_dep)) * 100)) / 1000
            min_dp = -(max(abs(min_dp), abs(max_dp)))
            max_dp = (max(abs(min_dp), abs(max_dp)))

            if float(max_dp) == 0.0:
                max_dp = 0.002
                min_dp = -0.002

            min_ln = (min(adj_lon) * 1000 - np.ceil((max(adj_lon) - min(adj_lon)) * 100)) / 1000
            max_ln = (max(adj_lon) * 1000 + np.ceil((max(adj_lon) - min(adj_lon)) * 100)) / 1000
            min_ln = -(max(abs(min_ln), abs(max_ln)))
            max_ln = (max(abs(min_ln), abs(max_ln)))

            if float(max_ln) == 0.0:
                max_ln = 0.002
                min_ln = -0.002

            min_lt = (min(adj_lat) * 1000 - np.ceil((max(adj_lat) - min(adj_lat)) * 100)) / 1000
            max_lt = (max(adj_lat) * 1000 + np.ceil((max(adj_lat) - min(adj_lat)) * 100)) / 1000
            min_lt = -(max(abs(min_lt), abs(max_lt)))
            max_lt = (max(abs(min_lt), abs(max_lt)))

            if float(max_lt) == 0.0:
                max_lt = 0.002
                min_lt = -0.002

            min_rms = (min(rms) * 1000 - np.ceil((max(rms) - min(rms)) * 100)) / 1000
            max_rms = (max(rms) * 1000 + np.ceil((max(rms) - min(rms)) * 100)) / 1000
            min_rms = -(max(abs(min_rms), abs(max_rms)))
            max_rms = (max(abs(min_rms), abs(max_rms)))

            min_vel = (min_vel * 1000 - np.ceil((max_vel - min_vel) * 100)) / 1000
            max_vel = (max_vel * 1000 + np.ceil((max_vel - min_vel) * 100)) / 1000

            min_dvel = (min_dvel * 1000 - np.ceil((max_dvel - min_dvel) * 100)) / 1000
            max_dvel = (max_dvel * 1000 + np.ceil((max_dvel - min_dvel) * 100)) / 1000
            min_dvel = -(max(abs(min_dvel), abs(max_dvel)))
            max_dvel = (max(abs(min_dvel), abs(max_dvel)))

            max_nhyp = max(hyp_arr) + np.ceil(max(hyp_arr) / 10)

            max_dly = np.ceil(max(sta_dly_arr) * 10) / 10.0

            if abs(min(sta_dly_arr)) > max_dly:
                max_dly = np.ceil(abs(min(sta_dly_arr)) * 10) / 10.0

            min_dly = -max_dly

            intv_dly = np.ceil(max_dly * 50) / 100.0

            # TODO : Write variable plot batch using f strings
            if flag_iset:
                fvar.write('@echo off\n')
                fvar.write(f'set ps1=Hypo_adj-iSet-{iset}.ps\n')
                fvar.write(f'set ps2=Model_adj-iSet-{iset}.ps\n')
                fvar.write(f'set ps3=EQ_location_adj-iSet-{iset}.ps\n')
                fvar.write(f'set iset=Set-{iset}\n')
                fvar.write(f'set out1=%~dp0../../{plot_result}/{model_nm}-Hypo_adj-iSet-{iset}.png\n')
                fvar.write(f'set out2=%~dp0../../{plot_result}/{model_nm}-Model_adj-iSet-{iset}.png\n')
                fvar.write(f'set out3=%~dp0../../{plot_result}/{model_nm}-EQ_location_adj-iSet-{iset}.png\n')
                fvar.write(f'set init_cat=%~dp0{model_nm}-CatalogInitial-iSet_{iset}.dat\n')
                fvar.write(f'set fnal_cat=%~dp0{model_nm}-CatalogFinal-iSet_{iset}.dat\n')
                fvar.write(f'set hypo_adj=%~dp0{model_nm}-HypoAdjustment-iSet_{iset}.dat\n')
                fvar.write(f'set init_stat=%~dp0{model_nm}-StationStatistic-iSet_{i}.dat\n')
                fvar.write(f'set sts_stat=%~dp0{model_nm}-StationStatistic-iSet_{iset}.dat\n')
                fvar.write(f'set init_mod=%~dp0{model_nm}-ModelInitial-iSet_{iset}.dat\n')

                if i == 0:
                    fvar.write('set init_lgn=Initial Model\n')
                else:
                    fvar.write(f'set init_lgn=Initial Set-{iset}\n')

                if len(velestdata) >= 5:
                    for k in range(len(velestdata)):
                        fvar.write(f'set mod_adj{(k + 1)}=%~dp0{model_nm}-ModelAdjustment-iSet_{iset}_{(k + 1)*invratio}.dat\n')
                else:
                    for k in range(5):
                        fvar.write(f'set mod_adj{(k + 1)}=%~dp0{model_nm}-ModelAdjustment-iSet_{iset}_{(k + 1)*invratio}.dat\n')

                fvar.write('set dly_scl=' + str(min_dly) + '/' + str(max_dly) + '\n')
                fvar.write('set dly_itvl=' + str(intv_dly) + '\n')
                # data hyposenter before after
                fvar.write('set R_ot=' + str(min_x) + '/' + str(max_x) + '/' + str(min_ot) + '/' + str(max_ot) + '\n')
                fvar.write('set R_dp=' + str(min_x) + '/' + str(max_x) + '/' + str(min_dp) + '/' + str(max_dp) + '\n')
                fvar.write('set R_ln=' + str(min_x) + '/' + str(max_x) + '/' + str(min_ln) + '/' + str(max_ln) + '\n')
                fvar.write('set R_lt=' + str(min_x) + '/' + str(max_x) + '/' + str(min_lt) + '/' + str(max_lt) + '\n')
                # fvar.write('set R_ot=' + str(min_x) + '/' + str(max_x) + '/' + str(0) + '/' + str(max_ot) + '\n')
                # fvar.write('set R_dp=' + str(min_x) + '/' + str(max_x) + '/' + str(0) + '/' + str(max_dp) + '\n')
                # fvar.write('set R_ln=' + str(min_x) + '/' + str(max_x) + '/' + str(0) + '/' + str(max_ln) + '\n')
                # fvar.write('set R_lt=' + str(min_x) + '/' + str(max_x) + '/' + str(0) + '/' + str(max_lt) + '\n')
                fvar.write('set R_rms=' + str(min_x) + '/' + str(max_x) + '/' + str(0) + '/' + str(max_rms) + '\n')
                fvar.write('set R_mod=' + str(min_vel) + '/' + str(max_vel) + '/' + str(min(dep) - 2) + '/' + str(
                    max(dep) + 2) + '\n')
                fvar.write('set R_dmod=' + str(min_dvel) + '/' + str(max_dvel) + '/' + str(min(dep) - 2) + '/' + str(
                    max(dep) + 2) + '\n')
                fvar.write('set R_nhyp=' + str(min(dep) - 2) + '/' + str(max(dep) + 2) + '/' + str(-5) + '/' + str(
                    max_nhyp) + '\n')
                fvar.write('set R_nhyp2=' + str(-5) + '/' + str(max_nhyp) + '/' + str(min(dep) - 2) + '/' + str(
                    max(dep) + 2) + '\n')
                fvar.write('call velestplot_itt.bat')
                fvar.close()

                if gmtplot:
                    # Running GMT Plot
                    print("________________________________________________________")
                    time.sleep(1)
                    print(f"\nPlotting model '{model_nm}' iteration set {iset} . . .\n")
                    p = sp.Popen(['Plot-' + model_nm + '.bat'], shell=True, cwd=outdir)
                    p.communicate()
                    time.sleep(2)

            if flag_iset and len(velestdata) > itrmax * invratio:
                print(f'Sub iteration more than 5, periksa script plot')
            if flag_iset and len(velestdata) < itrmax:
                print(f'\nSub iteration less than {itrmax}, iteration stop...')
                result = np.asarray(sta_stats)
                np.savetxt(outvel + '/' + model_nm + '-StationStatistic-final.dat', result, fmt='%s', delimiter=" ")
                result = np.asarray(model_adj)
                np.savetxt(outvel + '/' + model_nm + '-ModelFinal.dat', result, fmt='%.6f', delimiter=" ")
                break
            if not next_set:
                print('\nBackup reach 4 times, iteration stop...')
                break
    else:
        print('Number of Velest main output, initial data and final data must match')

    itr_arr = np.array(itr_all)
    adj_ot_arr = np.array(adj_ot_all)
    adj_dep_arr = np.array(adj_dep_all)
    adj_lon_arr = np.array(adj_lon_all)
    adj_lat_arr = np.array(adj_lat_all)
    rms_arr = np.array(rms_all)

    # USING LIST MAP

    hypo_adj = np.vstack((itr_arr, adj_ot_arr, adj_dep_arr, adj_lon_arr, adj_lat_arr, rms_arr)).T
    result = np.asarray(hypo_adj)
    np.savetxt(outvel + '/' + model_nm + '-HypoAdjustment-all.dat', result, fmt='%.6f', delimiter=" ")

    fvarall = open(outvel + '/Plot_all-' + model_nm + '.bat', 'w')
    fvel_mod = open(outvel + '/' + model_nm + '-vel_mod.bat', 'w')
    fvel_leg = open(outvel + '/' + model_nm + '-vel_leg.bat', 'w')
    fvel_adj = open(outvel + '/' + model_nm + '-vel_adj.bat', 'w')

    fvarall.write('@echo off\n')
    fvarall.write('set model=' + model_nm + '\n')
    fvarall.write('set ps1=Hypo_adj-all.ps\n')
    fvarall.write('set ps2=Model_adj-all.ps\n')
    fvarall.write('set ps3=EQ_location_adj-all.ps\n')
    fvarall.write('set ps4=Model_InitnFinal.ps\n')
    fvarall.write(f'set out1=../{plot_result}/{model_nm}-Hypo_adj-all.png\n')
    fvarall.write(f'set out2=../{plot_result}/{model_nm}-Model_adj-all.png\n')
    fvarall.write(f'set out3=../{plot_result}/{model_nm}-EQ_location_adj-all.png\n')
    fvarall.write(f'set out4=../{plot_result}/{model_nm}-Model_InitnFinal.png\n')
    fvarall.write('set init_cat=' + model_nm + '-CatalogInitial.dat' + '\n')
    fvarall.write('set fnal_cat=' + model_nm + '-CatalogFinal.dat' + '\n')
    fvarall.write('set hypo_adj=' + model_nm + '-HypoAdjustment-all.dat' + '\n')
    # fvarall.write('set init_stat=' + model_nm + '-StationStatistic-iSet_' + str(i) + '.dat'+'\n')
    fvarall.write('set sts_stat=' + model_nm + '-StationStatistic-final.dat' + '\n')
    fvarall.write('set init_lgn=Initial Model' + '\n')
    fvarall.write('set init_mod=' + model_nm + '-ModelInitial.dat' + '\n')

    min_it = min(itr_all)

    if min_it < 0:
        min_x = min_it
    else:
        min_x = 0
    max_it = max(itr_all)
    max_x = max_it + 1

    min_ot = (min(adj_ot_all) * 1000 - np.ceil((max(adj_ot_all) - min(adj_ot_all)) * 100)) / 1000
    max_ot = (max(adj_ot_all) * 1000 + np.ceil((max(adj_ot_all) - min(adj_ot_all)) * 100)) / 1000
    min_ot = -(max(abs(min_ot), abs(max_ot)))
    max_ot = (max(abs(min_ot), abs(max_ot)))
    if float(max_ot) == 0.0:
        max_ot = 0.002
        min_ot = -0.002

    min_dp = (min(adj_dep_all) * 1000 - np.ceil((max(adj_dep_all) - min(adj_dep_all)) * 100)) / 1000
    max_dp = (max(adj_dep_all) * 1000 + np.ceil((max(adj_dep_all) - min(adj_dep_all)) * 100)) / 1000
    min_dp = -(max(abs(min_dp), abs(max_dp)))
    max_dp = (max(abs(min_dp), abs(max_dp)))

    if float(max_dp) == 0.0:
        max_dp = 0.002
        min_dp = -0.002

    min_ln = (min(adj_lon_all) * 1000 - np.ceil((max(adj_lon_all) - min(adj_lon_all)) * 100)) / 1000
    max_ln = (max(adj_lon_all) * 1000 + np.ceil((max(adj_lon_all) - min(adj_lon_all)) * 100)) / 1000
    min_ln = -(max(abs(min_ln), abs(max_ln)))
    max_ln = (max(abs(min_ln), abs(max_ln)))

    if float(max_ln) == 0.0:
        max_ln = 0.002
        min_ln = -0.002

    min_lt = (min(adj_lat_all) * 1000 - np.ceil((max(adj_lat_all) - min(adj_lat_all)) * 100)) / 1000
    max_lt = (max(adj_lat_all) * 1000 + np.ceil((max(adj_lat_all) - min(adj_lat_all)) * 100)) / 1000
    min_lt = -(max(abs(min_lt), abs(max_lt)))
    max_lt = (max(abs(min_lt), abs(max_lt)))

    if float(max_lt) == 0.0:
        max_lt = 0.002
        min_lt = -0.002

    min_rms = (min(rms_all) * 1000 - np.ceil((max(rms_all) - min(rms_all)) * 100)) / 1000
    max_rms = (max(rms_all) * 1000 + np.ceil((max(rms_all) - min(rms_all)) * 100)) / 1000
    min_rms = -(max(abs(min_rms), abs(max_rms)))
    max_rms = (max(abs(min_rms), abs(max_rms)))

    min_vel = (min_vel * 1000 - np.ceil((max_vel - min_vel) * 100)) / 1000
    max_vel = (max_vel * 1000 + np.ceil((max_vel - min_vel) * 100)) / 1000

    min_dvel = (min_veladj * 1000 - np.ceil((max_veladj - min_veladj) * 100)) / 1000
    max_dvel = (max_veladj * 1000 + np.ceil((max_veladj - min_veladj) * 100)) / 1000
    min_dvel = -(max(abs(min_dvel), abs(max_dvel)))
    max_dvel = (max(abs(min_dvel), abs(max_dvel)))

    max_nhyp = max_hyp + np.ceil(max_hyp / 10)

    max_dly = np.ceil(max(sta_dly) * 10) / 10.0

    if abs(min(sta_dly)) > max_dly:
        max_dly = np.ceil(abs(min(sta_dly)) * 10) / 10.0

    min_dly = -max_dly

    intv_dly = np.ceil(max_dly * 50) / 100.0

    # clr = color_cycle
    # fvel_mod.write('gawk "{print $2, $1}" %init_mod% | psxy -J -R -W1,black -O -K -t30 >> %ps2%\n')
    fvel_leg.write('psbasemap -JX4/10 -R0.5/9/-0.5/22 -BWnSe -O -K >> %ps2%\n')
    fvel_adj.write('echo  0 ' + str(max(dep) + 2) + ' > midline.tmp\n')
    fvel_adj.write('echo  0 ' + str(min(dep) - 2) + ' >> midline.tmp\n')
    fvel_adj.write('psxy midline.tmp -J -R -W0.1,black,- -O -K >> %ps2%\n')

    for k in range(int(np.ceil(len(itr_all)/invratio))):

        if k == 0:
            fvel_mod.write('gawk "{print $2, $1}" %init_mod% | psxy -J -R -W1,black,- -O -K -t30 >> %ps2%\n')

            if allitt <= 20:
                fvel_leg.write(f'echo 1 {np.ceil(len(itr_all)/invratio) + 1} > leg.tmp\n')
                fvel_leg.write(f'echo 3 {np.ceil(len(itr_all)/invratio) + 1} >> leg.tmp\n')
                fvel_leg.write(f'echo 3.5 {np.ceil(len(itr_all)/invratio) + 1} %init_lgn% | pstext -J -R -F+f11p,Helvetica,black+jLM -O -K >> %ps2%\n')
                fvel_leg.write('psxy leg.tmp -J -R -W1.4,black,- -t30 -N -O -K >> %ps2%\n')
            else:
                fvel_leg.write('echo 1 22  > leg.tmp\n')
                fvel_leg.write('echo 3 22  >> leg.tmp\n')
                fvel_leg.write('echo 3.5 22 Initial Model | pstext -J -R '
                               '-F+f9p,Helvetica,black+jLM -N -O -K >> %ps2%\n')
                fvel_leg.write('psxy leg.tmp -J -R -W1.4,black,- -t30 -N -O -K >> %ps2%\n')

        else:
            clr, transp = color_cycle(k-1)
            fvarall.write(f'set mod_adj{k}={model_nm}-ModelAdjustment-iter{k*invratio}.dat\n')
            # transp = np.ceil(k / (len(clr) - 1)) * 15 + 15
            fvel_mod.write(f'gawk "{{print $2, $1}}" %mod_adj{k}% | psxy -J -R -W1,{clr} -O -K -t{transp} >> %ps2%\n')
            fvel_adj.write(f'gawk "{{print $3, $1}}" %mod_adj{k}% | '
                           f'psxy -J -R -SB0.22b0 -G{clr} -t{transp} -O -K >> %ps2%\n')
            if allitt <= 20:
                fvel_leg.write(f'echo 3 {np.ceil(len(itr_all)/invratio) + 1 - k} | '
                               f'psxy -J -R -SB0.2b1 -G{clr} -t{transp} -O -K >> %ps2%\n')
                fvel_leg.write(f'echo 3.5 {np.ceil(len(itr_all)/invratio) + 1 - k} Iterasi {k*invratio} | '
                               f'pstext -J -R -F+f11p,Helvetica,black+jLM -O -K >> %ps2%\n')
            else:
                fvel_leg.write(f'echo 3 {22 - k * 21 / np.ceil(len(itr_all)/invratio)} | '
                               f'psxy -J -R -SB0.2b1 -G{clr} -t{transp} -O -K >> %ps2%\n')
                fvel_leg.write(f'echo 3.5 {22 - k * 21 / np.ceil(len(itr_all)/invratio)} Iterasi {k*invratio} | '
                               f'pstext -J -R -F+f9p,Helvetica,black+jLM -O -K >> %ps2%\n')

            # if transp >= 95:
            #     print('Too much data, check plot script')

        if k == (int(np.ceil(len(itr_all)/invratio)) - 1):
            fvarall.write('set mod_adj' + str(k) + '=' + model_nm + '-ModelAdjustment-iter' + str(k*invratio) + '.dat' + '\n')
            fvarall.write('set fnal_mod=' + model_nm + '-ModelFinal.dat' + '\n')
            fvel_mod.write('gawk "{print $2, $1}" %fnal_mod% | psxy -J -R -W1,black -O -K -t35 >> %ps2%\n')
            fvel_leg.write('echo 3 1 | psxy -J -R -SB0.2b1 -Gblack -t30 -O -K >> %ps2%\n')
            fvel_adj.write('gawk "{print $3, $1}" %fnal_mod% | psxy -J -R -SB0.22b0 -Gblack -t35 -O -K >> %ps2%\n')
            if allitt <= 20:
                fvel_leg.write('echo 3.5 1 Final Model | pstext -J -R -F+f11p,Helvetica,black+jLM -O -K >> %ps2%\n')
            else:
                fvel_leg.write('echo 3.5 1 Final Model | pstext -J -R -F+f9p,Helvetica,black+jLM -O -K >> %ps2%\n')

    fvarall.write('set dly_scl=' + str(min_dly) + '/' + str(max_dly) + '\n')
    fvarall.write('set dly_itvl=' + str(intv_dly) + '\n')
    # data hyposenter before after
    fvarall.write('set R_ot=' + str(min_x) + '/' + str(max_x) + '/' + str(min_ot) + '/' + str(max_ot) + '\n')
    fvarall.write('set R_dp=' + str(min_x) + '/' + str(max_x) + '/' + str(min_dp) + '/' + str(max_dp) + '\n')
    fvarall.write('set R_ln=' + str(min_x) + '/' + str(max_x) + '/' + str(min_ln) + '/' + str(max_ln) + '\n')
    fvarall.write('set R_lt=' + str(min_x) + '/' + str(max_x) + '/' + str(min_lt) + '/' + str(max_lt) + '\n')
    # fvarall.write('set R_ot=' + str(min_x) + '/' + str(max_x) + '/' + str(0) + '/' + str(max_ot) + '\n')
    # fvarall.write('set R_dp=' + str(min_x) + '/' + str(max_x) + '/' + str(0) + '/' + str(max_dp) + '\n')
    # fvarall.write('set R_ln=' + str(min_x) + '/' + str(max_x) + '/' + str(0) + '/' + str(max_ln) + '\n')
    # fvarall.write('set R_lt=' + str(min_x) + '/' + str(max_x) + '/' + str(0) + '/' + str(max_lt) + '\n')
    fvarall.write('set R_rms=' + str(min_x) + '/' + str(max_x) + '/' + str(0) + '/' + str(max_rms) + '\n')
    fvarall.write(
        'set R_mod=' + str(min_vel) + '/' + str(max_vel) + '/' + str(min(dep) - 2) + '/' + str(max(dep) + 2) + '\n')
    fvarall.write(
        'set R_dmod=' + str(min_dvel) + '/' + str(max_dvel) + '/' + str(min(dep) - 2) + '/' + str(max(dep) + 2) + '\n')
    fvarall.write(
        'set R_nhyp=' + str(min(dep) - 2) + '/' + str(max(dep) + 2) + '/' + str(-5) + '/' + str(max_nhyp) + '\n')
    fvarall.write(
        'set R_nhyp2=' + str(-5) + '/' + str(max_nhyp) + '/' + str(min(dep) - 2) + '/' + str(max(dep) + 2) + '\n')
    fvarall.write('call velestplot_all.bat')
    fvarall.close()

    # for k in range(len(itr_all)):
    #     fvel_mod.write('gawk "{print $2, $1}" %init_mod% | psxy -J -R -W1,black -O -K -t30 >> %ps2%\n')

    fvel_mod.close()
    fvel_leg.close()
    fvel_adj.close()

    if gmtplot:
        # Running GMT Plot
        print("========================================================")
        time.sleep(1)
        print(f"\nPlotting model all itteration result for model '{model_nm} . . .'\n")
        p = sp.Popen(['Plot_all-' + model_nm + '.bat'], shell=True, cwd=outvel)
        p.communicate()
        time.sleep(3)
        print(f'\nResult generated on "{outvel}/{plot_result}"'
              f'\n========================================================')
        time.sleep(2)

    log = f'Success generate {iset} set, {allitt} iterasi'
    print('\n' + ids + log)
    file = open('log.txt', 'w')
    file.write(ids + log)
    file.close()

    return velestdata, init_data, final_data


def damping_test(model=None, phase=None, stacor=None, which_dmp=None,
                 min_damp=1, max_damp=900, num_damp=10, plot=False):
    """
    :param model: input model file
    :param phase: input phase file
    :param stacor: input phase file
    :type model: full path
    :type phase: full path
    :type stacor: full path
    :param which_dmp: damping parameter to test (ot, xy, z, vel, sta)
    :param min_damp: minimum damp value to test
    :param max_damp: maximum damp value to test
    :param num_damp: number damp value to test
    :param plot: plot damping test graph or not
    """

    if model is None:
        my_dir = os.getcwd()
        mod_dir = os.path.join('input', 'mod')

        if not os.path.exists(os.path.join('input')):
            os.makedirs('input')
        if not os.path.exists(mod_dir):
            os.makedirs(mod_dir)

        inpmod = []
        mod_nm = []
        for m in os.listdir(os.path.join(my_dir, mod_dir)):
            if os.path.isfile(os.path.join(my_dir, mod_dir, m)):
                inpmod.append(os.path.join(mod_dir, m))
                mod_nm.append(m[:-4])

        if len(inpmod) == 0:
            sys.exit(f'\nPlease place your input model in folder "input/mod/" . . .\n')
        else:
            print(f'\nList model:\n\n')
            for i, m in zip(range(len(mod_nm)), mod_nm):
                print(f'{i + 1} {mod_nm[i]}')
            mod_number = int(input(f'\nSelect model number:\n\n'))
            model = inpmod[mod_number - 1]
            # mod_nm = mod_nm[mod_number-1]

    if which_dmp is None:
        which_dmp = int(input('\nDamping parameter to test?\n\n'
                              '1. othet (Origin Time Damping Factor)\n'
                              '2. xythet (Epicenter Damping Factor)\n'
                              '3. zthet (Depth Damping Factor)\n'
                              '4. vthet (Velocity Damping Factor)\n'
                              '5. stathet (Station Correction Damping Factor)\n\n'))

    mod_nm = os.path.basename(model)[:-4]

    logfile = open('log_damptest.txt', 'w')

    if which_dmp == 1:
        damp_type = 'Origin Time'
    elif which_dmp == 2:
        damp_type = 'Epicenter'
    elif which_dmp == 3:
        damp_type = 'Depth'
    elif which_dmp == 4:
        damp_type = 'Velocity Model'
    elif which_dmp == 5:
        damp_type = 'Station Correction'
    else:
        sys.exit('Wrong damping parameter!')

    out_dir = os.path.join('output', 'damping_test')

    if not os.path.exists('output'):
        os.makedirs('output')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cons = max_damp ** (1 / (num_damp - 1))
    lst_dmp = cons ** np.arange(1, num_damp)

    if cons > min_damp:
        lst_dmp = np.insert(lst_dmp, 0, 1)

    lst_dmp = lst_dmp.round(decimals=1)

    # lst_dmp = np.linspace(min_damp, max_damp, int(num_damp), dtype=int)

    damp = [999, 999, 999, 999, 999]

    set_model = model
    if phase is None:
        pha_dir = os.path.join('input', 'pha')
        if not os.path.exists(pha_dir):
            os.makedirs(pha_dir)
        set_phafl = os.path.join(pha_dir, f'phase_{mod_nm}.cnv')
    else:
        set_phafl = phase
    if stacor is None:
        sta_dir = os.path.join('input', 'sta')
        if not os.path.exists(sta_dir):
            os.makedirs(sta_dir)
        set_stafl = os.path.join(sta_dir, f'stacor_{mod_nm}.out')
    else:
        set_stafl = stacor
    set_outmn = os.path.join(out_dir, f'mainprint_damptest{mod_nm}.out')
    set_outph = os.path.join(out_dir, f'finalhypo_damptest{mod_nm}.cnv')
    set_outst = os.path.join(out_dir, f'stacorrect_damptest{mod_nm}.out')

    if not os.path.exists(set_stafl) or not os.path.exists(set_phafl) or not os.path.exists(set_model):
        sys.exit(f'Check required file: model {set_model}, station {set_stafl} and phase {set_phafl}')

    eqs_num = CNV_EvtCount(set_phafl)

    log = f'\nTest {damp_type} Damping Factor With Model {mod_nm}\n\n'
    print(log)
    logfile.write(log)

    model_var = []
    data_var = []
    str_damp = []

    for dmp in lst_dmp:

        damp[which_dmp - 1] = dmp

        log = f'>> Use damping value = "{dmp}"'
        print(log)
        logfile.write(log)

        cmnout = open('velest.cmn', 'w')

        with open(os.path.join('input', 'base.cmn')) as f:

            hint_eqsnm = '*** neqs   nshot   rotate'
            hint_damps = '***   othet   xythet    zthet    vthet   stathet'
            hint_usecr = '*** nsinv   nshcor   nshfix     iuseelev    iusestacorr'
            hint_stcrt = '*** delmin   ittmax   invertratio'
            hint_model = '*** Modelfile:'
            hint_stafl = '*** Stationfile:'
            hint_phafl = '*** File with Earthquake data:'
            hint_outmn = '*** Main print output file:'
            hint_outst = '*** File with new station corrections:'
            hint_outph = '*** File with final hypocenters in *.cnv format:'

            flag_eqsnm = False
            flag_damps = False
            flag_usecr = False
            flag_stcrt = False
            flag_model = False
            flag_stafl = False
            flag_phafl = False
            flag_outmn = False
            flag_outst = False
            flag_outph = False

            for l in f:

                if hint_eqsnm in l:
                    flag_eqsnm = True
                if flag_eqsnm and hint_eqsnm not in l:
                    ln = l.split()
                    ln[0] = eqs_num
                    l = f"   {ln[0]:5d}  {int(ln[1]):5d}  {float(ln[2]):7.1f}\n"
                    flag_eqsnm = False

                if hint_usecr in l:
                    flag_usecr = True
                if flag_usecr and hint_usecr not in l:
                    ln = l.split()
                    ln[4] = 1
                    l = f"       {ln[0]}       {ln[1]}       {ln[2]}           {ln[3]}            {ln[4]}\n"
                    flag_usecr = False

                if hint_damps in l:
                    flag_damps = True
                if flag_damps and hint_damps not in l:
                    l = (f"      {damp[0]}      {damp[1]}       {damp[2]}"
                         f"      {damp[3]}      {damp[4]}\n")
                    flag_damps = False

                if hint_stcrt in l:
                    flag_stcrt = True
                if flag_stcrt and hint_stcrt not in l:
                    l = l.split()
                    l = f"    {float(l[0]):.3f}      1          1\n"
                    flag_stcrt = False

                if hint_model in l:
                    flag_model = True
                if flag_model and hint_model not in l:
                    l = set_model + '\n'
                    flag_model = False

                if hint_stafl in l:
                    flag_stafl = True
                if flag_stafl and hint_stafl not in l:
                    l = set_stafl + '\n'
                    flag_stafl = False

                if hint_phafl in l:
                    flag_phafl = True
                if flag_phafl and hint_phafl not in l:
                    l = set_phafl + '\n'
                    flag_phafl = False

                if hint_outmn in l:
                    flag_outmn = True
                if flag_outmn and hint_outmn not in l:
                    l = set_outmn + '\n'
                    flag_outmn = False

                if hint_outst in l:
                    flag_outst = True
                if flag_outst and hint_outst not in l:
                    l = set_outst + '\n'
                    flag_outst = False

                if hint_outph in l:
                    flag_outph = True
                if flag_outph and hint_outph not in l:
                    l = set_outph + '\n'
                    flag_outph = False

                cmnout.write(l)

        cmnout.close()

        proc = sp.Popen('velest', stdout=sp.PIPE)

        try:

            output = proc.communicate(timeout=200)[0]

            if 'error' in str(output):
                log = ('\n error ' + str(dt.now().strftime("%d-%b-%Y %H:%M:%S")) +
                       '\n\n >> next damping value\n_____________________________________\n')
                print(log)
                logfile.write(log)
                break

            log = ('\n finished ' + str(dt.now().strftime("%d-%b-%Y %H:%M:%S")) +
                   '\n_____________________________________\n')
            print(log)
            logfile.write(log)

            x, y = ReadVelestVar(set_outmn)
            s = str(dmp)

        except sp.TimeoutExpired:

            proc.terminate()

            log = ('\n process timeout ' + str(dt.now().strftime("%d-%b-%Y %H:%M:%S")) +
                   '\n\n >> next damping value\n_____________________________________\n')
            print(log)
            logfile.write(log)
            break

        model_var.append(x)
        data_var.append(y)
        str_damp.append(s)

    model_var = np.array(model_var)
    data_var = np.array(data_var)
    str_damp = np.array(str_damp)

    if plot:
        plt.figure()
        plt.ion()
        plt.show()
        plt.plot(model_var, data_var)
        for x, y, s in zip(model_var, data_var, str_damp):
            plt.text(x, y, s)
        plt.title(f'Damping Test {damp_type} {mod_nm}')
        plt.xlabel('Model variance (km/s)^2')
        plt.ylabel('Data variance (s)^2')
        plt.draw()
        plt.pause(0.001)

    return model_var, data_var, str_damp


def damp2gmt(mod, dat, str_damp, which_damp, mod_nm):
            
    out_dir = os.path.join('output', 'damping_test')
    plot_result = 'plot_result'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(plot_result):
        os.makedirs(plot_result)
    
    if which_damp == 1:
        damp_type = 'Origin Time'
        dmtype = 'OT'
    elif which_damp == 2:
        damp_type = 'Epicenter'
        dmtype = 'Loc'
    elif which_damp == 3:
        damp_type = 'Depth'
        dmtype = 'Dep'
    elif which_damp == 4:
        damp_type = 'Velocity Model'
        dmtype = 'VM'
    elif which_damp == 5:
        damp_type = 'Station Correction'
        dmtype = 'Sta'
    else:
        sys.exit('Wrong damping parameter!')

    damp_test = np.vstack((mod, dat, str_damp)).T
    result = np.asarray(damp_test)
    np.savetxt(os.path.join(out_dir, f'damptest{dmtype}_{mod_nm}.dat'), result, fmt="%s", delimiter=" ")

    dev = np.ceil((max(mod) - min(mod)) * 100000) / 1000000

    min_mod = min(mod) - dev
    max_mod = max(mod) + dev
    
    if min_mod == max_mod:
        min_mod = min_mod - min_mod * 0.1
        max_mod = max_mod + max_mod * 0.1
    
    dev = np.ceil((max(dat) - min(dat)) * 100000) / 1000000

    min_dat = min(dat) - dev
    max_dat = max(dat) + dev
    
    if min_dat == max_dat:
        min_dat = min_dat - min_dat * 0.1
        max_dat = max_dat + max_dat * 0.1

    bat = open(f'{out_dir}/Plot_damptest{dmtype}-{mod_nm}.bat', 'w')
    bat.write(f'@echo off\n')
    bat.write(f'set ps=damp_test.ps\n')
    bat.write(f'set out=%~dp0../../{plot_result}/{mod_nm}-DampTest{dmtype}.jpg\n')
    bat.write(f'set data={f"damptest{dmtype}_{mod_nm}.dat"}\n')
    bat.write(f'set R={min_mod}/{max_mod}/{min_dat}/{max_dat}\n')
    bat.write(f'set J=10/15\n')
    bat.write(f'psbasemap -JX%J% -R%R% -BWSne+t"Damping Test {damp_type} {mod_nm}" -Bxa+l"Model variance '
              f'(km/s)@+2@+" -Bya+l"Data variance (s)@+2@+" -K --FONT_TITLE=14 --FONT_LABEL=13 > %ps%\n')
    bat.write(f'gawk "{{print $1, $2}}" %data% | psxy -J -R -W1,blue -O -K >> %ps%\n')
    bat.write(f'gawk "{{print $1, $2, $3}}" %data% | pstext -J -R -F+f12p,Helvetica,black+jLM -O -K >> %ps%\n')
    for varmd, vardt, varsr in zip(mod, dat, str_damp):
        if vardt == min(dat):
            bat.write(f'echo {varmd} {vardt} {varsr} | pstext -J -R -F+f12p,Helvetica,black+jLM '
                      f'-Gwhite -W0.5,black -O >> %ps%\n')
    bat.write(f'psconvert %ps% -Tj -E512 -A0.2 -P -F%out%\n')
    bat.write(f'del gmt.* *.tmp *.cpt *.ps\n')
    bat.close()


def adjust_layer(vel, dep, step=2, dev=0.05, plot=True, flag_v=False):

    # vel = [4.357, 4.44, 6.943, 6.944, 6.945, 6.946, 6.947, 6.948, 6.949, 6.95, 6.951, 6.952, 7.05, 7.051, 7.052,
    # 7.053, 7.054, 7.055, 7.056, 7.057, 7.058, 7.059, 7.06, 7.061, 7.062, 7.063, 7.064, 7.065, 7.066, 7.067, 7.068,
    # 7.069, 7.07, 7.071, 7.072, 7.073, 7.074, 7.075, 7.076, 7.077, 7.078, 7.079, 7.08, 7.081, 7.082, 7.083, 7.084,
    # 7.085, 7.238, 7.297, 7.351, 7.352, 7.376, 7.404, 7.405, 7.406, 7.426, 7.427, 7.428, 7.429, 7.43, 7.431, 7.432,
    # 7.433, 7.434, 7.435, 7.436, 7.437, 7.438, 7.439, 7.44, 7.441, 8.287, 8.31, 8.314, 8.315, 8.327, 8.393, 8.394,
    # 8.395, 8.396, 8.397, 8.398, 8.429, 8.43, 8.431, 8.432, 8.433, 8.434, 8.435, 8.436, 8.437, 8.438, 8.442, 8.443,
    # 8.448, 8.449, 8.45, 8.487, 8.488]
    # dep = [-1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
    # 9.0, 10.0, 10.0, 11.0, 11.0, 12.0, 12.0, 13.0, 13.0, 14.0, 14.0, 15.0, 15.0, 16.0, 16.0, 17.0, 17.0, 18.0,
    # 18.0, 19.0, 19.0, 20.0, 20.0, 21.0, 21.0, 22.0, 22.0, 23.0, 23.0, 24.0, 24.0, 25.0, 25.0, 26.0, 26.0, 27.0,
    # 27.0, 28.0, 28.0, 29.0, 29.0, 30.0, 30.0, 31.0, 31.0, 32.0, 32.0, 33.0, 33.0, 34.0, 34.0, 35.0, 35.0, 36.0,
    # 36.0, 37.0, 37.0, 38.0, 38.0, 39.0, 39.0, 40.0, 40.0, 41.0, 41.0, 42.0, 42.0, 43.0, 43.0, 44.0, 44.0, 45.0,
    # 45.0, 46.0, 46.0, 47.0, 47.0, 48.0, 48.0, 49.0]
    # step = 2
    # dev = 0.05
    # plot = True
    # flag_v = False

    start = 0
    new_vel = vel.copy()

    if flag_v:
        print(f'\nInitial vel and depth:\n\n{vel}\n\n{dep}')

    for i in range(0, len(new_vel), step):

        for j in range(len(new_vel), start, -step):

            stop = j

            lay = new_vel[start:stop]

            dvdt = statistics.stdev(lay)
            mndt = statistics.mean(lay)

            # if statistics.stdev(lay) < dev:
            if dvdt < dev and mndt - dev <= vel[j - 1] <= mndt + dev:

                for k in range(start, stop):
                    new_vel[k] = round(statistics.mean(lay), 3)

                if flag_v:
                    print(f'\n adjust depth {dep[start]} - {dep[stop - 1]}:\n  {lay}\n  to\n  {new_vel[start:stop]}')

                start = stop
                break

            elif stop - start <= step:

                for k in range(start, stop):
                    new_vel[k] = round(statistics.mean(lay), 3)

                if flag_v:
                    print(f'\n adjust depth {dep[start]} - {dep[stop - 1]}:\n  {lay}\n  to\n  {new_vel[start:stop]}')

                start = stop
                break

        if start > len(new_vel) - step:
            break

    if plot:
        plt.figure()
        plt.ion()
        plt.show()
        xdep = [d * -1 for d in dep]
        plt.plot(vel, xdep)
        plt.plot(new_vel, xdep)
        plt.draw()
        plt.pause(0.001)

    return new_vel, dep


def simple_model(vel, dep, damp=None):

    top_vel = ''
    top_dep = ''
    top_dam = ''
    simple_vel = []
    simple_dep = []
    simple_dam = []

    if damp is None:
        # only give top layer (nlloc style)
        for i, v, d in zip(range(len(vel)), vel, dep):

            if i == 0:

                simple_vel.append(v)
                simple_dep.append(d)

            elif v != top_vel:

                simple_vel.append(v)
                simple_dep.append(d)

            top_vel = v

        return simple_vel, simple_dep

    else:
        # give top and bottom layer (velest style)
        for i, v, d, dm in zip(range(len(vel)), vel, dep, damp):

            if i == 0:

                simple_vel.append(v)
                simple_dep.append(d)
                simple_dam.append(dm)

            elif v != top_vel:

                simple_vel.append(top_vel)
                simple_dep.append(top_dep)
                simple_dam.append(top_dam)

                simple_vel.append(v)
                simple_dep.append(d)
                simple_dam.append(dm)

            top_vel = v
            top_dep = d
            top_dam = dm

        simple_vel.append(top_vel)
        simple_dep.append(top_dep)
        simple_dam.append(top_dam)

        return simple_vel, simple_dep, simple_dam


def Run_Velest(itrmax=5, invratio=1, time_out=120, use_stacor=False, itr_set=False, damp_test=False, auto_damp=False):
    """
    output gmt plot file in batch. Only works on gmt for windows
    :param itrmax: maksimum iteration
    :param invratio: invert ratio (read velest manual)
    :param time_out: time length to terminate velest run (instable solution)
    :param use_stacor: using station correction input file (on folder input/sta/stacor_"modname".out)
                       or using standart station input (on folder input/station.dat)
    :param itr_set: running in several set mode (output initial set with layer adjustment
                    became input for next set until minumum RMS) itrmax=max iteration each set
    :param damp_test: run damping test velocity model before running velest
    :param auto_damp: automatic select damping value based on minimum data variance or manually choose after test
    """

    uuid = get_machine_uuid()
    if uuid:
        if not is_license_valid(uuid):
            sys.exit("❌ The program cannot run on unauthorized machines.")
    else:
        sys.exit("⚠️ Could not retrieve machine information. Please contact the developer.")

    my_dir = os.getcwd()
    mod_dir = os.path.join('input', 'mod')
    pha_dir = os.path.join('input', 'pha')
    sta_dir = os.path.join('input', 'sta')

    if not os.path.exists(os.path.join('input')):
        os.makedirs('input')
    if not os.path.exists(mod_dir):
        os.makedirs(mod_dir)

    if not os.path.exists(os.path.join(mod_dir, 'unfinish')):
        os.makedirs(os.path.join(mod_dir, 'unfinish'))
    if not os.path.exists(os.path.join(mod_dir, 'finish')):
        os.makedirs(os.path.join(mod_dir, 'finish'))
    if not os.path.exists(os.path.join(mod_dir, 'error')):
        os.makedirs(os.path.join(mod_dir, 'error'))

    inpmod = []
    mod_nm = []
    for m in os.listdir(os.path.join(my_dir, mod_dir)):
        if os.path.isfile(os.path.join(my_dir, mod_dir, m)):
            inpmod.append(os.path.join(mod_dir, m))
            mod_nm.append(m[:-4])

    if itr_set:
        logfile = open('VelestSet_log.txt', 'w')
    else:
        logfile = open('Velest_log.txt', 'w')
        if not os.path.exists(pha_dir):
            os.makedirs(pha_dir)
        if not os.path.exists(sta_dir):
            os.makedirs(sta_dir)

    if len(inpmod) > 0:
        if itr_set:
            log = f'\n__Velest Set Runner by eQ Halauwet__\n' \
                  f'\nVelest will run several iteration set for {len(inpmod)} input model:\n'
        else:
            log = f'\n__Velest Runner by eQ Halauwet__\n' \
                  f'\nVelest will run maximum {itrmax} iteration for {len(inpmod)} input model:\n'
    else:
        log = f'\n__Velest Runner by eQ Halauwet__\n' \
              f'\nPlease place your input model in folder "input/mod/" . . .\n'
        print(log)
        sys.exit(0)

    print(log)
    logfile.write(log)

    for m in mod_nm:
        log = ' * ' + m
        print(log)
        logfile.write(log)

    if not os.path.exists('output'):
        os.makedirs('output')

    for mod, num, md_nm in zip(inpmod, range(len(inpmod)), mod_nm):
        if not os.path.exists(os.path.join('output', md_nm)):
            os.makedirs(os.path.join('output', md_nm))
        out_dir = os.path.join('output', md_nm)

        set_model = []
        set_stafl = []
        set_phafl = []
        set_outmn = []
        set_outph = []
        set_outst = []
        set_outmd = []

        log = f'\n\n\n>>> {num + 1}. Input model {md_nm}:\n'
        print(log)
        logfile.write(log)

        iset = 0
        j = 0
        while j < 1:

            iset += 1
            i = iset - 1

            if itr_set:
                log = f'\n * Iteration set {iset} . . .'
                print(log)
                logfile.write(log)
                
            if iset == 1:
            
                set_model = [mod]

                if use_stacor:
                
                    set_phafl = [os.path.join(pha_dir, f'phase_{md_nm}.cnv')]
                    set_stafl = [os.path.join(sta_dir, f'stacor_{md_nm}.out')]
                    
                else:
                
                    set_phafl = [os.path.join('input', 'phase.cnv')]
                    set_stafl = [os.path.join('input', 'station.dat')]

                if itr_set:
                
                    set_outmn = [os.path.join(out_dir, f'mainprint{str(iset).zfill(2)}_{md_nm}.out')]
                    set_outph = [os.path.join(out_dir, f'finalhypo{str(iset).zfill(2)}_{md_nm}.cnv')]
                    set_outst = [os.path.join(out_dir, f'stacorrect{str(iset).zfill(2)}_{md_nm}.out')]
                    set_outmd = [os.path.join(out_dir, f'outmodel{str(iset).zfill(2)}_{md_nm}.mod')]
                    
                else:
                
                    set_outmn = [os.path.join(out_dir, f'mainprint_min1D{md_nm}.out')]
                    set_outph = [os.path.join(out_dir, f'finalhypo_min1D{md_nm}.cnv')]
                    set_outst = [os.path.join(out_dir, f'stacorrect_min1D{md_nm}.out')]
                    set_outmd = [os.path.join(out_dir, f'outmodel_min1D{md_nm}.mod')]

            else:

                set_model.append(set_outmd[i - 1])
                set_stafl.append(set_outst[i - 1])
                set_phafl.append(set_outph[i - 1])
                set_outmn.append(os.path.join(out_dir, f'mainprint{str(iset).zfill(2)}_{md_nm}.out'))
                set_outph.append(os.path.join(out_dir, f'finalhypo{str(iset).zfill(2)}_{md_nm}.cnv'))
                set_outst.append(os.path.join(out_dir, f'stacorrect{str(iset).zfill(2)}_{md_nm}.out'))
                set_outmd.append(os.path.join(out_dir, f'outmodel{str(iset).zfill(2)}_{md_nm}.mod'))

            # check input file
            if not os.path.exists(set_stafl[i]) or not os.path.exists(set_phafl[i]):
            
                print(f'Check required file: station {set_stafl[i]} and phase {set_phafl[i]}')
                break

            eqs_num = CNV_EvtCount(set_phafl[i])
            
            if itr_set:
            
                damp_ot = 0.01
                damp_xy = 0.01
                damp_zt = 0.01
                damp_vt = 0.1
                damp_st = 0.01
                
            else:
                
                damp_ot = 0.01
                damp_xy = 0.01
                damp_zt = 0.01
                damp_vt = 1.0
                damp_st = 0.10
                        
                if damp_test:
                
                    model_var, data_var, str_damp = damping_test(model=set_model[i], phase=set_phafl[i],
                                                                 stacor=set_stafl[i], which_dmp=4, max_damp=500,
                                                                 num_damp=20, plot=True)
                    damp2gmt(model_var, data_var, str_damp, 4, md_nm)
                    
                    if auto_damp:
                    
                        for varmd, vardt, varsr in zip(model_var, data_var, str_damp):
                            if vardt == min(data_var):
                                damp_vt = float(varsr)
                                
                    else:
                        
                        damp_vt = float(input(f'Damping factor velocity model='))

            cmnout = open('velest.cmn', 'w')

            with open(os.path.join('input', 'base.cmn')) as f:

                hint_eqsnm = '*** neqs   nshot   rotate'
                hint_damps = '***   othet   xythet    zthet    vthet   stathet'
                hint_usecr = '*** nsinv   nshcor   nshfix     iuseelev    iusestacorr'
                hint_stcrt = '*** delmin   ittmax   invertratio'
                hint_model = '*** Modelfile:'
                hint_stafl = '*** Stationfile:'
                hint_phafl = '*** File with Earthquake data:'
                hint_outmn = '*** Main print output file:'
                hint_outst = '*** File with new station corrections:'
                hint_outph = '*** File with final hypocenters in *.cnv format:'

                flag_eqsnm = False
                flag_usecr = False
                flag_damps = False
                flag_stcrt = False
                flag_model = False
                flag_stafl = False
                flag_phafl = False
                flag_outmn = False
                flag_outst = False
                flag_outph = False

                for l in f:

                    if hint_eqsnm in l:
                        flag_eqsnm = True
                    if flag_eqsnm and hint_eqsnm not in l:
                        ln = l.split()
                        ln[0] = eqs_num
                        l = f"   {ln[0]:5d}  {int(ln[1]):5d}  {float(ln[2]):7.1f}\n"
                        flag_eqsnm = False

                    if hint_usecr in l:
                        flag_usecr = True
                    if flag_usecr and hint_usecr not in l:
                        ln = l.split()
                        if iset == 1:
                            if itr_set:
                                ln[4] = 0
                            else:
                                ln[4] = 1
                        else:
                            ln[4] = 1
                        l = f"       {ln[0]}       {ln[1]}       {ln[2]}           {ln[3]}            {ln[4]}\n"
                        flag_usecr = False

                    if hint_damps in l:
                        flag_damps = True
                    if flag_damps and hint_damps not in l:
                        l = f"      {damp_ot:5}   {damp_xy:5}     {damp_zt:5}    {damp_vt:5}    {damp_st:5}\n"
                        flag_damps = False

                    if hint_stcrt in l:
                        flag_stcrt = True
                    if flag_stcrt and hint_stcrt not in l:
                        l = l.split()
                        l[1] = itrmax
                        l[2] = invratio
                        l = f"    {float(l[0]):.3f}      {l[1]}          {l[2]}\n"
                        flag_stcrt = False

                    if hint_model in l:
                        flag_model = True
                    if flag_model and hint_model not in l:
                        l = set_model[i] + '\n'
                        flag_model = False

                    if hint_stafl in l:
                        flag_stafl = True
                    if flag_stafl and hint_stafl not in l:
                        l = set_stafl[i] + '\n'
                        flag_stafl = False

                    if hint_phafl in l:
                        flag_phafl = True
                    if flag_phafl and hint_phafl not in l:
                        l = set_phafl[i] + '\n'
                        flag_phafl = False

                    if hint_outmn in l:
                        flag_outmn = True
                    if flag_outmn and hint_outmn not in l:
                        l = set_outmn[i] + '\n'
                        flag_outmn = False

                    if hint_outst in l:
                        flag_outst = True
                    if flag_outst and hint_outst not in l:
                        l = set_outst[i] + '\n'
                        flag_outst = False

                    if hint_outph in l:
                        flag_outph = True
                    if flag_outph and hint_outph not in l:
                        l = set_outph[i] + '\n'
                        flag_outph = False

                    cmnout.write(l)

            cmnout.close()

            # Looping layer adjustment by velocity deviation
            if itr_set:
                lst_dev = np.linspace(0.05, 0.10, 3)
            else:
                lst_dev = [0.05]

            for k in range(len(lst_dev)):

                dev = lst_dev[k]
                if iset > 1:
                    log = ' adjust layer with dev <= %.3f' % dev
                    print(log)
                    logfile.write(log)

                proc = sp.Popen('velest', stdout=sp.PIPE)

                try:

                    output = proc.communicate(timeout=time_out)[0]

                    if 'error' in str(output):
                        
                        if k == len(lst_dev) - 1 and iset != 1:
                            # TODO: vel2gmt available result
                            nllmod_dir = 'outmodel_nlloc'
                            velmod_dir = 'outmodel_velest'

                            if not os.path.exists(nllmod_dir):
                                os.makedirs(nllmod_dir)
                            if not os.path.exists(velmod_dir):
                                os.makedirs(velmod_dir)
                                
                            del set_phafl[-1]
                            del set_outph[-1]
                            del set_outmn[-1]

                            velest2gmt(md_nm, set_phafl, set_outph, set_outmn, itrmax, invratio)

                            # READ OPTIMUM VELMOD (ITERATION WITH MIN RMS) FROM ITERATION SET BEFORE ERROR
                            itt, vel, dep, dam = ReadVelestOptmVel(set_outmn[i-1])
                            vel, dep = adjust_layer(vel, dep, step=2, dev=dev, plot=False, flag_v=False)
                            
                            # SET SIMPLE VELOCITY MODEL THEN WRITE TO NLLOC FORMAT
                            simple_v, simple_d = simple_model(vel, dep)
                            vpvs = 1.75

                            if itr_set:
                                mod_nlloc = open(os.path.join(nllmod_dir, f'velmod_nlloc_{md_nm}.dat'), 'w')
                            else:
                                mod_nlloc = open(os.path.join(nllmod_dir, f'velmod_nlloc_min1D{md_nm}.dat'), 'w')
                            mod_nlloc.write(
                                '# model layers (LAYER depth, Vp_top, Vp_grad, Vs_top, Vs_grad, p_top, p_grad)')
                            for v, d in zip(simple_v, simple_d):
                                l = f'\nLAYER {d:5.1f} {v:5.2f} 0.00   {v/vpvs:5.2f}  0.00  2.7 0.0'
                                mod_nlloc.write(l)

                            mod_nlloc.close()

                            # SET SIMPLE VELOCITY MODEL THEN WRITE TO VELEST FORMAT
                            simple_v, simple_d, simple_dam = simple_model(vel, dep, dam)

                            if itr_set:
                                modout = open(os.path.join(velmod_dir, f'outmodelfinal_{md_nm}.mod'), 'w')
                            else:
                                modout = open(os.path.join(velmod_dir, f'outmodelfinal_min1D{md_nm}.mod'), 'w')
                            modout.write(' Output model:\n')
                            modout.write(str(len(simple_v)) + '\n')

                            for vl, dp, dm in zip(simple_v, simple_d, simple_dam):
                                l = f'{vl:5.2f}     {dp:7.2f}  {dm:7.3f}\n'
                                modout.write(l)

                            modout.close()
                        
                        if iset == 1 or k == len(lst_dev) - 1:
                        
                            shutil.move(mod, os.path.join(my_dir, mod_dir, 'error', md_nm + '.mod'))
                            log = ('\n error ' + str(dt.now().strftime("%d-%b-%Y %H:%M:%S")) +
                                   '\n\n >> next model\n_____________________________________\n')
                            print(log)
                            logfile.write(log)
                            
                            j += 1
                            
                            break

                        else:
                            log = ('\n error ' + str(dt.now().strftime("%d-%b-%Y %H:%M:%S")) +
                                   '\n\n > readjust layer . . .\n')
                            print(log)
                            logfile.write(log)

                            itt, vel, dep, dam, next_set = ReadVelestVel(set_outmn[i - 1])

                            vel, dep = adjust_layer(vel, dep, step=2, dev=dev, plot=False, flag_v=False)

                            modout = open(set_outmd[i - 1], 'w')
                            modout.write(' Output model:\n')
                            modout.write(str(len(vel)) + '\n')

                            for vl, dp, dm in zip(vel, dep, dam):
                                l = f'{vl:5.2f}     {dp:7.2f}  {dm:7.3f}\n'
                                modout.write(l)

                            modout.close()

                            continue

                    log = ('\n finished ' + str(dt.now().strftime("%d-%b-%Y %H:%M:%S")) +
                           '\n_____________________________________\n')
                    print(log)
                    logfile.write(log)
                    
                    try:
                    
                        itt, vel, dep, dam, next_set = ReadVelestVel(set_outmn[i])
                        
                    except UnboundLocalError:
                        
                        if iset != 1:
                            
                            itt, vel, dep, dam, next_set = ReadVelestVel(set_outmn[i-1])
                        
                        else:
                            
                            shutil.move(mod, os.path.join(my_dir, mod_dir, 'error', md_nm + '.mod'))
                            log = ('\n error ' + str(dt.now().strftime("%d-%b-%Y %H:%M:%S")) +
                                   '\n\n >> next model\n_____________________________________\n')
                            print(log)
                            logfile.write(log)
                            
                            j += 1
                            
                            break

                    vel, dep = adjust_layer(vel, dep, step=2, dev=dev, plot=False, flag_v=False)

                    modout = open(set_outmd[i], 'w')
                    modout.write(' Output model:\n')
                    modout.write(str(len(vel)) + '\n')

                    for vl, dp, dm in zip(vel, dep, dam):
                        l = f'{vl:5.2f}     {dp:7.2f}  {dm:7.3f}\n'
                        modout.write(l)

                    modout.close()

                    if itr_set and itt < itrmax or itr_set is False or not next_set:

                        nllmod_dir = 'outmodel_nlloc'
                        velmod_dir = 'outmodel_velest'

                        if not os.path.exists(nllmod_dir):
                            os.makedirs(nllmod_dir)
                        if not os.path.exists(velmod_dir):
                            os.makedirs(velmod_dir)

                        velest2gmt(md_nm, set_phafl, set_outph, set_outmn, itrmax, invratio)

                        # READ OPTIMUM VELMOD (ITERATION WITH MIN RMS) FROM LAST ITERATION SET
                        itt, vel, dep, dam = ReadVelestOptmVel(set_outmn[i])
                        vel, dep = adjust_layer(vel, dep, step=2, dev=dev, plot=False, flag_v=False)
                        
                        # SET SIMPLE VELOCITY MODEL THEN WRITE TO NLLOC FORMAT
                        simple_v, simple_d = simple_model(vel, dep)
                        vpvs = 1.75

                        if itr_set:
                            mod_nlloc = open(os.path.join(nllmod_dir, f'velmod_nlloc_{md_nm}.dat'), 'w')
                        else:
                            mod_nlloc = open(os.path.join(nllmod_dir, f'velmod_nlloc_min1D{md_nm}.dat'), 'w')
                        mod_nlloc.write(
                            '# model layers (LAYER depth, Vp_top, Vp_grad, Vs_top, Vs_grad, p_top, p_grad)')
                        for v, d in zip(simple_v, simple_d):
                            l = f'\nLAYER {d:5.1f} {v:5.2f} 0.00   {v / vpvs:5.2f}  0.00  2.7 0.0'
                            mod_nlloc.write(l)

                        mod_nlloc.close()

                        # SET SIMPLE VELOCITY MODEL THEN WRITE TO VELEST FORMAT
                        simple_v, simple_d, simple_dam = simple_model(vel, dep, dam)

                        if itr_set:
                            modout = open(os.path.join(velmod_dir, f'outmodelfinal_{md_nm}.mod'), 'w')
                        else:
                            modout = open(os.path.join(velmod_dir, f'outmodelfinal_min1D{md_nm}.mod'), 'w')
                        modout.write(' Output model:\n')
                        modout.write(str(len(simple_v)) + '\n')

                        for vl, dp, dm in zip(simple_v, simple_d, simple_dam):
                            l = f'{vl:5.2f}     {dp:7.2f}  {dm:7.3f}\n'
                            modout.write(l)

                        modout.close()

                        j += 1

                    break

                except sp.TimeoutExpired:

                    proc.terminate()
                    
                    if k == len(lst_dev) - 1 and iset != 1:
                            # TODO: vel2gmt available result
                            nllmod_dir = 'outmodel_nlloc'
                            velmod_dir = 'outmodel_velest'

                            if not os.path.exists(nllmod_dir):
                                os.makedirs(nllmod_dir)
                            if not os.path.exists(velmod_dir):
                                os.makedirs(velmod_dir)
                                
                            del set_phafl[-1]
                            del set_outph[-1]
                            del set_outmn[-1]

                            velest2gmt(md_nm, set_phafl, set_outph, set_outmn, itrmax, invratio)

                            # READ OPTIMUM VELMOD (ITERATION WITH MIN RMS) FROM ITERATION SET BEFORE ERROR
                            itt, vel, dep, dam = ReadVelestOptmVel(set_outmn[i-1])
                            vel, dep = adjust_layer(vel, dep, step=2, dev=dev, plot=False, flag_v=False)
                            
                            # SET SIMPLE VELOCITY MODEL THEN WRITE TO NLLOC FORMAT
                            simple_v, simple_d = simple_model(vel, dep)
                            vpvs = 1.75

                            if itr_set:
                                mod_nlloc = open(os.path.join(nllmod_dir, f'velmod_nlloc_{md_nm}.dat'), 'w')
                            else:
                                mod_nlloc = open(os.path.join(nllmod_dir, f'velmod_nlloc_min1D{md_nm}.dat'), 'w')
                            mod_nlloc.write(
                                '# model layers (LAYER depth, Vp_top, Vp_grad, Vs_top, Vs_grad, p_top, p_grad)')
                            for v, d in zip(simple_v, simple_d):
                                l = f'\nLAYER {d:5.1f} {v:5.2f} 0.00   {v/vpvs:5.2f}  0.00  2.7 0.0'
                                mod_nlloc.write(l)

                            mod_nlloc.close()

                            # SET SIMPLE VELOCITY MODEL THEN WRITE TO VELEST FORMAT
                            simple_v, simple_d, simple_dam = simple_model(vel, dep, dam)

                            if itr_set:
                                modout = open(os.path.join(velmod_dir, f'outmodelfinal_{md_nm}.mod'), 'w')
                            else:
                                modout = open(os.path.join(velmod_dir, f'outmodelfinal_min1D{md_nm}.mod'), 'w')
                            modout.write(' Output model:\n')
                            modout.write(str(len(simple_v)) + '\n')

                            for vl, dp, dm in zip(simple_v, simple_d, simple_dam):
                                l = f'{vl:5.2f}     {dp:7.2f}  {dm:7.3f}\n'
                                modout.write(l)

                            modout.close()

                    if iset == 1 or k == len(lst_dev) - 1:
                        
                        log = ('\n process timeout ' + str(dt.now().strftime("%d-%b-%Y %H:%M:%S")) +
                               '\n\n >> next model\n_____________________________________\n')
                        print(log)
                        logfile.write(log)
                        shutil.move(mod, os.path.join(my_dir, mod_dir, 'unfinish', md_nm + '.mod'))
                        
                        j += 1
                        
                        break

                    else:
                        log = ('\n process timeout ' + str(dt.now().strftime("%d-%b-%Y %H:%M:%S")) +
                               '\n\n > readjust layer . . .\n')
                        print(log)
                        logfile.write(log)

                        itt, vel, dep, dam, next_set = ReadVelestVel(set_outmn[i - 1])

                        vel, dep = adjust_layer(vel, dep, step=2, dev=dev, plot=False, flag_v=False)

                        modout = open(set_outmd[i - 1], 'w')
                        modout.write(' Output model:\n')
                        modout.write(str(len(vel)) + '\n')

                        for vl, dp, dm in zip(vel, dep, dam):
                            l = f'{vl:5.2f}     {dp:7.2f}  {dm:7.3f}\n'
                            modout.write(l)

                        modout.close()
                        continue

    log = '\n Finished for all model\n_____________________________________\n'
    print(log)
    logfile.write(log)
    logfile.close()

Run_Velest(itrmax=10, itr_set=True)