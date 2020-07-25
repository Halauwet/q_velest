from Q_Velest import *

"""
tes RunVelest()
"""
def Run_Velest(itrmax=5, invratio=1, time_out=120, itr_set=False):

    my_dir = os.getcwd()
    mod_dir = os.path.join('input', 'mod')

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
                log = f'\n * Iteration set {iset} . . .\n'
                print(log)
                logfile.write(log)
            if iset == 1:
                set_model = [mod]  # .replace(os.sep, '/')]
                set_stafl = [os.path.join('input', 'station.dat')]
                if itr_set:
                    set_phafl = [os.path.join('input', 'phase.cnv')]
                    set_outmn = [os.path.join(out_dir, f'mainprint{str(iset).zfill(2)}_{md_nm}.out')]
                    set_outph = [os.path.join(out_dir, f'finalhypo{str(iset).zfill(2)}_{md_nm}.cnv')]
                    set_outst = [os.path.join(out_dir, f'stacorrect{str(iset).zfill(2)}_{md_nm}.out')]
                    set_outmd = [os.path.join(out_dir, f'outmodel{str(iset).zfill(2)}_{md_nm}.mod')]
                else:
                    set_phafl = [os.path.join('input', f'phase_{md_nm}.cnv')]
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

            cmnout = open('velest.cmn', 'w')

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

            with open(os.path.join('input', 'base.cmn')) as f:

                hint_usecr = '*** nsinv   nshcor   nshfix     iuseelev    iusestacorr'
                hint_damps = '***   othet   xythet    zthet    vthet   stathet'
                hint_stcrt = '*** delmin   ittmax   invertratio'
                hint_model = '*** Modelfile:'
                hint_stafl = '*** Stationfile:'
                hint_phafl = '*** File with Earthquake data:'
                hint_outmn = '*** Main print output file:'
                hint_outst = '*** File with new station corrections:'
                hint_outph = '*** File with final hypocenters in *.cnv format:'

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
                        l = (f"      {damp_ot}     {damp_xy}     {damp_zt}"
                             f"      {damp_vt}     {damp_st}\n")
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

            # print(set_model[i] + '\n\n' + set_stafl[i] + '\n\n' + set_phafl[i] + '\n\n' + set_outmn[i] +
            # '\n\n' + set_outph[i] + '\n\n' + set_outst[i] + '\n\n' + set_outmd[i])

            # Looping layer adjustment by velocity deviation
            if itr_set:
                lst_dev = np.linspace(0.05, 0.15, 3)
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

                        if iset == 1 or k == len(lst_dev) - 1:
                            shutil.move(mod, os.path.join(my_dir, mod_dir, 'error', md_nm + '.mod'))
                            log = (' error ' + str(dt.now().strftime("%d-%b-%Y %H:%M:%S")) +
                                   '\n\n >> next model\n_____________________________________\n')
                            # if num+1 == len(inpmod):
                            #     log += '\n Finished for all model \n_____________________________________\n'
                            print(log)
                            logfile.write(log)
                            j += 1
                            break

                        else:
                            log = (' error ' + str(dt.now().strftime("%d-%b-%Y %H:%M:%S")) +
                                   '\n\n > readjust layer . . .\n')
                            print(log)
                            logfile.write(log)

                            itt, vel, dep, dam = ReadVelestVel(set_outmn[i-1])

                            vel, dep = adjust_layer(vel, dep, step=2, dev=dev, plot=False, flag_v=False)

                            modout = open(set_outmd[i-1], 'w')
                            modout.write(' Output model:\n')
                            modout.write(str(len(vel)) + '\n')

                            for vl, dp, dm in zip(vel, dep, dam):
                                l = f'{vl:5.2f}     {dp:7.2f}  {dm:7.3f}\n'
                                modout.write(l)

                            modout.close()

                            continue

                    log = (' finished ' + str(dt.now().strftime("%d-%b-%Y %H:%M:%S")) +
                           '\n_____________________________________\n')
                    print(log)
                    logfile.write(log)

                    itt, vel, dep, dam = ReadVelestVel(set_outmn[i])

                    vel, dep = adjust_layer(vel, dep, step=2, dev=dev, plot=False, flag_v=False)

                    modout = open(set_outmd[i], 'w')
                    modout.write(' Output model:\n')
                    modout.write(str(len(vel)) + '\n')

                    for vl, dp, dm in zip(vel, dep, dam):
                        l = f'{vl:5.2f}     {dp:7.2f}  {dm:7.3f}\n'
                        modout.write(l)

                    modout.close()

                    if itr_set and itt < itrmax or itr_set is False:

                        nllmod_dir = 'outmodel_nlloc'
                        velmod_dir = 'outmodel_velest'

                        if not os.path.exists(nllmod_dir):
                            os.makedirs(nllmod_dir)
                        if not os.path.exists(velmod_dir):
                            os.makedirs(velmod_dir)

                        velest2gmt(md_nm, set_phafl, set_outph, set_outmn)

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

                    if iset == 1 or k == len(lst_dev) - 1:
                        log = (' process timeout ' + str(dt.now().strftime("%d-%b-%Y %H:%M:%S")) +
                               '\n\n >> next model\n_____________________________________\n')
                        print(log)
                        logfile.write(log)
                        shutil.move(mod, os.path.join(my_dir, mod_dir, 'unfinish', md_nm + '.mod'))
                        j += 1
                        break

                    else:
                        log = (' process timeout ' + str(dt.now().strftime("%d-%b-%Y %H:%M:%S")) +
                               '\n\n > readjust layer . . .\n')
                        print(log)
                        logfile.write(log)

                        itt, vel, dep, dam = ReadVelestVel(set_outmn[i-1])

                        vel, dep = adjust_layer(vel, dep, step=2, dev=dev, plot=False, flag_v=False)

                        modout = open(set_outmd[i-1], 'w')
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


tesRun_Velest(itrmax=10, itr_set=True)
