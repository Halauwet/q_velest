from Q_Velest import ReadVelestMain
import os
import statistics as st

my_dir = os.getcwd()
mod_dir = os.path.join('input', 'mod')

inpmod = []
mod_nm = []

for m in os.listdir(my_dir):

    if m[:9] == 'mainprint' and os.path.isfile(os.path.join(my_dir, m)):
        inpmod.append(m)
        mod_nm.append(m[12:-4])

zp = zip(inpmod, mod_nm)
zp = list(zp)
sorted_mod = sorted(zp, key=lambda x: x[1])

for mod, md_nm in sorted_mod:

    itt_data, init_data, final_data, optm_data, next_set, ids = ReadVelestMain(mod)

    datvar = ''
    msqrd = ''
    rms = ''
    avres = []
    mavres = ''
    vel = []
    min_vel = ''
    max_vel = ''

    if len(itt_data) > 0:

        for it in itt_data:

            d = itt_data[it]

            datvar = d['datvar']
            msqrd = d['sqr_res']
            rms = d['rms_res']
            avres = d['evt_highres']['avres']
            vel = d['vel_mod']['vel']

    elif len(init_data) > 0:

            d = init_data

            datvar = d['datvar']
            msqrd = d['sqr_res']
            rms = d['rms_res']
            avres = d['evt_highres']['avres']
            vel = d['vel_mod']['vel']

    if len(avres) > 0:

        mavres = st.mean(avres)
        mavres = f'{mavres:.6f}'

    if len(vel) > 0:

        min_vel = vel[2]
        max_vel = max(vel)

    print(f'{md_nm} {min_vel} {max_vel} {mavres} {datvar} {msqrd} {rms}')
