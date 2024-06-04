import os
from glob import glob
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize_scalar


def find_idx(arr, a):
    return (np.abs(np.array(arr) - a)).argmin()


def save_xy(fname, x, y, header=''):
    np.savetxt(fname, np.vstack((x, y)).T, header=header, comments='#')
    return


def load_xy(fname, skiprows=0):
    data = np.loadtxt(fname, skiprows=skiprows).T
    x = data[0]
    y = data[1]
    return x, y


def q2tt(q, energy):
    """ q: array or float, [A-1];  energy: [keV] """
    wl = 12.39842 / energy
    tt = 2 * 180 / np.pi * np.arcsin(q * wl / 4 / np.pi)
    return tt


def tt2q(tt, energy):
    """ q: array or float, [A-1];  energy: [keV] """
    wl = 12.39842 / energy
    q = 4 * np.pi / wl * np.sin(tt / 2 * np.pi / 180)
    return q


def convertpattern_q2tt(fname, energy, skiprows=0, dtype_out="xy"):
    x, y = load_xy(fname, skiprows=skiprows)
    x = q2tt(x, energy=energy)
    dtype = fname.split(".")[-1]
    fname_out = fname[:-(len(dtype) + 1)] + "_in_tt." + dtype_out
    save_xy(fname_out, x, y, header="converted from q to two theta, energy = {} keV\ntwo theta   intensity".format(energy))
    return

def convertpattern_tt2q(fname, energy, skiprows=0, dtype_out="xy"):
    x, y = load_xy(fname, skiprows=skiprows)
    x = tt2q(x, energy=energy)
    dtype = fname.split(".")[-1]
    fname_out = fname[:-(len(dtype) + 1)] + "_in_q." + dtype_out
    save_xy(fname_out, x, y, header="converted from two theta to q, energy = {} keV\ntwo theta   intensity".format(energy))
    return


def avg(path2data, dtype=".dat", fname_timetemp="", temp0=30, avg_num=60, overwrite=False):

    fnames = glob(os.path.join(path2data, '*' + dtype))
    fnames.sort()

    path2avg = os.path.join(os.path.dirname(os.path.dirname(path2data)), "avg{}_iqs".format(avg_num), os.path.basename(path2data))
    if not os.path.isdir(path2avg):
        os.makedirs(path2avg)

    timetempdir = "time_vs_temp"
    if not fname_timetemp:
        fname_timetemp = os.path.join(os.path.dirname(path2data), timetempdir, "timetemps_" + os.path.basename(path2data) + ".txt")
    times, temps = load_xy(fname_timetemp)
    idx0 = find_idx(temps, temp0)
    idxend = -((len(fnames)-idx0) % avg_num)
    print("average {} from file idx {} to {}".format(os.path.basename(path2avg), idx0, idxend))

    temps_crop = temps[idx0:idxend:avg_num]
    fnames_crop = fnames[idx0:idxend]  # cut before idx0 and every pattern which doesnt have enough for time resolution
    for temp, fnames_to_avg in zip(temps_crop, np.reshape(fnames_crop, (int(len(fnames_crop) / avg_num), avg_num))):
        fname_avg = os.path.join(path2avg,
                                 os.path.basename(fnames_to_avg[0][:-len(dtype)]) + "_avg_" + str(avg_num) + dtype)
        if os.path.isfile(fname_avg) and not overwrite:
            print("skippidiskippy")
            continue
        else:
            y_avgs = []
            for fname in fnames_to_avg:
                x, y = load_xy(fname)
                y_avgs.append(y)
            y_avg = np.mean(y_avgs, axis=0)
            header = 'averaged {} files\n'.format(avg_num)
            header += 'from path: {}\nto {}\n'.format(path2data, path2avg)
            header += 'temperature (°C): {}\n'.format(temp)
            header += 'q (A^-1)    I (a.u.)'
            save_xy(fname_avg, x, y_avg, header=header)
            # print(fname_avg)
    return


def avg_all_in_dir(parentdir, overwrite=False):
    subdirs = [os.path.join(parentdir, subdir) for subdir in os.listdir(parentdir)\
               if os.path.isdir(os.path.join(parentdir, subdir))]
    timetempdir = "time_vs_temp"
    skipsubdirs = [os.path.join(parentdir, timetempdir)]
    for skipsubdir in skipsubdirs:
        subdirs.remove(skipsubdir)
    for dir in subdirs:
        avg(dir, overwrite=overwrite)
    return


def bkg_sub(path2data, path2bkg, path2sub="", x_range=[1.05, 1.35], dtype=".dat", fancyapproach=False, bkgscale=1.0):

    if not path2sub:
        path2sub = os.path.join(os.path.dirname(os.path.dirname(path2data)), "bkgsub_iqs", os.path.basename(path2data))
    if not os.path.isdir(path2sub):
        os.makedirs(path2sub)

    fnames = glob(os.path.join(path2data, '*' + dtype))
    fnames.sort()

    fnames_bkg = glob(os.path.join(path2bkg, '*' + dtype))
    fnames_bkg.sort()
    for i in range(len(fnames) - len(fnames_bkg)):
        fnames_bkg.append(fnames_bkg[-1])

    x0, y0 = load_xy(fnames[0])
    idx_range = [find_idx(x0, xi) for xi in x_range]
    scale_approach_value = 0.00001

    from scipy.optimize import minimize_scalar
    def initial_scale(y1, y2):
        def func(s):
            return np.mean(abs(y1 - s * y2))
        res = minimize_scalar(func)
        return res['x']

    for fname, fname_bkg in zip(fnames, fnames_bkg):
        x, y = load_xy(fname)
        x_bkg, y_bkg = load_xy(fname_bkg)
        with open(fname, 'r') as f:
            temp = float(f.readlines()[3].split()[-1])

        if fancyapproach:
            # here fancy subtracton by finding scale so in the givin area background goes as close to zero as possible without beeing negative
            s = initial_scale(y[idx_range[0]:idx_range[1]], y_bkg[idx_range[0]:idx_range[1]])
            # s = round(s, len(str(scale_approach_value).split('.')[1]))
            # s = round(s, abs(int(np.log10(scale_approach_value))))
            y_sub = y - s * y_bkg

            if np.any(y_sub[idx_range[0]:idx_range[1]] <= 0.0):
                while np.any(y_sub[idx_range[0]:idx_range[1]] <= 0.0):
                    s -= scale_approach_value
                    y_sub = y - s * y_bkg
            else:
                while np.all(y_sub[idx_range[0]:idx_range[1]] > 0.0):
                    s += scale_approach_value
                    y_sub = y - s * y_bkg
                s -= scale_approach_value

            header = 'background subtraction by minimizing background subtracted file but not going below 0 between\n'
            header += 'q > {} and q < {} A^-1\n'.format(*x_range)
            header += 'scale: {} ±{}\n'.format(s, scale_approach_value)
            header += 'temperature (°C): {}\n'.format(temp)
            header += 'unsubtracted file: {}\n'.format(fname)
            header += 'background file: {}\n'.format(fname_bkg)
            header += 'q (A^-1)    I (a.u.)'

        else:
            # for fixed bkg scale
            s = bkgscale
            y_sub = y - s * y_bkg

            header = 'normal background subtraction\n'
            header += '\n'.format(*x_range)
            header += 'scale: {}\n'.format(s)
            header += 'temperature (°C): {}\n'.format(temp)
            header += 'unsubtracted file: {}\n'.format(fname)
            header += 'background file: {}\n'.format(fname_bkg)
            header += 'q (A^-1)    I (a.u.)'

        fname_sub = os.path.join(path2sub, os.path.basename(fname[:-len(dtype)]) + "_bkgsub" + dtype)

        save_xy(fname_sub, x, y_sub, header=header)

    return


def make_pdf(fname_iq, fname_gr, composition, qmin, qmax, qmax_inst, rpoly, rmax=30.0):

    command = [
        'pdfgetx3',
        str(fname_iq),
        '--force=True',
        '--format=QA',
        '--mode=xray',
        '--composition=' + str(composition),
        '--qmin=' + str(qmin),
        '--qmax=' + str(qmax),
        '--qmaxinst=' + str(qmax_inst),
        '--rpoly=' + str(rpoly),
        '--rmax=' + str(rmax),
        '-t=gr',
        '-o=' + str(fname_gr),
    ]

    subprocess.run(command)

    return


def make_pdf_all_in_dir(path2data, path2pdf="", dtype=".dat",
                        composition="Cu3PdN", qmin=0.7, qmax=10.3, qmax_inst=24.0, rpoly=0.9, rmax=30.0):

    print(path2data)

    if not path2pdf:
        path2pdf = os.path.join(os.path.dirname(os.path.dirname(path2data)), "pdfs", os.path.basename(path2data))
    if not os.path.isdir(path2pdf):
        os.makedirs(path2pdf)

    fnames = glob(os.path.join(path2data, '*' + dtype))
    fnames.sort()

    for fname_iq in fnames:

        fname_gr = os.path.join(path2pdf, os.path.basename(fname_iq)[:-len(dtype)] + ".gr")

        make_pdf(fname_iq, fname_gr, composition=composition, qmin=qmin, qmax=qmax, qmax_inst=qmax_inst, rpoly=rpoly, rmax=rmax)
        # print(os.path.basename(fname_gr))

    print(path2pdf)

    return


def bkg_sub_n_makepdf_fitglasspeak(fname_data, fname_bkg, fname_bkgsub, fname_gr):

    # fname_bkg = "/Users/admin/data/23_bt_2304_accell/avg60_iqs/bkg_bnh2_140_sdd600/bkg_bnh2_140_sdd600-01848_avg_60.dat"
    # fname_data = "/Users/admin/data/23_bt_2304_accell/avg60_iqs/cupdn_140_3_sdd600/cupdn_140_3_sdd600-03734_avg_60.dat"
    # fname_bkgsub = "/Users/admin/data/23_bt_2304_accell/bkgsub_iqs/cupdn_140_3_sdd600_glassfit/cupdn_140_3_sdd600-06914_avg_60.dat"
    # fname_gr = "/Users/admin/data/23_bt_2304_accell/pdfs/cupdn_140_3_sdd600_glassfit/cupdn_140_3_sdd600-06914_avg_60.gr"

    if not os.path.isdir(os.path.dirname(fname_bkgsub)):
        os.makedirs(os.path.dirname(fname_bkgsub))
    if not os.path.isdir(os.path.dirname(fname_gr)):
        os.makedirs(os.path.dirname(fname_gr))

    q, i_bkg = load_xy(fname_bkg)
    q, i_data = load_xy(fname_data)

    r_min = 1.2
    r_max = 1.8

    s_arr = []
    a_arr = []
    x0_arr = []
    fwhm_arr = []
    m_arr = []
    y0_arr = []

    def scale():

        def gaussian(x, x0, a, fwhm, m, y0):
            return a * np.exp(- (x - x0) ** 2 / 2 / (fwhm / 2.35482) ** 2) + m * x + y0

        def func(s):

            save_xy(fname_bkgsub, q, i_data - i_bkg * s)
            # make_pdf(fname_bkgsub, fname_gr, "Cu3PdN", qmin=0.7, qmax=10.3, qmax_inst=24.0, rpoly=0.9)
            make_pdf(fname_bkgsub, fname_gr, "Cu3PdN", qmin=0.4, qmax=10.3, qmax_inst=17.5, rpoly=0.9)
            r, g = load_xy(fname_gr, skiprows=23)
            idx_min = find_idx(r, r_min)
            idx_max = find_idx(r, r_max)

            initial_guess = [1.6, 0.1, 0.5, 0.1, -3.0]
            nobound = (-np.inf, np.inf)
            restraints = np.array([(1.5, 1.7), (0.0, np.inf), (0.3, 0.8), (-1, 1), nobound]).T
            popt, pcov = curve_fit(gaussian, r[idx_min:idx_max + 1], g[idx_min:idx_max + 1], p0=initial_guess, bounds=restraints)
            a = popt[1]
            a_err = np.sqrt(pcov[1, 1])

            s_arr.append(s)
            x0_arr.append(popt[0])
            a_arr.append(popt[1])
            fwhm_arr.append(popt[2])
            m_arr.append(popt[3])
            y0_arr.append(popt[4])
            return a

        res = minimize_scalar(func)
        return res['x']

    s = scale()
    # print(s)

    with open(fname_data, 'r') as f:
        temp = float(f.readlines()[3].split()[-1])
    header = 'glassfit background subtraction\n'
    header += 'scale: {}\n'.format(s)
    header += 'temperature (°C): {}\n'.format(temp)
    header += 'unsubtracted file: {}\n'.format(fname_data)
    header += 'background file: {}\n'.format(fname_bkg)
    header += 'q (A^-1)    I (a.u.)'
    save_xy(fname_bkgsub, q, i_data - i_bkg * s)
    # make_pdf(fname_bkgsub, fname_gr, "Cu3PdN", qmin=0.7, qmax=10.3, qmax_inst=24.0, rpoly=0.9)
    make_pdf(fname_bkgsub, fname_gr, "Cu3PdN", qmin=0.4, qmax=10.3, qmax_inst=17.5, rpoly=0.9)

    # r, g = load_xy(fname_gr, skiprows=23)
    #
    # fit_dict = {
    #     "s": {"values": s_arr},
    #     "a": {"values": a_arr},
    #     "x0": {"values": x0_arr},
    #     "fwhm": {"values": fwhm_arr},
    #     "m": {"values": m_arr},
    #     "y0": {"values": y0_arr},
    # }
    #
    # def key_ax(ax, key):
    #     ax.plot(fit_dict[key]["values"], label=key)
    #     ax.set_xlabel("iteration")
    #     ax.set_ylabel(key)
    #     return ax
    #
    # keys = ["s", "a", "x0", "fwhm", "m", "y0"]
    # figsize = (6, 2 * len(keys))
    # fig, axs = plt.subplots(len(keys), 1, sharex=True, sharey=False, gridspec_kw={'hspace': 0}, figsize=figsize)
    # lines = []
    # labels = []
    # for i, keys in enumerate(keys):
    #     axs[i] = key_ax(axs[i], keys)
    #     axs[i].legend(frameon=False)
    #     # lines_i, labels_i = axs[i].get_legend_handles_labels()
    #     # [lines.append(line) for line in lines_i if line not in lines]
    #     # [labels.append(label) for label in labels_i if label not in labels]
    # # axs[0].legend(lines, labels, loc='best', frameon=False)
    # fig.align_ylabels()
    # plt.show()
    #
    # plt.plot(r, g)
    # plt.xlim(0, 7.5)
    # plt.title(s)
    # plt.show()
    return s

def bkg_sub_n_makepdf_all_in_dir_glassfit(path2data, path2bkg, path2sub="", path2pdf="", dtype=".dat"):

    if not path2sub:
        path2sub = os.path.join(os.path.dirname(os.path.dirname(path2data)), "bkgsub_iqs", os.path.basename(path2data) + "_glassfit")
    if not os.path.isdir(path2sub):
        os.makedirs(path2sub)
    if not path2pdf:
        path2pdf = os.path.join(os.path.dirname(os.path.dirname(path2data)), "pdfs", os.path.basename(path2data) + "_glassfit")
    if not os.path.isdir(path2pdf):
        os.makedirs(path2pdf)

    fnames = glob(os.path.join(path2data, '*' + dtype))
    fnames.sort()

    fnames_bkg = glob(os.path.join(path2bkg, '*' + dtype))
    fnames_bkg.sort()
    for i in range(len(fnames) - len(fnames_bkg)):
        fnames_bkg.append(fnames_bkg[-1])

    s_arr = []
    for fname, fname_bkg in zip(fnames, fnames_bkg):

        print(fname)

        fname_bkgsub = os.path.join(path2sub, os.path.basename(fname[:-len(dtype)]) + "_bkgsub" + dtype)
        fname_gr = os.path.join(path2pdf, os.path.basename(fname[:-len(dtype)]) + ".gr")

        s = bkg_sub_n_makepdf_fitglasspeak(fname, fname_bkg, fname_bkgsub, fname_gr)
        s_arr.append(s)

    plt.plot(s_arr)
    plt.ylabel("scale")
    plt.xlabel("time (min)")
    plt.show()


def bkg_sub_n_makepdf_bymaxint(fname_data, fname_bkg, fname_bkgsub, fname_gr):

    if not os.path.isdir(os.path.dirname(fname_bkgsub)):
        os.makedirs(os.path.dirname(fname_bkgsub))
    if not os.path.isdir(os.path.dirname(fname_gr)):
        os.makedirs(os.path.dirname(fname_gr))

    q, i_bkg = load_xy(fname_bkg)
    q, i_data = load_xy(fname_data)

    r_min = 1.9
    r_max = 4.0

    def scale():

        def func(s):

            save_xy(fname_bkgsub, q, i_data - i_bkg * s)
            # make_pdf(fname_bkgsub, fname_gr, "Cu3PdN", qmin=0.7, qmax=10.3, qmax_inst=24.0, rpoly=0.9)
            make_pdf(fname_bkgsub, fname_gr, "Cu3PdN", qmin=0.4, qmax=10.3, qmax_inst=17.5, rpoly=0.9)
            r, g = load_xy(fname_gr, skiprows=23)
            idx_min = find_idx(r, r_min)
            idx_max = find_idx(r, r_max)

            return -np.max(g[idx_min:idx_max])

        res = minimize_scalar(func)
        return res['x']

    s = scale()
    print(s)

    with open(fname_data, 'r') as f:
        temp = float(f.readlines()[3].split()[-1])
    header = 'max pdf intensity (between {} adn {} A^-1) background subtraction\n'.format(r_min, r_max)
    header += 'scale: {}\n'.format(s)
    header += 'temperature (°C): {}\n'.format(temp)
    header += 'unsubtracted file: {}\n'.format(fname_data)
    header += 'background file: {}\n'.format(fname_bkg)
    header += 'q (A^-1)    I (a.u.)'
    save_xy(fname_bkgsub, q, i_data - i_bkg * s)
    make_pdf(fname_bkgsub, fname_gr, "Cu3PdN", qmin=0.4, qmax=10.3, qmax_inst=17.5, rpoly=0.9)

    return s

def bkg_sub_n_makepdf_all_in_dir_baymaxint(path2data, path2bkg, path2sub="", path2pdf="", dtype=".dat"):

    if not path2sub:
        path2sub = os.path.join(os.path.dirname(os.path.dirname(path2data)), "bkgsub_iqs", os.path.basename(path2data) + "_bymaxint")
    if not os.path.isdir(path2sub):
        os.makedirs(path2sub)
    if not path2pdf:
        path2pdf = os.path.join(os.path.dirname(os.path.dirname(path2data)), "pdfs", os.path.basename(path2data) + "_bymaxint")
    if not os.path.isdir(path2pdf):
        os.makedirs(path2pdf)

    fnames = glob(os.path.join(path2data, '*' + dtype))
    fnames.sort()

    fnames_bkg = glob(os.path.join(path2bkg, '*' + dtype))
    fnames_bkg.sort()
    for i in range(len(fnames) - len(fnames_bkg)):
        fnames_bkg.append(fnames_bkg[-1])

    s_arr = []
    for fname, fname_bkg in zip(fnames, fnames_bkg):

        print(fname)

        fname_bkgsub = os.path.join(path2sub, os.path.basename(fname[:-len(dtype)]) + "_bkgsub" + dtype)
        fname_gr = os.path.join(path2pdf, os.path.basename(fname[:-len(dtype)]) + ".gr")

        s = bkg_sub_n_makepdf_bymaxint(fname, fname_bkg, fname_bkgsub, fname_gr)
        s_arr.append(s)

    plt.plot(s_arr)
    plt.ylabel("scale")
    plt.xlabel("time (min)")
    plt.show()

path2data = "/Users/admin/Wolke/data/23_bt_2304_accell/avg60_iqs/cupdn_140_3_sdd600"
path2bkg = "/Users/admin/Wolke/data/23_bt_2304_accell/avg60_iqs/bkg_bnh2_140_sdd600"
path2sub = "/Users/admin/Wolke/data/23_bt_2304_accell/bkgsub_iqs/cupdn_140_3_sdd600_bymaxint"
path2pdf = "/Users/admin/Wolke/data/23_bt_2304_accell/pdfs/cupdn_140_3_sdd600_bymaxint"
bkg_sub_n_makepdf_all_in_dir_baymaxint(path2data, path2bkg)


def avg_specific(path2data, dtype=".dat", avg_startnum=0, avg_endnum=120):
    fnames = glob(os.path.join(path2data, '*' + dtype))
    fnames.sort()
    path2avg = "/Users/admin/data/23_bt_2304_accell/abcheckauis/starting_point_wobkgsub/iqs"

    avg_num = avg_endnum - avg_startnum
    fnames_to_avg = fnames[avg_startnum:avg_endnum]
    fname_avg = os.path.join(path2avg,
                             os.path.basename(fnames_to_avg[0][:-len(dtype)]) + "_avg_" + str(avg_num) + dtype)

    y_avgs = []
    for fname in fnames_to_avg:
        x, y = load_xy(fname)
        y_avgs.append(y)
    y_avg = np.mean(y_avgs, axis=0)
    header = 'averaged {} files\n'.format(avg_num)
    header += 'from path: {}\nto {}\n'.format(path2data, path2avg)
    header += 'q (A^-1)    I (a.u.)'
    save_xy(fname_avg, x, y_avg, header=header)
# datadirdir = "/Users/admin/data/23_bt_2304_accell/raw_iqs/"
# dir = "sh_pdn_180_1"
# path2data = os.path.join(datadirdir, dir)
# avg_specific(path2data)


# path2data = "/Users/admin/data/23_bt_2304_accell/raw_iqs/bkg_bnh2_160_1"
# avg(path2data, overwrite=True)
# path2bkg = "/Users/admin/data/23_bt_2304_accell/raw_iqs/cufen_160_1"
# avg(path2bkg, overwrite=True)
# bkg_sub_n_makepdf_all_in_dir_glassfit("/Users/admin/data/23_bt_2304_accell/avg60_iqs/cufen_160_1", "/Users/admin/data/23_bt_2304_accell/avg60_iqs/bkg_bnh2_160_1")
print('kukuk')

# path2bkg = "/Users/admin/data/23_bt_2304_accell/raw_iqs/lk_bg_fe3s4_180_1"
# avg(path2bkg, overwrite=True)
# path2data = "/Users/admin/data/23_bt_2304_accell/raw_iqs/lk_fe3s4_180_1"
# avg(path2data, overwrite=True)
# bkg_sub_n_makepdf_all_in_dir_glassfit("/Users/admin/data/23_bt_2304_accell/avg60_iqs/lk_fe3s4_180_1", "/Users/admin/data/23_bt_2304_accell/avg60_iqs/lk_bg_fe3s4_180_1")
# path2bkg = "/Users/admin/data/23_bt_2304_accell/raw_iqs/lk_bg_fe3s4_100_1"
# avg(path2bkg, overwrite=True)
# path2data = "/Users/admin/data/23_bt_2304_accell/raw_iqs/lk_fe3s4_100_1"
# avg(path2data, overwrite=True)
# bkg_sub_n_makepdf_all_in_dir_glassfit("/Users/admin/data/23_bt_2304_accell/avg60_iqs/lk_fe3s4_100_1", "/Users/admin/data/23_bt_2304_accell/avg60_iqs/lk_bg_fe3s4_100_1")



# path2bkg = "/Users/admin/data/23_bt_2304_accell/avg60_iqs/bkg_bnh2_140_sdd600"
# path2data = "/Users/admin/data/23_bt_2304_accell/avg60_iqs/cupdn_140_3_sdd600"
# bkg_sub(path2data, path2bkg, fancyapproach=False, bkgscale=1.0)

# path2data = "/Users/admin/data/23_bt_2304_accell/bkgsub_iqs/cupdn_140_3_sdd600"
# make_pdf_all_in_dir(path2data, composition="Cu3PdN", qmin=0.4, qmax=10.3, qmax_inst=17.5, rpoly=0.9)


# path2data = "/Users/admin/data/23_bt_2304_accell/bkgsub_iqs/cu3n_140_1"
# make_pdf_all_in_dir(path2data, composition="Cu3N", qmin=0.7, qmax=10.3, qmax_inst=24.0, rpoly=0.9)
# path2data = "/Users/admin/data/23_bt_2304_accell/bkgsub_iqs/cu3n_140_2"
# make_pdf_all_in_dir(path2data, composition="Cu3N", qmin=0.7, qmax=10.3, qmax_inst=24.0, rpoly=0.9)
# path2data = "/Users/admin/data/23_bt_2304_accell/bkgsub_iqs/cupdn_140_1"
# make_pdf_all_in_dir(path2data, composition="Cu3PdN", qmin=0.7, qmax=10.3, qmax_inst=24.0, rpoly=0.9)
# path2data = "/Users/admin/data/23_bt_2304_accell/bkgsub_iqs/cupdn_140_2"
# make_pdf_all_in_dir(path2data, composition="Cu3PdN", qmin=0.7, qmax=10.3, qmax_inst=24.0, rpoly=0.9)
# path2data = "/Users/admin/data/23_bt_2304_accell/bkgsub_iqs/sh_pdn_140_2"
# make_pdf_all_in_dir(path2data, composition="Pd", qmin=0.7, qmax=10.3, qmax_inst=24.0, rpoly=0.9)

# make_pdf("/Users/admin/data/23_bt_2304_accell/calibauis/lab6_ac_cell_385-00067.dat",
#          "/Users/admin/data/23_bt_2304_accell/calibauis/lab6_ac_cell_385-00067.dat",
#          composition="LaB6", qmin=0.7, qmax=10.3, qmax_inst=24.0, rpoly=0.9)

# path2data = "/Users/admin/data/23_bt_2304_accell/bkgsub_iqs/cupdn_140_3_sdd600"
# path2pdf = "/Users/admin/data/23_bt_2304_accell/pdfs/cupdn_140_3_sdd600_qmax16p3"
# make_pdf_all_in_dir(path2data, composition="Cu3PdN", qmin=0.7, qmax=16.3, qmax_inst=18.3, rpoly=0.9)
#
# make_pdf("/Users/admin/data/23_bt_2304_accell/calibauis/lab6_ac_cell_385-00067.dat",
#          "/Users/admin/data/23_bt_2304_accell/calibauis/lab6_ac_cell_385-00067_qmax16p3.gr",
#          composition="LaB6", qmin=0.7, qmax=16.3, qmax_inst=18.3, rpoly=0.9)

print('hi')
