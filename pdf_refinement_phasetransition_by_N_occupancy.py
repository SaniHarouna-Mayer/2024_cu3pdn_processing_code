import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import shutil
from glob import glob
from diffpy.srfit.fitbase import FitContribution, FitRecipe
from diffpy.srfit.fitbase import FitResults
from diffpy.srfit.fitbase import Profile
from diffpy.srfit.pdf import PDFParser, PDFGenerator
from diffpy.structure import Structure, loadStructure
from scipy.optimize import least_squares
from time import time

# data_dir = '/Users/admin/Wolke/data/23_bt_2304_accell/pdfs/cupdn_140_3_sdd600_glassfit'
data_dir = '/Users/admin/Wolke/data/23_bt_2304_accell/pdfs/cupdn_140_3_sdd600_qmax16p3'
data_files = glob(os.path.join(data_dir, '*.gr'))
data_files.sort()
fit_id = "240220_cpn_growth_1phase_N_occ_highadp0p35"

stru_file_1 = '/Users/admin/Wolke/data/structure_files/cpn/cu3pdn.cif'
stru1 = loadStructure(stru_file_1)
spacegroup_1 = 'Pm-3m'

run_parallel = True

pdf_rmin = 1.8
pdf_rmax = 30
pdf_rstep = 0.01

# # qmax = 10.3 A^-1
# qmin = 0.4
# qmax = 10.3
# qdamp = 0.0281
# qbroad = 0.0222

# qmax = 16.3 A^-1
qmin = 0.4  # eig 0.4, difference?
qmax = 16.3
qdamp = 0.0457
qbroad = 0.0270

initial_values_0 = {
    's1_i': 0.025,
    'a1_i': 3.777,
    'uiso_i': 0.05,
    'delta21_i': 3.5,
    'psize1_i': 30,
    'N_occ_i': 1.0
}

ref_values = initial_values_0


def plotresults(recipe, figname=""):
    r = recipe.pdf.profile.x
    g = recipe.pdf.profile.y
    gcalc = recipe.pdf.profile.ycalc
    diffzero = -0.65 * max(g) * np.ones_like(g)
    diff = g - gcalc + diffzero
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use('/Users/admin/Wolke/code/mpl-stylesheets/sani_style')
    fig, ax = plt.subplots(1, 1)
    ax.plot(r, g, ls="None", marker="o", ms=5, mew=0.2, mfc="None", label="G(r) Data")
    ax.plot(r, gcalc, lw=1.3, label="G(r) Fit")
    ax.plot(r, diff, lw=1.2, label="G(r) diff")
    ax.plot(r, diffzero, lw=1.0, ls="--", c="black")
    ax.set_xlabel(r"r ($\mathrm{\AA}$)")
    ax.set_ylabel(r"G ($\mathrm{\AA}$$^{-2}$)")
    if idx:
        ax.set_title(idx)
    ax.tick_params(axis="both", which="major", top=True, right=True)
    ax.set_xlim(pdf_rmin, pdf_rmax)
    ax.legend()
    plt.tight_layout()
    plt.show()
    if figname:
        fig.savefig(figname, format="pdf")

def printresults(recipe, sigfigs=4):
    for name, value in zip(recipe.names, recipe.values):
        print("{}: {}".format(name, "%.*g" % (sigfigs, value)))
    print("Rw: {}".format("%.*g" % (sigfigs, FitResults(recipe).rw)))

def makerecipe(stru1, data_file):

    pdfprofile = Profile()
    pdfparser = PDFParser()
    pdfparser.parseFile(data_file)
    pdfprofile.loadParsedData(pdfparser)
    pdfprofile.setCalculationRange(xmin=pdf_rmin, xmax=pdf_rmax, dx=pdf_rstep)

    pdfgenerator_1 = PDFGenerator("G1")
    pdfgenerator_1.setQmin(qmin)
    pdfgenerator_1.setQmax(qmax)
    pdfgenerator_1.setStructure(stru1, periodic=True)
    pdfgenerator_1.qdamp.value = qdamp
    pdfgenerator_1.qbroad.value = qbroad

    # for atom in pdfgenerator_1.phase.getScatterers():
    #     if atom.element == "Pd" or atom.element == "Cu":
    #         atom.Uiso.value = 0.01
    #     else:
    #         atom.Uiso.value = 0.5

    pdfcontribution = FitContribution("pdf")
    pdfcontribution.setProfile(pdfprofile, xname="r")
    pdfcontribution.addProfileGenerator(pdfgenerator_1)

    from diffpy.srfit.pdf.characteristicfunctions import sphericalCF
    pdfcontribution.registerFunction(sphericalCF, name="f1", argnames=["r", "psize1"])

    pdfcontribution.setEquation("s1*G1*f1")

    if run_parallel:
        import psutil
        import multiprocessing
        syst_cores = multiprocessing.cpu_count()
        cpu_percent = psutil.cpu_percent()
        avail_cores = np.floor((100 - cpu_percent) / (100.0 / syst_cores))
        ncpu = int(np.max([1, avail_cores]))
        pdfgenerator_1.parallel(ncpu)

    recipe = FitRecipe()
    recipe.addContribution(pdfcontribution)
    recipe.clearFitHooks()

    sig = 1e-4  # parameter for degree of restrain

    recipe.addVar(pdfcontribution.s1, value=initial_values_0['s1_i'], tag="s1")
    recipe.restrain("s1", lb=0.0, sig=sig)

    recipe.addVar(pdfcontribution.psize1, value=initial_values_0['psize1_i'], tag="psize1")
    recipe.restrain("psize1", lb=15, sig=sig)

    from diffpy.srfit.structure import constrainAsSpaceGroup
    spacegrouppars_1 = constrainAsSpaceGroup(pdfgenerator_1.phase, spacegroup=spacegroup_1)
    for par in spacegrouppars_1.latpars:
        recipe.addVar(par, value=initial_values_0['a1_i'], fixed=False, name="a1", tag="a1")
    # latdev = 0.03  # ± restrained deviation for lattice constant form initial value
    # recipe.restrain("a1", lb=a1_i * (1 - latdev), ub=a1_i * (1 + latdev), sig=sig)

    def initial_uiso(element):
        if element == "Pd" or element == "Cu":
            return 0.01
        else:
            return 0.35
    for par in spacegrouppars_1.adppars:
        element = par.par.obj.element
        name = element + "_" + par.par.name + "_uiso1"
        recipe.addVar(par, value=initial_uiso(par.par.obj.element), fixed=False, name=name, tag=element + "_uiso1")
        recipe.restrain(name, lb=0.0001, ub=0.5, sig=sig)

    recipe.addVar(pdfgenerator_1.delta2, name="delta21", value=initial_values_0['delta21_i'], tag="d21")
    recipe.restrain("delta21", lb=1.5, ub=10, sig=sig)

    recipe.addVar(spacegrouppars_1.scatterers[0].occ, value=initial_values_0['N_occ_i'], fixed=False, name="N_occ", tag="N_occ1")
    recipe.restrain("N_occ", lb=0.0, ub=1.0, sig=sig)

    return recipe

def newdata_to_recipe(recipe, data_file):

    pdfprofile = Profile()
    pdfparser = PDFParser()
    pdfparser.parseFile(data_file)
    pdfprofile.loadParsedData(pdfparser)
    pdfprofile.setCalculationRange(xmin=pdf_rmin, xmax=pdf_rmax, dx=pdf_rstep)

    recipe.pdf.setProfile(pdfprofile, xname="r")

    return recipe


def main(data_file, prev_recipe=False, idx=None, verbose=2):
    """
    :param data_file: (str) pdf pattern to refine
    :param prev_recipe: False --> do refinement with initial values as defined in dictionary initial_values_0
                        recipe object --> take initial parameters from a previous refinement for sequentialization
    :param idx: (int) index of datafile in list of data_files to refine
    :param verbose: (int) manage the output:
    0: no output,
    1: print and plot after every refinement iteration but dont safe results,
    2: only safe results and plot final refinement of every idx and
    3: printing and plotting and saving.
    :return:
    """
    # make folders to save fit result
    resdir = os.path.join(os.getcwd(), fit_id, "res")
    fitdir = os.path.join(os.getcwd(), fit_id, "fit")
    figdir = os.path.join(os.getcwd(), fit_id, "fig")
    folders = [resdir, fitdir, figdir]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    if not prev_recipe:
        recipe = makerecipe(stru1, data_file)
    else:
        recipe = newdata_to_recipe(prev_recipe, data_file)

    idx = idx if idx else np.nan
    if idx:
        print("\n")
        print("===================================================")
        print("======================== {} =======================".format(idx))

    def refine(par_tags, included_phases, verbose=2):
        if par_tags == "all" or "all" in par_tags:
            recipe.free("all")
        else:
            recipe.fix("all")
            for par_tag in par_tags:
                for phase_num in included_phases:
                    if par_tag.startswith("uiso"):  # free every uiso
                        uiso_par_tags = [uiso_par_tag for uiso_par_tag in recipe.fixednames if "uiso" in uiso_par_tag]
                        for uiso_par_tag in uiso_par_tags:
                            recipe.free(uiso_par_tag)
                    else:
                        try:
                            recipe.free(par_tag + phase_num)
                        except ValueError:
                            print("Ooops, tag <{}> not assigned!".format(par_tag + phase_num))
        if verbose == 1 or verbose == 3:
            print("---------------------------------------------------")
            print("before fit iteration")
            printresults(recipe)
            t_before_fitit = time()
        least_squares(recipe.residual, recipe.values, x_scale="jac", ftol=1e-6)
        if verbose == 1 or verbose == 3:
            print("after fit iteration")
            printresults(recipe)
            print("fit iteration run time {} s".format(int(round(time() - t_before_fitit, 0))))
            plotresults(recipe)

    def refine_wo_nuiso(par_tags, included_phases, verbose=2, thresh_n_occ=0.01):
        if par_tags == "all" or "all" in par_tags:
            recipe.free("all")
        else:
            recipe.fix("all")
            for par_tag in par_tags:
                for phase_num in included_phases:
                    if par_tag.startswith("uiso"):  # free every uiso
                        uiso_par_tags = [uiso_par_tag for uiso_par_tag in recipe.fixednames if "uiso" in uiso_par_tag]
                        for uiso_par_tag in uiso_par_tags:
                            recipe.free(uiso_par_tag)
                    else:
                        try:
                            recipe.free(par_tag + phase_num)
                        except ValueError:
                            print("Ooops, tag <{}> not assigned!".format(par_tag + phase_num))
        if verbose == 1 or verbose == 3:
            print("---------------------------------------------------")
            print("before fit iteration")
            printresults(recipe)
            t_before_fitit = time()
        if recipe.N_occ.value <= thresh_n_occ:
            recipe.fix("N_Uiso_uiso1")
        least_squares(recipe.residual, recipe.values, x_scale="jac", ftol=1e-6)
        if verbose == 1 or verbose == 3:
            print("after fit iteration")
            printresults(recipe)
            print("fit iteration run time {} s".format(int(round(time() - t_before_fitit, 0))))
            plotresults(recipe)



    refine(["s"], ["1"], verbose=verbose)

    included_phases_0 = ["1"]

    refine(["s", "psize"], included_phases_0, verbose=verbose)

    refine(["s", "a"], included_phases_0, verbose=verbose)

    refine(["s", "N_occ"], included_phases_0, verbose=verbose)

    # refine(["s", "uiso"], included_phases_0, verbose=verbose)
    refine(["s", "Pd_uiso", "Cu_uiso"], included_phases_0, verbose=verbose)

    refine(["s", "d2"], included_phases_0, verbose=verbose)

    # refine(["s", "psize", "a", "Pd_uiso", "Cu_uiso", "d2"], included_phases_0, verbose=verbose)

    # refine(["s", "psize", "a", "uiso", "d2", "N_occ"], included_phases_0, verbose=verbose)
    refine(["s", "psize", "a", "Pd_uiso", "Cu_uiso", "d2", "N_occ"], included_phases_0, verbose=verbose)


    # thresh_n_occ = 0.01
    #
    # refine(["s"], ["1"], verbose=verbose)
    #
    # included_phases_0 = ["1"]
    #
    # refine(["s", "psize"], included_phases_0, verbose=verbose)
    #
    # refine(["s", "a"], included_phases_0, verbose=verbose)
    #
    # refine(["s", "N_occ"], included_phases_0, verbose=verbose)
    #
    # refine(["s", "Pd_uiso", "Cu_uiso"], included_phases_0, verbose=verbose)
    #
    # refine_wo_nuiso(["s", "uiso"], included_phases_0, verbose=verbose)
    #
    # refine(["s", "d2"], included_phases_0, verbose=verbose)
    #
    # # refine(["s", "psize", "a", "Pd_uiso", "Cu_uiso", "d2"], included_phases_0, verbose=verbose)
    #
    # refine_wo_nuiso(["s", "psize", "a", "uiso", "d2", "N_occ"], included_phases_0, verbose=verbose)
    # # refine(["s", "psize", "a", "Pd_uiso", "Cu_uiso", "d2", "N_occ"], included_phases_0, verbose=verbose)


    if verbose == 2 or verbose == 3:
        print("---------------------- Final ----------------------")
        plotresults(recipe, os.path.join(figdir, fit_id + "_{}".format(idx) + ".pdf"))
        res = FitResults(recipe)
        res.printResults()
        res.saveResults(os.path.join(resdir, fit_id + "_{}".format(idx) + ".res"))
        profile = recipe.pdf.profile
        profile.savetxt(os.path.join(fitdir, fit_id + "_{}".format(idx) + ".fit"))
    print("===================================================")

    return recipe


if __name__ == "__main__":
    start_number = 13
    end_number = 111

    # start_number = 14
    # end_number = start_number

    verbose = 2
    t0 = time()
    tprevious = t0
    prev_recipe = False  # sequential refinement

    if not os.path.exists(os.path.join(os.getcwd(), fit_id)):
        os.makedirs(os.path.join(os.getcwd(), fit_id))
    shutil.copy2(os.path.realpath(__file__), os.path.join(os.getcwd(), fit_id, fit_id + ".py"))  # save source code

    for idx, data_file in enumerate(data_files[start_number:end_number+1]):
        idx = idx + start_number
        prev_recipe = main(data_file, prev_recipe=prev_recipe, idx=idx, verbose=verbose)
        print("run time idx {}: {} s".format(idx, int(round(time() - tprevious, 0))))
        tprevious = time()
    print("total run time: {} s".format(int(round(time() - t0, 0))))
    print("done.....nice")

