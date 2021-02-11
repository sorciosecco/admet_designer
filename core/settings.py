
import os, sys

maindir = os.path.dirname(os.path.dirname(__file__))
python_path = sys.executable

# enviroment variables
os.environ["DIR_INSTALL"] = maindir
os.environ["MD_LICENSE"] = maindir
os.environ["MD_LICENSE_FILE"] = "md_licenses.txt"

# executables
exec_bla = os.path.join(maindir, 'moka_suite', 'blabber_sd')
exec_tau = os.path.join(maindir, 'moka_suite', 'tauthor')
exec_vs3 = os.path.join(maindir, 'volsurf3', 'volsurf3_cli.sh')
exec_bab = "/".join(python_path.split("/")[:-1]+["babel"])

# These will be set during the first call of the main function
FIT=None
PREDICT=None
RESPONSE=None
MODEL=None
SEED=None
SAVEVARS=None
VERBOSE=None

# These will be set during the first call of the specific sub-function
BALANCE=False
CROSSVAL=None
LATENT=None
METHOD=None
NUMBER=None
PERCENTAGE=None
STRATEGY=False

BACKFEEL=False
CPUS=None
ERROR=False
HIGHMW=None
HIGHNA=None
HIGHRESP=None
HIGHTHRESHOLD=None
INFILE=None
LOWMW=None
LOWNA=None
LOWRESP=None
LOWTHRESHOLD=None
GRIDSEARCH=False
MULTICLASS=False
NPARA=None
OUTFILE=None
PROBACUTOFF=None
SAVEMODEL=False
SAVEPRED=False

VARS=None
<<<<<<< HEAD
#RESPONSE_DICT=None
=======
>>>>>>> calc_vsplus_descriptors

# These are used within the workflow, somewhere
N=0
PROG=0
NAMES=None
workdir=''

