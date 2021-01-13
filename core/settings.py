
import os

maindir = os.path.dirname(os.path.dirname(__file__))

# enviroment variables
os.environ["DIR_INSTALL"] = maindir

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
HIGHTHRESHOLD=None
LOWTHRESHOLD=None
GRIDSEARCH=False
MULTICLASS=False
NPARA=None
PROBACUTOFF=None
SAVEMODEL=False
SAVEPRED=False

VARS=None

# These are used within the workflow, somewhere
N=0
NAMES=None
workdir=''

