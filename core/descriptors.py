
import os, shutil, tempfile
import multiprocessing as mp

from core import settings
from core.tools import split_multisdf
from core.commands import *
from core.vs3 import run_volsurf3


m, q = mp.Manager(), mp.Queue()
dir_correct, edirs_list = os.path.join(os.getcwd(), "CORRECT"), [os.path.join(os.getcwd(), "E-tauthor"), os.path.join(os.getcwd(), "E-blabber"), os.path.join(os.getcwd(), "E-volsurf")]


class Process:
    
    def __init__(self, infile, tmpdir, error_dirs):
        self.infile=infile
        self.tmpdir=tmpdir
        self.error_dirs=error_dirs
        self.molname=open(self.infile, 'r').readline().strip()
        self.TAGS=Process.get_moltags(self)
        self.errorcode=0
        self.message=""
        self.outfile=""
    
    def get_moltags(self):
        if self.infile.endswith(".sdf"):
            L=open(self.infile, 'r').readlines()
            mol_tags=[(L[i].rstrip(">\n").split("<")[1], L[i+1].strip()) for i in range(len(L)) if L[i].startswith(">")]
        return mol_tags
        
    def tauthor(self):
        isdf, osdf = self.infile, self.infile.replace(".sdf", "t.sdf")
        returncode = run_tauthor(infile=isdf, outfile=osdf, wdir=self.tmpdir)
        if returncode==0:
            self.outfile = osdf
            #with open(self.outfile, 'r') as osdf2:
                #lines=osdf2.readlines()
                #self.TAGS.append(("T%_H2O-pH7.4", [lines[l+1].strip() for l in range(len(lines)) if lines[l].find("water%") != -1][0]))
        else:
            self.errorcode=1
            if settings.ERROR: shutil.copy(isdf, self.error_dirs[0])
        self.message += "\tt=%s" % (self.errorcode)
        self.infile = osdf
    
    def blabber(self):
        if self.errorcode==0:
            isdf, osdf = self.infile, self.infile.replace(".sdf", "p.sdf")
            returncode = run_blabber(infile=isdf, outfile=osdf, wdir=self.tmpdir)
            if returncode==0:
                self.outfile = osdf
                #with open(self.outfile, 'r') as osdf2:
                    #lines=osdf2.readlines()
                    #try:
                        #abd = float([lines[l+1].split('%')[0] for l in range(len(lines)) if lines[l].find("ABUNDANCE") != -1][0])
                    #except ValueError:
                        #abd = 100
                    #self.TAGS.append(("P%_H2O-pH7.4", str(abd)))
            else:
                self.errorcode=1
                if settings.ERROR: shutil.copy(isdf, self.error_dirs[1])
            self.infile = osdf
        self.message += "\tp=%s" % (self.errorcode)
    
    def babel(self):
        if self.errorcode==0:
            isdf, osdf, osmi = self.infile, self.infile.replace(".sdf", "c.sdf"), self.infile.replace(".sdf", ".smi")
            returncode = run_babel(infile=isdf, outfile=osdf, wdir=self.tmpdir)
            if returncode==0: self.outfile=osdf
                #returncode = obabel(infile=osdf, outfile=osmi, wdir=self.tmpdir, code_n=4)
                #if returncode==0:
                #with open(osmi, 'r') as smi:
                    #line = smi.readline().strip().split('\t')
                    #self.name_smiles = (line[-1], line[0])
                
                #else:
                    #self.errorcode=1
                    #if settings.ERROR:
                        #shutil.copy(isdf, self.error_dirs[2])
            else: self.errorcode=1
                #if settings.ERROR:
                    #shutil.copy(isdf, self.error_dirs[2])
        self.message += "\tb=%s" % (self.errorcode)
    
    def volsurf(self):
        if self.errorcode==0:
            returncode, ocsv, vs_message = run_volsurf3(iname=self.outfile.split('.')[0], name=self.molname, isdf=self.outfile,  wdir=self.tmpdir)
            if returncode==0:
                self.outfile = ocsv.split('/')[-1]
            else:
                self.errorcode=1
                if settings.ERROR: shutil.copy(self.outfile, self.error_dirs[2])
        self.message += "\tv=%s" % (self.errorcode)
    
    def save_vs_results(self, outdir):
        with open(self.outfile, 'r') as f:
            lines = f.readlines()
        header, line = str.split(lines[0].strip(), ';'), str.split(lines[1].strip(), ';')
        header2, line2 = header + [t[0] for t in self.TAGS], line + [t[1] for t in self.TAGS]
        ofile = open(os.path.join(outdir, self.outfile.replace('.sdf','.csv')), 'w')
        ofile.write(';'.join(header2)+'\n')
        ofile.write(';'.join([str(l) for l in line2])+'\n')
        ofile.close()
    
    # list of functions
    lof = [tauthor, blabber, babel, volsurf]


# the function that launches the class process
def run_class_proc(mol):
    origdir, workdir = os.getcwd(), tempfile.mkdtemp(prefix='TMP')
    shutil.move(mol, workdir)
    os.chdir(workdir)
    
    Run=Process(infile=mol, tmpdir=workdir, error_dirs=edirs_list)
    for f in list(range(4)): Run.lof[f](Run)
    
    if Run.errorcode == 0: Run.save_vs_results(outdir=dir_correct)
    
    q.put(1)
    size=q.qsize()
    if settings.VERBOSE == 1:
        print('['+str(round(size*100/settings.N,2))+' %] error codes ==>'+Run.message)
    else:
        prog = int(size*100/settings.N)
        if prog != settings.PROG:
            print('perc. completed: %s => progress %s/%s' % (prog, size, settings.N))
            settings.PROG = prog
    
    os.chdir(origdir)
    shutil.rmtree(workdir)
    

def sdf_main_import_func(args):
    settings.INFILE, settings.OUTFILE, settings.ERROR, settings.CPUS = args.infile, args.outfile, args.error, args.cpus
    
    # define the number of cores to use
    if settings.CPUS==None: ncpus=mp.cpu_count()
    else: ncpus=args.cpus
    
    # create temporary directories and define processes
    os.mkdir(dir_correct)
    if settings.ERROR:
        for d in edirs_list: os.mkdir(d)
    
    # prepare single molfiles
    SDF_FILES = split_multisdf(isdf=settings.INFILE)
    settings.N = len(SDF_FILES)
    
    # run processes on single molfiles
    if ncpus==1:
        i=1
        for mol in SDF_FILES:
            run_class_proc(mol)
            i+=1
    else:
        pool=mp.Pool(ncpus)
        pool.map_async(run_class_proc, SDF_FILES)
        pool.close()
        pool.join()
    
    # save results
    print("\nNow saving...\n")
    i=0
    ofile = open(os.path.join(os.getcwd(), settings.OUTFILE), 'w')
    for file_i in sorted(os.listdir(dir_correct)):
        lines=open(os.path.join(os.getcwd(), dir_correct, file_i), 'r').readlines()
        if i==0:
            for line in lines: ofile.write(line)
        else:
            ofile.write(lines[-1])
        i+=1
    ofile.close()
    shutil.rmtree(dir_correct)
    
    j=0
    if settings.ERROR:
        for d in edirs_list:
            if len(os.listdir(d)) != 0:
                oerr=open(os.path.join(os.getcwd(), d.split('/')[-1]+'.sdf'), 'w')
                for file_i in os.listdir(d):
                    for line in open(os.path.join(os.getcwd(), d, file_i), 'r'):
                        oerr.write(line)
                    j+=1
                oerr.close()
            shutil.rmtree(d)
    
    print('\nInput structures: %s\nCorrect: %s\nErrors: %s\n' % (len(SDF_FILES), i, j))
    
