
import os, subprocess

def execute(command, outfile, foldir=os.getcwd()):
    try:
        subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        if outfile in os.listdir(foldir):
            if os.stat(os.path.join(foldir, outfile)).st_size == 0: returncode=-3
            else: returncode=0
        else:
            returncode=-2
    except:
        returncode=-1
    return returncode

def split_multisdf(isdf):
    L=[]
    name=isdf.split(".")[0]
    n = len([1 for line in open(isdf, "r") if line.find("$$$$") != -1])
    infile=open(isdf, "r")
    for i in range(n):
        outfile=open('%s_%0.10d.sdf' % (name, i), "w")
        while 13:
            line=infile.readline()
            outfile.write(line)
            if line.strip() == "$$$$":
                break
        outfile.close()
        L.append('%s_%0.10d.sdf' % (name, i))
    infile.close()
    return L
