
Usage = """NAME.py INPUT OUTPUT"""
import os
import sys
import re
from Bio.Blast import NCBIXML
from Bio import SeqIO

max_length = int(os.getenv('MAX_LENGTH', 80))

transFile = sys.argv[2]
seqs = {}
for seq_record in SeqIO.parse(transFile, "fasta"):
    seqs[seq_record.id] = seq_record.seq._data


aa_position_file = sys.argv[4]
aa_position_handle = open(aa_position_file, "w")  #


def coverage_query_alignment_obj(alignmentObj, e_value_thresh=1000000):
    this_matches = 0
    for hsp in alignmentObj.hsps:
        if hsp.expect <= e_value_thresh:
            this_query = blast_record.query_length
            this_matches += hsp.identities
    if this_matches > 0:
        query_percent = (this_matches * 100.) / this_query
        query_percent = round(query_percent, 2)
        subject_percent = (this_matches * 100.) / alignment.length
        subject_percent = round(subject_percent, 2)

    coverage_percent = max(query_percent, subject_percent)
    return coverage_percent



def identity_query_alignment_obj(hsp, e_value_thresh=1000000):
    if hsp.expect <= e_value_thresh:
        this_percent = hsp.identities * 100 / hsp.align_length
    return this_percent


def adj(step, up, aa, down, up_start, aa_start, down_start):
    if type(up) == bytes:
        up = up.decode('utf-8')
    if type(aa) == bytes:
        aa = aa.decode('utf-8')
    if type(down) == bytes:
        down = down.decode('utf-8')

    newup = up.upper()
    newaa = aa.upper()
    newdown = down.upper()

    newaa_start = aa_start
    newaa_end = aa_start + len(newaa)

    if step > -3:
        step = 5
    else:
        step = abs(step)

    gt = [i.start() for i in re.finditer("TG", ''.join(reversed(up[-step:])))]

    ag = [i.start() for i in re.finditer("AG", down[:step])]
    aa_gt = [i.start() for i in re.finditer("GT", aa[:step])]
    aa_ag = [i.start() for i in re.finditer("GA", ''.join(reversed(aa[-step:])))]

    if gt:
        if ag:
            newup = up[:-gt[0] - 2]
            newdown = down[ag[0] + 2:]
            newaa = up[-gt[0] - 2:] + aa + down[:ag[0] + 2]
            # Update the start and end positions of newaa
            newaa_start = up_start + len(up) - gt[0] - 2
            newaa_end = newaa_start + len(newaa)
        else:
            newup = up[:-gt[0] - 2]
            newdown = down
            newaa = up[-gt[0] - 2:] + aa
            # Update the start and end positions of newaa
            newaa_start = up_start + len(up) - gt[0] - 2
            newaa_end = newaa_start + len(newaa)
    else:
        if ag:
            newup = up
            newdown = down[ag[0] + 2:]
            newaa = aa + down[:ag[0] + 2]
            # Update the start and end positions of newaa
            newaa_start = aa_start
            newaa_end = newaa_start + len(newaa)
        else:
            if aa_gt:
                if aa_ag:
                    if len(aa) > 4 and aa_ag[0] != 0:
                        newup = up + aa[:aa_gt[0]]
                        newdown = aa[-aa_ag[0]:] + down
                        newaa = aa[aa_gt[0]:-aa_ag[0]]
                        # Update the start and end positions of newaa
                        newaa_start = aa_start + aa_gt[0]
                        newaa_end = newaa_start + len(newaa)
                else:
                    newup = up + aa[:aa_gt[0]]
                    newdown = down
                    newaa = aa[aa_gt[0]:]
                    # Update the start and end positions of newaa
                    newaa_start = aa_start + aa_gt[0]
                    newaa_end = newaa_start + len(newaa)
            else:
                if aa_ag:
                    if aa_ag[0] != 0:
                        newup = up
                        newdown = aa[-aa_ag[0]:] + down
                        newaa = aa[:-aa_ag[0]]
                        # Update the start and end positions of newaa
                        newaa_start = aa_start
                        newaa_end = newaa_start + len(newaa)

    if newup[-1:] + newaa[:1] == "GT":
        newup = newup[:-1]
        newaa = "G" + newaa
    if newaa[-1:] + newdown[:1] == "AG":
        newaa = newaa + "G"
        newdown = newdown[1:]

    return newup + "," + newaa + "," + newdown, newaa_start, newaa_end


def compare(thishsp, lasthsp):
    if lasthsp:
        (thisqStart, thisqEnd, thisqName, thissName, thissStart, thissEnd, thisdiffslen, thisidentity, thiscoverage,
         thisqlen, thisslen) = thishsp
        (lastqStart, lastqEnd, lastqName, lastsName, lastsStart, lastsEnd, lastdiffslen, lastidentity, lastcoverage,
         lastqlen, lastslen) = lasthsp
        if int(thisdiffslen) > 0 and int(lastdiffslen) > 0:
            if int(thisqEnd) > int(lastqEnd) and int(thissEnd) > int(lastsEnd):
                gapQ = int(thisqStart) - int(lastqEnd)
                gapS = int(thissStart) - int(lastsEnd)
                mingap = min(gapQ, gapS)
                maxgap = max(gapQ, gapS)
                diffgap = maxgap - mingap
                if mingap <= 1 and mingap >= -10 and diffgap > 1 and diffgap <= 500:
                    startq = 1
                    starts = 1
                    if int(lastqEnd) - max_length > 1:
                        startq = int(lastqEnd) - max_length
                    if int(lastsEnd) - max_length > 1:
                        starts = int(lastsEnd) - max_length
                    if gapQ > gapS:
                        if gapQ <= 0:
                            newseq, newaa_start, newaa_end = adj(
                                mingap,
                                seqs[thisqName][startq:int(lastqEnd)],
                                seqs[thisqName][int(lastqEnd):int(lastqEnd) + diffgap],
                                seqs[thisqName][int(lastqEnd) + diffgap:int(lastqEnd) + diffgap + max_length],
                                startq,
                                int(lastqEnd),
                                int(lastqEnd) + diffgap
                            )
                            aa_position_handle.write(f"{thisqName}+{thissName},{newaa_start}-{newaa_end}\n")
                            return thisqName + "+" + thissName + "," + newseq
                        else:
                            newseq, newaa_start, newaa_end = adj(
                                mingap,
                                seqs[thisqName][startq:int(lastqEnd)],
                                seqs[thisqName][int(lastqEnd):int(thisqStart)],
                                seqs[thisqName][int(thisqStart):int(thisqStart) + max_length],
                                startq,
                                int(lastqEnd),
                                int(thisqStart)
                            )
                            aa_position_handle.write(f"{thisqName}+{thissName},{newaa_start}-{newaa_end}\n")
                            return thisqName + "+" + thissName + "," + newseq
                    else:
                        if gapS <= 0:
                            newseq, newaa_start, newaa_end = adj(
                                mingap,
                                seqs[thissName][starts:int(lastsEnd)],
                                seqs[thissName][int(lastsEnd):int(lastsEnd) + diffgap],
                                seqs[thissName][int(lastsEnd) + diffgap:int(lastsEnd) + diffgap + max_length],
                                starts,
                                int(lastsEnd),
                                int(lastsEnd) + diffgap
                            )
                            aa_position_handle.write(f"{thisqName}+{thissName},{newaa_start}-{newaa_end}\n")
                            return thisqName + "+" + thissName + "," + newseq
                        else:
                            newseq, newaa_start, newaa_end = adj(
                                mingap,
                                seqs[thissName][starts:int(lastsEnd)],
                                seqs[thissName][int(lastsEnd):int(thissStart)],
                                seqs[thissName][int(thissStart):int(thissStart) + max_length],
                                starts,
                                int(lastsEnd),
                                int(thissStart)
                            )
                            aa_position_handle.write(f"{thisqName}+{thissName},{newaa_start}-{newaa_end}\n")
                            return thisqName + "+" + thissName + "," + newseq
                else:
                    return False
    else:
        return False


def sort_key(s):
    if s:
        try:
            c = re.findall('^\d+', s)[0]
        except:
            c = -1
        return int(c)


def strsort(alist):
    alist.sort(key=sort_key)
    return alist


inPutFile = sys.argv[1]
outPutFile = sys.argv[3]
result_handle = open(inPutFile, "r")
out_handle = open(outPutFile, "w")
hspsss = open("hsp.txt", "w")
allinfo = open("allinfo.txt", "w")
cover = open("coveragebotname.txt", "w")


header = ['QueryName', 'SubjectName', 'QhStart', 'QhEnd', 'ShStart', 'ShEnd', 'hsplength', 'QueryIdentity',
          'queryLength', 'subjectLength', 'HSP']
out_handle.write("\t".join(header) + '\n')


Blast_records = NCBIXML.parse(result_handle)


for blast_record in Blast_records:
    queryName = blast_record.query
    queryLength = int(blast_record.query_length)
    subjectName = ""
    subjectLength = ""
    queryIdentityPercent = 0
    for alignment, description in zip(blast_record.alignments, blast_record.descriptions):
        hspNum = int(len(alignment.hsps))
        subjectName = alignment.title
        subjectNames = subjectName.split(' ')
        subjectName = subjectNames[1]
        subjectLength = int(alignment.length)
        querycoveragePercent = coverage_query_alignment_obj(alignment)
        breakflag = 0
        #
        if queryName != subjectName and hspNum > 1 and querycoveragePercent > 70:
            if queryLength >= 100 and subjectLength >= queryLength:
                allhsp = []
                lasthsp = []
                thishsp = []
                for hsp in alignment.hsps:
                    queryIdentityPercent = identity_query_alignment_obj(hsp)
                    hspLength = hsp.align_length
                    qhStart = hsp.query_start
                    qhEnd = hsp.query_end
                    shStart = hsp.sbjct_start
                    shEnd = hsp.sbjct_end
                    if queryIdentityPercent > 90:
                        hspList = [str(qhStart), str(qhEnd), queryName, subjectName, str(shStart), str(shEnd),
                                   str(hspLength), str(round(queryIdentityPercent, 2)),
                                   str(round(querycoveragePercent, 2)), str(queryLength), str(subjectLength)]
                        hspstr = '\t'.join(hspList)
                        allhsp.append(hspstr)
                    else:
                        breakflag = 1
                allhsp_sort = strsort(allhsp)
                print(queryName, subjectName, allhsp_sort, file=hspsss)
                if breakflag:
                    break

                if len(allhsp_sort) > 1:
                    as_num = 0
                    for hsp in allhsp_sort:
                        thishsp = hsp.split('\t')
                        figure = compare(thishsp, lasthsp)
                        if figure:
                            as_num += 1

                            thishsp[0], thishsp[1], thishsp[2], thishsp[3] = thishsp[2], thishsp[3], thishsp[0], \
                                thishsp[1]
                            lasthsp[0], lasthsp[1], lasthsp[2], lasthsp[3] = lasthsp[2], lasthsp[3], lasthsp[0], \
                                lasthsp[1]

                            out_handle.write(
                                "\t".join(lasthsp) + '\thsp1\t' + figure + '\n' + "\t".join(thishsp) + '\thsp2\n')

                            thishsp[2], thishsp[3], thishsp[0], thishsp[1] = thishsp[0], thishsp[1], thishsp[2], \
                                thishsp[3]
                            lasthsp[2], lasthsp[3], lasthsp[0], lasthsp[1] = lasthsp[0], lasthsp[1], lasthsp[2], \
                                lasthsp[3]
                            print(figure, sep=",")
                        lasthsp = thishsp


result_handle.close()
out_handle.close()
aa_position_handle.close()
