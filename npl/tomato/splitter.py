#!/usr/bin/env python

import os
import sys
from collections import defaultdict, OrderedDict

NAME_PADDING_LEN=20
DEFAULT_BLOCK_SIZE=50000
DEFAULT_OUT_DIR="data"

def parseGff(inGff):
	gff = defaultdict(OrderedDict)
	with open(inGff, 'r') as fhd:
		for line in fhd:
			line = line.strip()
			if len(line) == 0: continue
			if line[0] == "#": continue
			print line
			#SL2.50ch10      .       Euchromatin     55455388        65527505        .       .       .       .
			chrom, tp, cls, start, end = line.split("\t")[:5]
			cls = cls.replace(" ", "_")
			start, end = int(start), int(end)
			gff[chrom][(start, end)] = cls
	#print gff
	return gff

def paseFasta(inFas, gff, block_size=DEFAULT_BLOCK_SIZE, out_dir=DEFAULT_OUT_DIR):
	stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
	with open(inFas, 'r') as fhd:
		seqName = None
		seq     = None
		for line in fhd:
			line = line.strip()
			if len(line) == 0: continue
			if line[0] == ">":
				if seqName is not None:
					if seqName in gff:
						parseSeq(seqName, seq, gff, stats, block_size=block_size, out_dir=out_dir)
						#break
				seqName = line[1:].split()[0]
				seq     = []
			else:
				seq.append(line)
		parseSeq(seqName, seq, gff, stats, block_size=block_size, out_dir=out_dir)

	print
	print "Stats"
	for grp, c1 in sorted(stats.iteritems()):
		print "  {:22s}: {:12,d}".format( grp, sum([ sum([x for x in w.itervalues()]) for w in c1.itervalues() ]) )
		for k2, c2 in sorted(c1.iteritems()):
			print "   {:21s}: {:12,d}".format( k2, sum([v for v in c2.itervalues()]) )
			for k3, v in sorted(c2.iteritems()):
				print "    {:20s}: {:12,d}".format(k3,v)

def parseSeq(seqName, seq, gff, stats, block_size=DEFAULT_BLOCK_SIZE, out_dir=DEFAULT_OUT_DIR):
	if seqName is None: return
	if seq     is None: return
	if len(seq) == 0: return
	seq   = "".join(seq)
	total = len(seq)
	print "Chrom {} Length {:12,d}".format(seqName, len(seq)), ("*" if seqName in gff else "")
	if seqName in gff:
		coords = gff[seqName]
		for (start, end), cls in sorted(coords.iteritems()):
			size       = end - start
			num_blocks = int(size % block_size)
			print " Start {:12,d} End {:12,d} Size {:12,d} Blocks {:12,d} Class {:20s} {}".format( start, end, size, num_blocks, cls, ("*" if size > 1 else "" ))
			if size > 1:
				stats['by_chrom'][seqName][cls] += num_blocks
				stats['by_class'][cls][seqName] += num_blocks
				out_path = os.path.join(out_dir, cls)
				if not os.path.exists(out_path):
					os.makedirs(out_path)
				out_base_name = ("{:_<"+str(NAME_PADDING_LEN)+"s}_{:012d}_{:012d}").format(seqName, start, end)
				out_base_path = os.path.join(out_path, out_base_name)
				print "  out path {} seq name {}".format( out_path, out_base_name )
				for bn, s in enumerate(xrange(start, end, block_size)):
					e    = s + block_size
					l    = e - s
					if e <= end:
						frag = seq[s:e]
						assert len(frag) == block_size
						outfile = "{}_{:012d}_{:012d}_{:012d}.seq".format( out_base_path, bn, s, e )
						print "   #{:12,d} begin {:12,d} end {:12,d} size {:12,d} total {:12,d} file {}".format(bn, s, e, l, total, outfile)
						open(outfile, 'w').write(frag)
				#break

def splitFasta(inGff, inFas, block_size=DEFAULT_BLOCK_SIZE, out_dir=DEFAULT_OUT_DIR):
	gff = parseGff(inGff)
	paseFasta(inFas, gff, block_size=block_size, out_dir=out_dir)

def main():
	inGff      =     sys.argv[1]
	inFas      =     sys.argv[2]
#	block_size = int(sys.argv[3])
	splitFasta(inGff, inFas, block_size=DEFAULT_BLOCK_SIZE, out_dir=DEFAULT_OUT_DIR)

if __name__ == '__main__':
	main()
