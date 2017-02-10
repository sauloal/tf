#!/usr/bin/env python

import os
import sys
import math
import struct
import json
import socket
import time

from collections import defaultdict, OrderedDict

__CTIME__          = time.strftime("%c")
__HOSTNAME__       = socket.gethostname()
__VERSION__        = '1.0'

__DEBUG__          = False

NAME_PADDING_LEN   =    20
DEFAULT_BLOCKSIZE  = 50000
DEFAULT_OUT_DIR    = "data"

replacer = {
	'A': chr(int(math.pow(2, 1))),
	'a': chr(int(math.pow(2, 1))),
	'C': chr(int(math.pow(2, 3))),
	'c': chr(int(math.pow(2, 3))),
	'G': chr(int(math.pow(2, 5))),
	'g': chr(int(math.pow(2, 5))),
	'T': chr(int(math.pow(2, 7))),
	't': chr(int(math.pow(2, 7))),
}

rcer = {
	'A': 'T',
	'a': 'T',
	'C': 'G',
	'c': 'G',
	'G': 'C',
	'g': 'C',
	'T': 'A',
	't': 'A',
}

#print replacer
#print [(k,ord(r)) for k, r in replacer.iteritems()]
#quit()

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
			cls        = cls.replace(" ", "_")
			start, end = int(start), int(end)
			gff[chrom][(start, end)] = cls

	#print gff

	return gff

def paseFasta(inFas, gff, blockSize=DEFAULT_BLOCKSIZE, out_dir=DEFAULT_OUT_DIR):
	stats     = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
	filenames = []

	with open(inFas, 'r') as fhd:
		seqName = None
		seq     = None
		for line in fhd:
			line = line.strip()
			if len(line) == 0: continue
			if line[0] == ">":
				if seqName is not None:
					if seqName in gff:
						parseSeq(inFas, seqName, seq, gff, stats, filenames, blockSize=blockSize, out_dir=out_dir)
						#break
				seqName = line[1:].split()[0]
				seq     = []
			else:
				seq.append(line)
		parseSeq(inFas, seqName, seq, gff, stats, filenames, blockSize=blockSize, out_dir=out_dir)

	print

	print "Stats"

	for grp, c1 in sorted(stats.iteritems()):
		print "  {:22s}: {:12,d}".format( grp, sum([ sum([x for x in w.itervalues()]) for w in c1.itervalues() ]) )
		for k2, c2 in sorted(c1.iteritems()):
			print "   {:21s}: {:12,d}".format( k2, sum([v for v in c2.itervalues()]) )
			for k3, v in sorted(c2.iteritems()):
				print "    {:20s}: {:12,d}".format(k3,v)


	outs = {
		"all": [ os.path.join(out_dir, "files_{}.csv".format(os.path.basename(inFas))), None ]
	}

	for cls in stats['by_class']:
		outs[ cls ] = [ os.path.join(out_dir, "files_{}_{}.csv".format(os.path.basename(inFas),cls)), None ]

	for cls, (fn, n) in outs.iteritems():
		print "printing CSV {} for class {}".format(fn, cls)
		outs[ cls ][1] = open( fn, 'w' )

	for fn, cls in filenames:
		outs['all'][1].write(",".join((fn,cls)) + "\n")
		outs[cls  ][1].write(",".join((fn,cls)) + "\n")

	for cls in outs:
		outs[ cls ][1].close()

def rc(seq):
	return "".join([rcer.get(s, 'N') for s in reversed(seq)])

def parseSeq(inputFile, seqName, seq, gff, stats, filenames, blockSize=DEFAULT_BLOCKSIZE, out_dir=DEFAULT_OUT_DIR):
	if seqName is None: return
	if seq     is None: return
	if len(seq) == 0: return

	output_dir = os.path.join(out_dir, os.path.basename(inputFile))

	seq        = "".join(seq)
	total      = len(seq)

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	print "Chrom {} Length {:12,d}".format(seqName, len(seq)), ("*" if seqName in gff else "")
	ser = 0
	if seqName in gff:
		coords = gff[seqName]
		for (start, end), cls in sorted(coords.iteritems()):
			size       = end - start
			num_blocks = int(size % blockSize)
			print " Start {:12,d} End {:12,d} Size {:12,d} Blocks {:12,d} Class {:20s} {}".format( start, end, size, num_blocks, cls, ("*" if size > 1 else "" ))
			if size > 1:
				stats['by_chrom'][seqName][cls] += num_blocks
				stats['by_class'][cls][seqName] += num_blocks
				out_base_name = ("{:_<"+str(NAME_PADDING_LEN)+"s}_{:012d}_{:012d}").format(seqName, start, end)
				print "  class {} seq name {}".format( cls, out_base_name )
				for bn, s in enumerate(xrange(start, end, blockSize / 2)):
					e    = s + blockSize
					l    = e - s
					if e <= end:
						frag      = seq[s:e]
						assert len(frag) == blockSize

						for ext, piece in (('fwd',frag), ('rev',rc(frag))):
							basename  = "{}_{:012d}_{:012d}_{:012d}_{}.seq".format( out_base_name, bn, s, e, ext )
							subdir    = os.path.join(output_dir                 , cls     )
							filename  = os.path.join(os.path.basename(inputFile), cls       , basename)
							outfile   = os.path.join(subdir                     , basename)
							ser      += 1
							if not os.path.exists(subdir):
								os.makedirs(subdir)
							print "   #{:12,d} begin {:12,d} end {:12,d} size {:12,d} total {:12,d} file {}".format(bn, s, e, l, total, outfile)
							filenames.append((filename, cls))
							sequenceHandler(outfile   ,
									inputFile  = inputFile,
									blockSize  = blockSize,
									seqName    = seqName  ,
									serial     = ser      ,
									seg_serial = bn       ,
									seg_start  = start    ,
									seg_end    = end      ,
									start      = s        ,
									end        = e        ,
									group      = cls).write(toOneHot(frag ))
				#break
				pass

class group_id_generator(object):
	def __init__(self):
		self.dic = OrderedDict()

	def __getitem__(self, key):
		if   key is None:
			return None

		elif key not in self.dic:
			self.dic[key] = len(self.dic)

		return self.dic[key]

gid = group_id_generator()

class sequenceHandler(object):
	keys = ['inputFile' , 'blockSize' , 'seqName'   , 'serial'    ,
			'seg_serial', 'seg_start' , 'seg_end'   , 'start'     ,
			'end'       , 'group'     , 'groupId'   ,
			'ctime'     , 'host'      , 'version']

	def __init__(self, dbFile,
				 inputFile  = None,
				 blockSize  = None,
				 seqName    = None,
				 serial     = None,
				 seg_serial = None,
				 seg_start  = None,
				 seg_end    = None,
				 start      = None,
				 end        = None,
				 group      = None,
				 verbose    = False):
		self.dbFile     = dbFile
		self.inputFile  = os.path.abspath(inputFile) if inputFile is not None else None
		self.blockSize  = blockSize
		self.seqName    = seqName
		self.serial     = serial
		self.seg_serial = seg_serial
		self.seg_start  = seg_start
		self.seg_end    = seg_end
		self.start      = start
		self.end        = end
		self.group      = group
		self.groupId    = gid[group]
		self.verbose    = verbose
		self.ctime      = __CTIME__
		self.host       = __HOSTNAME__
		self.version    = __VERSION__

		if __DEBUG__:
			self.verbose = True

	def asDict(self):
		return dict([(k,getattr(self, k)) for k in self.keys])

	def _updateDict(self, msg):
		for k in self.keys:
			setattr(self, k, msg[k])

		if self.groupId is None:
			self.groupId = gid(self.group)

	def write(self, frag):
		if self.verbose:
			print "writing", self.dbFile

		assert self.blockSize == len(frag)

		with open(self.dbFile, 'wb') as fhd:
			fhd.write(frag)

		with open(self.dbFile+'.json', 'wb') as fhd:
			self._genHeader(fhd)

		if __DEBUG__:
			self.read()

	def read(self):
		frag = None

		if self.verbose:
			print "reading", self.dbFile

		with open(self.dbFile + '.json', 'rb') as fhd:
			self._readHeader(fhd)

		with open(self.dbFile, 'rb') as fhd:
			frag = fhd.read()
			assert self.blockSize == len(frag)

		return frag

	def _genHeader(self, fhd):
		msg     = self.asDict()
		j       = json.dumps(msg, separators=(',',':'), indent=None, sort_keys=True)
		# msgLen  = len(j)
		# txt     = struct.pack('Q', msgLen)
		# txt    += j
		txt    = j

		if self.verbose:
			print "saving header", j

		fhd.write(txt)

	def _readHeader(self, fhd):
		# msgLen  = struct.unpack('Q', fhd.read(8))[0]

		#print "msgLen", msgLen

		# j       = fhd.read(msgLen)
		j       = fhd.read()

		#print "J", j

		msg     = json.loads(j)

		if self.verbose:
			print "loaded header", j

		self._updateDict(msg)

		assert self.version == __VERSION__

def toOneHot(frag):
	return "".join([replacer.get(f, '\x00') for f in frag])

def splitFasta(inGff, inFas, blockSize=DEFAULT_BLOCKSIZE, out_dir=DEFAULT_OUT_DIR):
	gff = parseGff(inGff)
	paseFasta(inFas, gff, blockSize=blockSize, out_dir=out_dir)

def main():
	inGff      =     sys.argv[1]
	inFas      =     sys.argv[2]
#	blockSize = int(sys.argv[3])
	splitFasta(inGff, inFas, blockSize=DEFAULT_BLOCKSIZE, out_dir=DEFAULT_OUT_DIR)

if __name__ == '__main__':
	main()
