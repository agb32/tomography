import sys
import util.dm
util.dm.dmProjectionQuick("params.py",rmx="rmxTomo.fits",rmxOutName="rmx.fits",reconIdStr="recon",initDict={"ndm":int(sys.argv[1])})
