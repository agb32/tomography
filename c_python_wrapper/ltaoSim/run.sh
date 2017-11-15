
#!/bin/sh
python ltaoPoke.py params.py --param="this.tomoRecon.abortAfterPoke=1;this.tomoRecon.reconmxFilename='rmxTomo.fits'" --init="ndm=1"
python projection.py 1
python ltao.py params.py
