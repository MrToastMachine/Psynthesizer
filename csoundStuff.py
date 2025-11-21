import ctcsound
import time

csd = """
<CsoundSynthesizer>
<CsOptions>
-odac
</CsOptions>
<CsInstruments>
instr 1
    aSig oscil 0.5, 440
    out aSig
endin
</CsInstruments>
<CsScore>
i1 0 2
</CsScore>
</CsoundSynthesizer>
"""

cs = ctcsound.Csound()
cs.compileCsdText(csd)
cs.start()
cs.perform()
time.sleep(2.5)
cs.stop()
cs.cleanup()
