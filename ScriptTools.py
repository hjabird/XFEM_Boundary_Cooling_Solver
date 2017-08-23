

print(
"""


 .M""'bgd `7MM""'Mq.         `YMM'   `MP' `7MM"'"YMM `7MM"'"YMM  `7MMM.     ,MMF'
,MI    "Y   MM   `MM. __,      VMb.  ,P     MM    `7   MM    `7    MMMb    dPMM
`MMb.       MM   ,M9 `7MM       `MM.M'      MM   d     MM   d      M YM   ,M MM
  `YMMNq.   MMmmdM9    MM         MMb       MM""MM     MMmmMM      M  Mb  M' MM
.     `MM   MM         MM mmmmm ,M'`Mb.     MM   Y     MM   Y  ,   M  YM.P'  MM
Mb     dM   MM         MM      ,P   `MM.    MM         MM     ,M   M  `YM'   MM
P"Ybmmd"  .JMML.     .JMML.  .MM:.  .:MMa..JMML.     .JMMmmmmMMM .JML. `'  .JMML.



""")
print("SP1-XFEM: A XFEM based approximation to heat transfer with\n\
an SP1 radiation approximation.")
print("Copyright Hugh Bird 2017")
print("Experimental software")
print()

print()
print()

# REDEF PRINT
old_print = print
try:
    import inspect
    def ln_print(*kwarg):
        ln_num = inspect.currentframe().f_back.f_lineno
        old_print("Script line " + str(ln_num)+":\t", end="")
        old_print(*kwarg)
    print=ln_print
except:
    pass
    
