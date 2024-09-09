import json 


f1 = 0.627


inv_pr = 2/f1
inv_rec = inv_pr-2
rec = 1 / inv_rec

print(inv_pr, inv_rec, rec)

print(2 / (1 / rec + 1))